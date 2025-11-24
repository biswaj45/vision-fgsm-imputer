"""
Training script for TinyUNet model.
Includes training loop, checkpointing, and TensorBoard logging.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import yaml
from typing import Dict, Any, Optional
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.unet_tiny import create_tiny_unet
from models.autoencoder import create_autoencoder
from training.dataset import create_dataloaders
from training.transforms import get_training_transforms, get_validation_transforms
from training.utils import fgsm


class Trainer:
    """Trainer class for anti-deepfake model."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = 'auto'
    ):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
            device: Device to train on ('auto', 'cuda', 'cpu')
        """
        self.config = config
        
        # Auto-detect device
        if device == 'auto':
            device_config = config.get('device', 'auto')
            if device_config == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = device_config
        else:
            self.device = device
        
        # Create model
        model_type = config.get('model_type', 'unet')
        if model_type == 'unet':
            self.model = create_tiny_unet().to(device)
        else:
            self.model = create_autoencoder().to(device)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Create output directories
        self.output_dir = Path(config.get('output_dir', './outputs'))
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Enable CUDA optimizations
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Mixed precision training
        self.use_amp = config.get('mixed_precision', False) and self.device == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Mixed precision training enabled")
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        print(f"Training on device: {self.device.upper()}")
        print(f"Model parameters: {self.model.count_parameters():,}")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training dataloader
            epoch: Current epoch number
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        epsilon = self.config.get('epsilon', 0.02)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, images in enumerate(pbar):
            images = images.to(self.device)
            
            # Generate FGSM targets on-the-fly
            with torch.no_grad():
                fgsm_targets = fgsm(images, self.model, epsilon=epsilon)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    target_perturbation = fgsm_targets - images
                    loss = self.criterion(outputs, target_perturbation)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(images)
                target_perturbation = fgsm_targets - images
                loss = self.criterion(outputs, target_perturbation)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
            
            # Log to TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: Validation dataloader
            epoch: Current epoch number
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        epsilon = self.config.get('epsilon', 0.02)
        
        pbar = tqdm(val_loader, desc="Validation")
        
        for images in pbar:
            images = images.to(self.device)
            
            # Generate FGSM targets
            fgsm_targets = fgsm(images, self.model, epsilon=epsilon)
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss
            target_perturbation = fgsm_targets - images
            loss = self.criterion(outputs, target_perturbation)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss: {val_loss:.6f}")
        
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f'epoch_{epoch}.pth'
        torch.save(checkpoint, epoch_path)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int
    ):
        """
        Main training loop.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of epochs to train
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader, epoch)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
        
        total_time = time.time() - start_time
        print(f"\nTraining complete! Total time: {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        self.writer.close()


def main():
    """Main training function."""
    # Load config
    config_path = Path(__file__).parent.parent / 'config' / 'training_config.yaml'
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            'model_type': 'unet',
            'batch_size': 16,
            'num_epochs': 50,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'epsilon': 0.02,
            'image_size': 256,
            'num_workers': 4,
            'train_dir': './data/train',
            'val_dir': './data/val',
            'output_dir': './outputs',
            'max_samples': None
        }
    
    # Create dataloaders
    train_transform = get_training_transforms(config['image_size'])
    val_transform = get_validation_transforms(config['image_size'])
    
    train_loader, val_loader = create_dataloaders(
        train_dir=config['train_dir'],
        val_dir=config['val_dir'],
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        max_samples=config.get('max_samples')
    )
    
    # Create trainer and train
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader, config['num_epochs'])


if __name__ == "__main__":
    main()
