"""
Gradio web application for anti-deepfake protection.

Workflow:
- Training: GPU accelerated (Colab T4) - see training/train_unet.py
- Inference (Demo): CPU optimized (<400ms target) for deployment

This demo uses CPU by default for production-ready deployment.
"""

import gradio as gr
import numpy as np
import cv2
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.predict import NoiseImputer
from demo_utils import (
    create_side_by_side,
    compute_difference_heatmap,
    format_inference_time,
    get_image_info,
    add_protection_badge,
    Timer
)


class AntiDeepfakeApp:
    """Gradio app for anti-deepfake protection."""
    
    def __init__(
        self,
        model_path: str = None,
        model_type: str = 'unet',
        epsilon: float = 0.02,
        force_cpu: bool = True
    ):
        """
        Initialize app.
        
        Args:
            model_path: Path to trained model
            model_type: Model type ('unet' or 'autoencoder')
            epsilon: Perturbation magnitude
            force_cpu: Force CPU inference for demo (True by default for deployment)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.epsilon = epsilon
        self.imputer = None
        
        # Load model if path provided
        if model_path and Path(model_path).exists():
            try:
                # Use CPU for Gradio demo (fast inference, deployment-ready)
                # Training uses GPU, but demo optimized for CPU deployment
                device = 'cpu' if force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
                self.imputer = NoiseImputer(
                    model_path=model_path,
                    model_type=model_type,
                    epsilon=epsilon,
                    device=device
                )
                print(f"✓ Model loaded successfully from {model_path} on {device.upper()}")
                if force_cpu:
                    print("  [Demo Mode] Optimized for CPU deployment (<400ms target)")
            except Exception as e:
                print(f"⚠ Warning: Could not load model: {e}")
                print("Running in demo mode without trained model")
        else:
            print("⚠ No model path provided or file not found")
            print("Running in demo mode")
    
    def process_image(
        self,
        image: np.ndarray,
        epsilon: float,
        show_heatmap: bool,
        add_badge: bool
    ) -> tuple:
        """
        Process uploaded image.
        
        Args:
            image: Input image from Gradio (H, W, C) RGB
            epsilon: Perturbation magnitude
            show_heatmap: Whether to show difference heatmap
            add_badge: Whether to add protection badge
        
        Returns:
            Tuple of (output_image, info_text)
        """
        if image is None:
            return None, "Please upload an image"
        
        with Timer() as timer:
            # Use model if available, otherwise use simple noise
            if self.imputer is not None:
                try:
                    perturbed_img = self.imputer.impute_from_array(image)
                except Exception as e:
                    return None, f"Error during inference: {e}"
            else:
                # Demo mode: add simple random noise
                noise = np.random.randn(*image.shape) * (epsilon * 255)
                perturbed_img = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        inference_time = timer.get_elapsed_ms()
        
        # Add protection badge if requested
        if add_badge:
            perturbed_img = add_protection_badge(perturbed_img)
        
        # Create visualization
        if show_heatmap:
            heatmap = compute_difference_heatmap(image, perturbed_img)
            output = np.hstack([image, perturbed_img, heatmap])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            h, w = image.shape[:2]
            cv2.putText(output, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(output, "Protected", (w + 10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(output, "Difference", (2*w + 10, 30), font, 0.7, (255, 255, 255), 2)
        else:
            output = create_side_by_side(image, perturbed_img, add_labels=True)
        
        # Create info text
        info = f"✓ Processing complete!\n\n"
        info += f"⏱ Inference time: {format_inference_time(inference_time)}\n"
        info += f"🎯 Epsilon: {epsilon:.3f}\n"
        info += f"📊 {get_image_info(image)}\n"
        
        if inference_time < 400:
            info += f"\n✅ Target met: <400ms"
        else:
            info += f"\n⚠ Warning: Slower than target (400ms)"
        
        return output, info
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface."""
        
        with gr.Blocks(title="Anti-Deepfake Protection", theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
                # 🛡️ Anti-Deepfake Image Protection
                
                Upload an image to add invisible perturbations that protect against deepfake manipulation.
                The perturbations are imperceptible to humans but can disrupt AI-based face manipulation.
                
                **Features:**
                - Fast CPU inference (<400ms target)
                - Minimal visual impact
                - FGSM-based perturbations
                """
            )
            
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        label="Upload Image",
                        type="numpy",
                        height=400
                    )
                    
                    with gr.Row():
                        epsilon_slider = gr.Slider(
                            minimum=0.01,
                            maximum=0.1,
                            value=self.epsilon,
                            step=0.005,
                            label="Perturbation Strength (ε)",
                            info="Higher = stronger protection but more visible"
                        )
                    
                    with gr.Row():
                        show_heatmap = gr.Checkbox(
                            label="Show Difference Heatmap",
                            value=True
                        )
                        add_badge = gr.Checkbox(
                            label="Add Protection Badge",
                            value=False
                        )
                    
                    process_btn = gr.Button("🛡️ Protect Image", variant="primary", size="lg")
                
                with gr.Column():
                    output_image = gr.Image(
                        label="Result",
                        type="numpy",
                        height=400
                    )
                    info_output = gr.Textbox(
                        label="Processing Info",
                        lines=10,
                        max_lines=15
                    )
            
            # Examples
            gr.Markdown("### 📸 Example Images")
            gr.Markdown("Upload your own images to test the protection.")
            
            # Process button click
            process_btn.click(
                fn=self.process_image,
                inputs=[input_image, epsilon_slider, show_heatmap, add_badge],
                outputs=[output_image, info_output]
            )
            
            # Model info footer
            model_status = "✓ Model loaded" if self.imputer else "⚠ Demo mode (no model)"
            device_info = self.imputer.device.upper() if self.imputer else "CPU"
            
            gr.Markdown(
                f"""
                ---
                **Model Status:** {model_status} | **Model Type:** {self.model_type} | **Device:** {device_info}
                
                *Train on GPU (Colab T4) • Fast CPU inference for deployment (<400ms)*
                """
            )
        
        return demo
    
    def launch(self, **kwargs):
        """Launch the Gradio app."""
        demo = self.create_interface()
        demo.launch(**kwargs)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Anti-Deepfake Protection Demo")
    parser.add_argument(
        '--model_path',
        type=str,
        default='outputs/checkpoints/best.pth',
        help='Path to trained model weights'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='unet',
        choices=['unet', 'autoencoder'],
        help='Model type'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.02,
        help='Perturbation magnitude'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create public share link'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port to run on'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for inference (default: CPU for fast deployment)'
    )
    
    args = parser.parse_args()
    
    # Create and launch app
    # Note: Training uses GPU (Colab T4), but demo defaults to CPU for deployment
    app = AntiDeepfakeApp(
        model_path=args.model_path if Path(args.model_path).exists() else None,
        model_type=args.model_type,
        epsilon=args.epsilon,
        force_cpu=not args.gpu  # Default to CPU for demo
    )
    
    print("\n" + "="*60)
    print("Starting Anti-Deepfake Protection Demo")
    print("="*60)
    
    app.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()
