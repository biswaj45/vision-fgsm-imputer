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
from typing import Tuple, Optional

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
from app.deepfake_tester import DeepfakeTester, create_comparison_visualization


class AntiDeepfakeApp:
    """Gradio app for anti-deepfake protection."""
    
    def __init__(
        self,
        model_path: str = None,
        model_type: str = 'unet',
        epsilon: float = 0.02,
        force_cpu: bool = False
    ):
        """
        Initialize app.
        
        Args:
            model_path: Path to trained model
            model_type: Model type ('unet' or 'autoencoder')
            epsilon: Perturbation magnitude
            force_cpu: Force CPU inference (False = auto-detect GPU)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.epsilon = epsilon
        self.imputer = None
        self.deepfake_tester = None
        
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
            Tuple of (output_image, info_text, original_state, protected_state)
        """
        if image is None:
            return None, "Please upload an image", None, None
        
        with Timer() as timer:
            # Use model if available, otherwise use simple noise
            if self.imputer is not None:
                try:
                    perturbed_img = self.imputer.impute_from_array(image, epsilon=epsilon)
                except Exception as e:
                    return None, f"Error during inference: {e}", None, None
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
            # Show only protected image
            output = perturbed_img
        
        # Create info text
        info = f"✓ Processing complete!\n\n"
        info += f"⏱ Inference time: {format_inference_time(inference_time)}\n"
        info += f"🎯 Epsilon: {epsilon:.3f}\n"
        info += f"📊 {get_image_info(image)}\n"
        info += f"\n💡 Go to 'Test Protection' tab to verify effectiveness"
        
        if inference_time < 400:
            info += f"\n✅ Target met: <400ms"
        else:
            info += f"\n⚠ Warning: Slower than target (400ms)"
        
        # Return original and protected for testing tab
        return output, info, image.copy(), perturbed_img.copy()
    
    def load_deepfake_model(self) -> str:
        """Load the deepfake testing model."""
        if self.deepfake_tester is None:
            self.deepfake_tester = DeepfakeTester(device='auto')  # Auto-detect GPU
        
        success, msg = self.deepfake_tester.load_model()
        return f"{'✅' if success else '❌'} {msg}"
    
    def run_protection_test(
        self,
        original: np.ndarray,
        protected: np.ndarray,
        prompt: str
    ) -> Tuple[np.ndarray, str]:
        """Run complete protection test."""
        if original is None or protected is None:
            return None, "❌ Please protect an image in the first tab before testing"
        
        if not prompt or not prompt.strip():
            return None, "❌ Please enter a manipulation prompt (e.g., 'wearing sunglasses in garden')"
        
        if self.deepfake_tester is None or not self.deepfake_tester.model_loaded:
            return None, "❌ Please load the test model first (click '1️⃣ Load Test Model')"
        
        results_text = "🧪 **Protection Test Results**\n\n"
        results_text += f"**Prompt:** \"{prompt}\"\n\n"
        
        # Step 1: Test original image
        results_text += "**Step 1:** Testing deepfake on ORIGINAL image...\n"
        df_original, status_orig, metrics_orig = self.deepfake_tester.test_manipulation(
            original, prompt, strength=0.5, num_steps=35
        )
        results_text += f"{status_orig}\n"
        if metrics_orig:
            results_text += f"  - MSE: {metrics_orig['mse']:.2f}\n"
            results_text += f"  - PSNR: {metrics_orig['psnr']:.2f} dB\n"
            results_text += f"  - Quality: {'Good' if metrics_orig.get('std', 0) > 30 else 'Poor'}\n\n"
        
        # Step 2: Test protected image
        results_text += "**Step 2:** Testing deepfake on PROTECTED image...\n"
        df_protected, status_prot, metrics_prot = self.deepfake_tester.test_manipulation(
            protected, prompt, strength=0.5, num_steps=35
        )
        results_text += f"{status_prot}\n"
        if metrics_prot:
            results_text += f"  - MSE: {metrics_prot['mse']:.2f}\n"
            results_text += f"  - PSNR: {metrics_prot['psnr']:.2f} dB\n"
            results_text += f"  - Quality: {'Good' if metrics_prot.get('std', 0) > 30 else 'Corrupted'}\n\n"
        
        # Step 3: Verdict
        results_text += "**Final Verdict:**\n"
        
        # First check if original generation worked
        if df_original is None or metrics_orig.get('std', 0) < 30:
            results_text += "⚠️ **TEST INCONCLUSIVE**\n"
            results_text += "Original image generation failed. This could mean:\n"
            results_text += "- Prompt is not clear enough\n"
            results_text += "- Image quality is too low\n"
            results_text += "Try a more specific prompt like:\n"
            results_text += "  'person wearing sunglasses and hat'\n"
            results_text += "  'smiling person with different hairstyle'\n"
        elif metrics_prot and metrics_orig:
            orig_quality = metrics_orig.get('std', 0) > 30
            prot_quality = metrics_prot.get('std', 0) > 30
            
            if orig_quality and not prot_quality:
                results_text += "🛡️ **PROTECTION WORKING!**\n"
                results_text += "✅ Original: High quality deepfake generated\n"
                results_text += "❌ Protected: Generation corrupted/failed\n"
                results_text += f"MSE difference: {metrics_prot['mse'] - metrics_orig['mse']:.0f}\n"
            elif orig_quality and prot_quality:
                mse_diff = metrics_prot['mse'] - metrics_orig['mse']
                if mse_diff > 3000:
                    results_text += "🛡️ **PROTECTION PARTIALLY WORKING**\n"
                    results_text += f"Protected image degraded quality by {mse_diff:.0f} MSE\n"
                    results_text += "Consider increasing epsilon to 0.20-0.25\n"
                else:
                    results_text += "⚠️ **PROTECTION INSUFFICIENT**\n"
                    results_text += "Both images generated high quality deepfakes.\n"
                    results_text += "**Action needed:** Increase epsilon to 0.20-0.30\n"
            else:
                results_text += "⚠️ **TEST INCONCLUSIVE**\n"
                results_text += "Both generations failed - may need better prompt\n"
        
        # Create visualization
        comparison = create_comparison_visualization(
            original, protected, df_original, df_protected
        )
        
        return comparison, results_text
    
    def _create_protection_tab(self, original_state, protected_state):
        """Create the image protection tab."""
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
                        maximum=0.3,
                        value=0.15,
                        step=0.01,
                        label="Perturbation Strength (ε)",
                        info="Higher = stronger protection. 0.15-0.20 recommended for diffusion models."
                    )
                
                with gr.Row():
                    show_heatmap = gr.Checkbox(
                        label="Show Comparison View (Original | Protected | Difference)",
                        value=False,
                        info="Uncheck to show only the protected image"
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
        
        # Process button click - now returns 4 values including state
        process_btn.click(
            fn=self.process_image,
            inputs=[input_image, epsilon_slider, show_heatmap, add_badge],
            outputs=[output_image, info_output, original_state, protected_state]
        )
    
    def _create_testing_tab(self, original_state, protected_state):
        """Create the deepfake testing tab."""
        gr.Markdown("""
        ### 🧪 Test Protection with Real Deepfake Generation
        
        Uses **Stable Diffusion** to generate actual deepfakes and test protection effectiveness.
        **GPU recommended** - Takes ~60s on T4 GPU, ~5min on CPU.
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Images auto-populated from Protection tab**")
                test_original_display = gr.Image(label="Original Image (from tab 1)", type="numpy", height=180, interactive=False)
                test_protected_display = gr.Image(label="Protected Image (from tab 1)", type="numpy", height=180, interactive=False)
                
                test_prompt = gr.Textbox(
                    label="Deepfake Prompt (Be Specific!)",
                    placeholder="Examples:\n• 'person wearing sunglasses and casual hat'\n• 'smiling person with short blonde hair'\n• 'person in formal suit with glasses'",
                    lines=3,
                    value="person wearing sunglasses and casual hat in outdoor setting"
                )
                
                with gr.Row():
                    load_model_btn = gr.Button("1️⃣ Load Stable Diffusion", variant="secondary")
                    test_btn = gr.Button("2️⃣ Generate Deepfakes & Test", variant="primary")
                
                model_status_box = gr.Textbox(label="Model Status", lines=2, value="Click 'Load Stable Diffusion' first")
            
            with gr.Column():
                test_output = gr.Image(label="Comparison (2x2 Grid)", type="numpy", height=550)
                test_results = gr.Textbox(label="Test Results", lines=14)
        
        # Auto-populate displays when state changes
        original_state.change(
            fn=lambda x: x,
            inputs=[original_state],
            outputs=[test_original_display]
        )
        protected_state.change(
            fn=lambda x: x,
            inputs=[protected_state],
            outputs=[test_protected_display]
        )
        
        # Event handlers
        load_model_btn.click(
            fn=self.load_deepfake_model,
            inputs=[],
            outputs=[model_status_box]
        )
        
        test_btn.click(
            fn=self.run_protection_test,
            inputs=[original_state, protected_state, test_prompt],
            outputs=[test_output, test_results]
        )
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface."""
        
        with gr.Blocks(title="Anti-Deepfake Protection", theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
                # 🛡️ Anti-Deepfake Image Protection
                
                Upload an image to add invisible perturbations that protect against deepfake manipulation.
                The perturbations are imperceptible to humans but can disrupt AI-based face manipulation.
                
                **Features:**
                - Fast GPU inference (protection & testing)
                - Minimal visual impact on protected images
                - FGSM-based perturbations
                - Real deepfake testing with Stable Diffusion
                """
            )
            
            # State variables to pass images between tabs
            original_state = gr.State(value=None)
            protected_state = gr.State(value=None)
            
            with gr.Tabs():
                # Tab 1: Protection
                with gr.Tab("🛡️ Protect Images"):
                    self._create_protection_tab(original_state, protected_state)
                
                # Tab 2: Test Protection
                with gr.Tab("🧪 Test Protection"):
                    self._create_testing_tab(original_state, protected_state)
            
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
        help='Use GPU for inference (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Create and launch app
    # Auto-detect GPU by default for best performance
    app = AntiDeepfakeApp(
        model_path=args.model_path if Path(args.model_path).exists() else None,
        model_type=args.model_type,
        epsilon=args.epsilon,
        force_cpu=False  # Auto-detect GPU
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
