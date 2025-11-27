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
from app.demo_utils import (
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
                print("Running in LIMITED mode - protection disabled, testing only")
        else:
            if model_path:
                print(f"⚠ Model file not found: {model_path}")
            else:
                print("ℹ️  No model path provided")
            print("\n" + "="*60)
            print("RUNNING IN LIMITED MODE")
            print("="*60)
            print("• Protection tab: DISABLED (no trained model)")
            print("• Testing tab: ENABLED (face swap testing works)")
            print("\nTo enable full functionality, train the model first:")
            print("  !python training/train_unet.py")
            print("Then restart with:")
            print("  !python app/gradio_app.py --model_path outputs/checkpoints/best.pth --share")
            print("="*60 + "\n")
    
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
                    # Update epsilon if different from initialization
                    if epsilon != self.imputer.epsilon:
                        self.imputer.epsilon = epsilon
                    perturbed_img = self.imputer.impute_from_array(image)
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
        source_face: np.ndarray
    ) -> Tuple[np.ndarray, str]:
        """Run complete protection test with face swapping."""
        if original is None or protected is None:
            return None, "❌ Please protect an image in the first tab before testing"
        
        if source_face is None:
            return None, "❌ Please upload a source face image to swap in"
        
        if self.deepfake_tester is None or not self.deepfake_tester.model_loaded:
            return None, "❌ Please load Face Swapper first (click '1️⃣ Load Face Swapper')"
        
        results_text = "🧪 **Face Swap Protection Test**\n\n"
        results_text += "**Method:** InsightFace inswapper (real deepfake technology)\n\n"
        
        # Step 1: Swap face on original image
        results_text += "**Step 1:** Swapping face on ORIGINAL image...\n"
        df_original, status_orig, metrics_orig = self.deepfake_tester.test_manipulation(
            target_image=original,
            source_image=source_face
        )
        results_text += f"{status_orig}\n"
        if metrics_orig and metrics_orig.get('mse', 0) > 0:
            results_text += f"  - MSE: {metrics_orig.get('mse', 0):.2f} ({metrics_orig.get('swap_strength', 'Unknown')} change)\n"
            results_text += f"  - PSNR: {metrics_orig.get('psnr', 0):.2f} dB\n"
            results_text += f"  - Quality: {'Good' if metrics_orig.get('std', 0) > 30 else 'Poor'}\n\n"
        
        # Step 2: Swap face on protected image  
        results_text += "**Step 2:** Swapping face on PROTECTED image...\n"
        df_protected, status_prot, metrics_prot = self.deepfake_tester.test_manipulation(
            target_image=protected,
            source_image=source_face
        )
        results_text += f"{status_prot}\n"
        if metrics_prot and metrics_prot.get('mse', 0) > 0:
            results_text += f"  - MSE: {metrics_prot.get('mse', 0):.2f} ({metrics_prot.get('swap_strength', 'Unknown')} change)\n"
            results_text += f"  - PSNR: {metrics_prot.get('psnr', 0):.2f} dB\n\n"
        
        # Step 3: Verdict
        results_text += "**Final Verdict:**\n"
        
        # First check if original face swap worked
        if df_original is None or metrics_orig.get('mse', 0) < 500:
            results_text += "⚠️ **TEST INCONCLUSIVE**\n"
            results_text += "Face swap on original image failed or too weak.\n"
            results_text += "Possible reasons:\n"
            results_text += "- No clear face detected in source or target\n"
            results_text += "- Faces at extreme angles\n"
            results_text += "- Try a different source face (frontal view)\n"
        elif metrics_prot and metrics_orig:
            mse_original = metrics_orig['mse']
            mse_protected = metrics_prot['mse']
            
            # For face swap: successful swap should have MSE > 500
            # Protected should either fail (MSE < 100) or create artifacts (MSE > 15000)
            
            if metrics_prot.get('corruption_detected'):
                results_text += "✅ **PROTECTION WORKING!**\n"
                if mse_protected < 100:
                    results_text += f"Face swap FAILED on protected image (MSE: {mse_protected:.0f})\n"
                else:
                    results_text += f"Face swap created heavy artifacts (MSE: {mse_protected:.0f})\n"
                results_text += "**Result:** Protection successfully disrupted deepfake generation!\n"
            elif mse_protected > 15000:
                results_text += "✅ **STRONG PROTECTION**\n"
                results_text += f"Face swap heavily corrupted (MSE: {mse_protected:.0f} vs {mse_original:.0f})\n"
            elif mse_protected < 100:
                results_text += "✅ **EXCELLENT PROTECTION**\n"
                results_text += f"Face swap completely failed on protected image!\n"
            elif mse_original > 500 and mse_protected > 500:
                results_text += "⚠️ **PROTECTION INSUFFICIENT**\n"
                results_text += f"Both images had successful face swaps (Original MSE: {mse_original:.0f}, Protected MSE: {mse_protected:.0f})\n"
                results_text += "**Action needed:** Increase epsilon to 0.25-0.30\n"
            else:
                results_text += "⚠️ **TEST INCONCLUSIVE**\n"
                results_text += "Face swap results unclear. Try different source face.\n"
        
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
        ### 🧪 Test Protection with Real Face Swap
        
        Uses **InsightFace inswapper** - the SAME technology used by DeepFaceLab, Roop, and FaceSwap!
        Upload a source face → it will be swapped onto your target images.
        **GPU recommended** - Takes ~5-10s on T4 GPU, ~30s on CPU.
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Target Images (auto-populated from Protection tab)**")
                test_original_display = gr.Image(label="Original Image (from tab 1)", type="numpy", height=180, interactive=False)
                test_protected_display = gr.Image(label="Protected Image (from tab 1)", type="numpy", height=180, interactive=False)
                
                gr.Markdown("**Source Face (upload any face to swap in)**")
                source_face_input = gr.Image(
                    label="Source Face Image",
                    type="numpy",
                    height=200,
                    sources=["upload", "clipboard"],
                    elem_id="source_face"
                )
                gr.Markdown("*💡 Tip: Use a clear frontal face photo for best results*")
                
                with gr.Row():
                    load_model_btn = gr.Button("1️⃣ Load Face Swapper", variant="secondary")
                    test_btn = gr.Button("2️⃣ Swap Faces & Test", variant="primary")
                
                model_status_box = gr.Textbox(label="Model Status", lines=2, value="Click 'Load Face Swapper' first")
            
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
            inputs=[original_state, protected_state, source_face_input],
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
                - Real face swap testing with InsightFace inswapper (DeepFaceLab technology)
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
