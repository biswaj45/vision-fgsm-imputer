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
    
    def load_deepfake_model(self, model_choice: str) -> str:
        """Load the deepfake testing model."""
        if "SimSwap" in model_choice:
            # Load SimSwap
            from app.simswap_tester import SimSwapTester
            self.deepfake_tester = SimSwapTester(device='auto')
            success, msg = self.deepfake_tester.load_model()
            return f"{'✅' if success else '❌'} {msg}"
        else:
            # Load InsightFace (default)
            if self.deepfake_tester is None:
                self.deepfake_tester = DeepfakeTester(device='auto')
            success, msg = self.deepfake_tester.load_model()
            return f"{'✅' if success else '❌'} {msg}"
    
    def run_protection_test(
        self,
        original: np.ndarray,
        protected: np.ndarray,
        source_face: np.ndarray,
        model_choice: str
    ) -> Tuple[np.ndarray, str]:
        """Run complete protection test with face swapping."""
        if original is None or protected is None:
            return None, "❌ Please protect an image in the first tab before testing"
        
        if source_face is None:
            return None, "❌ Please upload a source face image to swap in"
        
        if self.deepfake_tester is None or not self.deepfake_tester.model_loaded:
            return None, "❌ Please load Face Swapper first (click '1️⃣ Load Face Swapper')"
        
        model_name = "SimSwap" if "SimSwap" in model_choice else "InsightFace inswapper"
        results_text = "🧪 **Face Swap Protection Test**\n\n"
        results_text += f"**Method:** {model_name} (real deepfake technology)\n\n"
        
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
        # Successful swap typically has MSE > 200 (face changed)
        if df_original is None or metrics_orig.get('mse', 0) < 100:
            results_text += "⚠️ **TEST INCONCLUSIVE**\n"
            results_text += "Face swap on original image failed or too weak.\n"
            results_text += "Possible reasons:\n"
            results_text += "- No clear face detected in source or target\n"
            results_text += "- Faces at extreme angles\n"
            results_text += "- Try a different source face (frontal view)\n"
        elif metrics_prot and metrics_orig:
            mse_original = metrics_orig['mse']
            mse_protected = metrics_prot['mse']
            swap_orig = metrics_orig.get('swap_strength', 'Unknown')
            swap_prot = metrics_prot.get('swap_strength', 'Unknown')
            
            # Check if protection degraded the swap quality
            if metrics_prot.get('corruption_detected'):
                results_text += "✅ **PROTECTION WORKING!**\n"
                results_text += f"Face swap detected as corrupted/artificial\n"
                results_text += "**Result:** Protection successfully disrupted deepfake generation!\n"
            elif swap_prot == 'Failed':
                results_text += "✅ **EXCELLENT PROTECTION**\n"
                results_text += f"Face swap completely failed on protected image!\n"
                results_text += f"Original: MSE={mse_original:.0f} ({swap_orig}), Protected: MSE={mse_protected:.0f} (Failed)\n"
            elif swap_orig in ['Strong', 'Medium'] and swap_prot == 'Weak':
                results_text += "✅ **PROTECTION WORKING**\n"
                results_text += f"Swap quality degraded: {swap_orig} → {swap_prot}\n"
                results_text += f"Original MSE: {mse_original:.0f}, Protected MSE: {mse_protected:.0f}\n"
            elif mse_protected > mse_original * 1.5:
                results_text += "✅ **PARTIAL PROTECTION**\n"
                results_text += f"Protected image shows more artifacts (MSE increased by {((mse_protected/mse_original)-1)*100:.0f}%)\n"
            elif swap_orig == swap_prot and abs(mse_protected - mse_original) < 200:
                results_text += "⚠️ **PROTECTION INSUFFICIENT**\n"
                results_text += f"Both swaps similar quality ({swap_orig}): Original MSE={mse_original:.0f}, Protected MSE={mse_protected:.0f}\n"
                results_text += "**Action needed:** Increase epsilon to 0.25-0.30\n"
            else:
                results_text += "⚠️ **UNCLEAR RESULT**\n"
                results_text += f"Original: MSE={mse_original:.0f} ({swap_orig}), Protected: MSE={mse_protected:.0f} ({swap_prot})\n"
        
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
        
        Choose your face swapping method:
        - **InsightFace (inswapper)**: Fast, good quality
        - **SimSwap**: Slower but sharper, no blur
        
        Upload a source face → it will be swapped onto your target images.
        **GPU recommended** - Takes ~5-10s on T4 GPU, ~30s on CPU.
        """)
        
        with gr.Row():
            with gr.Column():
                # Model selection
                model_choice = gr.Radio(
                    choices=["InsightFace (inswapper)", "SimSwap"],
                    value="SimSwap",
                    label="Face Swap Model",
                    info="SimSwap is recommended for better quality"
                )
                
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
            inputs=[model_choice],
            outputs=[model_status_box]
        )
        
        test_btn.click(
            fn=self.run_protection_test,
            inputs=[original_state, protected_state, source_face_input, model_choice],
            outputs=[test_output, test_results]
        )
    
    def _create_face_swap_demo_tab(self):
        """Create standalone face swap demo tab with direct uploads."""
        gr.Markdown("""
        ### 🎭 Face Swap Demo - Upload Any Images
        
        **Upload source and target images directly to test face swapping.**
        
        - **SOURCE**: Person whose face identity to extract
        - **TARGET**: Person whose face will be replaced  
        - **RESULT**: TARGET image with SOURCE person's face
        
        *Try with/without protection to see the difference!*
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 📤 Upload Images")
                
                source_img = gr.Image(
                    label="SOURCE (Identity Donor)",
                    type="numpy",
                    height=250,
                    sources=["upload", "clipboard"]
                )
                
                target_img = gr.Image(
                    label="TARGET (Face to Replace)",
                    type="numpy",
                    height=250,
                    sources=["upload", "clipboard"]
                )
                
                with gr.Row():
                    model_select = gr.Radio(
                        choices=["InsightFace (inswapper)", "SimSwap"],
                        value="SimSwap",
                        label="Face Swap Model"
                    )
                
                with gr.Row():
                    apply_protection = gr.Checkbox(
                        label="Apply FGSM Protection to Target",
                        value=False,
                        info="Test protection effectiveness"
                    )
                    epsilon_demo = gr.Slider(
                        minimum=0.10,
                        maximum=0.30,
                        value=0.15,
                        step=0.05,
                        label="Protection Strength (ε)",
                        visible=False
                    )
                
                # Show epsilon slider when protection is enabled
                apply_protection.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[apply_protection],
                    outputs=[epsilon_demo]
                )
                
                with gr.Row():
                    load_swap_model_btn = gr.Button("1️⃣ Load Model", variant="secondary")
                    run_swap_btn = gr.Button("2️⃣ Swap Faces", variant="primary", size="lg")
                
                model_status = gr.Textbox(
                    label="Model Status",
                    value="Click 'Load Model' first",
                    lines=2
                )
            
            with gr.Column():
                gr.Markdown("#### 📊 Results")
                
                result_grid = gr.Image(
                    label="Comparison",
                    type="numpy",
                    height=500
                )
                
                metrics_display = gr.Textbox(
                    label="Metrics & Analysis",
                    lines=12
                )
        
        # Event handlers
        load_swap_model_btn.click(
            fn=self.load_deepfake_model,
            inputs=[model_select],
            outputs=[model_status]
        )
        
        run_swap_btn.click(
            fn=self._run_face_swap_demo,
            inputs=[source_img, target_img, model_select, apply_protection, epsilon_demo],
            outputs=[result_grid, metrics_display]
        )
    
    def _run_face_swap_demo(
        self,
        source: np.ndarray,
        target: np.ndarray,
        model_choice: str,
        apply_protection: bool,
        epsilon: float
    ):
        """Run face swap demo with optional protection."""
        
        if source is None or target is None:
            return None, "❌ Please upload both SOURCE and TARGET images"
        
        if not self.deepfake_model_loaded:
            return None, "❌ Please load face swap model first (click '1️⃣ Load Model')"
        
        try:
            import cv2
            from app.demo_utils import create_side_by_side
            
            # Step 1: Optionally protect target
            target_protected = None
            if apply_protection and self.imputer:
                target_protected = self.imputer.impute_from_array(target)
            
            # Step 2: Run face swap on original
            swapped_orig, status_orig, metrics_orig = self.deepfake_tester.test_manipulation(
                target, source
            )
            
            # Step 3: Run face swap on protected (if enabled)
            swapped_prot = None
            metrics_prot = None
            if apply_protection and target_protected is not None:
                swapped_prot, status_prot, metrics_prot = self.deepfake_tester.test_manipulation(
                    target_protected, source
                )
            
            # Create visualization
            if apply_protection and swapped_prot is not None:
                # 2x2 grid: Source | Target | Swap Original | Swap Protected
                h, w = 300, 300
                
                source_resized = cv2.resize(source, (w, h))
                target_resized = cv2.resize(target, (w, h))
                swapped_orig_resized = cv2.resize(swapped_orig, (w, h)) if swapped_orig is not None else np.zeros((h, w, 3), dtype=np.uint8)
                swapped_prot_resized = cv2.resize(swapped_prot, (w, h))
                
                # Add labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(source_resized, 'SOURCE', (10, 30), font, 0.8, (255, 255, 255), 2)
                cv2.putText(target_resized, 'TARGET', (10, 30), font, 0.8, (255, 255, 255), 2)
                cv2.putText(swapped_orig_resized, 'SWAP (Original)', (10, 30), font, 0.7, (0, 255, 0), 2)
                cv2.putText(swapped_prot_resized, f'SWAP (Protected ε={epsilon:.2f})', (10, 30), font, 0.6, (255, 0, 0), 2)
                
                # Stack
                top_row = np.hstack([source_resized, target_resized])
                bottom_row = np.hstack([swapped_orig_resized, swapped_prot_resized])
                grid = np.vstack([top_row, bottom_row])
                
                # Format metrics
                results = "## 📊 COMPARISON RESULTS\n\n"
                results += "### Original Target (No Protection):\n"
                results += f"- **Swap Strength**: {metrics_orig.get('swap_strength', 'N/A')}\n"
                results += f"- **MSE**: {metrics_orig.get('mse', 0):.2f}\n"
                results += f"- **PSNR**: {metrics_orig.get('psnr', 0):.2f} dB\n"
                results += f"- **Corruption**: {metrics_orig.get('corruption_detected', False)}\n\n"
                
                results += f"### Protected Target (ε={epsilon:.2f}):\n"
                results += f"- **Swap Strength**: {metrics_prot.get('swap_strength', 'N/A')}\n"
                results += f"- **MSE**: {metrics_prot.get('mse', 0):.2f}\n"
                results += f"- **PSNR**: {metrics_prot.get('psnr', 0):.2f} dB\n"
                results += f"- **Corruption**: {metrics_prot.get('corruption_detected', False)}\n\n"
                
                # Verdict
                results += "### 🎯 Protection Effectiveness:\n"
                orig_strength = metrics_orig.get('swap_strength', 'Strong')
                prot_strength = metrics_prot.get('swap_strength', 'Weak')
                
                if prot_strength == 'Failed' or metrics_prot.get('corruption_detected'):
                    results += "✅ **EXCELLENT**: Protection prevented face swap!\n"
                elif orig_strength == 'Strong' and prot_strength in ['Weak', 'Medium']:
                    results += "✅ **GOOD**: Protection degraded swap quality\n"
                else:
                    results += "⚠️ **LIMITED**: Try higher epsilon (0.25-0.30)\n"
                
            else:
                # Single swap result
                h = 400
                source_resized = cv2.resize(source, (h, h))
                target_resized = cv2.resize(target, (h, h))
                swapped_resized = cv2.resize(swapped_orig, (h, h)) if swapped_orig is not None else np.zeros((h, h, 3), dtype=np.uint8)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(source_resized, 'SOURCE', (10, 30), font, 1, (255, 255, 255), 2)
                cv2.putText(target_resized, 'TARGET', (10, 30), font, 1, (255, 255, 255), 2)
                cv2.putText(swapped_resized, 'RESULT', (10, 30), font, 1, (0, 255, 0), 2)
                
                grid = np.hstack([source_resized, target_resized, swapped_resized])
                
                results = "## 📊 FACE SWAP RESULTS\n\n"
                results += f"**Status**: {status_orig}\n\n"
                results += "### Metrics:\n"
                results += f"- **Swap Strength**: {metrics_orig.get('swap_strength', 'N/A')}\n"
                results += f"- **MSE**: {metrics_orig.get('mse', 0):.2f}\n"
                results += f"- **PSNR**: {metrics_orig.get('psnr', 0):.2f} dB\n"
                results += f"- **Corruption**: {metrics_orig.get('corruption_detected', False)}\n\n"
                results += "*💡 Enable 'Apply FGSM Protection' to test defense effectiveness*"
            
            return grid, results
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
            return None, error_msg
    
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
                
                # Tab 3: Face Swap Demo
                with gr.Tab("🎭 Face Swap Demo"):
                    self._create_face_swap_demo_tab()
            
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
