import sys
import gradio as gr
import torch
from PIL import Image
from pathlib import Path

from inference import DiffusionCLIPInferencer

# Global dictionary to hold pre-loaded models
LOADED_INFERENCERS = {}
EDIT_TYPES = ['night', 'rainy', 'foggy', 'snowy']  # ['night', 'rainy', 'foggy', 'snowy', 'rusty', 'vines']


def preload_models():
    """Initializes and caches an inferencer for each edit type."""
    print("Pre-loading all models. This may take a moment...")

    # Automatically detect the number of available GPUs to distribute models
    if not torch.cuda.is_available():
        print("Warning: CUDA not detected. All models will run on CPU.", file=sys.stderr)
        num_gpus = 0
    else:
        num_gpus = torch.cuda.device_count()
        print(f"Detected {num_gpus} available GPU(s) for model distribution.")

    for i, edit_type in enumerate(EDIT_TYPES):
        # Assign device in a round-robin fashion if GPUs are available
        device = f"cuda:{i % num_gpus}" if num_gpus > 0 else "cpu"
        print(f"  Loading model for: '{edit_type}' onto device '{device}'")
        try:
            # Pass the assigned device to the inferencer
            inferencer = DiffusionCLIPInferencer(
                edit_type=edit_type,
                degree_of_change=1.0, # Default value, will be updated by slider
                t_0=301,
                fine_tuned_epoch=9,
                device=device # Pass specific device ID
            )
            LOADED_INFERENCERS[edit_type] = inferencer
        except Exception as e:
            print(f"  Failed to load model for {edit_type}: {e}", file=sys.stderr)
    print("All models pre-loaded successfully!")


def run_gradio_demo():
    """
    Sets up and launches the Gradio web interface using gr.Blocks for more control.
    """
    # --- Prepare the list of test images for the dropdown ---
    test_data_dir = Path('./data/traffic_Data/TEST')
    test_image_paths = sorted([str(p) for p in test_data_dir.glob("*.png")])

    def process_image_edit(image_path, uploaded_image, edit_type, degree_of_change):
        """
        This function is called by the Gradio interface on user submission.
        It uses pre-loaded models to run inference.
        """
        if uploaded_image is not None: image = uploaded_image
        elif image_path is not None: image = Image.open(image_path).convert("RGB")
        else: return "Please select or upload an image!", None
        
        if edit_type not in LOADED_INFERENCERS:
            raise gr.Error(f"Model for '{edit_type}' is not available. Please check the console for loading errors.")
        
        # 1. Get the pre-loaded inferencer
        inferencer = LOADED_INFERENCERS[edit_type]
        
        # 2. Update the degree of change from the slider
        inferencer.degree_of_change = degree_of_change
        
        # 3. Run inference on the selected image path
        print(f"Running inference for '{edit_type}' on {image_path}...")
        edited_pil = inferencer.inference(
            image=image,
            n_inv_step=40,
            n_test_step=12
        )
        
        # 4. Return only the edited image
        return edited_pil

    # --- Build the UI with gr.Blocks ---
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# DiffusionCLIP Image Editor")
        gr.Markdown("Choose or upload a test image and an edit type to apply a transformation.")
        
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("Choose from test set"):
                        image_selector = gr.Dropdown(
                            choices=test_image_paths,
                            label="Choose a Test Image",
                        )
                    with gr.TabItem("Upload a image"):
                        image_uploader = gr.Image(type="pil", label="Upload Your Image")

                edit_type_selector = gr.Dropdown(
                    EDIT_TYPES, 
                    label="Choose an Edit Type", 
                    value='snowy'
                )
                degree_slider = gr.Slider(
                    minimum=0.0, 
                    maximum=1.0, 
                    step=0.1, 
                    value=0.8, 
                    label="Degree of Change"
                )
                submit_button = gr.Button("Apply Edit", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    input_display = gr.Image(type="pil", label="Input Image")
                    output_display = gr.Image(type="pil", label="Edited Result")

        # --- Define Event Handlers ---
        def update_traffic_preview_from_dropdown(image_path):
            if image_path:
                image = Image.open(image_path).convert("RGB")
                return image.resize((512, 512)), None, None
            return None, None, None

        def update_traffic_preview_from_upload(uploaded_image):
            if uploaded_image:
                image = uploaded_image.convert("RGB")
                return image.resize((512, 512)), None, None
            return None, None, None

        # When the user selects a new image from the dropdown, update the input display instantly.
        image_selector.change(
            fn=update_traffic_preview_from_dropdown,
            inputs=image_selector,
            outputs=[input_display, image_uploader, output_display]
        )

        image_uploader.change(
            fn=update_traffic_preview_from_upload,
            inputs=image_uploader,
            outputs=[input_display, image_selector, output_display]
        )

        # When the user clicks the submit button, run the main processing function.
        submit_button.click(
            fn=process_image_edit,
            inputs=[image_selector, image_uploader, edit_type_selector, degree_slider],
            outputs=output_display
        )

    print("Launching Gradio demo...")
    demo.launch(share=True)


if __name__ == '__main__':

    # Pre-load models before launching the demo
    preload_models()
    
    # Launch the Gradio demo
    run_gradio_demo()
