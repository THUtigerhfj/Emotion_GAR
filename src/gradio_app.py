import os
import threading
from functools import lru_cache

import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification

from inference import VITAttentionGradRollout, enhance_rollout_mask, show_mask_on_image


# Guard model execution to avoid race conditions / OOM under heavy concurrent traffic.
INFERENCE_LOCK = threading.Lock()


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return max(int(value), minimum)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def load_runtime():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_path = os.path.abspath(os.path.join(script_dir, "..", "models", "emotion_vit"))
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(
            f"Model not found at '{local_model_path}'. Run 'python src/download_model.py' first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(local_model_path)
    model = ViTForImageClassification.from_pretrained(
        local_model_path,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    return device, processor, model


def run_inference_ui(input_image):
    if input_image is None:
        return None, "Please upload or drag an image first."

    try:
        device, processor, model = load_runtime()
    except Exception as exc:
        return None, f"Failed to load model: {exc}"

    raw_img = input_image.convert("RGB")
    inputs = processor(images=raw_img, return_tensors="pt")
    input_tensor = inputs["pixel_values"].to(device)

    with INFERENCE_LOCK:
        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs.logits
            predicted_idx = logits.argmax(-1).item()

        predicted_emotion = model.config.id2label[predicted_idx]

        rollout_generator = VITAttentionGradRollout(model, discard_ratio=0.2, head_fusion="mean", large_reweight="none")
        input_tensor.requires_grad = True
        mask = rollout_generator(input_tensor, predicted_idx)

    img_resized = raw_img.resize((224, 224))
    np_img_rgb = np.array(img_resized)

    mask_resized = np.array(Image.fromarray(np.uint8(mask * 255)).resize((np_img_rgb.shape[1], np_img_rgb.shape[0]), Image.BILINEAR), dtype=np.float32) / 255.0
    mask_resized = enhance_rollout_mask(mask_resized)
    rollout_rgb = show_mask_on_image(np_img_rgb, mask_resized)

    result_text = f"Predicted emotion: {predicted_emotion.upper()} (Index: {predicted_idx})"
    return Image.fromarray(rollout_rgb), result_text


def build_ui():
    with gr.Blocks(title="Emotion Grad Rollout") as demo:
        gr.Markdown("## Emotion Prediction + Attention Rollout")

        with gr.Row():
            input_image = gr.Image(
                type="pil",
                label="Input Image (Drag or Upload)",
                height=360,
            )
            output_image = gr.Image(
                type="pil",
                label="Attention Rollout",
                height=360,
            )

        with gr.Row():
            run_button = gr.Button("Process", variant="primary", scale=1)

        prediction_text = gr.Textbox(
            label="Prediction",
            interactive=False,
            lines=1,
        )

        run_button.click(
            fn=run_inference_ui,
            inputs=[input_image],
            outputs=[output_image, prediction_text],
            concurrency_limit=1,
        )

    return demo


if __name__ == "__main__":
    app = build_ui()

    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = _env_int("GRADIO_SERVER_PORT", 7860, minimum=1)
    queue_size = _env_int("GRADIO_QUEUE_MAX_SIZE", 10, minimum=1)
    queue_workers = _env_int("GRADIO_QUEUE_WORKERS", 1, minimum=1)
    app_max_threads = _env_int("GRADIO_MAX_THREADS", 16, minimum=1)
    share = _env_bool("GRADIO_SHARE", default=False)

    # Queue requests to smooth bursts and reduce crashes under multi-user traffic.
    app.queue(max_size=queue_size, default_concurrency_limit=queue_workers)
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        max_threads=app_max_threads,
    )
