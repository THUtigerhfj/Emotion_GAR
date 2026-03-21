import os
import shutil
import threading
from collections import Counter
from functools import lru_cache

import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification

from inference import (
    VITAttentionGradRollout,
    detect_and_crop_faces,
    draw_face_boxes,
    enhance_rollout_mask,
    show_mask_on_image,
)


# Guard model execution to avoid race conditions / OOM under heavy concurrent traffic.
INFERENCE_LOCK = threading.Lock()


def _prepare_retinaface_local_weights():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    project_weight = os.path.join(project_root, "models", "retinaface", "retinaface.h5")

    # DeepFace internally appends ".deepface" to DEEPFACE_HOME.
    # If DEEPFACE_HOME already ends with ".deepface", normalize it to parent to avoid
    # accidental nested paths like ".deepface/.deepface/weights".
    deepface_home = os.getenv("DEEPFACE_HOME", project_root)
    if os.path.basename(os.path.normpath(deepface_home)) == ".deepface":
        deepface_home = os.path.dirname(os.path.normpath(deepface_home))

    # Prefer project-local DeepFace cache so runtime is deterministic and portable.
    os.environ["DEEPFACE_HOME"] = deepface_home
    runtime_weight = os.path.join(deepface_home, ".deepface", "weights", "retinaface.h5")

    if os.path.exists(project_weight) and not os.path.exists(runtime_weight):
        os.makedirs(os.path.dirname(runtime_weight), exist_ok=True)
        shutil.copy2(project_weight, runtime_weight)

    return project_weight, runtime_weight


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

    project_weight, runtime_weight = _prepare_retinaface_local_weights()
    if not os.path.exists(runtime_weight):
        raise FileNotFoundError(
            "RetinaFace weights not found locally. "
            "Run 'python src/download_model.py --retinaface-only' first, or place file at "
            f"'{project_weight}'."
        )

    # Prime RetinaFace import once; weights are loaded lazily on first detect call.
    from retinaface import RetinaFace

    rollout = VITAttentionGradRollout(
        model,
        discard_ratio=0.2,
        head_fusion="mean",
        large_reweight="none",
    )
    return device, processor, model, rollout, RetinaFace


def run_inference_ui(input_image):
    if input_image is None:
        return None, [], "Please upload or drag an image first."

    try:
        device, processor, model, rollout_generator, _ = load_runtime()
    except Exception as exc:
        return None, [], f"Failed to load model/runtime: {exc}"

    raw_img = input_image.convert("RGB")

    try:
        detections, crops_rgb = detect_and_crop_faces(raw_img, expand_ratio=0.3)
    except Exception as exc:
        return None, [], f"RetinaFace detection failed: {exc}"

    boxed = draw_face_boxes(raw_img, detections) if detections else np.asarray(raw_img, dtype=np.uint8)
    boxed_image = Image.fromarray(boxed)

    if not detections:
        return boxed_image, [], "Detected faces: 0\n\nNo face found in the input image."

    gallery_items = []
    emotion_counter = Counter()

    with INFERENCE_LOCK:
        for idx, crop_rgb in enumerate(crops_rgb, start=1):
            face_img = Image.fromarray(crop_rgb)
            inputs = processor(images=face_img, return_tensors="pt")
            input_tensor = inputs["pixel_values"].to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs.logits
                predicted_idx = logits.argmax(-1).item()

            predicted_emotion = model.config.id2label[predicted_idx]
            emotion_counter[predicted_emotion] += 1

            input_tensor.requires_grad = True
            mask = rollout_generator(input_tensor, predicted_idx)

            h, w = crop_rgb.shape[:2]
            mask_resized = np.array(
                Image.fromarray(np.uint8(mask * 255)).resize((w, h), Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0
            mask_resized = enhance_rollout_mask(mask_resized)
            rollout_rgb = show_mask_on_image(crop_rgb, mask_resized)

            gallery_items.append((Image.fromarray(crop_rgb), f"Face {idx}: Crop"))
            gallery_items.append((Image.fromarray(rollout_rgb), f"Face {idx}: GAR | {predicted_emotion.upper()}"))

    total_faces = len(crops_rgb)
    summary_lines = [f"Detected faces: {total_faces}", "", "Emotion distribution:"]
    for emotion, count in emotion_counter.most_common():
        pct = (count / total_faces) * 100.0
        summary_lines.append(f"- {emotion.upper()}: {count}/{total_faces} ({pct:.1f}%)")

    return boxed_image, gallery_items, "\n".join(summary_lines)


def build_ui():
    with gr.Blocks(title="Emotion Grad Rollout") as demo:
        gr.Markdown("## Multi-Face Emotion + GAR")
        gr.Markdown("Upload one image. The app detects all faces, expands each box by 30%, and runs emotion + GAR per face.")

        input_image = gr.Image(
            type="pil",
            label="Input Image",
            height=420,
        )

        run_button = gr.Button("Detect + Analyze", variant="primary")

        detection_image = gr.Image(
            type="pil",
            label="RetinaFace Detection (Green: raw box, Yellow: expanded crop box)",
            height=420,
        )

        with gr.Accordion("Per-face outputs (scrollable)", open=False):
            per_face_gallery = gr.Gallery(
                label="Pairs: [Crop] and [GAR + Prediction]",
                columns=2,
                object_fit="contain",
                height=500,
                preview=False,
            )

        prediction_text = gr.Textbox(
            label="Summary",
            interactive=False,
            lines=8,
        )

        run_button.click(
            fn=run_inference_ui,
            inputs=[input_image],
            outputs=[detection_image, per_face_gallery, prediction_text],
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
