import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification

from inference import (
    VITAttentionGradRollout,
    enhance_rollout_mask,
    show_mask_on_image,
    save_image_unicode_safe,
)


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES


def iter_images(root_dir: Path):
    for path in root_dir.rglob("*"):
        if is_image_file(path):
            yield path


def build_summary_image(original_rgb, rollout_rgb, predicted_label, confidence, true_label):
    """Build one image containing original, rollout, and class info."""
    h, w = original_rgb.shape[:2]

    info_panel = np.full((h, w, 3), 245, dtype=np.uint8)
    title = "Classification"
    pred_text = f"Predicted: {predicted_label}"
    conf_text = f"Confidence: {confidence:.2%}"
    true_text = f"Folder label: {true_label}" if true_label else "Folder label: N/A"

    cv2.putText(info_panel, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(info_panel, pred_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(info_panel, conf_text, (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(info_panel, true_text, (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)

    original_labeled = original_rgb.copy()
    rollout_labeled = rollout_rgb.copy()
    cv2.putText(original_labeled, "Original", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(rollout_labeled, "Grad Rollout", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return np.concatenate([original_labeled, rollout_labeled, info_panel], axis=1)


def process_one_image(
    image_path: Path,
    output_path: Path,
    model,
    processor,
    grad_rollout,
    device,
):
    raw_img = Image.open(str(image_path)).convert("RGB")

    inputs = processor(images=raw_img, return_tensors="pt")
    input_tensor = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_idx = probs.argmax(-1).item()
        confidence = probs[0, predicted_idx].item()

    predicted_emotion = model.config.id2label[predicted_idx]

    input_tensor.requires_grad = True
    mask = grad_rollout(input_tensor, predicted_idx)

    img_resized = raw_img.resize((224, 224))
    original_rgb = np.array(img_resized)

    mask_resized = cv2.resize(mask, (224, 224))
    mask_resized = enhance_rollout_mask(mask_resized)
    rollout_rgb = show_mask_on_image(original_rgb, mask_resized)

    true_label = image_path.parent.name if image_path.parent != image_path.parent.parent else ""

    summary_rgb = build_summary_image(
        original_rgb=original_rgb,
        rollout_rgb=rollout_rgb,
        predicted_label=predicted_emotion,
        confidence=confidence,
        true_label=true_label,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_bgr = cv2.cvtColor(summary_rgb, cv2.COLOR_RGB2BGR)
    save_image_unicode_safe(str(output_path), summary_bgr)

    return predicted_emotion, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Run batch inference on /data/observe and save original + class + grad rollout visualizations."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "observe"),
        help="Directory containing observe images (can include emotion subfolders).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "outputs" / "observe_inference"),
        help="Directory to save combined visualization images.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "models" / "emotion_vit"),
        help="Local model directory.",
    )
    parser.add_argument("--discard_ratio", type=float, default=0.2)
    parser.add_argument("--head_fusion", type=str, default="mean", choices=["mean", "max", "min"])
    parser.add_argument(
        "--large_reweight",
        type=str,
        default="none",
        choices=["none", "sqrt", "power", "log"],
    )
    parser.add_argument("--large_quantile", type=float, default=0.95)
    parser.add_argument("--large_power", type=float, default=0.5)
    args = parser.parse_args()

    if not (0.0 <= args.discard_ratio <= 1.0):
        raise ValueError("--discard_ratio must be between 0 and 1.")
    if not (0.0 < args.large_quantile < 1.0):
        raise ValueError("--large_quantile must be between 0 and 1 (exclusive).")
    if not (0.0 < args.large_power <= 1.0):
        raise ValueError("--large_power must be in (0, 1].")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    image_paths = sorted(iter_images(input_dir))
    if not image_paths:
        print(f"No image files found in: {input_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    print(f"Found {len(image_paths)} images in {input_dir}")

    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    model = ViTForImageClassification.from_pretrained(
        str(model_dir),
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    grad_rollout = VITAttentionGradRollout(
        model,
        discard_ratio=args.discard_ratio,
        head_fusion=args.head_fusion,
        large_reweight=args.large_reweight,
        large_quantile=args.large_quantile,
        large_power=args.large_power,
    )

    success = 0
    for i, image_path in enumerate(image_paths, start=1):
        rel = image_path.relative_to(input_dir)
        out_name = rel.with_suffix("").name + "_summary.png"
        out_path = output_dir / rel.parent / out_name

        try:
            predicted_emotion, confidence = process_one_image(
                image_path=image_path,
                output_path=out_path,
                model=model,
                processor=processor,
                grad_rollout=grad_rollout,
                device=device,
            )
            success += 1
            print(
                f"[{i}/{len(image_paths)}] OK: {image_path.name} -> {predicted_emotion} ({confidence:.2%})"
            )
        except Exception as exc:
            print(f"[{i}/{len(image_paths)}] FAIL: {image_path} | {exc}")

    print(f"Done. {success}/{len(image_paths)} images processed.")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
