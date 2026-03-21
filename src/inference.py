import argparse
import sys
import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification


def _parse_retinaface_bbox(face_data):
    """Extract [x1, y1, x2, y2] bbox from RetinaFace response entry."""
    if isinstance(face_data, dict):
        if "facial_area" in face_data and len(face_data["facial_area"]) >= 4:
            box = face_data["facial_area"]
            return int(box[0]), int(box[1]), int(box[2]), int(box[3])
        if "bbox" in face_data and len(face_data["bbox"]) >= 4:
            box = face_data["bbox"]
            return int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if isinstance(face_data, (list, tuple)) and len(face_data) >= 4:
        return int(face_data[0]), int(face_data[1]), int(face_data[2]), int(face_data[3])
    return None


def detect_and_crop_faces(image_rgb, expand_ratio=0.3):
    """Detect faces with RetinaFace and return expanded crops.

    Returns:
        detections: list of dicts with keys raw_bbox and expanded_bbox
        crops_rgb: list of np.ndarray, one expanded crop per detection
    """
    if not (0.0 <= expand_ratio <= 1.0):
        raise ValueError("expand_ratio must be between 0 and 1.")

    from retinaface import RetinaFace

    img = np.asarray(image_rgb, dtype=np.uint8)
    h, w = img.shape[:2]
    resp = RetinaFace.detect_faces(img)

    if not isinstance(resp, dict) or len(resp) == 0:
        return [], []

    detections = []
    crops_rgb = []
    for _, face_data in resp.items():
        parsed = _parse_retinaface_bbox(face_data)
        if parsed is None:
            continue
        x1, y1, x2, y2 = parsed

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        expand_w = int(bw * expand_ratio)
        expand_h = int(bh * expand_ratio)

        ex1 = max(0, x1 - expand_w)
        ey1 = max(0, y1 - expand_h)
        ex2 = min(w, x2 + expand_w)
        ey2 = min(h, y2 + expand_h)

        if ex2 <= ex1 or ey2 <= ey1:
            continue

        crop = img[ey1:ey2, ex1:ex2].copy()
        detections.append(
            {
                "raw_bbox": (x1, y1, x2, y2),
                "expanded_bbox": (ex1, ey1, ex2, ey2),
            }
        )
        crops_rgb.append(crop)

    return detections, crops_rgb


def draw_face_boxes(image_rgb, detections):
    """Draw RetinaFace raw and expanded boxes for visualization."""
    canvas = np.asarray(image_rgb, dtype=np.uint8).copy()
    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["raw_bbox"]
        ex1, ey1, ex2, ey2 = det["expanded_bbox"]
        cv2.rectangle(canvas, (ex1, ey1), (ex2, ey2), (255, 196, 0), 2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"Face {idx}",
            (ex1, max(12, ey1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 196, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def reweight_large_attention(x, method="sqrt", upper_quantile=0.95, power=0.5):
    """Compress only upper-tail large values while preserving smaller ones."""
    if method == "none":
        return x
    if method not in {"sqrt", "power", "log"}:
        raise ValueError("large_reweight must be one of: none, sqrt, power, log.")
    if not (0.0 < upper_quantile < 1.0):
        raise ValueError("upper_quantile must be between 0 and 1 (exclusive).")

    # Compute threshold per sample over all token-pair entries.
    flat = x.view(x.size(0), -1)
    tau = torch.quantile(flat, upper_quantile, dim=1, keepdim=True)
    tau = tau.view(x.size(0), 1, 1)

    delta = torch.clamp(x - tau, min=0.0)
    if method == "sqrt":
        compressed = tau + torch.sqrt(delta + 1e-8)
    elif method == "power":
        if not (0.0 < power <= 1.0):
            raise ValueError("large_power must be in (0, 1].")
        compressed = tau + torch.pow(delta + 1e-8, power)
    else:
        compressed = tau + torch.log1p(delta)

    return torch.where(x > tau, compressed, x)


def grad_rollout(
    attentions,
    gradients,
    discard_ratio,
    head_fusion="max",
    large_reweight="sqrt",
    large_quantile=0.95,
    large_power=0.5,
):
    if not attentions:
        raise RuntimeError("No attention maps captured for rollout.")
    if not (0.0 <= discard_ratio <= 1.0):
        raise ValueError("discard_ratio must be between 0 and 1.")
    if head_fusion not in {"mean", "max", "min"}:
        raise ValueError("head_fusion must be one of: mean, max, min.")

    result = torch.eye(attentions[0].size(-1)) # create identity with shape of attention matrix
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):
            weights = grad
            weighted_attention = attention * weights
            if head_fusion == "mean":
                attention_heads_fused = weighted_attention.mean(dim=1)
            elif head_fusion == "max":
                attention_heads_fused = weighted_attention.max(dim=1).values
            else:
                attention_heads_fused = weighted_attention.min(dim=1).values
            attention_heads_fused[attention_heads_fused < 0] = 0

            # Keep small values unchanged and only compress extreme upper-tail peaks.
            attention_heads_fused = reweight_large_attention(
                attention_heads_fused,
                method=large_reweight,
                upper_quantile=large_quantile,
                power=large_power,
            )

            # Drop the lowest attentions, but don't drop class-token self-attention.
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            num_discard = int(flat.size(-1) * discard_ratio)
            if num_discard > 0:
                _, indices = flat.topk(num_discard, dim=-1, largest=False) # num_discard smallest
                for batch_idx in range(flat.size(0)):
                    drop_indices = indices[batch_idx]
                    drop_indices = drop_indices[drop_indices != 0] # keep [CLS] token
                    if drop_indices.numel() > 0:
                        flat[batch_idx, drop_indices] = 0

            I = torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device)
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            result = torch.matmul(a.to(result.device), result)
    
    # Look at the total attention between the class token and the image patches
    mask = result[0, 0, 1:]
    # Reshape the 1D array back to a 2D grid patch sizes.
    # Images are 224x224, patches are 16x16 -> 14x14 grid
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = np.maximum(mask, 0)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


def enhance_rollout_mask(mask):
    """Discard extreme spikes and locally repair discarded holes using local dxd median filter. Also set a percentage of low-value pixels to zero at the beginning."""
    low_percent = 20  # Set lowest 20% to zero
    local_d = 21
    outlier_z = 6.0
    fill_ksize = int(local_d) if int(local_d) % 2 == 1 else int(local_d) + 1  # Ensure odd kernel size
    fill_ksize = max(fill_ksize, 3)

    mask = np.asarray(mask, dtype=np.float32)
    mask = np.clip(mask, 0.0, None)

    # Set lowest X% to zero
    low_thresh = np.percentile(mask, low_percent)
    mask[mask <= low_thresh] = 0.0

    median = np.median(mask)
    mad = np.median(np.abs(mask - median))
    robust_sigma = 1.4826 * mad

    if robust_sigma < 1e-8:
        return np.clip(mask / (mask.max() + 1e-8), 0.0, 1.0)

    # Value-based discard: values above upper_cap are set to zero.
    upper_cap = median + outlier_z * robust_sigma
    discarded = mask > upper_cap
    mask = np.where(discarded, 0.0, mask)
    mask = mask / (upper_cap + 1e-8)

    # Expand discarded points to their local_d neighborhoods, then median-filter once.
    if discarded.any():
        mask_uint8 = np.uint8(np.clip(mask * 255, 0, 255))
        median_filtered = cv2.medianBlur(mask_uint8, fill_ksize).astype(np.float32) / 255.0

        kernel = np.ones((fill_ksize, fill_ksize), dtype=np.uint8)
        repair_region = cv2.dilate(discarded.astype(np.uint8), kernel, iterations=1).astype(bool)

        mask = np.where(repair_region, median_filtered, mask)

    return np.clip(mask, 0.0, 1.0)

class VITAttentionGradRollout:
    def __init__(
        self,
        model,
        discard_ratio=0.2,
        head_fusion="mean",
        large_reweight="sqrt",
        large_quantile=0.95,
        large_power=0.5,
    ):
        self.model = model
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion
        self.large_reweight = large_reweight
        self.large_quantile = large_quantile
        self.large_power = large_power

    def __call__(self, input_tensor, category_index):
        self.model.zero_grad(set_to_none=True)

        # Forward pass with attention outputs enabled
        outputs = self.model(input_tensor, output_attentions=True)
        logits = outputs.logits
        attentions = list(outputs.attentions)

        if not attentions:
            raise RuntimeError(
                "Model returned no attentions. Ensure output_attentions=True is supported."
            )

        for att in attentions:
            att.retain_grad()
        
        # Prepare category mask on the correct device
        category_mask = torch.zeros_like(logits)
        category_mask[:, category_index] = 1
        
        # Backward pass restricted to category mask
        loss = (logits * category_mask).sum()
        loss.backward()

        gradients = [att.grad for att in attentions]
        if any(g is None for g in gradients):
            raise RuntimeError("Some attention gradients are None; cannot compute grad rollout.")

        # Move intermediate tensors to CPU to reduce GPU memory pressure
        attentions_cpu = [a.detach().cpu() for a in attentions]
        gradients_cpu = [g.detach().cpu() for g in gradients]

        return grad_rollout(
            attentions_cpu,
            gradients_cpu,
            self.discard_ratio,
            self.head_fusion,
            self.large_reweight,
            self.large_quantile,
            self.large_power,
        )

def show_mask_on_image(img, mask):
    img = np.asarray(img, dtype=np.float32) / 255.0
    mask = np.asarray(mask, dtype=np.float32)
    mask = np.clip(mask, 0.0, 1.0)

    # OpenCV colormaps are BGR; convert to RGB to match the input image here.
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Blend using the mask itself as per-pixel alpha.
    alpha = np.expand_dims(mask, axis=-1)
    cam = img * (1.0 - alpha) + heatmap * alpha
    return np.uint8(np.clip(cam, 0.0, 1.0) * 255)


def save_image_unicode_safe(path, image_bgr):
    """Save an image robustly even when the path contains non-ASCII characters."""
    ok = cv2.imwrite(path, image_bgr)
    if ok:
        return

    ext = os.path.splitext(path)[1] or ".png"
    success, encoded = cv2.imencode(ext, image_bgr)
    if not success:
        raise RuntimeError(f"Failed to encode image for '{path}'.")

    # OpenCV can fail on some Windows Unicode paths; write encoded bytes directly.
    encoded.tofile(path)
    if not os.path.exists(path):
        raise RuntimeError(f"Failed to save visualization to '{path}'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Input image path')
    parser.add_argument('--discard_ratio', type=float, default=0.2, help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--head_fusion', type=str, default='mean', choices=['mean', 'max', 'min'], help='How to fuse attention heads before rollout')
    parser.add_argument('--large_reweight', type=str, default='none', choices=['none', 'sqrt', 'power', 'log'], help='Reweight only very large attention values to avoid extreme spikes')
    parser.add_argument('--large_quantile', type=float, default=0.95, help='Quantile threshold above which large attention values are compressed (0-1)')
    parser.add_argument('--large_power', type=float, default=0.5, help='Exponent for --large_reweight power mode, in (0,1]')
    args = parser.parse_args()

    if not (0.0 <= args.discard_ratio <= 1.0):
        raise ValueError("--discard_ratio must be between 0 and 1.")
    if not (0.0 < args.large_quantile < 1.0):
        raise ValueError("--large_quantile must be between 0 and 1 (exclusive).")
    if not (0.0 < args.large_power <= 1.0):
        raise ValueError("--large_power must be in (0, 1].")

    # Determine CPU / GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on device: {device}")

    # Load HF model and processor from local cache
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_path = os.path.abspath(os.path.join(script_dir, "..", "models", "emotion_vit"))
    
    if not os.path.exists(local_model_path):
        print(f"Error: Model not found at '{local_model_path}'.")
        print("Please run 'python src/download_model.py' first.")
        sys.exit(1)

    print(f"Loading cached model from {local_model_path}...")
    processor = AutoImageProcessor.from_pretrained(local_model_path)
    # Use eager attention so output_attentions returns real tensors (SDPA can return None here).
    model = ViTForImageClassification.from_pretrained(
        local_model_path,
        attn_implementation="eager",
    )
    
    # Place model on device
    model.to(device)
    model.eval()

    # Initialize Rollout Generator
    grad_rollout = VITAttentionGradRollout(
        model,
        discard_ratio=args.discard_ratio,
        head_fusion=args.head_fusion,
        large_reweight=args.large_reweight,
        large_quantile=args.large_quantile,
        large_power=args.large_power,
    )

    # Prepare Image
    print(f"Reading image from {args.image_path}...")
    raw_img = Image.open(args.image_path).convert('RGB')
    
    # Process using HF processor to get exact tensor needed by the model
    # Move tensor to correct device
    inputs = processor(images=raw_img, return_tensors="pt")
    input_tensor = inputs['pixel_values'].to(device)

    # Inference: forward pass to determine dominant category
    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits
        predicted_idx = logits.argmax(-1).item()
        
    predicted_emotion = model.config.id2label[predicted_idx]
    print(f"\n--- Result ---")
    print(f"Predicted emotion: {predicted_emotion.upper()} (Index: {predicted_idx})")

    # Generate Attention Gradient Rollout Map
    print(f"Generating Gradient Rollout mask for '{predicted_emotion}'...")
    # Enable gradients locally for the input tensor
    input_tensor.requires_grad = True
    mask = grad_rollout(input_tensor, predicted_idx)

    # Visualization Setup
    img_resized = raw_img.resize((224, 224))
    np_img = np.array(img_resized)[:, :, ::-1] # Convert RGB to BGR for OpenCV
    
    # Resize the 14x14 map to 224x224 smoothly
    mask_resized = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask_resized = enhance_rollout_mask(mask_resized)
    cam = show_mask_on_image(np_img[:, :, ::-1], mask_resized) # Pass RGB array to the overlay function
    
    # Save Output
    cam_bgr = cam[:, :, ::-1] # Back to BGR to save to disk
    out_dir = os.path.join(script_dir, "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    basename = os.path.basename(args.image_path)
    filename_without_ext = os.path.splitext(basename)[0]
    ratio_str = str(args.discard_ratio).replace('.', 'p')
    out_filename = os.path.join(
        out_dir,
        f"{filename_without_ext}_rollout_{predicted_emotion}_d{ratio_str}_{args.head_fusion}_{args.large_reweight}.png",
    )
    
    save_image_unicode_safe(out_filename, cam_bgr)
    print(f"✅ Saved visualization to '{out_filename}'!")

if __name__ == '__main__':
    main()
