"""
SegFormer-B0 model loader and inference for UAVid dataset.
Automatically loads fine-tuned checkpoint if available,
otherwise falls back to pretrained weights with a clear warning.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
import time
from PIL import Image
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from config import CLASS_NAMES, CLASS_COLORS, CLASS_MAP, OUTPUT_DIR

NUM_CLASSES    = 8
MODEL_NAME     = "nvidia/mit-b0"
IMAGE_SIZE     = 512
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoints" / "segformer_b0_uavid_best.pth"


def build_segformer_model(checkpoint_path: Path = None):
    from transformers import SegformerForSemanticSegmentation, SegformerConfig

    config = SegformerConfig.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label={str(i): n for i, n in enumerate(CLASS_NAMES)},
        label2id={n: i for i, n in enumerate(CLASS_NAMES)},
        ignore_mismatched_sizes=True,
    )

    resolved = checkpoint_path or CHECKPOINT_PATH

    if resolved.exists():
        print(f"  Loading fine-tuned checkpoint: {resolved}")
        model = SegformerForSemanticSegmentation(config)
        ckpt  = torch.load(str(resolved), map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        epoch    = ckpt.get("epoch", "?")
        val_miou = ckpt.get("val_miou", 0.0)
        print(f"  Checkpoint epoch={epoch} | val_mIoU={val_miou*100:.2f}%")
    else:
        print(f"  WARNING: No fine-tuned checkpoint found at {resolved}")
        print(f"  Loading pretrained MiT-B0 backbone (predictions will be imprecise).")
        print(f"  Run `python src/train_segformer.py` to fine-tune first.")
        model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAME,
            config=config,
            ignore_mismatched_sizes=True,
        )

    model.eval()
    return model


def preprocess_image(image: Image.Image, size: int = IMAGE_SIZE):
    from torchvision import transforms
    image_rgb = image.convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    return tf(image_rgb).unsqueeze(0), image_rgb


def mask_to_color(pred_mask: np.ndarray) -> np.ndarray:
    h, w       = pred_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CLASS_COLORS):
        color_mask[pred_mask == cls_id] = color
    return color_mask


def compute_coverage_stats(pred_mask: np.ndarray) -> dict:
    total = pred_mask.size
    stats = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        count = int((pred_mask == cls_id).sum())
        stats[cls_name] = {
            "pixel_count":  count,
            "coverage_pct": round(count / total * 100, 4),
        }
    return stats


def run_inference(model, image: Image.Image) -> dict:
    t0          = time.time()
    tensor, img_rgb = preprocess_image(image)
    orig_w, orig_h  = img_rgb.size

    with torch.no_grad():
        outputs = model(pixel_values=tensor)
        logits  = outputs.logits

    logits_up  = F.interpolate(logits, size=(orig_h, orig_w),
                                mode="bilinear", align_corners=False)
    pred_mask  = logits_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    color_mask = mask_to_color(pred_mask)
    overlay    = (np.array(img_rgb) * 0.45 + color_mask * 0.55).astype(np.uint8)
    stats      = compute_coverage_stats(pred_mask)
    elapsed    = time.time() - t0

    sorted_cls = sorted(stats.items(), key=lambda x: x[1]["coverage_pct"], reverse=True)
    dominant   = sorted_cls[0][0]
    top3       = ", ".join([f"{c} ({s['coverage_pct']:.1f}%)" for c, s in sorted_cls[:3]])
    present    = [c for c, s in stats.items() if s["coverage_pct"] > 1.0]

    insight = (
        f"SegFormer-B0 prediction. "
        f"Dominant: {dominant} ({stats[dominant]['coverage_pct']:.1f}%). "
        f"Top 3: {top3}. Present (>1%): {', '.join(present)}. "
        f"Vegetation: {stats['Tree']['coverage_pct'] + stats['Low vegetation']['coverage_pct']:.1f}%. "
        f"Road: {stats['Road']['coverage_pct']:.1f}%. Inference time: {elapsed:.2f}s."
    )

    print(f"  Inference: {elapsed:.2f}s | dominant={dominant} ({stats[dominant]['coverage_pct']:.1f}%)")
    print(f"  Top 3: {top3}")

    return {
        "pred_mask":  pred_mask,
        "color_mask": color_mask,
        "overlay":    overlay,
        "stats":      stats,
        "insight":    insight,
        "elapsed":    elapsed,
    }


def compute_metrics_from_masks(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    total    = pred_mask.size
    pix_acc  = float((pred_mask == gt_mask).sum()) / total

    iou_per_class = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        pred_c = (pred_mask == cls_id)
        gt_c   = (gt_mask   == cls_id)
        inter  = (pred_c & gt_c).sum()
        union  = (pred_c | gt_c).sum()
        iou_per_class[cls_name] = float(inter) / float(union) if union > 0 else None

    valid    = [v for v in iou_per_class.values() if v is not None]
    miou     = float(np.mean(valid)) if valid else 0.0

    freq_w   = {cls: float((gt_mask == i).sum()) / total
                for i, cls in enumerate(CLASS_NAMES)}
    fw_iou   = sum(freq_w[c] * (iou_per_class[c] or 0.0) for c in CLASS_NAMES)

    return {
        "pixel_accuracy": round(pix_acc * 100, 4),
        "mean_iou":       round(miou * 100, 4),
        "fw_iou":         round(fw_iou * 100, 4),
        "per_class_iou":  {c: round(v * 100, 4) if v is not None else 0.0
                           for c, v in iou_per_class.items()},
    }


def generate_insight_text(image_name: str, stats: dict, split: str = "inference") -> str:
    sorted_cls = sorted(stats.items(),
                        key=lambda x: x[1]["coverage_pct"] if isinstance(x[1], dict) else x[1],
                        reverse=True)
    get_pct = lambda c: stats[c]["coverage_pct"] if isinstance(stats[c], dict) else stats[c]
    dominant = sorted_cls[0][0]
    top3     = ", ".join([f"{c} ({get_pct(c):.1f}%)" for c, _ in sorted_cls[:3]])
    present  = [c for c in CLASS_NAMES if get_pct(c) > 1.0]
    absent   = [c for c in CLASS_NAMES if get_pct(c) < 0.1]
    veg      = get_pct("Tree") + get_pct("Low vegetation")
    scene    = ("Urban Dense"  if get_pct("Building") > 30 else
                "Mixed Urban"  if get_pct("Building") > 10 else
                "Natural/Open")
    return (
        f"Image: {image_name} [{split}]. "
        f"Dominant: {dominant} ({get_pct(dominant):.1f}%). "
        f"Top 3: {top3}. "
        f"Present (>1%): {', '.join(present) if present else 'none'}. "
        f"Absent (<0.1%): {', '.join(absent) if absent else 'none'}. "
        f"Scene: {scene}. Vegetation: {veg:.1f}%. "
        f"Road: {get_pct('Road'):.1f}%. "
        f"Moving car: {get_pct('Moving car'):.2f}%. "
        f"Static car: {get_pct('Static car'):.2f}%. "
        f"Human: {get_pct('Human'):.3f}%."
    )