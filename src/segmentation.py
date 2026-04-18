"""
Semantic Segmentation Pipeline for UAVid Dataset.
Uses SegFormer-B0 for real inference + label-based ground truth comparison.
Computes real evaluation metrics: pixel accuracy, per-class IoU, mIoU, FW-IoU.
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

sys.path.append(str(Path(__file__).parent))
from config import (
    TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS,
    TEST_IMAGES, CLASS_MAP, CLASS_NAMES, CLASS_COLORS,
    OUTPUT_DIR, TOLERANCE
)
from segformer_model import (
    build_segformer_model, run_inference, compute_metrics_from_masks,
    generate_insight_text, mask_to_color, NUM_CLASSES
)


def color_to_class_id(label_rgb_array):
    h, w        = label_rgb_array.shape[:2]
    class_mask  = np.zeros((h, w), dtype=np.uint8)
    for cls_id, (cls_name, color) in enumerate(CLASS_MAP.items()):
        mask = (
            (np.abs(label_rgb_array[:, :, 0].astype(int) - color[0]) <= TOLERANCE) &
            (np.abs(label_rgb_array[:, :, 1].astype(int) - color[1]) <= TOLERANCE) &
            (np.abs(label_rgb_array[:, :, 2].astype(int) - color[2]) <= TOLERANCE)
        )
        class_mask[mask] = cls_id
    return class_mask


def compute_per_class_stats(class_mask, total_pixels):
    stats = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        count    = int((class_mask == cls_id).sum())
        coverage = count / total_pixels * 100
        stats[cls_name] = {"pixel_count": count, "coverage_pct": round(coverage, 4)}
    return stats


def print_segmentation_report(image_name, split, width, height, total_pixels, class_stats):
    print(f"\n  Image     : {image_name}")
    print(f"  Split     : {split}")
    print(f"  Dimensions: {width} x {height} px ({total_pixels:,} total pixels)")
    print(f"\n  {'Class':<22} {'Pixels':>12} {'Coverage %':>12}")
    print(f"  {'-'*48}")
    sorted_cls = sorted(class_stats.items(), key=lambda x: x[1]["coverage_pct"], reverse=True)
    for cls_name, s in sorted_cls:
        bar = "#" * int(s["coverage_pct"] / 2)
        print(f"  {cls_name:<22} {s['pixel_count']:>12,} {s['coverage_pct']:>11.2f}%  {bar}")


def print_metrics_report(metrics):
    print(f"\n  Pixel Accuracy : {metrics['pixel_accuracy']:.4f}%")
    print(f"  Mean IoU       : {metrics['mean_iou']:.4f}%")
    print(f"  FW-IoU         : {metrics['fw_iou']:.4f}%")
    print(f"\n  {'Class':<22} {'IoU %':>10}")
    print(f"  {'-'*34}")
    for cls_name, iou in metrics["per_class_iou"].items():
        bar = "#" * int(iou / 5)
        print(f"  {cls_name:<22} {iou:>10.4f}%  {bar}")


def analyze_single_label(label_path, image_name, split):
    label_rgb    = np.array(Image.open(label_path).convert("RGB"))
    h, w         = label_rgb.shape[:2]
    total_pixels = h * w
    class_mask   = color_to_class_id(label_rgb)
    stats        = compute_per_class_stats(class_mask, total_pixels)
    insight      = generate_insight_text(image_name, stats, split)
    return {
        "image_name":   image_name,
        "split":        split,
        "width":        w,
        "height":       h,
        "total_pixels": total_pixels,
        "class_stats":  stats,
        "insight":      insight,
        "class_mask":   class_mask,
    }


def visualize_segmentation(image_path, label_path, pred_mask, gt_mask, image_name, save_path):
    img        = np.array(Image.open(image_path).convert("RGB"))
    lbl        = np.array(Image.open(label_path).convert("RGB"))
    pred_color = mask_to_color(pred_mask)
    gt_color   = mask_to_color(gt_mask)
    overlay_p  = (img * 0.5 + pred_color * 0.5).astype(np.uint8)
    overlay_g  = (img * 0.5 + gt_color   * 0.5).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    axes[0].imshow(img);       axes[0].set_title("Original Image");        axes[0].axis("off")
    axes[1].imshow(lbl);       axes[1].set_title("Ground Truth Label");    axes[1].axis("off")
    axes[2].imshow(overlay_p); axes[2].set_title("SegFormer-B0 Pred");    axes[2].axis("off")
    axes[3].imshow(overlay_g); axes[3].set_title("GT Overlay (ref)");     axes[3].axis("off")

    patches = [mpatches.Patch(color=tuple(c/255 for c in v), label=k) for k, v in CLASS_MAP.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    plt.suptitle(f"Segmentation: {image_name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close()


def run_batch_segmentation(split="train", max_samples=None, visualize_n=2, save_insights=True):
    print(f"\nBatch Segmentation - {split.upper()}")
    print("-" * 60)

    img_dir = TRAIN_IMAGES if split == "train" else VAL_IMAGES
    lbl_dir = TRAIN_LABELS if split == "train" else VAL_LABELS

    if not img_dir.exists():
        print(f"  Directory not found: {img_dir}")
        return []

    img_paths = sorted(img_dir.glob("*.png"))
    if max_samples:
        img_paths = img_paths[:max_samples]
    print(f"  Processing {len(img_paths)} images")

    print("  Loading SegFormer-B0 model...")
    model = build_segformer_model()
    print("  Model loaded")

    all_insights  = []
    all_pixel_acc = []
    iou_accum     = defaultdict(list)
    freq_accum    = defaultdict(float)
    total_px_all  = 0

    for idx, img_path in enumerate(tqdm(img_paths, desc=f"Segmenting {split}")):
        lbl_path = lbl_dir / img_path.name
        if not lbl_path.exists():
            continue

        image      = Image.open(img_path).convert("RGB")
        gt_label   = np.array(Image.open(lbl_path).convert("RGB"))
        gt_mask    = color_to_class_id(gt_label)
        h, w       = gt_mask.shape
        total_px   = h * w
        total_px_all += total_px

        result     = run_inference(model, image)
        pred_mask  = result["pred_mask"]
        stats      = result["stats"]

        import torch.nn.functional as F
        import torch
        pred_resized = torch.tensor(pred_mask).unsqueeze(0).unsqueeze(0).float()
        pred_resized = F.interpolate(pred_resized, size=(h, w), mode="nearest")
        pred_final   = pred_resized.squeeze().numpy().astype(np.uint8)

        metrics    = compute_metrics_from_masks(pred_final, gt_mask)
        gt_stats   = compute_per_class_stats(gt_mask, total_px)
        insight    = generate_insight_text(img_path.name, gt_stats, split)

        all_pixel_acc.append(metrics["pixel_accuracy"])
        for cls_name, iou in metrics["per_class_iou"].items():
            if iou > 0:
                iou_accum[cls_name].append(iou)
        for cls_id, cls_name in enumerate(CLASS_NAMES):
            freq_accum[cls_name] += float((gt_mask == cls_id).sum())

        all_insights.append({
            "image_name": img_path.name,
            "split":      split,
            "insight":    insight,
            "class_stats": {k: v["coverage_pct"] for k, v in gt_stats.items()},
            "metrics":    metrics,
        })

        if idx < visualize_n:
            print_segmentation_report(img_path.name, split, w, h, total_px, gt_stats)
            print_metrics_report(metrics)
            save_path = OUTPUT_DIR / f"seg_overlay_{split}_{idx}.png"
            visualize_segmentation(img_path, lbl_path, pred_final, gt_mask, img_path.name, save_path)
            print(f"  Saved: {save_path}")

    if not all_insights:
        print("  No results.")
        return []

    mean_pix_acc = float(np.mean(all_pixel_acc))
    miou_vals    = {c: float(np.mean(v)) for c, v in iou_accum.items() if v}
    miou         = float(np.mean(list(miou_vals.values()))) if miou_vals else 0.0
    freq_total   = sum(freq_accum.values())
    fw_iou       = sum((freq_accum[c] / freq_total) * miou_vals.get(c, 0.0) for c in CLASS_NAMES)

    print(f"\n  Aggregate Results ({split}, {len(all_insights)} images)")
    print(f"  Pixel Accuracy  : {mean_pix_acc:.4f}%")
    print(f"  Mean IoU (mIoU) : {miou:.4f}%")
    print(f"  FW-IoU          : {fw_iou:.4f}%")
    print(f"\n  {'Class':<22} {'mIoU %':>10}")
    print(f"  {'-'*34}")
    for cls_name in CLASS_NAMES:
        val = miou_vals.get(cls_name, 0.0)
        bar = "#" * int(val / 5)
        print(f"  {cls_name:<22} {val:>10.4f}%  {bar}")

    aggregate_metrics = {
        "split":            split,
        "images_evaluated": len(all_insights),
        "pixel_accuracy":   round(mean_pix_acc, 4),
        "mean_iou":         round(miou, 4),
        "fw_iou":           round(fw_iou, 4),
        "per_class_iou":    {c: round(miou_vals.get(c, 0.0), 4) for c in CLASS_NAMES},
        "per_class_freq_pct": {c: round(freq_accum[c] / freq_total * 100, 4) for c in CLASS_NAMES},
    }

    metrics_path = OUTPUT_DIR / f"metrics_{split}.json"
    with open(metrics_path, "w") as f:
        json.dump(aggregate_metrics, f, indent=2)
    print(f"\n  Metrics saved: {metrics_path}")

    if save_insights:
        out_path = OUTPUT_DIR / f"insights_{split}.json"
        with open(out_path, "w") as f:
            json.dump(all_insights, f, indent=2)
        print(f"  Insights saved: {out_path}")

    plot_aggregate_coverage(split, all_insights)
    return all_insights


def plot_aggregate_coverage(split, all_insights):
    if not all_insights:
        return
    coverage_by_class = defaultdict(list)
    for item in all_insights:
        for cls, pct in item["class_stats"].items():
            coverage_by_class[cls].append(pct)

    df  = pd.DataFrame(coverage_by_class)
    fig, ax = plt.subplots(figsize=(14, 5))
    df.plot(kind="area", stacked=True, ax=ax,
            color=[tuple(c/255 for c in CLASS_MAP[cls]) for cls in CLASS_NAMES], alpha=0.85)
    ax.set_xlabel("Image Index")
    ax.set_ylabel("Coverage (%)")
    ax.set_title(f"Stacked GT Class Coverage - {split.capitalize()}")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"seg_stacked_coverage_{split}.png"
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Stacked coverage chart saved: {save_path}")


def run_test_analysis():
    print(f"\nTest Set Analysis (no labels)")
    print("-" * 60)

    if not TEST_IMAGES.exists():
        print(f"  Directory not found: {TEST_IMAGES}")
        return []

    img_paths = sorted(TEST_IMAGES.glob("*.png"))
    print(f"  Found {len(img_paths)} test images")

    print("  Loading SegFormer-B0 model...")
    model = build_segformer_model()

    insights = []
    for idx, img_path in enumerate(tqdm(img_paths, desc="Test inference")):
        image  = Image.open(img_path).convert("RGB")
        result = run_inference(model, image)
        stats  = result["stats"]

        insight = (
            f"Image: {img_path.name} [test]. "
            f"SegFormer-B0 prediction (no GT label). "
            f"Dominant: {max(stats, key=lambda c: stats[c]['coverage_pct'])} "
            f"({max(stats.values(), key=lambda s: s['coverage_pct'])['coverage_pct']:.1f}%). "
            f"Road: {stats['Road']['coverage_pct']:.1f}%. "
            f"Building: {stats['Building']['coverage_pct']:.1f}%. "
            f"Inference time: {result['elapsed']:.2f}s."
        )

        insights.append({
            "image_name":  img_path.name,
            "split":       "test",
            "insight":     insight,
            "class_stats": {k: v["coverage_pct"] for k, v in stats.items()},
        })

        if idx < 3:
            print(f"  [{idx+1}] {img_path.name} | {result['insight'][:120]}...")

    out_path = OUTPUT_DIR / "insights_test.json"
    with open(out_path, "w") as f:
        json.dump(insights, f, indent=2)
    print(f"  Test insights saved: {out_path} ({len(insights)} images)")
    return insights


def plot_evaluation_metrics(metrics_train, metrics_val):
    if not metrics_train or not metrics_val:
        return
    classes   = CLASS_NAMES
    iou_train = [metrics_train["per_class_iou"].get(c, 0) for c in classes]
    iou_val   = [metrics_val["per_class_iou"].get(c, 0)   for c in classes]

    x = np.arange(len(classes))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("UAVid SegFormer-B0 Evaluation Metrics", fontsize=13, fontweight="bold")

    axes[0].bar(x - w/2, iou_train, w, label="Train", color="#2196F3", alpha=0.85, edgecolor="black", linewidth=0.5)
    axes[0].bar(x + w/2, iou_val,   w, label="Val",   color="#FF5722", alpha=0.85, edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("IoU (%)")
    axes[0].set_title("Per-Class IoU: Train vs Val")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    for xi, (t, v) in enumerate(zip(iou_train, iou_val)):
        axes[0].text(xi - w/2, t + 0.3, f"{t:.1f}", ha="center", fontsize=7)
        axes[0].text(xi + w/2, v + 0.3, f"{v:.1f}", ha="center", fontsize=7)

    summary_labels = ["Pixel Acc", "mIoU", "FW-IoU"]
    train_vals = [metrics_train["pixel_accuracy"], metrics_train["mean_iou"], metrics_train["fw_iou"]]
    val_vals   = [metrics_val["pixel_accuracy"],   metrics_val["mean_iou"],   metrics_val["fw_iou"]]

    xi2 = np.arange(len(summary_labels))
    axes[1].bar(xi2 - w/2, train_vals, w, label="Train", color="#2196F3", alpha=0.85, edgecolor="black", linewidth=0.5)
    axes[1].bar(xi2 + w/2, val_vals,   w, label="Val",   color="#FF5722", alpha=0.85, edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(xi2)
    axes[1].set_xticklabels(summary_labels)
    axes[1].set_ylabel("Score (%)")
    axes[1].set_title("Summary Metrics")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    for xi, (t, v) in enumerate(zip(train_vals, val_vals)):
        axes[1].text(xi - w/2, t + 0.3, f"{t:.1f}", ha="center", fontsize=9)
        axes[1].text(xi + w/2, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "metrics_comparison.png"
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Metrics chart saved: {save_path}")


def run_full_segmentation_pipeline():
    print("UAVid SegFormer-B0 Segmentation Pipeline")
    print("=" * 60)

    train_insights = run_batch_segmentation("train", max_samples=None, visualize_n=2)
    val_insights   = run_batch_segmentation("val",   max_samples=None, visualize_n=2)
    test_insights  = run_test_analysis()

    metrics_train_path = OUTPUT_DIR / "metrics_train.json"
    metrics_val_path   = OUTPUT_DIR / "metrics_val.json"
    metrics_train = json.load(open(metrics_train_path)) if metrics_train_path.exists() else {}
    metrics_val   = json.load(open(metrics_val_path))   if metrics_val_path.exists()   else {}
    plot_evaluation_metrics(metrics_train, metrics_val)

    all_insights = train_insights + val_insights + test_insights
    all_path     = OUTPUT_DIR / "insights_all.json"
    with open(all_path, "w") as f:
        json.dump(all_insights, f, indent=2)

    print(f"\nPipeline complete.")
    print(f"  Train: {len(train_insights)} | Val: {len(val_insights)} | Test: {len(test_insights)}")
    print(f"  Total insight documents: {len(all_insights)}")
    print(f"  All insights: {all_path}")
    return all_insights


if __name__ == "__main__":
    run_full_segmentation_pipeline()
