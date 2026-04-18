"""
Exploratory Data Analysis for UAVid Dataset.
Covers dataset statistics, class distribution, image properties, and visual analysis.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

sys.path.append(str(Path(__file__).parent))
from config import (
    TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS,
    TEST_IMAGES, CLASS_MAP, CLASS_NAMES, CLASS_COLORS,
    COLOR_TO_CLASS, OUTPUT_DIR, TOLERANCE
)


def collect_image_paths(split="train"):
    splits = {
        "train": (TRAIN_IMAGES, TRAIN_LABELS),
        "val":   (VAL_IMAGES,   VAL_LABELS),
        "test":  (TEST_IMAGES,  None),
    }
    img_dir, lbl_dir = splits[split]
    img_paths, lbl_paths = [], []

    if not img_dir.exists():
        print(f"Directory not found: {img_dir}")
        return img_paths, lbl_paths

    for img_file in sorted(img_dir.glob("*.png")):
        img_paths.append(img_file)
        if lbl_dir:
            lbl_file = lbl_dir / img_file.name
            lbl_paths.append(lbl_file if lbl_file.exists() else None)

    return img_paths, lbl_paths


def get_dataset_summary():
    print("\n--- Dataset Summary ---")
    summary = {}
    for split in ["train", "val", "test"]:
        imgs, lbls = collect_image_paths(split)
        valid_lbls = [l for l in lbls if l is not None] if lbls else []
        summary[split] = {"images": len(imgs), "labels": len(valid_lbls)}
        print(f"  {split:5s} | images: {len(imgs):4d} | labels: {len(valid_lbls):4d}")

    total_imgs = sum(v["images"] for v in summary.values())
    total_lbls = sum(v["labels"] for v in summary.values())
    print(f"  {'TOTAL':5s} | images: {total_imgs:4d} | labels: {total_lbls:4d}")
    return summary


def analyze_image_properties(split="train", max_samples=20):
    print(f"\n--- Image Properties ({split}) ---")
    img_paths, _ = collect_image_paths(split)
    if not img_paths:
        print("  No images found.")
        return {}

    samples = img_paths[:max_samples]
    widths, heights, channels_list, filesizes = [], [], [], []

    for p in tqdm(samples, desc="Reading image properties"):
        try:
            img = Image.open(p)
            w, h = img.size
            c = len(img.getbands())
            fs = p.stat().st_size / 1024
            widths.append(w)
            heights.append(h)
            channels_list.append(c)
            filesizes.append(fs)
        except Exception as e:
            print(f"  Error reading {p.name}: {e}")

    stats = {
        "width":    {"mean": np.mean(widths),    "min": np.min(widths),    "max": np.max(widths)},
        "height":   {"mean": np.mean(heights),   "min": np.min(heights),   "max": np.max(heights)},
        "filesize_kb": {"mean": np.mean(filesizes), "min": np.min(filesizes), "max": np.max(filesizes)},
    }

    print(f"  Width   : mean={stats['width']['mean']:.0f}  min={stats['width']['min']}  max={stats['width']['max']}")
    print(f"  Height  : mean={stats['height']['mean']:.0f}  min={stats['height']['min']}  max={stats['height']['max']}")
    print(f"  Filesize: mean={stats['filesize_kb']['mean']:.1f} KB  min={stats['filesize_kb']['min']:.1f} KB  max={stats['filesize_kb']['max']:.1f} KB")
    print(f"  Channels: {set(channels_list)}")
    return stats


def pixel_to_class(pixel, tolerance=TOLERANCE):
    best_class, best_dist = "Background clutter", float("inf")
    for cls_name, color in CLASS_MAP.items():
        dist = sum((int(pixel[i]) - color[i]) ** 2 for i in range(3))
        if dist < best_dist:
            best_dist = dist
            best_class = cls_name
    return best_class


def compute_class_distribution(label_path):
    label = np.array(Image.open(label_path).convert("RGB"))
    h, w = label.shape[:2]
    total_pixels = h * w
    class_pixels = defaultdict(int)

    for cls_name, color in CLASS_MAP.items():
        mask = (
            (np.abs(label[:, :, 0].astype(int) - color[0]) <= TOLERANCE) &
            (np.abs(label[:, :, 1].astype(int) - color[1]) <= TOLERANCE) &
            (np.abs(label[:, :, 2].astype(int) - color[2]) <= TOLERANCE)
        )
        class_pixels[cls_name] = int(mask.sum())

    return {k: v / total_pixels * 100 for k, v in class_pixels.items()}


def analyze_class_distribution(split="train", max_samples=30):
    print(f"\n--- Class Distribution Analysis ({split}, up to {max_samples} images) ---")
    img_paths, lbl_paths = collect_image_paths(split)
    valid_pairs = [(i, l) for i, l in zip(img_paths, lbl_paths) if l is not None]
    valid_pairs = valid_pairs[:max_samples]

    if not valid_pairs:
        print("  No labeled images found.")
        return {}

    all_distributions = defaultdict(list)
    for _, lbl_path in tqdm(valid_pairs, desc="Computing class distributions"):
        dist = compute_class_distribution(lbl_path)
        for cls, pct in dist.items():
            all_distributions[cls].append(pct)

    summary = {}
    print(f"\n  {'Class':<22} {'Mean %':>8} {'Std %':>8} {'Min %':>8} {'Max %':>8}")
    print(f"  {'-'*56}")
    for cls in CLASS_NAMES:
        vals = all_distributions[cls]
        mean_v, std_v = np.mean(vals), np.std(vals)
        min_v, max_v = np.min(vals), np.max(vals)
        summary[cls] = {"mean": mean_v, "std": std_v, "min": min_v, "max": max_v}
        print(f"  {cls:<22} {mean_v:>8.2f} {std_v:>8.2f} {min_v:>8.2f} {max_v:>8.2f}")

    dominant = max(summary, key=lambda c: summary[c]["mean"])
    rarest   = min(summary, key=lambda c: summary[c]["mean"])
    print(f"\n  Dominant class : {dominant} ({summary[dominant]['mean']:.2f}%)")
    print(f"  Rarest class   : {rarest}  ({summary[rarest]['mean']:.2f}%)")
    return summary, all_distributions


def analyze_sequence_structure():
    print("\n--- Sequence Structure Analysis ---")
    for split in ["train", "val", "test"]:
        img_paths, _ = collect_image_paths(split)
        sequences = defaultdict(list)
        for p in img_paths:
            seq_id = p.stem.split("_")[0]
            sequences[seq_id].append(p.name)
        print(f"  {split:5s} | sequences: {len(sequences)} | "
              f"frames range: {min(len(v) for v in sequences.values()) if sequences else 0}"
              f"-{max(len(v) for v in sequences.values()) if sequences else 0}")
        for seq, frames in sorted(sequences.items())[:3]:
            print(f"    {seq}: {len(frames)} frames")
        if len(sequences) > 3:
            print(f"    ... and {len(sequences)-3} more sequences")


def plot_class_distribution(summary_dict, split="train"):
    if not summary_dict:
        return
    classes = CLASS_NAMES
    means = [summary_dict[c]["mean"] for c in classes]
    stds  = [summary_dict[c]["std"]  for c in classes]
    colors_norm = [tuple(c/255 for c in CLASS_MAP[cls]) for cls in classes]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"UAVid Class Distribution - {split.capitalize()} Set", fontsize=14, fontweight="bold")

    bars = axes[0].barh(classes, means, xerr=stds, color=colors_norm, edgecolor="black", linewidth=0.6, capsize=4)
    axes[0].set_xlabel("Mean Coverage (%)")
    axes[0].set_title("Mean Class Coverage with Std Dev")
    axes[0].invert_yaxis()
    for bar, mean in zip(bars, means):
        axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                     f"{mean:.1f}%", va="center", fontsize=9)

    pie_colors = [tuple(c/255 for c in CLASS_MAP[cls]) for cls in classes]
    wedges, texts, autotexts = axes[1].pie(
        means, labels=None, colors=pie_colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.82,
        wedgeprops={"edgecolor": "white", "linewidth": 0.8}
    )
    for at in autotexts:
        at.set_fontsize(8)
    axes[1].legend(wedges, classes, loc="lower left", fontsize=8, bbox_to_anchor=(-0.1, -0.1))
    axes[1].set_title("Proportional Class Coverage")

    plt.tight_layout()
    save_path = OUTPUT_DIR / f"eda_class_distribution_{split}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_class_variability(all_distributions, split="train"):
    if not all_distributions:
        return
    data_for_box = [all_distributions[cls] for cls in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(data_for_box, labels=CLASS_NAMES, patch_artist=True, notch=False)
    colors_norm = [tuple(c/255 for c in CLASS_MAP[cls]) for cls in CLASS_NAMES]
    for patch, color in zip(bp["boxes"], colors_norm):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_ylabel("Coverage (%)")
    ax.set_title(f"Class Coverage Variability Across Images - {split.capitalize()} Set")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"eda_class_variability_{split}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_sample_images(split="train", n=4):
    img_paths, lbl_paths = collect_image_paths(split)
    valid_pairs = [(i, l) for i, l in zip(img_paths, lbl_paths) if l is not None][:n]
    if not valid_pairs:
        return

    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    fig.suptitle(f"UAVid Samples - {split.capitalize()} Set", fontsize=13, fontweight="bold")

    for row, (img_path, lbl_path) in enumerate(valid_pairs):
        img = np.array(Image.open(img_path).convert("RGB"))
        lbl = np.array(Image.open(lbl_path).convert("RGB"))
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"Image: {img_path.name}", fontsize=9)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(lbl)
        axes[row, 1].set_title(f"Label: {lbl_path.name}", fontsize=9)
        axes[row, 1].axis("off")

    patches = [mpatches.Patch(color=tuple(c/255 for c in v), label=k) for k, v in CLASS_MAP.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"eda_samples_{split}.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pixel_intensity_analysis(split="train", max_samples=10):
    img_paths, _ = collect_image_paths(split)
    samples = img_paths[:max_samples]
    if not samples:
        return

    channel_means = {"R": [], "G": [], "B": []}
    for p in samples:
        img = np.array(Image.open(p).convert("RGB")).astype(float)
        channel_means["R"].append(img[:, :, 0].mean())
        channel_means["G"].append(img[:, :, 1].mean())
        channel_means["B"].append(img[:, :, 2].mean())

    print(f"\n--- Pixel Intensity Statistics ({split}, {len(samples)} samples) ---")
    for ch, vals in channel_means.items():
        print(f"  Channel {ch}: mean={np.mean(vals):.2f}  std={np.std(vals):.2f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(samples))
    ax.plot(x, channel_means["R"], color="red",   label="Red",   marker="o", markersize=4)
    ax.plot(x, channel_means["G"], color="green", label="Green", marker="s", markersize=4)
    ax.plot(x, channel_means["B"], color="blue",  label="Blue",  marker="^", markersize=4)
    ax.set_xlabel("Image Index")
    ax.set_ylabel("Mean Pixel Intensity")
    ax.set_title(f"Per-Channel Mean Intensity Across Images - {split.capitalize()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"eda_intensity_{split}.png"
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def compute_class_cooccurrence(split="train", max_samples=30):
    print(f"\n--- Class Co-occurrence Matrix ({split}) ---")
    img_paths, lbl_paths = collect_image_paths(split)
    valid_pairs = [(i, l) for i, l in zip(img_paths, lbl_paths) if l is not None][:max_samples]

    n = len(CLASS_NAMES)
    cooccur = np.zeros((n, n), dtype=float)

    for _, lbl_path in tqdm(valid_pairs, desc="Computing co-occurrence"):
        dist = compute_class_distribution(lbl_path)
        present = [i for i, cls in enumerate(CLASS_NAMES) if dist[cls] > 0.5]
        for i in present:
            for j in present:
                cooccur[i, j] += 1

    cooccur_norm = cooccur / len(valid_pairs) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cooccur_norm, annot=True, fmt=".0f", cmap="YlOrRd",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.4, cbar_kws={"label": "Co-occurrence (%)"}
    )
    ax.set_title(f"Class Co-occurrence Matrix - {split.capitalize()} (% of images)")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"eda_cooccurrence_{split}.png"
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def run_full_eda():
    print("UAVid Dataset - Full Exploratory Data Analysis")
    print("=" * 60)

    get_dataset_summary()
    analyze_sequence_structure()

    for split in ["train", "val"]:
        analyze_image_properties(split, max_samples=20)
        plot_pixel_intensity_analysis(split, max_samples=10)
        result = analyze_class_distribution(split, max_samples=30)
        if result:
            summary, all_dists = result
            plot_class_distribution(summary, split)
            plot_class_variability(all_dists, split)
            compute_class_cooccurrence(split, max_samples=30)
        plot_sample_images(split, n=4)

    print("\nEDA complete. All outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    run_full_eda()
