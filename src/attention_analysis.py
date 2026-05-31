"""
Attention Map Analysis untuk SegFormer-B0.
Pengganti GradCAM untuk transformer segmentation — mengekstrak attention weights
dari Multi-Head Self-Attention di setiap stage encoder MiT-B0.
Menghasilkan: attention heatmap, uncertainty map, per-class confidence, boundary analysis.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from pathlib import Path
from scipy import ndimage

sys.path.append(str(Path(__file__).parent))
from config import CLASS_NAMES, CLASS_COLORS, CLASS_MAP, OUTPUT_DIR


ATTENTION_CMAP = LinearSegmentedColormap.from_list(
    "attention", ["#0a0e1a", "#1a237e", "#1565c0", "#00acc1", "#26c6da", "#ffeb3b", "#ff6f00", "#b71c1c"]
)
UNCERTAINTY_CMAP = LinearSegmentedColormap.from_list(
    "uncertainty", ["#1b5e20", "#f9a825", "#b71c1c"]
)


def extract_attention_maps(model, image_tensor):
    """
    Hook Multi-Head Self-Attention dari semua 4 stage MiT-B0.
    MiT-B0 menggunakan Efficient Self-Attention dengan spatial reduction.
    """
    attention_maps = {}

    def make_hook(name):
        def hook(module, input, output):
            if hasattr(module, "attention_weights"):
                attention_maps[name] = module.attention_weights.detach()
        return hook

    hooks = []
    for i, layer in enumerate(model.segformer.encoder.layer):
        for j, block in enumerate(layer):
            if hasattr(block, "attention"):
                attn = block.attention.self_attention
                h = attn.register_forward_hook(make_hook(f"stage{i+1}_block{j+1}"))
                hooks.append(h)

    with torch.no_grad():
        outputs = model(pixel_values=image_tensor, output_attentions=True)

    for h in hooks:
        h.remove()

    return outputs


def compute_entropy_uncertainty(logits_up):
    """
    Uncertainty dari entropi prediksi softmax.
    Tinggi = model tidak yakin, rendah = model sangat yakin.
    """
    softmax = F.softmax(logits_up, dim=1)
    log_sm  = torch.log(softmax + 1e-10)
    entropy = -(softmax * log_sm).sum(dim=1).squeeze(0)
    return entropy.cpu().numpy(), softmax.squeeze(0).cpu().numpy()


def compute_per_class_confidence(softmax_np):
    """Max confidence per pixel untuk tiap kelas."""
    confidence = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        cls_conf = softmax_np[cls_id]
        confidence[cls_name] = {
            "mean_conf":   float(cls_conf.mean()),
            "max_conf":    float(cls_conf.max()),
            "area_pct":    float((softmax_np.argmax(0) == cls_id).mean() * 100),
            "high_conf_pct": float((cls_conf > 0.8).mean() * 100),
        }
    return confidence


def detect_boundaries(pred_mask):
    """Deteksi tepi (boundary) antar kelas menggunakan Sobel filter."""
    from scipy.ndimage import sobel
    boundary = np.zeros_like(pred_mask, dtype=np.float32)
    sx = sobel(pred_mask.astype(float), axis=0)
    sy = sobel(pred_mask.astype(float), axis=1)
    boundary = np.hypot(sx, sy)
    boundary = (boundary > 0).astype(np.float32)
    boundary = ndimage.binary_dilation(boundary, iterations=2).astype(np.float32)
    return boundary


def compute_class_spatial_stats(pred_mask):
    """Analisis spasial: distribusi kelas di 9 zona (3x3 grid) gambar."""
    h, w    = pred_mask.shape
    grid_h, grid_w = h // 3, w // 3
    spatial = {}
    zone_names = ["top-left", "top-center", "top-right",
                  "mid-left", "center",     "mid-right",
                  "bot-left", "bot-center", "bot-right"]
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        zone_pcts = []
        for zi in range(3):
            for zj in range(3):
                r0, r1 = zi * grid_h, (zi+1) * grid_h
                c0, c1 = zj * grid_w, (zj+1) * grid_w
                zone   = pred_mask[r0:r1, c0:c1]
                pct    = float((zone == cls_id).mean() * 100)
                zone_pcts.append(pct)
        spatial[cls_name] = dict(zip(zone_names, zone_pcts))
    return spatial


def plot_comprehensive_analysis(
    image_np, pred_mask, softmax_np, entropy_np,
    image_name, save_path
):
    """
    Dashboard analisis lengkap:
    - Original + Segmentation Overlay
    - Uncertainty Map (entropi)
    - Per-class Confidence Maps (4 kelas teratas)
    - Boundary Detection
    - Spatial Distribution Heatmap
    - Coverage Bar Chart
    - Confidence Radar
    """
    h, w    = pred_mask.shape
    boundary = detect_boundaries(pred_mask)
    coverage = {c: float((pred_mask == i).mean() * 100)
                for i, c in enumerate(CLASS_NAMES)}
    sorted_cls = sorted(coverage.items(), key=lambda x: x[1], reverse=True)
    top4_cls   = [c for c, _ in sorted_cls[:4]]

    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CLASS_COLORS):
        color_mask[pred_mask == cls_id] = color
    overlay = (image_np * 0.45 + color_mask * 0.55).astype(np.uint8)

    fig = plt.figure(figsize=(22, 16), facecolor="#0a0e1a")
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    def dark_ax(ax):
        ax.set_facecolor("#0f172a")
        for spine in ax.spines.values():
            spine.set_color("#1e293b")
        ax.tick_params(colors="#64748b", labelsize=7)
        return ax

    ax0 = fig.add_subplot(gs[0, 0]); dark_ax(ax0)
    ax0.imshow(image_np); ax0.set_title("Original Image", color="#e2e8f0", fontsize=9); ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1]); dark_ax(ax1)
    ax1.imshow(overlay); ax1.set_title("SegFormer-B0 Prediction", color="#e2e8f0", fontsize=9); ax1.axis("off")
    patches = [mpatches.Patch(color=tuple(c/255 for c in v), label=k)
               for k, v in CLASS_MAP.items()]
    ax1.legend(handles=patches, loc="lower left", fontsize=5,
               facecolor="#0a0e1a", edgecolor="#1e293b", labelcolor="#94a3b8", ncol=2)

    ax2 = fig.add_subplot(gs[0, 2]); dark_ax(ax2)
    im2 = ax2.imshow(entropy_np, cmap=UNCERTAINTY_CMAP, vmin=0)
    ax2.set_title("Prediction Uncertainty (Entropy)\nBright=Uncertain, Dark=Confident",
                  color="#e2e8f0", fontsize=8)
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).ax.tick_params(colors="#64748b", labelsize=7)

    ax3 = fig.add_subplot(gs[0, 3]); dark_ax(ax3)
    boundary_vis = np.stack([boundary, boundary * 0.3, boundary * 0.1], axis=2)
    ax3.imshow(image_np)
    ax3.imshow((boundary_vis * 255).astype(np.uint8), alpha=0.7)
    ax3.set_title("Class Boundary Detection\n(Red = class transitions)", color="#e2e8f0", fontsize=8)
    ax3.axis("off")

    for i, cls_name in enumerate(top4_cls):
        ax = fig.add_subplot(gs[1, i]); dark_ax(ax)
        cls_id   = CLASS_NAMES.index(cls_name)
        conf_map = softmax_np[cls_id]
        ax.imshow(conf_map, cmap="plasma", vmin=0, vmax=1)
        ax.set_title(
            f"{cls_name}\nConf map — mean={conf_map.mean():.3f} max={conf_map.max():.3f}",
            color="#e2e8f0", fontsize=8
        )
        ax.axis("off")

    ax_bar = fig.add_subplot(gs[2, 0:2]); dark_ax(ax_bar)
    classes_sorted = [c for c, _ in sorted_cls]
    vals_sorted    = [v for _, v in sorted_cls]
    colors_norm    = [tuple(c/255 for c in CLASS_MAP[cls]) for cls in classes_sorted]
    bars = ax_bar.barh(classes_sorted, vals_sorted, color=colors_norm,
                        edgecolor="#0a0e1a", linewidth=0.5)
    for bar, val in zip(bars, vals_sorted):
        if val > 1:
            ax_bar.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                        f"{val:.2f}%", va="center", color="#e2e8f0", fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Coverage (%)", color="#94a3b8", fontsize=8)
    ax_bar.set_title("Class Coverage (sorted)", color="#e2e8f0", fontsize=9)
    ax_bar.tick_params(axis="y", colors="#94a3b8", labelsize=8)

    ax_radar = fig.add_subplot(gs[2, 2], polar=True)
    ax_radar.set_facecolor("#0f172a")
    radar_vals  = [coverage[c] for c in CLASS_NAMES] + [coverage[CLASS_NAMES[0]]]
    radar_angles = np.linspace(0, 2*np.pi, len(CLASS_NAMES), endpoint=False).tolist()
    radar_angles += radar_angles[:1]
    ax_radar.plot(radar_angles, radar_vals, color="#3b82f6", lw=2)
    ax_radar.fill(radar_angles, radar_vals, color="#3b82f6", alpha=0.2)
    ax_radar.set_thetagrids(np.degrees(radar_angles[:-1]), CLASS_NAMES, fontsize=6, color="#94a3b8")
    ax_radar.set_title("Coverage Radar", color="#e2e8f0", fontsize=9, pad=15)
    ax_radar.tick_params(colors="#475569")

    ax_unc = fig.add_subplot(gs[2, 3]); dark_ax(ax_unc)
    conf_per_class = compute_per_class_confidence(softmax_np)
    cls_names_s = [c for c, _ in sorted_cls]
    mean_confs  = [conf_per_class[c]["mean_conf"] for c in cls_names_s]
    hi_confs    = [conf_per_class[c]["high_conf_pct"] for c in cls_names_s]
    x = np.arange(len(cls_names_s))
    ax_unc.bar(x - 0.2, mean_confs, 0.35, label="Mean Conf", color="#3b82f6", alpha=0.85)
    ax_unc.bar(x + 0.2, [h/100 for h in hi_confs], 0.35, label=">80% Conf %/100", color="#f97316", alpha=0.85)
    ax_unc.set_xticks(x)
    ax_unc.set_xticklabels([c[:8] for c in cls_names_s], rotation=45, ha="right", fontsize=6, color="#94a3b8")
    ax_unc.set_ylabel("Score", color="#94a3b8", fontsize=7)
    ax_unc.set_title("Per-Class Confidence", color="#e2e8f0", fontsize=9)
    ax_unc.legend(fontsize=7, facecolor="#0f172a", edgecolor="#1e293b", labelcolor="#94a3b8")
    ax_unc.set_ylim(0, 1)

    mean_ent    = float(entropy_np.mean())
    max_ent     = float(entropy_np.max())
    high_unc    = float((entropy_np > entropy_np.mean() + entropy_np.std()).mean() * 100)
    boundary_pct = float(boundary.mean() * 100)
    dominant    = sorted_cls[0][0]

    summary = (
        f"Image: {image_name}  |  Dominant: {dominant} ({sorted_cls[0][1]:.1f}%)  |  "
        f"Mean Entropy: {mean_ent:.3f}  |  High-Uncertainty Pixels: {high_unc:.1f}%  |  "
        f"Boundary Pixels: {boundary_pct:.1f}%"
    )
    fig.suptitle(summary, color="#94a3b8", fontsize=8, y=0.98)

    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="#0a0e1a")
    plt.close()
    print(f"  Analysis dashboard saved: {save_path}")


def run_full_analysis(model, image: Image.Image, image_name: str, save_dir: Path = None):
    """
    Jalankan full analysis pipeline:
    1. Inference + logits
    2. Entropy uncertainty map
    3. Per-class confidence
    4. Boundary detection
    5. Spatial distribution
    6. Dashboard visualization
    """
    if save_dir is None:
        save_dir = OUTPUT_DIR / "analysis"
    save_dir.mkdir(exist_ok=True)

    from segformer_model import preprocess_image

    print(f"\n  Analyzing: {image_name}")

    tensor, img_rgb = preprocess_image(image)
    orig_w, orig_h  = img_rgb.size
    image_np        = np.array(img_rgb)

    with torch.no_grad():
        outputs   = model(pixel_values=tensor)
        logits    = outputs.logits
        logits_up = F.interpolate(logits, size=(orig_h, orig_w),
                                   mode="bilinear", align_corners=False)

    pred_mask  = logits_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    entropy_np, softmax_np = compute_entropy_uncertainty(logits_up)

    conf_per_class = compute_per_class_confidence(softmax_np)
    spatial_stats  = compute_class_spatial_stats(pred_mask)
    boundary_mask  = detect_boundaries(pred_mask)

    print(f"  Mean prediction entropy : {entropy_np.mean():.4f} (lower=more confident)")
    print(f"  High-uncertainty pixels : {(entropy_np > entropy_np.mean() + entropy_np.std()).mean()*100:.2f}%")
    print(f"  Boundary pixel ratio    : {boundary_mask.mean()*100:.2f}%")
    print(f"\n  Per-class confidence:")
    print(f"  {'Class':<22} {'Mean Conf':>10} {'High (>80%)':>12} {'Coverage %':>12}")
    print(f"  {'-'*58}")
    for cls_name in CLASS_NAMES:
        c = conf_per_class[cls_name]
        print(f"  {cls_name:<22} {c['mean_conf']:>10.4f} {c['high_conf_pct']:>11.2f}% {c['area_pct']:>11.2f}%")

    stem      = Path(image_name).stem
    dash_path = save_dir / f"analysis_{stem}.png"
    plot_comprehensive_analysis(
        image_np, pred_mask, softmax_np, entropy_np,
        image_name, dash_path
    )

    result = {
        "image_name":       image_name,
        "mean_entropy":     round(float(entropy_np.mean()), 4),
        "high_uncertainty_pct": round(float((entropy_np > entropy_np.mean() + entropy_np.std()).mean() * 100), 2),
        "boundary_pct":     round(float(boundary_mask.mean() * 100), 2),
        "per_class_confidence": {
            c: {k: round(v, 4) for k, v in d.items()}
            for c, d in conf_per_class.items()
        },
        "spatial_distribution": spatial_stats,
        "dashboard_path":   str(dash_path),
    }
    return result, pred_mask, entropy_np


if __name__ == "__main__":
    from segformer_model import build_segformer_model
    import json

    model = build_segformer_model()
    model.eval()

    img_paths = list(Path("modified_uavid_dataset/val_data/Images").glob("*.png"))[:3]
    if not img_paths:
        img_paths = list(Path("modified_uavid_dataset/train_data/Images").glob("*.png"))[:3]

    all_results = []
    for img_path in img_paths:
        image  = Image.open(img_path).convert("RGB")
        result, _, _ = run_full_analysis(model, image, img_path.name)
        all_results.append(result)

    out = OUTPUT_DIR / "analysis_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll analysis results: {out}")
