"""
UAVid RAG Explorer - Streamlit Application
SegFormer-B0 semantic segmentation + RAG Q&A for UAVid dataset.
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path
from collections import defaultdict
import io

import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))
from config import (
    TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS,
    TEST_IMAGES, CLASS_MAP, CLASS_NAMES, CLASS_COLORS,
    OUTPUT_DIR, VECTORSTORE_DIR, EMBED_MODEL, COLLECTION_NAME, GEMINI_MODEL, TOLERANCE
)
from segformer_model import (
    build_segformer_model, run_inference, compute_metrics_from_masks,
    mask_to_color
)
from segmentation import color_to_class_id, compute_per_class_stats, generate_insight_text


st.set_page_config(
    page_title="UAVid Remote Sensing AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-deep:      #04060c;
    --bg-panel:     #0a0f1c;
    --bg-panel-2:   #0e1526;
    --bg-card:      #0d1320;
    --line:         #1a2436;
    --line-bright:  #25344d;
    --signal:       #00e5a0;
    --signal-dim:   rgba(0,229,160,0.14);
    --alert:        #ff6b4a;
    --alert-dim:    rgba(255,107,74,0.14);
    --data-blue:    #3b9eff;
    --text-bright:  #eef2f8;
    --text-mid:     #93a1b8;
    --text-dim:     #4b5874;
    --mono: 'JetBrains Mono', 'Space Mono', monospace;
    --display: 'Sora', sans-serif;
}

* { box-sizing: border-box; }

html, body, [class*="css"] { font-family: var(--display); }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes scanSweep {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(220%); }
}
@keyframes pulseDot {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,229,160,0.55); }
    50%      { opacity: 0.6; box-shadow: 0 0 0 6px rgba(0,229,160,0); }
}
@keyframes fillBar {
    from { width: 0%; }
}
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: 0.001ms !important; animation-iteration-count: 1 !important; transition-duration: 0.001ms !important; }
}

.main {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,229,160,0.06), transparent),
        radial-gradient(ellipse 60% 40% at 100% 100%, rgba(59,158,255,0.05), transparent),
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(255,255,255,0.012) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(255,255,255,0.012) 40px),
        var(--bg-deep);
    background-attachment: fixed;
}

/* ---------- Page header with radar-sweep signature ---------- */
.page-title-wrap {
    position: relative;
    overflow: hidden;
    border-radius: 14px;
    background: linear-gradient(135deg, var(--bg-panel) 0%, var(--bg-panel-2) 100%);
    border: 1px solid var(--line);
    padding: 22px 26px;
    margin-bottom: 28px;
    animation: fadeUp 0.5s ease both;
}
.page-title-wrap::before {
    content: "";
    position: absolute;
    top: 0; left: 0; height: 100%; width: 35%;
    background: linear-gradient(90deg, transparent, rgba(0,229,160,0.08), transparent);
    animation: scanSweep 5s ease-in-out infinite;
    pointer-events: none;
}
.page-eyebrow {
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--signal);
    text-transform: uppercase;
    letter-spacing: 0.22em;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.page-eyebrow .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--signal);
    animation: pulseDot 2s ease-in-out infinite;
}
.page-title {
    font-family: var(--display);
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--text-bright);
    letter-spacing: -0.02em;
    margin: 0;
    position: relative;
}
.page-sub {
    font-size: 0.88rem;
    color: var(--text-mid);
    margin-top: 6px;
    font-family: var(--mono);
    position: relative;
}

/* ---------- Metric cards ---------- */
.metric-card {
    background: linear-gradient(160deg, var(--bg-card) 0%, var(--bg-panel) 100%);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    margin: 4px 0;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s cubic-bezier(.2,.8,.2,1), border-color 0.25s ease, box-shadow 0.25s ease;
    animation: fadeUp 0.5s ease both;
}
.metric-card::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, transparent 40%, rgba(0,229,160,0.06) 50%, transparent 60%);
    background-size: 200% 100%;
    opacity: 0;
    transition: opacity 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: var(--line-bright);
    box-shadow: 0 10px 30px -12px rgba(0,229,160,0.18);
}
.metric-card:hover::after {
    opacity: 1;
    animation: shimmer 1.6s linear infinite;
}
.metric-value {
    font-family: var(--mono);
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--signal);
    position: relative;
}
.metric-label {
    font-size: 0.68rem;
    color: var(--text-mid);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 5px;
    position: relative;
}

/* ---------- Insight / answer cards ---------- */
.insight-card {
    background: linear-gradient(135deg, #08121a 0%, #0c1a22 55%, #0a1c20 100%);
    border: 1px solid rgba(0,229,160,0.22);
    border-radius: 12px;
    padding: 18px 22px;
    font-size: 0.87rem;
    color: #cbd5e1;
    line-height: 1.85;
    margin: 12px 0;
    position: relative;
    animation: fadeUp 0.5s ease both;
}
.insight-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, var(--signal), transparent);
}
.answer-card {
    background: var(--bg-card);
    border: 1px solid rgba(59,158,255,0.3);
    border-left: 4px solid var(--data-blue);
    border-radius: 12px;
    padding: 20px 24px;
    color: var(--text-bright);
    font-size: 0.92rem;
    line-height: 1.8;
    margin-top: 12px;
    animation: fadeUp 0.45s ease both;
}
.class-chip {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    margin: 2px;
    font-family: var(--mono);
}

/* ---------- Section headers ---------- */
.section-header {
    font-family: var(--mono);
    font-size: 0.76rem;
    color: var(--signal);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin: 24px 0 10px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--line);
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header::before {
    content: "//";
    color: var(--text-dim);
}

/* ---------- Animated coverage bars ---------- */
.cov-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 6px 0;
}
.cov-swatch {
    width: 13px; height: 13px;
    border-radius: 4px;
    flex-shrink: 0;
    box-shadow: 0 0 8px -1px currentColor;
}
.cov-name {
    font-family: var(--mono);
    font-size: 0.74rem;
    color: var(--text-mid);
    width: 148px;
    flex-shrink: 0;
}
.cov-track {
    flex: 1;
    height: 13px;
    background: var(--bg-deep);
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid var(--line);
}
.cov-fill {
    height: 100%;
    border-radius: 4px;
    animation: fillBar 1.1s cubic-bezier(.16,1,.3,1) both;
}
.cov-pct {
    font-family: var(--mono);
    font-size: 0.74rem;
    color: var(--text-bright);
    width: 54px;
    text-align: right;
    flex-shrink: 0;
}

/* ---------- Sidebar ---------- */
div[data-testid="stSidebarContent"] {
    background: linear-gradient(180deg, #030509 0%, #050810 100%);
    border-right: 1px solid var(--line);
}
div[data-testid="stSidebarContent"] label,
div[data-testid="stSidebarContent"] .stRadio label {
    color: var(--text-mid) !important;
    font-size: 0.88rem;
}
div[data-testid="stSidebarContent"] p { color: var(--text-dim); font-size: 0.8rem; }

.brand-block {
    animation: fadeUp 0.5s ease both;
}
.brand-title {
    font-family: var(--display);
    font-size: 1.05rem;
    color: var(--text-bright);
    font-weight: 700;
    margin-bottom: 2px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.brand-title .live-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--signal);
    animation: pulseDot 2s ease-in-out infinite;
}
.brand-sub {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--text-dim);
    letter-spacing: 0.08em;
    margin-bottom: 18px;
}
.spec-block {
    font-size: 0.78rem;
    color: var(--text-dim);
    line-height: 1.9;
}
.spec-block b {
    color: var(--text-mid);
    font-weight: 500;
    font-family: var(--mono);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ---------- Buttons ---------- */
.stButton button {
    background: linear-gradient(135deg, #0a3d2e 0%, #0d5c42 100%) !important;
    color: #d7ffef !important;
    border: 1px solid rgba(0,229,160,0.35) !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    padding: 9px 20px !important;
    transition: all 0.22s cubic-bezier(.2,.8,.2,1) !important;
    box-shadow: 0 0 0 0 rgba(0,229,160,0) !important;
}
.stButton button:hover {
    background: linear-gradient(135deg, #0d5c42 0%, #0a7a58 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px -6px rgba(0,229,160,0.45) !important;
    border-color: var(--signal) !important;
}
.stButton button:active { transform: translateY(0) !important; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    color: var(--text-dim) !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.stTabs [aria-selected="true"] {
    color: var(--signal) !important;
    border-bottom-color: var(--signal) !important;
}

/* ---------- Misc ---------- */
.upload-zone {
    background: var(--bg-card);
    border: 2px dashed rgba(0,229,160,0.3);
    border-radius: 14px;
    padding: 26px;
    text-align: center;
    color: var(--text-dim);
    transition: border-color 0.25s ease, background 0.25s ease;
}
.upload-zone:hover { border-color: var(--signal); background: var(--bg-panel-2); }

hr { border-color: var(--line) !important; }

::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--line-bright); border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: var(--signal-dim); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading SegFormer-B0 model...")
def get_segformer_model():
    return build_segformer_model()


@st.cache_resource(show_spinner="Loading embedding model...")
def get_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Loading vector store...")
def get_vectorstore():
    import chromadb
    chroma_path = str(VECTORSTORE_DIR / "chroma_uavid")
    if not (VECTORSTORE_DIR / "chroma_uavid").exists():
        return None
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return None


def hex_color(rgb_tuple):
    return "#{:02x}{:02x}{:02x}".format(*rgb_tuple)


# --- Version-safe image renderer -------------------------------------------------
# Different Streamlit versions support different "full width" kwargs for st.image:
#   - older versions: use_container_width=True
#   - newer versions: width="stretch"
#   - very new/old versions: neither (just omit the kwarg)
# We detect what's actually supported once at import time and reuse that everywhere,
# so this never breaks again regardless of which Streamlit version is deployed.
import inspect as _inspect
_IMAGE_PARAMS = set(_inspect.signature(st.image).parameters.keys())
if "use_container_width" in _IMAGE_PARAMS:
    _IMAGE_WIDTH_KW = {"use_container_width": True}
elif "width" in _IMAGE_PARAMS:
    _IMAGE_WIDTH_KW = {"width": "stretch"}
else:
    _IMAGE_WIDTH_KW = {}


def safe_image(container, image_src, caption=None):
    """Render an image with whatever full-width kwarg this Streamlit version supports.
    `container` can be `st` itself or any column/element (e.g. col1, cols[i])."""
    kwargs = dict(_IMAGE_WIDTH_KW)
    if caption is not None:
        kwargs["caption"] = caption
    try:
        container.image(image_src, **kwargs)
    except TypeError:
        # Last-resort fallback: no width kwarg at all.
        fallback_kwargs = {"caption": caption} if caption is not None else {}
        container.image(image_src, **fallback_kwargs)
# -----------------------------------------------------------------------------------


def render_class_bars(stats_dict):
    sorted_items = sorted(stats_dict.items(), key=lambda x: x[1]["coverage_pct"], reverse=True)
    for cls_name, s in sorted_items:
        pct   = s["coverage_pct"]
        color = CLASS_MAP[cls_name]
        hx    = hex_color(color)
        bar_w = max(pct * 1.8, 0.3)
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0">'
            f'<div style="width:14px;height:14px;background:{hx};border-radius:3px;flex-shrink:0"></div>'
            f'<span style="font-family:Space Mono,monospace;font-size:0.75rem;color:#94a3b8;width:150px;flex-shrink:0">{cls_name}</span>'
            f'<div style="background:{hx};opacity:0.85;width:{bar_w:.1f}%;height:12px;border-radius:2px;min-width:3px"></div>'
            f'<span style="font-family:Space Mono,monospace;font-size:0.75rem;color:#e2e8f0">{pct:.2f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_coverage_radar(stats_dict, title="Coverage Radar"):
    classes = CLASS_NAMES
    values  = [stats_dict[c]["coverage_pct"] if isinstance(stats_dict[c], dict)
               else stats_dict[c] for c in classes]
    values_plot = values + [values[0]]
    angles  = np.linspace(0, 2 * np.pi, len(classes), endpoint=False).tolist() + \
              [np.linspace(0, 2 * np.pi, len(classes), endpoint=False)[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0f172a")
    ax.plot(angles, values_plot, color="#3b82f6", linewidth=2)
    ax.fill(angles, values_plot, color="#3b82f6", alpha=0.2)
    ax.set_thetagrids(np.degrees(angles[:-1]), classes, fontsize=7, color="#94a3b8")
    ax.tick_params(colors="#64748b")
    ax.spines["polar"].set_color("#1e293b")
    ax.set_title(title, fontsize=9, color="#e2e8f0", pad=20)
    ax.yaxis.label.set_color("#64748b")
    for label in ax.get_yticklabels():
        label.set_color("#64748b")
    plt.tight_layout()
    return fig


def segment_and_display(image: Image.Image, source_label="Uploaded Image", gt_mask=None, run_analysis=False):
    model = get_segformer_model()

    with st.spinner("Running SegFormer-B0 inference..."):
        result  = run_inference(model, image)

    pred_mask  = result["pred_mask"]
    stats      = result["stats"]
    elapsed    = result["elapsed"]
    img_arr    = np.array(image.convert("RGB"))
    orig_h, orig_w = img_arr.shape[:2]

    import torch
    import torch.nn.functional as tfF
    pred_t       = torch.tensor(pred_mask).unsqueeze(0).unsqueeze(0).float()
    pred_resized = tfF.interpolate(pred_t, size=(orig_h, orig_w), mode="nearest")
    pred_final   = pred_resized.squeeze().numpy().astype(np.uint8)
    color_final  = mask_to_color(pred_final)
    overlay      = (img_arr * 0.45 + color_final * 0.55).astype(np.uint8)

    col1, col2, col3 = st.columns(3)
    try:
        safe_image(col1, img_arr.astype(np.uint8), caption="Input Image")
    except Exception as e:
        col1.warning(f"Could not render input image: {e}")
    try:
        safe_image(col2, color_final.astype(np.uint8), caption="SegFormer-B0 Mask")
    except Exception as e:
        col2.warning(f"Could not render mask: {e}")
    try:
        safe_image(col3, overlay.astype(np.uint8), caption="Overlay (55% seg)")
    except Exception as e:
        col3.warning(f"Could not render overlay: {e}")

    st.markdown('<div class="section-header">Class Coverage</div>', unsafe_allow_html=True)
    col_bar, col_radar = st.columns([1.3, 1])
    with col_bar:
        render_class_bars(stats)
    with col_radar:
        fig = render_coverage_radar(stats, source_label)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    dominant = max(stats, key=lambda c: stats[c]["coverage_pct"])
    veg      = stats["Tree"]["coverage_pct"] + stats["Low vegetation"]["coverage_pct"]
    cars     = stats["Moving car"]["coverage_pct"] + stats["Static car"]["coverage_pct"]
    scene    = ("Urban Dense" if stats["Building"]["coverage_pct"] > 30 else
                "Mixed Urban" if stats["Building"]["coverage_pct"] > 10 else "Natural/Open")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{stats[dominant]["coverage_pct"]:.1f}%</div><div class="metric-label">Dominant</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{veg:.1f}%</div><div class="metric-label">Vegetation</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{stats["Road"]["coverage_pct"]:.1f}%</div><div class="metric-label">Road</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-value">{cars:.2f}%</div><div class="metric-label">Vehicles</div></div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="metric-card"><div class="metric-value">{stats["Human"]["coverage_pct"]:.3f}%</div><div class="metric-label">Human</div></div>', unsafe_allow_html=True)
    c6.markdown(f'<div class="metric-card"><div class="metric-value">{elapsed:.2f}s</div><div class="metric-label">Inference</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Scene Intelligence</div>', unsafe_allow_html=True)
    sorted_cls = sorted(stats.items(), key=lambda x: x[1]["coverage_pct"], reverse=True)
    top3_str   = " | ".join([f"{c}: {s['coverage_pct']:.1f}%" for c,s in sorted_cls[:3]])
    present    = [c for c,s in stats.items() if s["coverage_pct"] > 1.0]
    absent     = [c for c,s in stats.items() if s["coverage_pct"] < 0.1]
    urban_idx  = (stats["Building"]["coverage_pct"] * 0.4 +
                  stats["Road"]["coverage_pct"] * 0.35 +
                  cars * 0.25)
    green_idx  = veg
    human_risk = ("DETECTED" if stats["Human"]["coverage_pct"] > 0.5 else
                  "TRACE" if stats["Human"]["coverage_pct"] > 0.05 else "NOT DETECTED")

    insight_html = f"""
    <div class="insight-card">
    <b style="color:#3b82f6;font-family:Space Mono,monospace">SCENE TYPE:</b> {scene}<br>
    <b style="color:#3b82f6;font-family:Space Mono,monospace">TOP CLASSES:</b> {top3_str}<br>
    <b style="color:#3b82f6;font-family:Space Mono,monospace">PRESENT (&gt;1%):</b> {", ".join(present) if present else "none"}<br>
    <b style="color:#3b82f6;font-family:Space Mono,monospace">ABSENT (&lt;0.1%):</b> {", ".join(absent) if absent else "none"}<br>
    <b style="color:#3b82f6;font-family:Space Mono,monospace">URBAN INDEX:</b> {urban_idx:.1f}% (Building+Road+Vehicle weighted)<br>
    <b style="color:#3b82f6;font-family:Space Mono,monospace">GREEN INDEX:</b> {green_idx:.1f}% (Tree+LowVegetation)<br>
    <b style="color:#3b82f6;font-family:Space Mono,monospace">HUMAN PRESENCE:</b> {human_risk} ({stats["Human"]["coverage_pct"]:.3f}%)<br>
    <b style="color:#3b82f6;font-family:Space Mono,monospace">MOVING VEHICLES:</b> {stats["Moving car"]["coverage_pct"]:.2f}% | PARKED: {stats["Static car"]["coverage_pct"]:.2f}%<br>
    <b style="color:#3b82f6;font-family:Space Mono,monospace">BACKGROUND CLUTTER:</b> {stats["Background clutter"]["coverage_pct"]:.2f}%
    </div>"""
    st.markdown(insight_html, unsafe_allow_html=True)

    if gt_mask is not None:
        metrics = compute_metrics_from_masks(pred_final, gt_mask)
        st.markdown('<div class="section-header">Evaluation vs Ground Truth</div>', unsafe_allow_html=True)
        m1,m2,m3,m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["pixel_accuracy"]:.2f}%</div><div class="metric-label">Pixel Accuracy</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["mean_iou"]:.2f}%</div><div class="metric-label">Mean IoU</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["fw_iou"]:.2f}%</div><div class="metric-label">FW-IoU</div></div>', unsafe_allow_html=True)
        best_cls = max(metrics["per_class_iou"], key=metrics["per_class_iou"].get)
        m4.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["per_class_iou"][best_cls]:.1f}%</div><div class="metric-label">Best: {best_cls[:8]}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Per-Class IoU Detail</div>', unsafe_allow_html=True)
        iou_sorted = sorted(metrics["per_class_iou"].items(), key=lambda x: x[1], reverse=True)
        for cls_name, iou_val in iou_sorted:
            color = CLASS_MAP[cls_name]
            hx    = "#{:02x}{:02x}{:02x}".format(*color)
            bar_w = max(iou_val * 1.5, 0.3)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin:3px 0">' +
                f'<div style="width:12px;height:12px;background:{hx};border-radius:2px;flex-shrink:0"></div>' +
                f'<span style="font-family:Space Mono,monospace;font-size:0.72rem;color:#94a3b8;width:140px">{cls_name}</span>' +
                f'<div style="background:{hx};opacity:0.8;width:{bar_w:.1f}%;height:10px;border-radius:2px;min-width:2px"></div>' +
                f'<span style="font-family:Space Mono,monospace;font-size:0.72rem;color:#e2e8f0">{iou_val:.2f}%</span>' +
                f'</div>',
                unsafe_allow_html=True
            )

    if run_analysis:
        st.markdown('<div class="section-header">Deep Analysis Dashboard</div>', unsafe_allow_html=True)
        with st.spinner("Generating attention + uncertainty analysis..."):
            from attention_analysis import run_full_analysis
            import tempfile, os
            save_dir = Path(OUTPUT_DIR) / "analysis"
            result_data, _, entropy_np = run_full_analysis(
                model, image, source_label, save_dir=save_dir
            )
            dash_path = Path(result_data["dashboard_path"])
            if dash_path.exists() and dash_path.stat().st_size > 0:
                try:
                    safe_image(st, str(dash_path))
                except Exception as e:
                    st.warning(f"Could not render dashboard image: {e}")

            st.markdown('<div class="section-header">Analysis Metrics</div>', unsafe_allow_html=True)
            a1,a2,a3 = st.columns(3)
            a1.markdown(f'<div class="metric-card"><div class="metric-value">{result_data["mean_entropy"]:.4f}</div><div class="metric-label">Mean Entropy (lower=better)</div></div>', unsafe_allow_html=True)
            a2.markdown(f'<div class="metric-card"><div class="metric-value">{result_data["high_uncertainty_pct"]:.1f}%</div><div class="metric-label">High-Uncertainty Pixels</div></div>', unsafe_allow_html=True)
            a3.markdown(f'<div class="metric-card"><div class="metric-value">{result_data["boundary_pct"]:.1f}%</div><div class="metric-label">Boundary Pixels</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header">Per-Class Confidence Detail</div>', unsafe_allow_html=True)
            import pandas as pd
            conf_rows = []
            for cls_name in CLASS_NAMES:
                c = result_data["per_class_confidence"][cls_name]
                conf_rows.append({
                    "Class": cls_name,
                    "Mean Confidence": f'{c["mean_conf"]:.4f}',
                    "Max Confidence":  f'{c["max_conf"]:.4f}',
                    "High Conf >80%":  f'{c["high_conf_pct"]:.2f}%',
                    "Predicted Area":  f'{c["area_pct"]:.2f}%',
                })
            st.dataframe(pd.DataFrame(conf_rows), use_container_width=True)

    return stats, pred_final

def page_live_inference():
    st.markdown('<div class="page-title">Live Inference</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Upload an image or use your camera for real-time SegFormer-B0 segmentation</div>', unsafe_allow_html=True)

    tab_upload, tab_camera, tab_dataset = st.tabs(["Upload Image", "Camera", "Dataset Sample"])

    with tab_upload:
        st.markdown('<div class="section-header">Upload UAV or Aerial Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg"],
            help="Upload any UAV aerial image or street-view image",
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.markdown(f"**File:** `{uploaded.name}` | **Size:** {image.size[0]}x{image.size[1]} px")
            segment_and_display(image, source_label=uploaded.name, run_analysis=st.session_state.get("run_deep", False))

    with tab_camera:
        st.markdown('<div class="section-header">Camera Capture</div>', unsafe_allow_html=True)
        st.info("Point your camera at a scene and capture for instant segmentation.")
        camera_img = st.camera_input("Take a photo")
        if camera_img:
            image = Image.open(camera_img).convert("RGB")
            segment_and_display(image, source_label="camera_capture.jpg", run_analysis=st.session_state.get("run_deep", False))

    with tab_dataset:
        st.markdown('<div class="section-header">Analyze from Dataset</div>', unsafe_allow_html=True)
        split_sel  = st.selectbox("Split", ["train", "val", "test"])
        img_dir    = TRAIN_IMAGES if split_sel == "train" else (VAL_IMAGES if split_sel == "val" else TEST_IMAGES)
        lbl_dir    = TRAIN_LABELS if split_sel == "train" else (VAL_LABELS if split_sel == "val" else None)

        if not img_dir.exists():
            st.warning(f"Directory not found: {img_dir}")
        else:
            img_paths = sorted(img_dir.glob("*.png"))
            if not img_paths:
                st.warning("No images found.")
            else:
                img_names = [p.name for p in img_paths]
                selected  = st.selectbox("Select image", img_names)
                img_path  = img_dir / selected

                if st.button("Run Segmentation", key="run_dataset"):
                    image   = Image.open(img_path).convert("RGB")
                    gt_mask = None
                    if lbl_dir:
                        lbl_path = lbl_dir / selected
                        if lbl_path.exists():
                            gt_arr  = np.array(Image.open(lbl_path).convert("RGB"))
                            gt_mask = color_to_class_id(gt_arr)
                    segment_and_display(image, source_label=selected, gt_mask=gt_mask, run_analysis=st.session_state.get("run_deep", False))


def page_dataset_overview():
    st.markdown('<div class="page-title">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">UAVid Modified Dataset - Exploratory Analysis</div>', unsafe_allow_html=True)

    train_imgs = sorted(TRAIN_IMAGES.glob("*.png")) if TRAIN_IMAGES.exists() else []
    val_imgs   = sorted(VAL_IMAGES.glob("*.png"))   if VAL_IMAGES.exists()   else []
    test_imgs  = sorted(TEST_IMAGES.glob("*.png"))  if TEST_IMAGES.exists()  else []
    train_lbls = sorted(TRAIN_LABELS.glob("*.png")) if TRAIN_LABELS.exists() else []
    val_lbls   = sorted(VAL_LABELS.glob("*.png"))   if VAL_LABELS.exists()   else []

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val in zip(
        [c1, c2, c3, c4, c5],
        ["Train Images", "Val Images", "Test Images", "Train Labels", "Val Labels"],
        [len(train_imgs), len(val_imgs), len(test_imgs), len(train_lbls), len(val_lbls)]
    ):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Class Color Legend</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (cls_name, color) in enumerate(CLASS_MAP.items()):
        hx = hex_color(color)
        cols[i % 4].markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0;padding:6px 10px;background:#0f172a;border-radius:6px;border:1px solid #1e293b">'
            f'<div style="width:18px;height:18px;background:{hx};border-radius:3px;border:1px solid #374151;flex-shrink:0"></div>'
            f'<span style="font-size:0.82rem;color:#cbd5e1">{cls_name}</span>'
            f'<span style="font-family:Space Mono,monospace;font-size:0.7rem;color:#475569;margin-left:auto">RGB{color}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-header">Pre-generated EDA Charts</div>', unsafe_allow_html=True)
    eda_files = sorted(OUTPUT_DIR.glob("eda_*.png"))
    if eda_files:
        cols = st.columns(2)
        for i, f in enumerate(eda_files[:8]):
            try:
                if f.stat().st_size == 0:
                    cols[i % 2].warning(f"Skipped (empty file): {f.name}")
                    continue
                with Image.open(f) as im:
                    im.verify()
                safe_image(cols[i % 2], str(f), caption=f.stem.replace("_", " ").title())
            except Exception as e:
                cols[i % 2].warning(f"Skipped (could not load {f.name}): {e}")
    else:
        st.info("Run `python src/eda.py` to generate EDA charts.")

    st.markdown('<div class="section-header">Segmentation Overlay Samples</div>', unsafe_allow_html=True)
    seg_files = sorted(OUTPUT_DIR.glob("seg_overlay_*.png"))
    if seg_files:
        for f in seg_files[:4]:
            try:
                if f.stat().st_size == 0:
                    st.warning(f"Skipped (empty file): {f.name}")
                    continue
                with Image.open(f) as im:
                    im.verify()
                safe_image(st, str(f), caption=f.stem.replace("_", " ").title())
            except Exception as e:
                st.warning(f"Skipped (could not load {f.name}): {e}")
    else:
        st.info("Run `python pipeline.py` to generate segmentation samples.")


def page_evaluation_metrics():
    st.markdown('<div class="page-title">Evaluation Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Real SegFormer-B0 metrics — Pixel Accuracy, mIoU, FW-IoU vs Ground Truth</div>', unsafe_allow_html=True)

    import pandas as pd

    loaded = {}
    for split in ["train", "val"]:
        p = OUTPUT_DIR / f"metrics_{split}.json"
        if p.exists():
            with open(p) as f:
                loaded[split] = json.load(f)

    if not loaded:
        st.warning("No metrics found. Run `python pipeline.py` first.")
        return

    for split, m in loaded.items():
        st.markdown(f'<div class="section-header">{split.capitalize()} Split — {m.get("images_evaluated", "?")} images</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><div class="metric-value">{m["pixel_accuracy"]:.2f}%</div><div class="metric-label">Pixel Accuracy</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value">{m["mean_iou"]:.2f}%</div><div class="metric-label">Mean IoU (mIoU)</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-value">{m["fw_iou"]:.2f}%</div><div class="metric-label">Freq-Weighted IoU</div></div>', unsafe_allow_html=True)
        st.markdown("")

    st.markdown('<div class="section-header">Per-Class IoU Comparison</div>', unsafe_allow_html=True)
    rows = []
    for cls in CLASS_NAMES:
        row = {"Class": cls}
        for split, m in loaded.items():
            row[f"IoU {split} (%)"] = m["per_class_iou"].get(cls, 0)
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format({c: "{:.4f}" for c in df.columns if c != "Class"}), use_container_width=True)

    st.markdown('<div class="section-header">IoU Bar Chart</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0f172a")
    x = np.arange(len(CLASS_NAMES))
    w = 0.35
    if "train" in loaded:
        iou_t = [loaded["train"]["per_class_iou"].get(c, 0) for c in CLASS_NAMES]
        ax.bar(x - w/2, iou_t, w, label="Train", color="#3b82f6", alpha=0.85, edgecolor="#1e40af", linewidth=0.5)
    if "val" in loaded:
        iou_v = [loaded["val"]["per_class_iou"].get(c, 0) for c in CLASS_NAMES]
        ax.bar(x + w/2, iou_v, w, label="Val",   color="#f97316", alpha=0.85, edgecolor="#c2410c", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=8, color="#94a3b8")
    ax.set_ylabel("IoU (%)", color="#94a3b8")
    ax.set_title("Per-Class IoU — SegFormer-B0", color="#e2e8f0", fontsize=11)
    ax.legend(labelcolor="#e2e8f0", facecolor="#1e293b", edgecolor="#334155")
    ax.grid(axis="y", alpha=0.2, color="#334155")
    ax.tick_params(colors="#64748b")
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    metrics_img = OUTPUT_DIR / "metrics_comparison.png"
    if metrics_img.exists() and metrics_img.stat().st_size > 0:
        try:
            safe_image(st, str(metrics_img), caption="Metrics Comparison Chart")
        except Exception as e:
            st.warning(f"Could not render metrics comparison chart: {e}")


def page_batch_insights():
    st.markdown('<div class="page-title">Batch Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Pre-computed segmentation insights across all images</div>', unsafe_allow_html=True)

    insight_path = OUTPUT_DIR / "insights_all.json"
    if not insight_path.exists():
        st.info("No insights file found. Run `python pipeline.py`.")
        return

    with open(insight_path) as f:
        insights = json.load(f)

    import pandas as pd
    rows = []
    for item in insights:
        stats = item.get("class_stats", {})
        rows.append({
            "Image":       item.get("image_name", ""),
            "Split":       item.get("split", ""),
            "Dominant":    max(stats, key=stats.get) if stats else "",
            "Road %":      round(stats.get("Road", 0), 2),
            "Building %":  round(stats.get("Building", 0), 2),
            "Tree %":      round(stats.get("Tree", 0), 2),
            "Human %":     round(stats.get("Human", 0), 4),
        })
    df = pd.DataFrame(rows)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Total Images</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{len(df[df["Split"]=="train"])}</div><div class="metric-label">Train</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{len(df[df["Split"]=="val"])}</div><div class="metric-label">Val</div></div>', unsafe_allow_html=True)

    split_filter = st.selectbox("Filter by split", ["all", "train", "val", "test"])
    if split_filter != "all":
        df = df[df["Split"] == split_filter]

    st.markdown('<div class="section-header">Image Table</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)

    st.markdown('<div class="section-header">Dominant Class Distribution</div>', unsafe_allow_html=True)
    dom_counts = df["Dominant"].value_counts()
    fig, ax    = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0f172a")
    colors_norm = [tuple(c/255 for c in CLASS_MAP.get(cls, (128,128,128))) for cls in dom_counts.index]
    ax.bar(dom_counts.index, dom_counts.values, color=colors_norm, edgecolor="#1e293b", linewidth=0.5)
    ax.set_ylabel("Number of images", color="#94a3b8")
    ax.set_title("Dominant Class per Image", color="#e2e8f0")
    ax.tick_params(axis="x", rotation=30, labelsize=8, colors="#94a3b8")
    ax.tick_params(axis="y", colors="#64748b")
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def page_rag():
    st.markdown('<div class="page-title">RAG Q&A System</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Ask anything about UAVid dataset using semantic retrieval + Gemini 2.5 Flash</div>', unsafe_allow_html=True)

    api_key = st.sidebar.text_input("Gemini API Key", type="password", placeholder="AIza...")

    if not api_key:
        st.markdown("""
        <div style="background:#0f172a;border:1px solid #1e3a5f;border-radius:10px;padding:20px">
        <p style="color:#94a3b8;margin:0">Enter your Gemini API Key in the sidebar to activate Q&A.</p>
        <p style="color:#64748b;font-size:0.82rem;margin-top:8px">Get a free key at <a href="https://aistudio.google.com" style="color:#3b82f6">aistudio.google.com</a></p>
        </div>
        """, unsafe_allow_html=True)
        return

    collection = get_vectorstore()
    if collection is None:
        st.error("Vector store not built yet. Run `python pipeline.py` first.")
        return

    encoder = get_encoder()
    doc_count = collection.count()
    st.success(f"Vector store ready — {doc_count} documents indexed")

    st.markdown('<div class="section-header">Suggested Queries</div>', unsafe_allow_html=True)
    suggestions = [
        "Which images have the highest road coverage?",
        "Show me scenes with significant human presence.",
        "Describe vegetation distribution in training set.",
        "Which images are dominated by buildings?",
        "Are there images with both moving and static cars?",
        "What is the average background clutter percentage?",
    ]

    chosen = ""
    cols = st.columns(3)
    for i, q in enumerate(suggestions):
        if cols[i % 3].button(q, key=f"sug_{i}", use_container_width=True):
            chosen = q

    st.markdown('<div class="section-header">Ask Your Question</div>', unsafe_allow_html=True)
    query = st.text_area(
        "Question",
        value=chosen,
        height=90,
        placeholder="E.g. Which images have the most trees and low vegetation?",
        label_visibility="collapsed",
    )
    top_k = st.slider("Documents to retrieve", 3, 10, 5)

    ask_clicked = st.button("Ask", type="primary", use_container_width=False)

    if ask_clicked and query.strip():
        from rag_pipeline import retrieve, format_context

        with st.spinner("Searching vector store..."):
            t0      = time.time()
            results = retrieve(query.strip(), collection, encoder, top_k)
            t_ret   = time.time() - t0

        st.markdown(f'<div class="section-header">Retrieved {top_k} Documents in {t_ret:.3f}s</div>', unsafe_allow_html=True)

        with st.expander("View retrieved source documents"):
            for i, (meta, dist) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
                sim = 1.0 - dist
                st.markdown(
                    f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:8px 12px;margin:4px 0">'
                    f'<span style="color:#3b82f6;font-family:Space Mono,monospace;font-size:0.75rem">[{i+1}]</span> '
                    f'<span style="color:#e2e8f0;font-size:0.82rem">{meta.get("image_name","N/A")}</span> '
                    f'<span style="color:#64748b;font-size:0.78rem"> | split={meta.get("split","?")} | sim={sim:.3f} | dominant={meta.get("dominant_class","?")}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        context = format_context(results)

        system_p = (
            "You are an expert remote sensing analyst specializing in UAV aerial imagery "
            "and semantic segmentation. You have access to detailed segmentation analysis "
            "of UAVid dataset images. Answer questions accurately and insightfully. "
            "When referencing images or statistics, be precise."
        )
        prompt = (
            f"{system_p}\n\n"
            f"Context from UAVid segmentation analysis:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Provide a detailed, insightful answer with specific statistics."
        )

        import google.generativeai as genai
        from config import GEMINI_MODELS

        answer     = None
        t_llm      = 0.0
        used_model = None

        model_try_order = GEMINI_MODELS

        for model_name in model_try_order:
            with st.spinner(f"Calling {model_name}..."):
                try:
                    genai.configure(api_key=api_key)
                    gemini   = genai.GenerativeModel(model_name)
                    t0       = time.time()
                    response = gemini.generate_content(prompt)
                    answer   = response.text
                    t_llm    = time.time() - t0
                    used_model = model_name
                    break
                except Exception as e:
                    err_str = str(e)
                    if any(x in err_str for x in ["429", "quota", "RESOURCE_EXHAUSTED", "rate"]):
                        st.warning(f"{model_name} quota exceeded — trying next model...")
                        import time as _t; _t.sleep(4)
                        continue
                    else:
                        st.error(f"API error ({model_name}): {e}")
                        break

        if answer:
            st.markdown(f'<div class="section-header">Answer via {used_model} — {t_llm:.2f}s</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)
        elif answer is None:
            st.error("All Gemini models returned quota errors. Wait a few minutes and try again, or check billing at https://aistudio.google.com")
            st.info("Free tier limit: 15 requests/min on gemini-1.5-flash. Try waiting 60s.")

def main():
    with st.sidebar:
        st.markdown(
            '<div style="font-family:Space Mono,monospace;font-size:1rem;color:#e2e8f0;font-weight:700;margin-bottom:4px">UAVid AI Explorer</div>'
            '<div style="font-size:0.75rem;color:#475569;margin-bottom:20px">Remote Sensing Analysis</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["Live Inference", "Dataset Overview", "Evaluation Metrics", "Batch Insights", "RAG Q&A"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        if page == "Live Inference":
            st.markdown(
                '<div style="font-size:0.78rem;color:#3b82f6;font-family:Space Mono,monospace;'
                'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px">Analysis Options</div>',
                unsafe_allow_html=True
            )
            run_deep = st.checkbox(
                "Deep Analysis",
                value=False,
                help="Uncertainty map, per-class confidence, boundary detection, spatial heatmap"
            )
            st.markdown(
                '<div style="font-size:0.72rem;color:#475569;margin-top:4px">'
                'Uncertainty + Confidence + Boundary</div>',
                unsafe_allow_html=True
            )
            st.session_state["run_deep"] = run_deep
        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.78rem;color:#475569;line-height:1.8">'
            '<b style="color:#64748b">Dataset</b><br>UAVid Modified<br><br>'
            '<b style="color:#64748b">Segmentation</b><br>SegFormer-B0<br><br>'
            '<b style="color:#64748b">LLM</b><br>Gemini 2.5 Flash<br><br>'
            '<b style="color:#64748b">Embeddings</b><br>all-MiniLM-L6-v2<br><br>'
            '<b style="color:#64748b">Vector DB</b><br>ChromaDB'
            '</div>',
            unsafe_allow_html=True
        )

    if page == "Live Inference":
        page_live_inference()
    elif page == "Dataset Overview":
        page_dataset_overview()
    elif page == "Evaluation Metrics":
        page_evaluation_metrics()
    elif page == "Batch Insights":
        page_batch_insights()
    elif page == "RAG Q&A":
        page_rag()


if __name__ == "__main__":
    main()
