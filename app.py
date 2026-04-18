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
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0a0e1a; }

.page-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 4px solid #3b82f6;
    padding-left: 16px;
    margin-bottom: 4px;
    letter-spacing: -0.5px;
}
.page-sub {
    font-size: 0.88rem;
    color: #64748b;
    padding-left: 22px;
    margin-bottom: 24px;
    font-family: 'Space Mono', monospace;
}
.metric-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 18px 16px;
    text-align: center;
    margin: 4px 0;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    color: #3b82f6;
}
.metric-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}
.insight-card {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border: 1px solid #1e40af;
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 0.87rem;
    color: #cbd5e1;
    line-height: 1.7;
    margin: 10px 0;
}
.answer-card {
    background: #0f172a;
    border: 1px solid #1d4ed8;
    border-left: 5px solid #3b82f6;
    border-radius: 10px;
    padding: 20px 24px;
    color: #e2e8f0;
    font-size: 0.92rem;
    line-height: 1.8;
    margin-top: 12px;
}
.class-chip {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    margin: 2px;
    font-family: 'Space Mono', monospace;
}
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #3b82f6;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin: 20px 0 8px 0;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 6px;
}
div[data-testid="stSidebarContent"] {
    background: #050810;
    border-right: 1px solid #1e293b;
}
div[data-testid="stSidebarContent"] label,
div[data-testid="stSidebarContent"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 0.88rem;
}
div[data-testid="stSidebarContent"] p {
    color: #64748b;
    font-size: 0.8rem;
}
.stButton button {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
}
.stButton button:hover {
    background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%) !important;
    transform: translateY(-1px) !important;
}
.upload-zone {
    background: #0f172a;
    border: 2px dashed #1e40af;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    color: #64748b;
}
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


def segment_and_display(image: Image.Image, source_label="Uploaded Image", gt_mask=None):
    model = get_segformer_model()

    with st.spinner("Running SegFormer-B0 inference..."):
        t0     = time.time()
        result = run_inference(model, image)
        elapsed = time.time() - t0

    pred_mask  = result["pred_mask"]
    color_mask = result["color_mask"]
    overlay    = result["overlay"]
    stats      = result["stats"]

    img_arr = np.array(image.convert("RGB"))
    orig_h, orig_w = img_arr.shape[:2]

    import torch
    import torch.nn.functional as F
    pred_t       = torch.tensor(pred_mask).unsqueeze(0).unsqueeze(0).float()
    pred_resized = F.interpolate(pred_t, size=(orig_h, orig_w), mode="nearest")
    pred_final   = pred_resized.squeeze().numpy().astype(np.uint8)

    col1, col2, col3 = st.columns(3)
    col1.image(img_arr,              caption="Input Image",            use_container_width=True)
    col2.image(mask_to_color(pred_final), caption="SegFormer-B0 Mask",  use_container_width=True)
    col3.image((img_arr * 0.45 + mask_to_color(pred_final) * 0.55).astype(np.uint8),
               caption="Overlay (55% seg)", use_container_width=True)

    st.markdown('<div class="section-header">Class Coverage</div>', unsafe_allow_html=True)
    col_bar, col_radar = st.columns([1.2, 1])
    with col_bar:
        render_class_bars(stats)
    with col_radar:
        fig = render_coverage_radar(stats, f"Coverage - {source_label}")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    dominant = max(stats, key=lambda c: stats[c]["coverage_pct"])
    veg      = stats["Tree"]["coverage_pct"] + stats["Low vegetation"]["coverage_pct"]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{stats[dominant]["coverage_pct"]:.1f}%</div><div class="metric-label">Dominant ({dominant[:8]})</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{veg:.1f}%</div><div class="metric-label">Total Vegetation</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{stats["Road"]["coverage_pct"]:.1f}%</div><div class="metric-label">Road Coverage</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-value">{elapsed:.2f}s</div><div class="metric-label">Inference Time</div></div>', unsafe_allow_html=True)

    insight = generate_insight_text(source_label, stats, "inference")
    st.markdown('<div class="section-header">AI Insight</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

    if gt_mask is not None:
        import torch, torch.nn.functional as F
        metrics = compute_metrics_from_masks(pred_final, gt_mask)
        st.markdown('<div class="section-header">Evaluation Metrics vs Ground Truth</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["pixel_accuracy"]:.2f}%</div><div class="metric-label">Pixel Accuracy</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["mean_iou"]:.2f}%</div><div class="metric-label">Mean IoU</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["fw_iou"]:.2f}%</div><div class="metric-label">FW-IoU</div></div>', unsafe_allow_html=True)

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
            segment_and_display(image, source_label=uploaded.name)

    with tab_camera:
        st.markdown('<div class="section-header">Camera Capture</div>', unsafe_allow_html=True)
        st.info("Point your camera at a scene and capture for instant segmentation.")
        camera_img = st.camera_input("Take a photo")
        if camera_img:
            image = Image.open(camera_img).convert("RGB")
            segment_and_display(image, source_label="camera_capture.jpg")

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
                    segment_and_display(image, source_label=selected, gt_mask=gt_mask)


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
            cols[i % 2].image(str(f), caption=f.stem.replace("_", " ").title(), use_container_width=True)
    else:
        st.info("Run `python src/eda.py` to generate EDA charts.")

    st.markdown('<div class="section-header">Segmentation Overlay Samples</div>', unsafe_allow_html=True)
    seg_files = sorted(OUTPUT_DIR.glob("seg_overlay_*.png"))
    if seg_files:
        for f in seg_files[:4]:
            st.image(str(f), caption=f.stem.replace("_", " ").title(), use_container_width=True)
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
    if metrics_img.exists():
        st.image(str(metrics_img), caption="Metrics Comparison Chart", use_container_width=True)


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
    st.markdown('<div class="page-sub">Ask anything about UAVid dataset using semantic retrieval + Gemini 2.0 Flash</div>', unsafe_allow_html=True)

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
        st.markdown(
            '<div style="font-size:0.78rem;color:#475569;line-height:1.8">'
            '<b style="color:#64748b">Dataset</b><br>UAVid Modified<br><br>'
            '<b style="color:#64748b">Segmentation</b><br>SegFormer-B0<br><br>'
            '<b style="color:#64748b">LLM</b><br>Gemini 2.0 Flash<br><br>'
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