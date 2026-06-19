"""
UAVid RAG Explorer - Streamlit Application
SegFormer-B0 semantic segmentation + RAG Q&A for UAVid dataset.
Ultimate modern design with glassmorphism, neon effects, and smooth animations.
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
import base64
from datetime import datetime

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


# ============================================================================
# ULTRA PREMIUM CSS WITH GLASSMORPHISM, NEON GLOW & ANIMATIONS
# ============================================================================
st.set_page_config(
    page_title="UAVid Remote Sensing AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ===== FONTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ===== KEYFRAME ANIMATIONS ===== */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(40px) scale(0.96); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-30px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-50px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes slideInRight {
    from { opacity: 0; transform: translateX(50px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes neonPulse {
    0%, 100% { 
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.15), 0 0 40px rgba(59, 130, 246, 0.05);
    }
    50% { 
        box-shadow: 0 0 40px rgba(59, 130, 246, 0.3), 0 0 80px rgba(59, 130, 246, 0.08), 0 0 120px rgba(139, 92, 246, 0.04);
    }
}

@keyframes neonPulseBlue {
    0%, 100% { 
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.2), 0 0 30px rgba(59, 130, 246, 0.05);
        border-color: rgba(59, 130, 246, 0.1);
    }
    50% { 
        box-shadow: 0 0 35px rgba(59, 130, 246, 0.4), 0 0 70px rgba(59, 130, 246, 0.1);
        border-color: rgba(59, 130, 246, 0.25);
    }
}

@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-12px); }
}

@keyframes floatSlow {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    33% { transform: translateY(-8px) rotate(1deg); }
    66% { transform: translateY(4px) rotate(-1deg); }
}

@keyframes rotateGlow {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes borderGlow {
    0%, 100% { border-color: rgba(59, 130, 246, 0.08); }
    50% { border-color: rgba(59, 130, 246, 0.2); }
}

@keyframes breathe {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
}

@keyframes sparkle {
    0% { transform: scale(0) rotate(0deg); opacity: 0; }
    50% { transform: scale(1.2) rotate(180deg); opacity: 1; }
    100% { transform: scale(0) rotate(360deg); opacity: 0; }
}

@keyframes dataStream {
    0% { background-position: 0% 0%; }
    100% { background-position: 200% 0%; }
}

/* ===== BACKGROUND WITH NEON ORBS ===== */
.main {
    background: #060a16;
    background-image: 
        radial-gradient(ellipse at 15% 20%, rgba(59, 130, 246, 0.07) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 80%, rgba(139, 92, 246, 0.07) 0%, transparent 55%),
        radial-gradient(ellipse at 50% 50%, rgba(6, 182, 212, 0.03) 0%, transparent 70%),
        radial-gradient(ellipse at 20% 80%, rgba(236, 72, 153, 0.02) 0%, transparent 50%);
    animation: gradientShift 20s ease-in-out infinite;
    position: relative;
}

/* ===== FLOATING NEON ORBS ===== */
.glow-orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(120px);
    pointer-events: none;
    z-index: 0;
    animation: floatSlow 14s ease-in-out infinite;
}

.glow-orb-1 {
    width: 600px;
    height: 600px;
    background: rgba(59, 130, 246, 0.04);
    top: -200px;
    right: -150px;
    animation-delay: 0s;
}

.glow-orb-2 {
    width: 500px;
    height: 500px;
    background: rgba(139, 92, 246, 0.03);
    bottom: -150px;
    left: -100px;
    animation-delay: -5s;
}

.glow-orb-3 {
    width: 400px;
    height: 400px;
    background: rgba(6, 182, 212, 0.03);
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    animation-delay: -10s;
}

/* ===== GLASSMORPHISM ===== */
.glass {
    background: rgba(15, 23, 42, 0.5);
    backdrop-filter: blur(24px) saturate(1.8);
    -webkit-backdrop-filter: blur(24px) saturate(1.8);
    border: 1px solid rgba(255, 255, 255, 0.03);
    border-radius: 16px;
    animation: neonPulse 5s ease-in-out infinite;
}

.glass-strong {
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(28px) saturate(2);
    -webkit-backdrop-filter: blur(28px) saturate(2);
    border: 1px solid rgba(59, 130, 246, 0.06);
    border-radius: 16px;
    animation: neonPulseBlue 6s ease-in-out infinite;
}

/* ===== SIDEBAR ===== */
div[data-testid="stSidebarContent"] {
    background: rgba(6, 10, 22, 0.94) !important;
    backdrop-filter: blur(32px) saturate(2);
    -webkit-backdrop-filter: blur(32px) saturate(2);
    border-right: 1px solid rgba(59, 130, 246, 0.04);
    padding: 24px 16px;
    position: relative;
    z-index: 10;
}

div[data-testid="stSidebarContent"]::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, 
        rgba(59, 130, 246, 0.03) 0%, 
        transparent 40%, 
        rgba(139, 92, 246, 0.03) 100%);
    pointer-events: none;
}

div[data-testid="stSidebarContent"]::after {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 1px;
    height: 100%;
    background: linear-gradient(180deg, 
        transparent 0%, 
        rgba(59, 130, 246, 0.1) 30%, 
        rgba(139, 92, 246, 0.1) 70%, 
        transparent 100%);
}

/* ===== SIDEBAR TITLE ===== */
.sidebar-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6, #60a5fa);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 4s ease-in-out infinite;
    letter-spacing: -0.5px;
    margin-bottom: 2px;
    position: relative;
}

.sidebar-title::after {
    content: '';
    position: absolute;
    bottom: -6px;
    left: 0;
    width: 44px;
    height: 2.5px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #f472b6);
    background-size: 200% 100%;
    animation: gradientShift 3s ease-in-out infinite;
    border-radius: 2px;
}

.sidebar-sub {
    font-size: 0.62rem;
    color: rgba(100, 116, 139, 0.5);
    margin-bottom: 20px;
    font-weight: 400;
    letter-spacing: 0.35em;
    text-transform: uppercase;
}

/* ===== SIDEBAR NAVIGATION ===== */
div[data-testid="stSidebarContent"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1) !important;
    padding: 10px 14px !important;
    border-radius: 10px !important;
    margin: 2px 0 !important;
    cursor: pointer !important;
    position: relative !important;
}

div[data-testid="stSidebarContent"] .stRadio label::before {
    content: '';
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%) scaleX(0);
    width: 3px;
    height: 50%;
    background: linear-gradient(180deg, #3b82f6, #8b5cf6);
    border-radius: 0 3px 3px 0;
    transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

div[data-testid="stSidebarContent"] .stRadio label:hover {
    background: rgba(59, 130, 246, 0.05) !important;
    color: #e2e8f0 !important;
    transform: translateX(5px);
}

div[data-testid="stSidebarContent"] .stRadio label:hover::before {
    transform: translateY(-50%) scaleX(1);
}

/* ===== PAGE TITLE ===== */
.page-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f1f5f9 15%, #60a5fa 45%, #a78bfa 70%, #f472b6 100%);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 6s ease-in-out infinite;
    margin-bottom: 2px;
    letter-spacing: -0.5px;
    animation: fadeInUp 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    padding-left: 4px;
    position: relative;
}

.page-title::after {
    content: '';
    position: absolute;
    bottom: -8px;
    left: 4px;
    width: 70px;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #f472b6);
    background-size: 200% 100%;
    animation: gradientShift 3s ease-in-out infinite;
    border-radius: 2px;
}

.page-sub {
    font-size: 0.9rem;
    color: #64748b;
    margin-bottom: 28px;
    font-weight: 400;
    animation: fadeInUp 0.8s cubic-bezier(0.4, 0, 0.2, 1) 0.15s both;
    padding-left: 4px;
    padding-top: 14px;
    letter-spacing: 0.02em;
}

/* ===== SECTION HEADER ===== */
.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #60a5fa;
    text-transform: uppercase;
    letter-spacing: 0.28em;
    margin: 28px 0 14px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(59, 130, 246, 0.04);
    display: flex;
    align-items: center;
    gap: 12px;
    animation: slideInLeft 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}

.section-header::before {
    content: '';
    width: 3px;
    height: 22px;
    background: linear-gradient(180deg, #3b82f6, #8b5cf6, #f472b6);
    border-radius: 2px;
    display: inline-block;
    flex-shrink: 0;
    animation: breathe 2.5s ease-in-out infinite;
}

.section-header::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 0;
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, 
        rgba(59, 130, 246, 0.2) 0%, 
        transparent 40%, 
        transparent 60%, 
        rgba(139, 92, 246, 0.2) 100%);
}

/* ===== METRIC CARDS ===== */
.metric-card {
    background: rgba(15, 23, 42, 0.35);
    backdrop-filter: blur(16px) saturate(1.4);
    -webkit-backdrop-filter: blur(16px) saturate(1.4);
    border: 1px solid rgba(59, 130, 246, 0.04);
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    margin: 4px 0;
    transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    animation: neonPulseBlue 5s ease-in-out infinite;
}

.metric-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, 
        rgba(59, 130, 246, 0.03) 0%, 
        transparent 40%, 
        rgba(139, 92, 246, 0.03) 60%, 
        transparent 100%);
    pointer-events: none;
}

.metric-card::after {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at center, 
        rgba(59, 130, 246, 0.02) 0%, 
        transparent 70%);
    opacity: 0;
    transition: opacity 0.6s ease;
    pointer-events: none;
}

.metric-card:hover {
    transform: translateY(-4px) scale(1.02);
    border-color: rgba(59, 130, 246, 0.15);
    box-shadow: 0 12px 48px rgba(59, 130, 246, 0.06);
}

.metric-card:hover::after {
    opacity: 1;
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    position: relative;
    z-index: 1;
}

.metric-label {
    font-size: 0.62rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-top: 6px;
    font-weight: 500;
    position: relative;
    z-index: 1;
}

/* ===== INSIGHT CARD ===== */
.insight-card {
    background: linear-gradient(135deg, 
        rgba(15, 23, 42, 0.6), 
        rgba(30, 41, 59, 0.25));
    backdrop-filter: blur(20px) saturate(1.6);
    -webkit-backdrop-filter: blur(20px) saturate(1.6);
    border: 1px solid rgba(59, 130, 246, 0.06);
    border-radius: 14px;
    padding: 20px 24px;
    font-size: 0.85rem;
    color: #cbd5e1;
    line-height: 2.1;
    margin: 10px 0;
    animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.insight-card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, 
        rgba(59, 130, 246, 0.02) 0%, 
        transparent 60%);
    animation: rotateGlow 25s linear infinite;
    pointer-events: none;
}

.insight-card b {
    color: #60a5fa;
    font-weight: 600;
    position: relative;
    z-index: 1;
}

/* ===== ANSWER CARD ===== */
.answer-card {
    background: rgba(15, 23, 42, 0.55);
    backdrop-filter: blur(20px) saturate(1.6);
    -webkit-backdrop-filter: blur(20px) saturate(1.6);
    border: 1px solid rgba(59, 130, 246, 0.06);
    border-left: 4px solid;
    border-image: linear-gradient(180deg, #3b82f6, #8b5cf6, #f472b6) 1;
    border-radius: 14px;
    padding: 24px 28px;
    color: #e2e8f0;
    font-size: 0.92rem;
    line-height: 2.1;
    margin-top: 12px;
    animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.answer-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, 
        rgba(59, 130, 246, 0.02) 0%, 
        transparent 40%, 
        rgba(139, 92, 246, 0.02) 100%);
    pointer-events: none;
}

/* ===== BUTTONS ===== */
.stButton button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 10px 28px !important;
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 28px rgba(59, 130, 246, 0.12) !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton button::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, 
        rgba(255, 255, 255, 0.06) 0%, 
        transparent 40%, 
        transparent 60%, 
        rgba(255, 255, 255, 0.03) 100%);
    pointer-events: none;
}

.stButton button::after {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at center, 
        rgba(255, 255, 255, 0.04) 0%, 
        transparent 70%);
    opacity: 0;
    transition: opacity 0.6s ease;
    pointer-events: none;
}

.stButton button:hover {
    transform: translateY(-3px) scale(1.03) !important;
    box-shadow: 0 8px 48px rgba(59, 130, 246, 0.25) !important;
}

.stButton button:hover::after {
    opacity: 1;
}

.stButton button:active {
    transform: translateY(0) scale(0.97) !important;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(15, 23, 42, 0.2);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255, 255, 255, 0.01);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 22px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    color: #94a3b8 !important;
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #e2e8f0 !important;
    background: rgba(59, 130, 246, 0.04) !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: rgba(59, 130, 246, 0.08) !important;
    color: #60a5fa !important;
    box-shadow: 0 0 40px rgba(59, 130, 246, 0.03) !important;
}

/* ===== FILE UPLOADER ===== */
.upload-zone {
    background: rgba(15, 23, 42, 0.15);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 2px dashed rgba(59, 130, 246, 0.08);
    border-radius: 14px;
    padding: 32px;
    text-align: center;
    color: #64748b;
    transition: all 0.5s ease;
    animation: borderGlow 4s ease-in-out infinite;
}

.upload-zone:hover {
    border-color: rgba(59, 130, 246, 0.25);
    background: rgba(15, 23, 42, 0.25);
}

/* ===== EXPANDER ===== */
.streamlit-expanderHeader {
    background: rgba(15, 23, 42, 0.15) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-radius: 10px !important;
    border: 1px solid rgba(59, 130, 246, 0.03) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    transition: all 0.4s ease !important;
}

.streamlit-expanderHeader:hover {
    background: rgba(15, 23, 42, 0.25) !important;
    border-color: rgba(59, 130, 246, 0.08) !important;
}

/* ===== DATA FRAME ===== */
.dataframe-container {
    background: rgba(15, 23, 42, 0.15);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-radius: 12px;
    border: 1px solid rgba(59, 130, 246, 0.03);
    padding: 4px;
}

/* ===== SLIDER ===== */
.stSlider div[data-baseweb="slider"] {
    background: rgba(59, 130, 246, 0.04) !important;
}

.stSlider div[role="slider"] {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    border: 2px solid rgba(255, 255, 255, 0.04) !important;
    box-shadow: 0 0 30px rgba(59, 130, 246, 0.1) !important;
    transition: all 0.4s ease !important;
}

.stSlider div[role="slider"]:hover {
    box-shadow: 0 0 50px rgba(59, 130, 246, 0.2) !important;
    transform: scale(1.12);
}

/* ===== TEXT INPUT ===== */
.stTextInput input, .stTextArea textarea {
    background: rgba(15, 23, 42, 0.25) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(59, 130, 246, 0.04) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: rgba(59, 130, 246, 0.2) !important;
    box-shadow: 0 0 50px rgba(59, 130, 246, 0.03) !important;
    background: rgba(15, 23, 42, 0.35) !important;
    transform: scale(1.005);
}

/* ===== SELECTBOX ===== */
.stSelectbox div[data-baseweb="select"] {
    background: rgba(15, 23, 42, 0.25) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-radius: 10px !important;
    border: 1px solid rgba(59, 130, 246, 0.04) !important;
    transition: all 0.4s ease !important;
}

.stSelectbox div[data-baseweb="select"]:hover {
    border-color: rgba(59, 130, 246, 0.12) !important;
}

/* ===== CHECKBOX ===== */
.stCheckbox label {
    color: #94a3b8 !important;
    font-weight: 500 !important;
    transition: color 0.4s ease !important;
}

.stCheckbox label:hover {
    color: #e2e8f0 !important;
}

.stCheckbox div[data-baseweb="checkbox"] div:first-child {
    border-color: #334155 !important;
    border-radius: 6px !important;
    transition: all 0.4s ease !important;
}

.stCheckbox div[data-baseweb="checkbox"] div:first-child[data-checked="true"] {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    border-color: #3b82f6 !important;
}

/* ===== ALERTS ===== */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.15);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #3b82f6, #8b5cf6);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #2563eb, #7c3aed);
}

/* ===== RESPONSIVE ===== */
@media (max-width: 768px) {
    .page-title {
        font-size: 1.4rem;
    }
    .metric-value {
        font-size: 1.2rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CACHED RESOURCES
# ============================================================================
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


# ============================================================================
# VERSION-SAFE IMAGE RENDERER
# ============================================================================
import inspect as _inspect
_IMAGE_PARAMS = set(_inspect.signature(st.image).parameters.keys())
if "use_container_width" in _IMAGE_PARAMS:
    _IMAGE_WIDTH_KW = {"use_container_width": True}
elif "width" in _IMAGE_PARAMS:
    _IMAGE_WIDTH_KW = {"width": "stretch"}
else:
    _IMAGE_WIDTH_KW = {}


def safe_image(container, image_src, caption=None):
    kwargs = dict(_IMAGE_WIDTH_KW)
    if caption is not None:
        kwargs["caption"] = caption
    try:
        container.image(image_src, **kwargs)
    except TypeError:
        fallback_kwargs = {"caption": caption} if caption is not None else {}
        container.image(image_src, **fallback_kwargs)


# ============================================================================
# RENDER FUNCTIONS
# ============================================================================
def render_class_bars(stats_dict):
    sorted_items = sorted(stats_dict.items(), key=lambda x: x[1]["coverage_pct"], reverse=True)
    for cls_name, s in sorted_items:
        pct = s["coverage_pct"]
        color = CLASS_MAP[cls_name]
        hx = hex_color(color)
        bar_w = max(pct * 1.8, 0.3)
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0;padding:2px 0;">'
            f'<div style="width:14px;height:14px;background:{hx};border-radius:4px;flex-shrink:0;border:1px solid rgba(255,255,255,0.03);box-shadow:0 0 15px {hx}22;"></div>'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#94a3b8;width:150px;flex-shrink:0;">{cls_name}</span>'
            f'<div style="flex:1;background:rgba(255,255,255,0.015);border-radius:4px;height:8px;overflow:hidden;min-width:20px;box-shadow:inset 0 1px 3px rgba(0,0,0,0.3);">'
            f'<div style="background:linear-gradient(90deg,{hx},{hx}dd);width:{bar_w:.1f}%;height:100%;border-radius:4px;transition:width 1s cubic-bezier(0.4,0,0.2,1);box-shadow:0 0 20px {hx}33;"></div>'
            f'</div>'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#e2e8f0;min-width:50px;text-align:right;">{pct:.2f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_coverage_radar(stats_dict, title="Coverage Radar"):
    classes = CLASS_NAMES
    values = [stats_dict[c]["coverage_pct"] if isinstance(stats_dict[c], dict)
              else stats_dict[c] for c in classes]
    values_plot = values + [values[0]]
    angles = np.linspace(0, 2 * np.pi, len(classes), endpoint=False).tolist() + \
             [np.linspace(0, 2 * np.pi, len(classes), endpoint=False)[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    
    ax.plot(angles, values_plot, color="#3b82f6", linewidth=3, alpha=0.9)
    ax.fill(angles, values_plot, color="#3b82f6", alpha=0.1)
    
    for i in range(len(angles)-1):
        ax.plot([angles[i], angles[i+1]], [values_plot[i], values_plot[i+1]], 
                color="#3b82f6", linewidth=8, alpha=0.05)
    
    ax.set_thetagrids(np.degrees(angles[:-1]), classes, 
                      fontsize=7, color="#94a3b8", fontweight="500")
    ax.tick_params(colors="#64748b")
    ax.spines["polar"].set_color("rgba(255,255,255,0.03)")
    ax.set_title(title, fontsize=9, color="#e2e8f0", pad=20, fontweight="600")
    
    for label in ax.get_yticklabels():
        label.set_color("#64748b")
    
    ax.grid(alpha=0.03, color="#64748b")
    plt.tight_layout()
    return fig


# ============================================================================
# CORE SEGMENTATION FUNCTION
# ============================================================================
def segment_and_display(image: Image.Image, source_label="Uploaded Image", gt_mask=None, run_analysis=False):
    model = get_segformer_model()

    with st.spinner("Running SegFormer-B0 inference..."):
        result = run_inference(model, image)

    pred_mask = result["pred_mask"]
    stats = result["stats"]
    elapsed = result["elapsed"]
    img_arr = np.array(image.convert("RGB"))
    orig_h, orig_w = img_arr.shape[:2]

    import torch
    import torch.nn.functional as tfF
    pred_t = torch.tensor(pred_mask).unsqueeze(0).unsqueeze(0).float()
    pred_resized = tfF.interpolate(pred_t, size=(orig_h, orig_w), mode="nearest")
    pred_final = pred_resized.squeeze().numpy().astype(np.uint8)
    color_final = mask_to_color(pred_final)
    overlay = (img_arr * 0.45 + color_final * 0.55).astype(np.uint8)

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
    col_bar, col_radar = st.columns([1.4, 1])
    with col_bar:
        render_class_bars(stats)
    with col_radar:
        fig = render_coverage_radar(stats, source_label)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    dominant = max(stats, key=lambda c: stats[c]["coverage_pct"])
    veg = stats["Tree"]["coverage_pct"] + stats["Low vegetation"]["coverage_pct"]
    cars = stats["Moving car"]["coverage_pct"] + stats["Static car"]["coverage_pct"]
    scene = ("Urban Dense" if stats["Building"]["coverage_pct"] > 30 else
             "Mixed Urban" if stats["Building"]["coverage_pct"] > 10 else "Natural/Open")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{stats[dominant]["coverage_pct"]:.1f}%</div><div class="metric-label">Dominant</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{veg:.1f}%</div><div class="metric-label">Vegetation</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{stats["Road"]["coverage_pct"]:.1f}%</div><div class="metric-label">Road</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-value">{cars:.2f}%</div><div class="metric-label">Vehicles</div></div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="metric-card"><div class="metric-value">{stats["Human"]["coverage_pct"]:.3f}%</div><div class="metric-label">Human</div></div>', unsafe_allow_html=True)
    c6.markdown(f'<div class="metric-card"><div class="metric-value">{elapsed:.2f}s</div><div class="metric-label">Inference</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Scene Intelligence</div>', unsafe_allow_html=True)
    sorted_cls = sorted(stats.items(), key=lambda x: x[1]["coverage_pct"], reverse=True)
    top3_str = " | ".join([f"{c}: {s['coverage_pct']:.1f}%" for c, s in sorted_cls[:3]])
    present = [c for c, s in stats.items() if s["coverage_pct"] > 1.0]
    absent = [c for c, s in stats.items() if s["coverage_pct"] < 0.1]
    urban_idx = (stats["Building"]["coverage_pct"] * 0.4 +
                 stats["Road"]["coverage_pct"] * 0.35 +
                 cars * 0.25)
    green_idx = veg
    human_risk = ("DETECTED" if stats["Human"]["coverage_pct"] > 0.5 else
                  "TRACE" if stats["Human"]["coverage_pct"] > 0.05 else "NOT DETECTED")

    insight_html = f"""
    <div class="insight-card">
    <b>SCENE TYPE:</b> {scene}<br>
    <b>TOP CLASSES:</b> {top3_str}<br>
    <b>PRESENT (>1%):</b> {", ".join(present) if present else "none"}<br>
    <b>ABSENT (<0.1%):</b> {", ".join(absent) if absent else "none"}<br>
    <b>URBAN INDEX:</b> {urban_idx:.1f}% (Building+Road+Vehicle weighted)<br>
    <b>GREEN INDEX:</b> {green_idx:.1f}% (Tree+LowVegetation)<br>
    <b>HUMAN PRESENCE:</b> {human_risk} ({stats["Human"]["coverage_pct"]:.3f}%)<br>
    <b>MOVING VEHICLES:</b> {stats["Moving car"]["coverage_pct"]:.2f}% | PARKED: {stats["Static car"]["coverage_pct"]:.2f}%<br>
    <b>BACKGROUND CLUTTER:</b> {stats["Background clutter"]["coverage_pct"]:.2f}%
    </div>"""
    st.markdown(insight_html, unsafe_allow_html=True)

    if gt_mask is not None:
        metrics = compute_metrics_from_masks(pred_final, gt_mask)
        st.markdown('<div class="section-header">Evaluation vs Ground Truth</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["pixel_accuracy"]:.2f}%</div><div class="metric-label">Pixel Accuracy</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["mean_iou"]:.2f}%</div><div class="metric-label">Mean IoU</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["fw_iou"]:.2f}%</div><div class="metric-label">FW-IoU</div></div>', unsafe_allow_html=True)
        best_cls = max(metrics["per_class_iou"], key=metrics["per_class_iou"].get)
        m4.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["per_class_iou"][best_cls]:.1f}%</div><div class="metric-label">Best: {best_cls[:8]}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Per-Class IoU Detail</div>', unsafe_allow_html=True)
        iou_sorted = sorted(metrics["per_class_iou"].items(), key=lambda x: x[1], reverse=True)
        for cls_name, iou_val in iou_sorted:
            color = CLASS_MAP[cls_name]
            hx = "#{:02x}{:02x}{:02x}".format(*color)
            bar_w = max(iou_val * 1.5, 0.3)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin:3px 0;padding:2px 0;">'
                f'<div style="width:12px;height:12px;background:{hx};border-radius:3px;flex-shrink:0;border:1px solid rgba(255,255,255,0.03);box-shadow:0 0 12px {hx}22;"></div>'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#94a3b8;width:140px;flex-shrink:0;">{cls_name}</span>'
                f'<div style="flex:1;background:rgba(255,255,255,0.015);border-radius:3px;height:6px;overflow:hidden;min-width:20px;box-shadow:inset 0 1px 3px rgba(0,0,0,0.3);">'
                f'<div style="background:linear-gradient(90deg,{hx},{hx}dd);width:{bar_w:.1f}%;height:100%;border-radius:3px;transition:width 1s cubic-bezier(0.4,0,0.2,1);box-shadow:0 0 16px {hx}33;"></div>'
                f'</div>'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#e2e8f0;min-width:50px;text-align:right;">{iou_val:.2f}%</span>'
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
            a1, a2, a3 = st.columns(3)
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
                    "Max Confidence": f'{c["max_conf"]:.4f}',
                    "High Conf >80%": f'{c["high_conf_pct"]:.2f}%',
                    "Predicted Area": f'{c["area_pct"]:.2f}%',
                })
            st.dataframe(pd.DataFrame(conf_rows), use_container_width=True)

    return stats, pred_final


# ============================================================================
# PAGE FUNCTIONS
# ============================================================================
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
        split_sel = st.selectbox("Split", ["train", "val", "test"])
        img_dir = TRAIN_IMAGES if split_sel == "train" else (VAL_IMAGES if split_sel == "val" else TEST_IMAGES)
        lbl_dir = TRAIN_LABELS if split_sel == "train" else (VAL_LABELS if split_sel == "val" else None)

        if not img_dir.exists():
            st.warning(f"Directory not found: {img_dir}")
        else:
            img_paths = sorted(img_dir.glob("*.png"))
            if not img_paths:
                st.warning("No images found.")
            else:
                img_names = [p.name for p in img_paths]
                selected = st.selectbox("Select image", img_names)
                img_path = img_dir / selected

                if st.button("Run Segmentation", key="run_dataset"):
                    image = Image.open(img_path).convert("RGB")
                    gt_mask = None
                    if lbl_dir:
                        lbl_path = lbl_dir / selected
                        if lbl_path.exists():
                            gt_arr = np.array(Image.open(lbl_path).convert("RGB"))
                            gt_mask = color_to_class_id(gt_arr)
                    segment_and_display(image, source_label=selected, gt_mask=gt_mask, run_analysis=st.session_state.get("run_deep", False))


def page_dataset_overview():
    st.markdown('<div class="page-title">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">UAVid Modified Dataset - Exploratory Analysis</div>', unsafe_allow_html=True)

    train_imgs = sorted(TRAIN_IMAGES.glob("*.png")) if TRAIN_IMAGES.exists() else []
    val_imgs = sorted(VAL_IMAGES.glob("*.png")) if VAL_IMAGES.exists() else []
    test_imgs = sorted(TEST_IMAGES.glob("*.png")) if TEST_IMAGES.exists() else []
    train_lbls = sorted(TRAIN_LABELS.glob("*.png")) if TRAIN_LABELS.exists() else []
    val_lbls = sorted(VAL_LABELS.glob("*.png")) if VAL_LABELS.exists() else []

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
            f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0;padding:8px 12px;background:rgba(15,23,42,0.25);backdrop-filter:blur(8px);border-radius:10px;border:1px solid rgba(59,130,246,0.03);transition:all 0.4s ease;animation:borderGlow 4s ease-in-out infinite;">'
            f'<div style="width:20px;height:20px;background:{hx};border-radius:4px;border:1px solid rgba(255,255,255,0.03);flex-shrink:0;box-shadow:0 0 20px {hx}22;"></div>'
            f'<span style="font-size:0.82rem;color:#cbd5e1;font-weight:500;">{cls_name}</span>'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#475569;margin-left:auto;">RGB{color}</span>'
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
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    
    x = np.arange(len(CLASS_NAMES))
    w = 0.35
    if "train" in loaded:
        iou_t = [loaded["train"]["per_class_iou"].get(c, 0) for c in CLASS_NAMES]
        bars = ax.bar(x - w/2, iou_t, w, label="Train", color="#3b82f6", alpha=0.8, edgecolor="none")
    if "val" in loaded:
        iou_v = [loaded["val"]["per_class_iou"].get(c, 0) for c in CLASS_NAMES]
        bars = ax.bar(x + w/2, iou_v, w, label="Val", color="#a78bfa", alpha=0.8, edgecolor="none")
    
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=8, color="#94a3b8", fontweight="500")
    ax.set_ylabel("IoU (%)", color="#94a3b8", fontweight="500")
    ax.set_title("Per-Class IoU — SegFormer-B0", color="#e2e8f0", fontsize=11, fontweight="600")
    
    legend = ax.legend(labelcolor="#e2e8f0", facecolor="rgba(15,23,42,0.3)", edgecolor="rgba(255,255,255,0.02)", labelspacing=0.5, framealpha=0.5)
    legend.get_frame().set_alpha(0.5)
    
    ax.grid(axis="y", alpha=0.03, color="#64748b")
    ax.tick_params(colors="#64748b")
    for spine in ax.spines.values():
        spine.set_color("rgba(255,255,255,0.02)")
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
            "Image": item.get("image_name", ""),
            "Split": item.get("split", ""),
            "Dominant": max(stats, key=stats.get) if stats else "",
            "Road %": round(stats.get("Road", 0), 2),
            "Building %": round(stats.get("Building", 0), 2),
            "Tree %": round(stats.get("Tree", 0), 2),
            "Human %": round(stats.get("Human", 0), 4),
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
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    
    colors_norm = [tuple(c/255 for c in CLASS_MAP.get(cls, (128, 128, 128))) for cls in dom_counts.index]
    bars = ax.bar(dom_counts.index, dom_counts.values, color=colors_norm, edgecolor="none", alpha=0.8)
    
    ax.set_ylabel("Number of images", color="#94a3b8", fontweight="500")
    ax.set_title("Dominant Class per Image", color="#e2e8f0", fontsize=11, fontweight="600")
    ax.tick_params(axis="x", rotation=30, labelsize=8, colors="#94a3b8")
    ax.tick_params(axis="y", colors="#64748b")
    for spine in ax.spines.values():
        spine.set_color("rgba(255,255,255,0.02)")
    ax.grid(axis="y", alpha=0.03, color="#64748b")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def page_rag():
    st.markdown('<div class="page-title">RAG Q&A System</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Ask anything about UAVid dataset using semantic retrieval + Gemini 2.5 Flash</div>', unsafe_allow_html=True)

    api_key = st.sidebar.text_input("Gemini API Key", type="password", placeholder="AIza...")

    if not api_key:
        st.markdown("""
        <div style="background:rgba(15,23,42,0.35);backdrop-filter:blur(16px);border:1px solid rgba(59,130,246,0.04);border-radius:14px;padding:24px;animation:neonPulseBlue 5s ease-in-out infinite;">
        <p style="color:#94a3b8;margin:0;font-weight:400;">Enter your Gemini API Key in the sidebar to activate Q&A.</p>
        <p style="color:#64748b;font-size:0.82rem;margin-top:8px;">Get a free key at <a href="https://aistudio.google.com" style="color:#60a5fa;text-decoration:none;font-weight:500;">aistudio.google.com</a></p>
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
            t0 = time.time()
            results = retrieve(query.strip(), collection, encoder, top_k)
            t_ret = time.time() - t0

        st.markdown(f'<div class="section-header">Retrieved {top_k} Documents in {t_ret:.3f}s</div>', unsafe_allow_html=True)

        with st.expander("View retrieved source documents"):
            for i, (meta, dist) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
                sim = 1.0 - dist
                st.markdown(
                    f'<div style="background:rgba(15,23,42,0.15);border:1px solid rgba(59,130,246,0.03);border-radius:10px;padding:10px 14px;margin:4px 0;transition:all 0.4s ease;animation:borderGlow 4s ease-in-out infinite;">'
                    f'<span style="color:#60a5fa;font-family:JetBrains Mono,monospace;font-size:0.75rem;font-weight:600;">[{i+1}]</span> '
                    f'<span style="color:#e2e8f0;font-size:0.82rem;font-weight:500;">{meta.get("image_name","N/A")}</span> '
                    f'<span style="color:#64748b;font-size:0.75rem;"> | split={meta.get("split","?")} | sim={sim:.3f} | dominant={meta.get("dominant_class","?")}</span>'
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

        answer = None
        t_llm = 0.0
        used_model = None

        model_try_order = GEMINI_MODELS

        for model_name in model_try_order:
            with st.spinner(f"Calling {model_name}..."):
                try:
                    genai.configure(api_key=api_key)
                    gemini = genai.GenerativeModel(model_name)
                    t0 = time.time()
                    response = gemini.generate_content(prompt)
                    answer = response.text
                    t_llm = time.time() - t0
                    used_model = model_name
                    break
                except Exception as e:
                    err_str = str(e)
                    if any(x in err_str for x in ["429", "quota", "RESOURCE_EXHAUSTED", "rate"]):
                        st.warning(f"{model_name} quota exceeded — trying next model...")
                        import time as _t
                        _t.sleep(4)
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


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-title">UAVid AI Explorer</div>'
            '<div class="sidebar-sub">Remote Sensing Analysis</div>',
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
                '<div style="font-size:0.68rem;color:#60a5fa;font-family:JetBrains Mono,monospace;'
                'text-transform:uppercase;letter-spacing:0.18em;margin-bottom:8px;font-weight:600;">Analysis Options</div>',
                unsafe_allow_html=True
            )
            run_deep = st.checkbox(
                "Deep Analysis",
                value=False,
                help="Uncertainty map, per-class confidence, boundary detection, spatial heatmap"
            )
            st.markdown(
                '<div style="font-size:0.62rem;color:#475569;margin-top:4px;">'
                'Uncertainty + Confidence + Boundary</div>',
                unsafe_allow_html=True
            )
            st.session_state["run_deep"] = run_deep
        
        st.markdown("---")
        
        st.markdown(
            '<div style="font-size:0.7rem;color:#475569;line-height:2.1;">'
            '<b style="color:#64748b;font-weight:600;">Dataset</b><br>UAVid Modified<br><br>'
            '<b style="color:#64748b;font-weight:600;">Segmentation</b><br>SegFormer-B0<br><br>'
            '<b style="color:#64748b;font-weight:600;">LLM</b><br>Gemini 2.5 Flash<br><br>'
            '<b style="color:#64748b;font-weight:600;">Embeddings</b><br>all-MiniLM-L6-v2<br><br>'
            '<b style="color:#64748b;font-weight:600;">Vector DB</b><br>ChromaDB'
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
