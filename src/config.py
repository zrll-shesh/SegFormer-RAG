import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "modified_uavid_dataset"

TRAIN_IMAGES = DATA_DIR / "train_data" / "Images"
TRAIN_LABELS = DATA_DIR / "train_data" / "Labels"
VAL_IMAGES   = DATA_DIR / "val_data"   / "Images"
VAL_LABELS   = DATA_DIR / "val_data"   / "Labels"
TEST_IMAGES  = DATA_DIR / "test_data"  / "Images"

OUTPUT_DIR      = BASE_DIR / "outputs"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

OUTPUT_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)

CLASS_MAP = {
    "Background clutter": (0,   0,   0),
    "Building":           (128, 0,   0),
    "Road":               (128, 64,  128),
    "Tree":               (0,   128, 0),
    "Low vegetation":     (128, 128, 0),
    "Moving car":         (64,  0,   128),
    "Static car":         (192, 0,   192),
    "Human":              (64,  64,  0),
}

CLASS_NAMES  = list(CLASS_MAP.keys())
CLASS_COLORS = list(CLASS_MAP.values())
COLOR_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}

# Model list in priority order - tries each until one succeeds
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]
GEMINI_MODEL = "gemini-2.0-flash"
EMBED_MODEL     = "all-MiniLM-L6-v2"
COLLECTION_NAME = "uavid_segments"
TOLERANCE       = 10

NUM_CLASSES = 8