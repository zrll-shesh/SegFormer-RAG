# UAVid RAG Explorer

Semantic segmentation analysis and RAG-based Q&A system for the Modified UAVid Dataset.

## Project Structure

```
uavid_rag_project/
├── src/
│   ├── config.py          # All paths, class definitions, constants
│   ├── eda.py             # Full exploratory data analysis
│   ├── segmentation.py    # Label-based segmentation + insight extraction
│   └── rag_pipeline.py    # ChromaDB + Gemini RAG pipeline
├── app.py                 # Streamlit web application
├── pipeline.py            # Full pipeline runner
├── requirements.txt
└── .env.example
```

## Setup

1. Clone and install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and add your Gemini API key:
```bash
cp .env.example .env
```
Get a free key at: https://aistudio.google.com

3. Place dataset in project root so structure is:
```
modified_uavid_dataset/
├── train_data/
│   ├── Images/
│   └── Labels/
├── val_data/
│   ├── Images/
│   └── Labels/
└── test_data/
    └── Images/
```

## Running

### Option A: Full pipeline (recommended first run)
```bash
python pipeline.py
```

### Option B: Step by step
```bash
python src/eda.py          # Run EDA only
python src/segmentation.py # Run segmentation + extract insights
python src/rag_pipeline.py # Build vector store + test queries
```

### Option C: Skip to Streamlit (if pipeline already ran)
```bash
streamlit run app.py
```

## Pipeline Arguments
```bash
python pipeline.py --skip-eda        # Skip EDA step
python pipeline.py --skip-seg        # Skip segmentation (use cached insights)
python pipeline.py --skip-rag-build  # Skip vector store rebuild
python pipeline.py --run-queries     # Run sample Q&A after build
```

## Architecture

```
UAVid Label PNG
     |
     v
Color -> Class ID mapping (pixel-level, 8 classes)
     |
     v
Per-image statistics (coverage %, pixel counts)
     |
     v
Insight text generation (structured natural language)
     |
     v
Sentence Embeddings (all-MiniLM-L6-v2, CPU-friendly)
     |
     v
ChromaDB Vector Store (persistent)
     |
     v
RAG Query: user question -> retrieve top-k docs -> Gemini 2.0 Flash
     |
     v
Streamlit UI
```

## Classes
| Class | Color (R,G,B) |
|---|---|
| Background clutter | (0,0,0) |
| Building | (128,0,0) |
| Road | (128,64,128) |
| Tree | (0,128,0) |
| Low vegetation | (128,128,0) |
| Moving car | (64,0,128) |
| Static car | (192,0,192) |
| Human | (64,64,0) |
