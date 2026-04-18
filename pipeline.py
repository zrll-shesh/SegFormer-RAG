"""
Main pipeline runner for UAVid RAG project.
Runs: EDA -> Segmentation (train+val+test) -> Evaluation Metrics -> Build Vector Store -> Sample Q&A
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent / "src"))

from src.eda import run_full_eda
from src.segmentation import run_full_segmentation_pipeline
from src.rag_pipeline import build_vectorstore, load_insights, run_sample_queries


def run_pipeline(skip_eda=False, skip_seg=False, skip_rag_build=False, run_queries=False):
    print("UAVid Full Pipeline")
    print("=" * 60)

    if not skip_eda:
        print("\nStep 1: Exploratory Data Analysis")
        run_full_eda()
    else:
        print("\nStep 1: EDA skipped")

    if not skip_seg:
        print("\nStep 2: Segmentation + Evaluation Metrics (train / val / test)")
        all_insights = run_full_segmentation_pipeline()
    else:
        print("\nStep 2: Segmentation skipped, loading existing insights")
        all_insights = load_insights()

    if not skip_rag_build:
        print("\nStep 3: Building Vector Store")
        collection = build_vectorstore(insights=all_insights, force_rebuild=True)
        print(f"  Vector store ready: {collection.count()} documents")
    else:
        print("\nStep 3: RAG build skipped")

    if run_queries:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            print("\nStep 4: Sample RAG Queries")
            run_sample_queries(api_key)
        else:
            print("\nStep 4: Skipped (no GEMINI_API_KEY in .env)")

    print("\nPipeline complete.")
    print("Run the Streamlit app with:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-eda",       action="store_true")
    parser.add_argument("--skip-seg",       action="store_true")
    parser.add_argument("--skip-rag-build", action="store_true")
    parser.add_argument("--run-queries",    action="store_true")
    args = parser.parse_args()

    run_pipeline(
        skip_eda=args.skip_eda,
        skip_seg=args.skip_seg,
        skip_rag_build=args.skip_rag_build,
        run_queries=args.run_queries,
    )
