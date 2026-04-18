"""
RAG Pipeline for UAVid Dataset.
Builds a vector store from segmentation insights and enables semantic Q&A
using ChromaDB + sentence-transformers + Google Gemini 2.0 Flash.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent))
from config import OUTPUT_DIR, VECTORSTORE_DIR, EMBED_MODEL, COLLECTION_NAME, GEMINI_MODEL


def load_insights(insight_path: Optional[Path] = None):
    if insight_path is None:
        insight_path = OUTPUT_DIR / "insights_all.json"

    if not insight_path.exists():
        docs = []
        for name in ["insights_train.json", "insights_val.json", "insights_test.json"]:
            p = OUTPUT_DIR / name
            if p.exists():
                with open(p) as f:
                    docs.extend(json.load(f))
        return docs

    with open(insight_path) as f:
        return json.load(f)


def build_documents(insights):
    documents = []
    metadatas = []
    ids       = []

    for idx, item in enumerate(insights):
        image_name = item.get("image_name", f"image_{idx}")
        split      = item.get("split", "unknown")
        insight    = item.get("insight", "")
        stats      = item.get("class_stats", {})

        stats_str = " | ".join([f"{cls}: {pct:.2f}%" for cls, pct in stats.items()])

        doc = (
            "[UAVid Image Analysis]\n"
            f"Image: {image_name}\n"
            f"Dataset split: {split}\n"
            f"Segmentation insight: {insight}\n"
            f"Detailed class coverage: {stats_str}\n"
        )

        dominant = max(stats, key=stats.get) if stats else "unknown"

        documents.append(doc)
        metadatas.append({
            "image_name":     image_name,
            "split":          split,
            "dominant_class": dominant,
            "road_pct":       round(float(stats.get("Road", 0)), 2),
            "building_pct":   round(float(stats.get("Building", 0)), 2),
            "tree_pct":       round(float(stats.get("Tree", 0)), 2),
            "human_pct":      round(float(stats.get("Human", 0)), 4),
        })
        ids.append(f"uavid_{split}_{idx:04d}")

    return documents, metadatas, ids


def build_vectorstore(insights=None, force_rebuild=False):
    from sentence_transformers import SentenceTransformer
    import chromadb

    chroma_path       = str(VECTORSTORE_DIR / "chroma_uavid")
    collection_exists = (VECTORSTORE_DIR / "chroma_uavid").exists()

    if collection_exists and not force_rebuild:
        print(f"\nLoading existing vector store from: {chroma_path}")
        client     = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(COLLECTION_NAME)
        print(f"  Documents in store: {collection.count()}")
        return collection

    print("\nBuilding vector store...")

    if insights is None:
        insights = load_insights()

    if not insights:
        raise ValueError("No insights found. Run segmentation pipeline first.")

    print(f"  Loading embedding model: {EMBED_MODEL}")
    encoder = SentenceTransformer(EMBED_MODEL)

    documents, metadatas, ids = build_documents(insights)
    print(f"  Total documents to index: {len(documents)}")

    print("  Generating embeddings...")
    t0         = time.time()
    embeddings = encoder.encode(documents, show_progress_bar=True, batch_size=16)
    elapsed    = time.time() - t0
    print(f"  Embedding done in {elapsed:.1f}s | shape: {embeddings.shape}")

    client = chromadb.PersistentClient(path=chroma_path)

    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Deleted old collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size  = 50
    num_batches = (len(documents) + batch_size - 1) // batch_size

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_embs = embeddings[i : i + batch_size].tolist()
        batch_meta = metadatas[i : i + batch_size]
        batch_ids  = ids[i : i + batch_size]

        collection.add(
            documents=batch_docs,
            embeddings=batch_embs,
            metadatas=batch_meta,
            ids=batch_ids,
        )
        batch_num = i // batch_size + 1
        print(f"  Indexed batch {batch_num}/{num_batches} ({len(batch_docs)} docs)")

    total = collection.count()
    print(f"  Vector store ready. Total indexed: {total}")
    return collection


def retrieve(query: str, collection, encoder, top_k: int = 5):
    query_vec = encoder.encode([query])[0].tolist()
    results   = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return results


def format_context(results):
    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    parts = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
        similarity = 1.0 - dist
        parts.append(
            f"[Source {i+1} | similarity={similarity:.3f} | image={meta.get('image_name','N/A')}]\n{doc}"
        )
    return "\n\n".join(parts)


def ask_gemini(query: str, context: str, api_key: str, model: str = GEMINI_MODEL):
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel(model)

    system_prompt = (
        "You are an expert remote sensing analyst specializing in UAV aerial imagery "
        "and semantic segmentation. You have access to detailed segmentation analysis "
        "of UAVid dataset images. Answer questions accurately and insightfully based on "
        "the provided context. When referencing specific images or statistics, be precise. "
        "If the context does not contain enough information, say so clearly."
    )

    full_prompt = (
        f"{system_prompt}\n\n"
        f"Context from UAVid segmentation analysis:\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        f"Provide a detailed, insightful answer with specific statistics where available."
    )

    response = gemini.generate_content(full_prompt)
    return response.text


def rag_pipeline(query: str, collection, encoder, api_key: str, top_k: int = 5):
    print(f"\nQuery: {query}")
    print("-" * 60)

    t0      = time.time()
    results = retrieve(query, collection, encoder, top_k)
    t_ret   = time.time() - t0
    print(f"  Retrieved {top_k} documents in {t_ret:.3f}s")

    for i, (meta, dist) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
        sim = 1.0 - dist
        print(f"  [{i+1}] {meta.get('image_name','N/A')} | "
              f"similarity={sim:.3f} | dominant={meta.get('dominant_class','N/A')}")

    context = format_context(results)

    t0     = time.time()
    answer = ask_gemini(query, context, api_key)
    t_llm  = time.time() - t0
    print(f"  LLM response in {t_llm:.2f}s")
    print(f"\nAnswer:\n{answer}")
    return answer, results


def load_rag_components():
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {EMBED_MODEL}")
    encoder = SentenceTransformer(EMBED_MODEL)

    print("Loading vector store...")
    collection = build_vectorstore()
    return encoder, collection


def run_sample_queries(api_key: str):
    encoder, collection = load_rag_components()

    sample_queries = [
        "Which images have the highest road coverage percentage?",
        "What are the scenes with significant human presence?",
        "Describe the vegetation distribution across the dataset.",
        "Which images are dominated by buildings?",
        "Are there images with both moving cars and static cars present?",
    ]

    for query in sample_queries:
        rag_pipeline(query, collection, encoder, api_key)
        print("\n" + "=" * 60)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY in your .env file")
    else:
        run_sample_queries(api_key)
