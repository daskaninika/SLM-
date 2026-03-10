r"""
build_rag_index.py - RAG Index Builder for SLM_wData Project
==============================================================
Chunks all training documents, builds a FAISS index using
sentence-transformers embeddings, and provides a retrieval function.

Two modes:
  1) BUILD mode (default):  Reads data/, chunks, embeds, saves index
  2) QUERY mode:            Loads saved index, retrieves relevant chunks

Usage (from project root C:\Users\kanidas\OneDrive - Cisco\Desktop\SLM_wData):
    venv\Scripts\activate

    # Build the index:
    python src/build_rag_index.py --build

    # Query the index:
    python src/build_rag_index.py --query "VXLAN BGP EVPN multisite design"

    # Interactive query mode:
    python src/build_rag_index.py --interactive

    # Build + immediately query:
    python src/build_rag_index.py --build --query "spine leaf topology"
"""

import os
import sys
import json
import argparse
import pickle
import re
import numpy as np

# ──────────────────────────────────────────────
# 1.  CONFIGURATION
# ──────────────────────────────────────────────

DEFAULT_DATA_DIR = "data"
DEFAULT_INDEX_DIR = "artifacts/rag_index"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality

# Chunking settings
CHUNK_SIZE = 500       # target words per chunk
CHUNK_OVERLAP = 50     # overlap words between chunks
MIN_CHUNK_WORDS = 30   # skip tiny chunks


# ──────────────────────────────────────────────
# 2.  TEXT CHUNKING
# ──────────────────────────────────────────────

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP,
               min_words=MIN_CHUNK_WORDS):
    """
    Split text into overlapping chunks by word count.
    Returns list of chunk strings.
    """
    words = text.split()
    if len(words) < min_words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        if len(chunk_words) >= min_words:
            chunks.append(chunk_text_str)

        if end >= len(words):
            break
        start = end - overlap

    return chunks


def load_and_chunk_documents(data_dir):
    """
    Load all .txt files from data_dir, chunk them.
    Returns list of dicts: {"text": ..., "source": ..., "chunk_id": ...}
    """
    all_chunks = []
    data_path = os.path.abspath(data_dir)

    if not os.path.exists(data_path):
        print(f"[ERROR] Data directory not found: {data_path}")
        return all_chunks

    txt_files = sorted([
        f for f in os.listdir(data_path)
        if f.endswith(".txt") and os.path.isfile(os.path.join(data_path, f))
    ])

    print(f"[INFO] Found {len(txt_files)} .txt files in {data_path}")

    total_chunks = 0
    for fname in txt_files:
        fpath = os.path.join(data_path, fname)
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        except Exception as e:
            print(f"  [WARN] Could not read {fname}: {e}")
            continue

        word_count = len(text.split())
        if word_count < MIN_CHUNK_WORDS:
            print(f"  [SKIP] {fname} ({word_count} words - too short)")
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": fname,
                "chunk_id": f"{fname}::chunk_{i:04d}",
            })

        total_chunks += len(chunks)
        print(f"  [OK]   {fname}: {word_count:,} words -> {len(chunks)} chunks")

    # Also load from data/core42_sections/ and data/core42_chunks/ if they exist
    for subdir_name in ["core42_sections", "core42_chunks"]:
        subdir_path = os.path.join(data_path, subdir_name)
        if os.path.exists(subdir_path):
            sub_files = sorted([
                f for f in os.listdir(subdir_path)
                if f.endswith(".txt")
            ])
            print(f"\n[INFO] Found {len(sub_files)} files in {subdir_name}/")
            for fname in sub_files:
                fpath = os.path.join(subdir_path, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read().strip()
                except Exception as e:
                    print(f"  [WARN] Could not read {fname}: {e}")
                    continue

                word_count = len(text.split())
                if word_count < MIN_CHUNK_WORDS:
                    continue

                chunks = chunk_text(text)
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        "text": chunk,
                        "source": f"{subdir_name}/{fname}",
                        "chunk_id": f"{subdir_name}/{fname}::chunk_{i:04d}",
                    })
                total_chunks += len(chunks)
                print(f"  [OK]   {subdir_name}/{fname}: {word_count:,} words -> {len(chunks)} chunks")

    print(f"\n[INFO] Total chunks created: {len(all_chunks)}")
    return all_chunks


# ──────────────────────────────────────────────
# 3.  EMBEDDING & INDEX BUILDING
# ──────────────────────────────────────────────

def build_index(data_dir, index_dir, embed_model_name):
    """Build FAISS index from document chunks."""
    import faiss
    from sentence_transformers import SentenceTransformer

    os.makedirs(index_dir, exist_ok=True)

    # Load and chunk documents
    all_chunks = load_and_chunk_documents(data_dir)
    if not all_chunks:
        print("[ERROR] No chunks created. Cannot build index.")
        return

    # Load embedding model
    print(f"\n[INFO] Loading embedding model: {embed_model_name}")
    embed_model = SentenceTransformer(embed_model_name)
    embed_dim = embed_model.get_sentence_embedding_dimension()
    print(f"[INFO] Embedding dimension: {embed_dim}")

    # Encode all chunks
    texts = [c["text"] for c in all_chunks]
    print(f"[INFO] Encoding {len(texts)} chunks... (this may take a minute)")
    embeddings = embed_model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,  # for cosine similarity via inner product
    )

    print(f"[INFO] Embeddings shape: {embeddings.shape}")

    # Build FAISS index (Inner Product = cosine similarity when normalized)
    index = faiss.IndexFlatIP(embed_dim)
    index.add(embeddings.astype(np.float32))
    print(f"[INFO] FAISS index built with {index.ntotal} vectors")

    # Save everything
    faiss.write_index(index, os.path.join(index_dir, "faiss_index.bin"))
    with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(all_chunks, f)
    with open(os.path.join(index_dir, "index_config.json"), "w") as f:
        json.dump({
            "embed_model": embed_model_name,
            "embed_dim": embed_dim,
            "num_chunks": len(all_chunks),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }, f, indent=2)

    print(f"\n[INFO] Index saved to {index_dir}/")
    print(f"       faiss_index.bin  ({index.ntotal} vectors)")
    print(f"       chunks.pkl       ({len(all_chunks)} chunk records)")
    print(f"       index_config.json")


# ──────────────────────────────────────────────
# 4.  RETRIEVAL
# ──────────────────────────────────────────────

class RAGRetriever:
    """Loads saved FAISS index and retrieves relevant chunks."""

    def __init__(self, index_dir, embed_model_name=None):
        import faiss
        from sentence_transformers import SentenceTransformer

        # Load config
        config_path = os.path.join(index_dir, "index_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = {}

        if embed_model_name is None:
            embed_model_name = self.config.get("embed_model", DEFAULT_EMBED_MODEL)

        # Load index
        index_path = os.path.join(index_dir, "faiss_index.bin")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Run --build first.")
        self.index = faiss.read_index(index_path)

        # Load chunks
        chunks_path = os.path.join(index_dir, "chunks.pkl")
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks file not found at {chunks_path}. Run --build first.")
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        # Load embedding model
        print(f"[INFO] Loading embedding model: {embed_model_name}")
        self.embed_model = SentenceTransformer(embed_model_name)

        print(f"[INFO] RAG retriever ready: {self.index.ntotal} chunks indexed")

    def retrieve(self, query, top_k=5):
        """
        Retrieve top-k most relevant chunks for a query.

        Returns list of dicts with keys: text, source, chunk_id, score
        """
        # Encode query
        query_embedding = self.embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    def retrieve_as_context(self, query, top_k=5, max_words=1500):
        """
        Retrieve chunks and concatenate into a context string.
        Useful for feeding into the SLM as a RAG prompt.
        """
        results = self.retrieve(query, top_k=top_k)
        context_parts = []
        word_count = 0

        for r in results:
            chunk_words = len(r["text"].split())
            if word_count + chunk_words > max_words:
                break
            context_parts.append(
                f"[Source: {r['source']} | Score: {r['score']:.3f}]\n{r['text']}"
            )
            word_count += chunk_words

        return "\n\n---\n\n".join(context_parts)


# ──────────────────────────────────────────────
# 5.  MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build and query RAG index for SDD generation."
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Build the FAISS index from documents in data/."
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Query string to retrieve relevant chunks."
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Interactive query mode."
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of chunks to retrieve (default: 5)."
    )
    parser.add_argument(
        "--data_dir", type=str, default=DEFAULT_DATA_DIR,
        help=f"Data directory (default: {DEFAULT_DATA_DIR})."
    )
    parser.add_argument(
        "--index_dir", type=str, default=DEFAULT_INDEX_DIR,
        help=f"Index directory (default: {DEFAULT_INDEX_DIR})."
    )
    parser.add_argument(
        "--embed_model", type=str, default=DEFAULT_EMBED_MODEL,
        help=f"Sentence-transformer model (default: {DEFAULT_EMBED_MODEL})."
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, args.data_dir)
    index_dir = os.path.join(project_root, args.index_dir)

    # ── BUILD ──
    if args.build:
        print("=" * 60)
        print("  BUILDING RAG INDEX")
        print("=" * 60)
        build_index(data_dir, index_dir, args.embed_model)
        print("\n[DONE] Index build complete.")
        if not args.query and not args.interactive:
            return

    # ── QUERY ──
    if args.query or args.interactive:
        print("\n" + "=" * 60)
        print("  RAG RETRIEVAL")
        print("=" * 60)

        retriever = RAGRetriever(index_dir, args.embed_model)

        if args.query:
            results = retriever.retrieve(args.query, top_k=args.top_k)
            print(f"\n[QUERY] \"{args.query}\"")
            print(f"[TOP {len(results)} RESULTS]")
            for i, r in enumerate(results, 1):
                print(f"\n--- Result {i} (score: {r['score']:.4f}) ---")
                print(f"Source: {r['source']}")
                print(f"Chunk:  {r['chunk_id']}")
                # Print first 200 chars
                preview = r['text'][:300].replace('\n', ' ')
                print(f"Text:   {preview}...")

        if args.interactive:
            print("\n  Type a query and press Enter to search.")
            print("  Type 'quit' or 'exit' to stop.")
            print("  Type 'full' to toggle showing full chunk text.")
            print("  Type 'topk=N' to change number of results.\n")

            show_full = False
            top_k = args.top_k

            while True:
                try:
                    user_input = input("[SEARCH]> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting.")
                    break

                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit"):
                    break
                if user_input.lower() == "full":
                    show_full = not show_full
                    print(f"  [Show full text: {show_full}]")
                    continue
                if user_input.lower().startswith("topk="):
                    try:
                        top_k = int(user_input.split("=")[1])
                        print(f"  [top_k set to {top_k}]")
                    except ValueError:
                        pass
                    continue

                results = retriever.retrieve(user_input, top_k=top_k)
                print(f"\n  [{len(results)} results for \"{user_input}\"]")
                for i, r in enumerate(results, 1):
                    print(f"\n  --- Result {i} (score: {r['score']:.4f}, source: {r['source']}) ---")
                    if show_full:
                        print(f"  {r['text']}")
                    else:
                        preview = r['text'][:300].replace('\n', ' ')
                        print(f"  {preview}...")
                print()

    if not args.build and not args.query and not args.interactive:
        print("[INFO] No action specified. Use --build, --query, or --interactive.")
        print("Examples:")
        print("  python src/build_rag_index.py --build")
        print('  python src/build_rag_index.py --query "VXLAN multisite BGP EVPN"')
        print("  python src/build_rag_index.py --interactive")
        print("  python src/build_rag_index.py --build --interactive")


if __name__ == "__main__":
    main()