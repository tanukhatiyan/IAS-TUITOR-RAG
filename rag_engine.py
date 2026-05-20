import os
import pickle
from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pdfplumber

# ── Config ──────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # ~90MB, CPU-friendly
CHROMA_DIR       = "./chroma_db"
COLLECTION_NAME  = "ias_books"
CHUNK_SIZE       = 500    # tokens (approx characters / 4)
CHUNK_OVERLAP    = 50


class RAGEngine:
    def __init__(self):
        print("Loading embedding model (first run downloads ~90MB)...")
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Vector store ready. Chunks stored: {self.collection.count()}")

    # ── PDF loading ──────────────────────────────────────────────────────────
    def load_pdf(self, pdf_path: str) -> str:
        """Extract all text from a PDF file."""
        text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text.append(t)
        return "\n".join(text)

    # ── Chunking ─────────────────────────────────────────────────────────────
    def chunk_text(self, text: str, source: str) -> List[dict]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        i = 0
        chunk_id = 0
        while i < len(words):
            chunk_words = words[i: i + CHUNK_SIZE]
            chunk_text  = " ".join(chunk_words)
            chunks.append({
                "id":   f"{Path(source).stem}_{chunk_id}",
                "text": chunk_text,
                "meta": {"source": Path(source).name, "chunk": chunk_id},
            })
            i += CHUNK_SIZE - CHUNK_OVERLAP
            chunk_id += 1
        return chunks

    # ── Ingest PDF ────────────────────────────────────────────────────────────
    def ingest_pdf(self, pdf_path: str) -> int:
        """Load, chunk, embed and store a PDF. Returns number of new chunks."""
        source_name = Path(pdf_path).name

        # Skip if already indexed
        existing = self.collection.get(where={"source": source_name})
        if existing["ids"]:
            return 0  # already done

        raw_text = self.load_pdf(pdf_path)
        chunks   = self.chunk_text(raw_text, pdf_path)

        texts = [c["text"] for c in chunks]
        ids   = [c["id"]   for c in chunks]
        metas = [c["meta"] for c in chunks]

        # Embed in batches of 64
        batch = 64
        embeddings = []
        for i in range(0, len(texts), batch):
            vecs = self.embedder.encode(texts[i: i + batch], show_progress_bar=False)
            embeddings.extend(vecs.tolist())

        self.collection.add(
            ids        = ids,
            documents  = texts,
            embeddings = embeddings,
            metadatas  = metas,
        )
        return len(chunks)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    def retrieve(self, query: str, k: int = 4) -> List[Tuple[str, str, float]]:
        """Return top-k (chunk_text, source, distance) for a query."""
        q_vec = self.embedder.encode([query])[0].tolist()
        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            out.append((doc, meta.get("source", "unknown"), dist))
        return out

    def list_sources(self) -> List[str]:
        """Return unique source file names in the store."""
        if self.collection.count() == 0:
            return []
        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted(set(m.get("source", "") for m in all_meta))

    def total_chunks(self) -> int:
        return self.collection.count()
