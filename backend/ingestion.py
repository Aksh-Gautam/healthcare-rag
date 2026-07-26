"""
Document ingestion pipeline for the Healthcare Knowledge Navigator.
Loads, chunks, embeds, and stores medical documents in ChromaDB.
"""
import os
import re
import hashlib
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default paths
KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge_base"
CHROMA_DB_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "medical_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def parse_document_metadata(content: str) -> tuple[dict, str]:
    """Parse YAML-like frontmatter metadata from document content."""
    metadata = {
        "title": "Unknown",
        "source": "Unknown",
        "category": "General",
        "date": "Unknown",
        "evidence_level": "Not specified",
        "specialty": "General",
    }

    # Check for frontmatter delimiters
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            body = parts[2].strip()

            for line in frontmatter.split("\n"):
                line = line.strip()
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower().replace(" ", "_")
                    value = value.strip().strip('"').strip("'")
                    if key in metadata:
                        metadata[key] = value

            return metadata, body

    return metadata, content


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[str]:
    """
    Split text into overlapping chunks using recursive character splitting.
    Tries to split on paragraph breaks, then sentences, then words.
    """
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Try splitting on double newlines (paragraphs) first
    separators = ["\n\n", "\n", ". ", " "]

    chunks = []
    current_text = text

    for separator in separators:
        if len(current_text) <= chunk_size:
            break

        new_chunks = []
        segments = current_text.split(separator)
        current_chunk = ""

        for segment in segments:
            test_chunk = current_chunk + separator + segment if current_chunk else segment

            if len(test_chunk) <= chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    new_chunks.append(current_chunk.strip())
                current_chunk = segment

        if current_chunk:
            new_chunks.append(current_chunk.strip())

        if all(len(c) <= chunk_size for c in new_chunks):
            # Apply overlap
            final_chunks = []
            for i, chunk in enumerate(new_chunks):
                if i > 0 and chunk_overlap > 0:
                    # Prepend end of previous chunk for context
                    prev_text = new_chunks[i - 1]
                    overlap_text = prev_text[-chunk_overlap:] if len(prev_text) > chunk_overlap else prev_text
                    chunk = overlap_text + " " + chunk
                final_chunks.append(chunk.strip())
            return [c for c in final_chunks if c]

    # Fallback: hard split by character count
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap

    return chunks


class DocumentIngestionPipeline:
    """Handles loading, chunking, embedding, and storing medical documents."""

    def __init__(
        self,
        knowledge_base_dir: Optional[Path] = None,
        chroma_db_dir: Optional[Path] = None,
    ):
        self.knowledge_base_dir = knowledge_base_dir or KNOWLEDGE_BASE_DIR
        self.chroma_db_dir = chroma_db_dir or CHROMA_DB_DIR

        # Initialize embedding model
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # Initialize ChromaDB
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_db_dir),
        )

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"ChromaDB initialized. Collection '{COLLECTION_NAME}' has "
            f"{self.collection.count()} documents."
        )

    def _generate_chunk_id(self, filename: str, chunk_index: int) -> str:
        """Generate a deterministic ID for a document chunk."""
        raw = f"{filename}::chunk_{chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()

    def ingest_file(self, filepath: Path) -> int:
        """Ingest a single document file. Returns number of chunks created."""
        logger.info(f"Ingesting: {filepath.name}")

        content = filepath.read_text(encoding="utf-8")
        metadata, body = parse_document_metadata(content)

        # Chunk the document body
        chunks = chunk_text(body)
        if not chunks:
            logger.warning(f"No chunks produced from {filepath.name}")
            return 0

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(filepath.name, i)
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "filename": filepath.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **metadata,
            })

        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"  → {len(chunks)} chunks indexed from {filepath.name}")
        return len(chunks)

    def ingest_directory(self, force_reingest: bool = False) -> dict:
        """Ingest all documents from the knowledge base directory."""
        if not self.knowledge_base_dir.exists():
            logger.error(f"Knowledge base directory not found: {self.knowledge_base_dir}")
            return {"files_processed": 0, "total_chunks": 0}

        # Check if already ingested
        existing_count = self.collection.count()
        if existing_count > 0 and not force_reingest:
            logger.info(
                f"Collection already has {existing_count} chunks. "
                "Use force_reingest=True to re-ingest."
            )
            return {"files_processed": 0, "total_chunks": existing_count, "skipped": True}

        # Clear existing data if re-ingesting
        if force_reingest and existing_count > 0:
            logger.info("Clearing existing collection for re-ingestion...")
            self.chroma_client.delete_collection(COLLECTION_NAME)
            self.collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        # Find all text files
        files = sorted(self.knowledge_base_dir.glob("*.txt"))
        logger.info(f"Found {len(files)} documents to ingest")

        total_chunks = 0
        for filepath in files:
            chunks = self.ingest_file(filepath)
            total_chunks += chunks

        result = {
            "files_processed": len(files),
            "total_chunks": total_chunks,
            "skipped": False,
        }
        logger.info(f"Ingestion complete: {result}")
        return result

    def ingest_uploaded_content(
        self,
        filename: str,
        content: str,
    ) -> int:
        """Ingest a document from uploaded content string."""
        metadata, body = parse_document_metadata(content)
        chunks = chunk_text(body)

        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(filename, i)
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **metadata,
            })

        embeddings = self.embedding_model.encode(documents).tolist()

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
    ) -> list[dict]:
        """Search for relevant document chunks."""
        query_embedding = self.embedding_model.encode([query]).tolist()

        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity score (0-1)
                relevance_score = max(0.0, 1.0 - distance / 2.0)

                chunks.append({
                    "chunk_id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "relevance_score": relevance_score,
                })

        return chunks

    def get_document_list(self) -> list[dict]:
        """Get a list of all unique documents indexed."""
        all_data = self.collection.get(include=["metadatas"])

        if not all_data or not all_data["metadatas"]:
            return []

        # Group by filename
        doc_map = {}
        for meta in all_data["metadatas"]:
            filename = meta.get("filename", "unknown")
            if filename not in doc_map:
                doc_map[filename] = {
                    "filename": filename,
                    "title": meta.get("title", "Unknown"),
                    "source": meta.get("source", "Unknown"),
                    "category": meta.get("category", "General"),
                    "evidence_level": meta.get("evidence_level", "Not specified"),
                    "specialty": meta.get("specialty", "General"),
                    "date": meta.get("date", "Unknown"),
                    "chunk_count": 0,
                }
            doc_map[filename]["chunk_count"] += 1

        return list(doc_map.values())

    def get_categories(self) -> list[str]:
        """Get unique document categories."""
        docs = self.get_document_list()
        categories = sorted(set(d["category"] for d in docs))
        return categories

    def get_total_chunks(self) -> int:
        """Get total number of indexed chunks."""
        return self.collection.count()

    def get_embeddings_for_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts. Used by XAI engine."""
        return self.embedding_model.encode(texts).tolist()
