import os
import json
import uuid
import uuid
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStoreConfig(BaseModel):
    collection_name: str = "prometheus_papers"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    batch_size: int = 64
    distance_metric: str = "cosine"
    persist_directory: str = "chroma_db"

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        if v < 100:
            raise ValueError("chunk_size must be at least 100")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info: Any) -> int:
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class DocumentChunk(BaseModel):
    chunk_id: str
    paper_id: str
    text: str
    metadata: Dict[str, Any]
    chunk_index: int


class TextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []

        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_len = len(word) + 1
            if current_length + word_len > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(w) + 1 for w in current_chunk)

            current_chunk.append(word)
            current_length += word_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


class DocumentProcessor:
    def __init__(self, splitter: TextSplitter) -> None:
        self.splitter = splitter

    def process_document(self, raw_doc: Dict[str, Any]) -> List[DocumentChunk]:
        paper_id = str(raw_doc.get("paper_id", uuid.uuid4()))
        title = raw_doc.get("title", "")
        abstract = raw_doc.get("abstract", "")
        authors = raw_doc.get("authors", [])
        pub_date = raw_doc.get("publication_date", "")

        full_text = f"{title}. {abstract}"
        text_chunks = self.splitter.split_text(full_text)

        author_names = [a.get("name", "") if isinstance(a, dict) else str(a) for a in authors]
        author_str = ", ".join(author_names)

        processed_chunks = []
        for idx, text_segment in enumerate(text_chunks):
            chunk_id = f"{paper_id}_chunk_{idx}"
            metadata = {
                "paper_id": paper_id,
                "title": title,
                "authors": author_str,
                "publication_date": pub_date,
                "chunk_index": idx,
                "total_chunks": len(text_chunks)
            }

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                paper_id=paper_id,
                text=text_segment,
                metadata=metadata,
                chunk_index=idx
            )
            processed_chunks.append(chunk)

        return processed_chunks


class EmbeddingClient:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


class ChromaDBManager:
    def __init__(self, persist_dir: Path, collection_name: str, metric: str) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.metric = metric
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Any:
        try:
            return self.client.get_collection(name=self.collection_name)
        except Exception:
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.metric}
            )

    def upsert_batch(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]],
                     embeddings: List[List[float]]) -> None:
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def query(self, query_embeddings: List[List[float]], n_results: int) -> Dict[str, Any]:
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

    def get_document_count(self) -> int:
        return self.collection.count()


class VectorStoreOrchestrator:
    def __init__(self, base_dir: Path, config: Optional[VectorStoreConfig] = None) -> None:
        self.base_dir = base_dir
        self.config = config or VectorStoreConfig()

        self.db_path = self.base_dir / self.config.persist_directory
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.splitter = TextSplitter(self.config.chunk_size, self.config.chunk_overlap)
        self.processor = DocumentProcessor(self.splitter)
        self.embedder = EmbeddingClient(self.config.embedding_model)
        self.db_manager = ChromaDBManager(
            persist_dir=self.db_path,
            collection_name=self.config.collection_name,
            metric=self.config.distance_metric
        )

    def _read_jsonl(self, filepath: Path) -> Iterator[Dict[str, Any]]:
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found at {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    yield json.loads(stripped_line)

    def _batch_iterator(self, iterator: Iterator[Any], batch_size: int) -> Iterator[List[Any]]:
        batch = []
        for item in iterator:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def load_and_index(self, filename: str) -> None:
        file_path = self.base_dir / filename
        all_chunks = []

        for raw_doc in self._read_jsonl(file_path):
            doc_chunks = self.processor.process_document(raw_doc)
            all_chunks.extend(doc_chunks)

        for chunk_batch in self._batch_iterator(iter(all_chunks), self.config.batch_size):
            self._process_and_store_batch(chunk_batch)

    def _process_and_store_batch(self, chunks: List[DocumentChunk]) -> None:
        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata for c in chunks]

        embeddings = self.embedder.generate_embeddings(texts)
        self.db_manager.upsert_batch(ids, texts, metadatas, embeddings)

    def query(self, query_text: str, k: int = 10) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.generate_embeddings([query_text])[0]
        raw_results = self.db_manager.query([query_embedding], n_results=k)

        formatted_results = []
        if not raw_results["ids"] or not raw_results["ids"][0]:
            return formatted_results

        ids = raw_results["ids"][0]
        documents = raw_results["documents"][0]
        metadatas = raw_results["metadatas"][0]
        distances = raw_results["distances"][0]

        for idx in range(len(ids)):
            metadata_copy = metadatas[idx].copy()
            metadata_copy.update({
                "chunk_id": ids[idx],
                "matched_text": documents[idx],
                "vector_distance": distances[idx],
                "vector_score": 1.0 / (1.0 + distances[idx])
            })
            formatted_results.append(metadata_copy)

        return formatted_results