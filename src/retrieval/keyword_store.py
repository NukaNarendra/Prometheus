import re
import json
import math
import pickle
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from pydantic import BaseModel
from rank_bm25 import BM25Okapi


class KeywordConfig(BaseModel):
    min_word_length: int = 3
    max_word_length: int = 50
    persist_filename: str = "bm25_index.pkl"
    registry_filename: str = "document_registry.json"


class EnglishStopWords:
    @staticmethod
    def get_stop_words() -> Set[str]:
        return {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren",
            "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
            "can", "cannot", "could", "couldn", "did", "didn", "do", "does", "doesn", "doing", "don", "down",
            "during", "each", "few", "for", "from", "further", "had", "hadn", "has", "hasn", "have", "haven",
            "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
            "into", "is", "isn", "it", "its", "itself", "let", "me", "more", "most", "mustn", "my", "myself",
            "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
            "ourselves", "out", "over", "own", "same", "shan", "she", "should", "shouldn", "so", "some", "such",
            "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
            "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn", "we", "were",
            "weren", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "won", "would",
            "wouldn", "you", "your", "yours", "yourself", "yourselves", "background", "methods", "results",
            "conclusions", "introduction", "discussion", "abstract", "summary"
        }


class TextTokenizer:
    def __init__(self, config: KeywordConfig) -> None:
        self.config = config
        self.stop_words = EnglishStopWords.get_stop_words()
        self.pattern = re.compile(r'\b[a-zA-Z0-9_]+\b')

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []

        text = text.lower()
        tokens = self.pattern.findall(text)

        filtered_tokens = []
        for token in tokens:
            if len(token) >= self.config.min_word_length and len(token) <= self.config.max_word_length:
                if token not in self.stop_words:
                    if not token.isnumeric():
                        filtered_tokens.append(token)

        return filtered_tokens


class DocumentRegistry:
    def __init__(self) -> None:
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.index_to_id: Dict[int, str] = {}

    def add_document(self, doc_id: str, metadata: Dict[str, Any], index: int) -> None:
        self.documents[doc_id] = metadata
        self.index_to_id[index] = doc_id

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.documents.get(doc_id)

    def get_document_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        doc_id = self.index_to_id.get(index)
        if doc_id:
            return self.get_document(doc_id)
        return None

    def serialize(self) -> Dict[str, Any]:
        return {
            "documents": self.documents,
            "index_to_id": self.index_to_id
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        self.documents = data.get("documents", {})
        self.index_to_id = {int(k): v for k, v in data.get("index_to_id", {}).items()}


class BM25EngineWrapper:
    def __init__(self) -> None:
        self.bm25: Optional[BM25Okapi] = None
        self.is_built = False

    def build_index(self, tokenized_corpus: List[List[str]]) -> None:
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.is_built = True

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        if not self.is_built or self.bm25 is None:
            raise RuntimeError("BM25 index not built yet")
        return self.bm25.get_scores(query_tokens)


class ScoreNormalizer:
    @staticmethod
    def normalize_min_max(scores: List[float]) -> List[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [0.0 for _ in scores]
        return [(s - min_score) / (max_score - min_score) for s in scores]


class KeywordStoreOrchestrator:
    def __init__(self, base_dir: Path, config: Optional[KeywordConfig] = None) -> None:
        self.base_dir = base_dir
        self.config = config or KeywordConfig()

        self.index_path = self.base_dir / "cache" / self.config.persist_filename
        self.registry_path = self.base_dir / "cache" / self.config.registry_filename

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self.tokenizer = TextTokenizer(self.config)
        self.registry = DocumentRegistry()
        self.engine = BM25EngineWrapper()

    def _read_jsonl(self, filepath: Path) -> List[Dict[str, Any]]:
        documents = []
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found at {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    documents.append(json.loads(stripped_line))
        return documents

    def load_and_index(self, filename: str) -> None:
        file_path = self.base_dir / filename
        raw_documents = self._read_jsonl(file_path)

        tokenized_corpus = []
        for idx, doc in enumerate(raw_documents):
            doc_id = doc.get("paper_id", f"unknown_id_{idx}")
            title = doc.get("title", "")
            abstract = doc.get("abstract", "")

            full_text = f"{title} {abstract}"
            tokens = self.tokenizer.tokenize(full_text)
            tokenized_corpus.append(tokens)

            metadata = {
                "paper_id": doc_id,
                "title": title,
                "authors": doc.get("authors", []),
                "publication_date": doc.get("publication_date", ""),
                "url": doc.get("url", ""),
                "abstract": abstract
            }
            self.registry.add_document(doc_id, metadata, idx)

        self.engine.build_index(tokenized_corpus)
        self.save_state()

    def save_state(self) -> None:
        if self.engine.is_built:
            with open(self.index_path, "wb") as f:
                pickle.dump(self.engine.bm25, f)

            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(self.registry.serialize(), f)

    def load_state(self) -> bool:
        if self.index_path.exists() and self.registry_path.exists():
            with open(self.index_path, "rb") as f:
                self.engine.bm25 = pickle.load(f)
                self.engine.is_built = True

            with open(self.registry_path, "r", encoding="utf-8") as f:
                registry_data = json.load(f)
                self.registry.deserialize(registry_data)
            return True
        return False

    def query(self, query_text: str, k: int = 10) -> List[Dict[str, Any]]:
        if not self.engine.is_built:
            success = self.load_state()
            if not success:
                raise RuntimeError("Keyword index not built and no saved state found")

        query_tokens = self.tokenizer.tokenize(query_text)
        if not query_tokens:
            return []

        raw_scores = self.engine.get_scores(query_tokens)
        normalized_scores = ScoreNormalizer.normalize_min_max(raw_scores.tolist())

        scored_indices = [(idx, score) for idx, score in enumerate(normalized_scores) if score > 0]
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        top_k_indices = scored_indices[:k]

        results = []
        for idx, score in top_k_indices:
            doc_metadata = self.registry.get_document_by_index(idx)
            if doc_metadata:
                result_item = doc_metadata.copy()
                result_item["keyword_score"] = score
                results.append(result_item)

        return results