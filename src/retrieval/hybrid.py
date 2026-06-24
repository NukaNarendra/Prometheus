import json
import math
from typing import List, Dict, Any, Set, Tuple, Optional
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.retrieval.vector_store import VectorStoreOrchestrator
from src.retrieval.keyword_store import KeywordStoreOrchestrator


class HybridRetrievalConfig:
    def __init__(self, vector_weight: float = 0.5, keyword_weight: float = 0.5, rrf_k: int = 60,
                 top_k: int = 10) -> None:
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        self.top_k = top_k
        self.validate()

    def validate(self) -> None:
        if self.vector_weight < 0 or self.vector_weight > 1:
            raise ValueError("vector_weight must be between 0 and 1")
        if self.keyword_weight < 0 or self.keyword_weight > 1:
            raise ValueError("keyword_weight must be between 0 and 1")
        if abs((self.vector_weight + self.keyword_weight) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")


class DocumentNormalizer:
    @staticmethod
    def generate_id(doc: Dict[str, Any]) -> str:
        if "paper_id" in doc and doc["paper_id"]:
            return str(doc["paper_id"])
        if "id" in doc and doc["id"]:
            return str(doc["id"])
        title = doc.get("title", "")
        abstract = doc.get("abstract", "")
        combined = str(title) + str(abstract)
        return str(hash(combined))


class ReciprocalRankFusion:
    def __init__(self, config: HybridRetrievalConfig) -> None:
        self.config = config

    def fuse(self, ranked_lists: List[List[Dict[str, Any]]], weights: List[float]) -> List[Dict[str, Any]]:
        if len(ranked_lists) != len(weights):
            raise ValueError("Number of ranked lists must match number of weights")

        fused_scores: Dict[str, float] = {}
        document_store: Dict[str, Dict[str, Any]] = {}

        for list_idx, ranked_list in enumerate(ranked_lists):
            weight = weights[list_idx]
            for rank, doc in enumerate(ranked_list):
                doc_id = DocumentNormalizer.generate_id(doc)
                if doc_id not in document_store:
                    document_store[doc_id] = doc
                    fused_scores[doc_id] = 0.0

                rrf_score = 1.0 / (self.config.rrf_k + rank + 1)
                fused_scores[doc_id] += rrf_score * weight

        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        fused_results = []
        for doc_id, score in sorted_docs:
            doc = document_store[doc_id].copy()
            doc["hybrid_score"] = score
            fused_results.append(doc)

        return fused_results[:self.config.top_k]


class HybridSearcher:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.config = HybridRetrievalConfig()
        self.vector_store = VectorStoreOrchestrator(data_dir)
        self.keyword_store = KeywordStoreOrchestrator(data_dir)
        self.fusion_engine = ReciprocalRankFusion(self.config)
        self.is_initialized = False

    def initialize(self) -> None:
        self.vector_store.load_and_index("train.jsonl")
        self.vector_store.load_and_index("test.jsonl")
        self.keyword_store.load_and_index("train.jsonl")
        self.keyword_store.load_and_index("test.jsonl")
        self.is_initialized = True

    def search(self, query: str) -> List[Dict[str, Any]]:
        if not self.is_initialized:
            raise RuntimeError("Hybrid searcher must be initialized before querying")

        vector_results = self.vector_store.query(query, k=self.config.top_k * 2)
        keyword_results = self.keyword_store.query(query, k=self.config.top_k * 2)

        ranked_lists = [vector_results, keyword_results]
        weights = [self.config.vector_weight, self.config.keyword_weight]

        return self.fusion_engine.fuse(ranked_lists, weights)


class RetrievalEvaluator:
    def __init__(self, searcher: HybridSearcher, test_file: Path) -> None:
        self.searcher = searcher
        self.test_file = test_file
        self.test_documents: List[Dict[str, Any]] = []
        self.load_test_data()

    def load_test_data(self) -> None:
        if not self.test_file.exists():
            raise FileNotFoundError("Test file does not exist")

        with open(self.test_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        self.test_documents.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    def extract_ground_truth_keywords(self, doc: Dict[str, Any]) -> List[str]:
        title = doc.get("title", "")
        return [title]

    def compute_dcg(self, relevances: List[int]) -> float:
        dcg = 0.0
        for i, rel in enumerate(relevances):
            dcg += (2 ** rel - 1) / math.log2(i + 2)
        return dcg

    def evaluate_ndcg(self, k: int) -> float:
        total_queries = min(50, len(self.test_documents))
        ndcg_sum = 0.0

        for doc in self.test_documents[:total_queries]:
            query_terms = self.extract_ground_truth_keywords(doc)
            if not query_terms:
                total_queries -= 1
                continue

            query = " ".join(query_terms)
            retrieved_docs = self.searcher.search(query)
            target_id = DocumentNormalizer.generate_id(doc)

            relevances = []
            for ret_doc in retrieved_docs[:k]:
                if DocumentNormalizer.generate_id(ret_doc) == target_id:
                    relevances.append(1)
                else:
                    relevances.append(0)

            ideal_relevances = sorted(relevances, reverse=True)
            dcg = self.compute_dcg(relevances)
            idcg = self.compute_dcg(ideal_relevances)

            if idcg > 0:
                ndcg_sum += dcg / idcg

        return ndcg_sum / total_queries if total_queries > 0 else 0.0

    def evaluate(self, k_values: List[int]) -> Dict[str, Dict[str, float]]:
        if not self.searcher.is_initialized:
            self.searcher.initialize()

        results: Dict[str, Dict[str, float]] = {f"Recall@{k}": {} for k in k_values}
        results.update({f"Precision@{k}": {} for k in k_values})

        total_queries = min(50, len(self.test_documents))

        for k in k_values:
            recall_sum = 0.0
            precision_sum = 0.0

            for doc in self.test_documents[:total_queries]:
                query_terms = self.extract_ground_truth_keywords(doc)
                if not query_terms:
                    total_queries -= 1
                    continue

                query = " ".join(query_terms)
                original_top_k = self.searcher.config.top_k
                self.searcher.config.top_k = k

                retrieved_docs = self.searcher.search(query)
                self.searcher.config.top_k = original_top_k

                retrieved_ids = [DocumentNormalizer.generate_id(r) for r in retrieved_docs]
                target_id = DocumentNormalizer.generate_id(doc)

                if target_id in retrieved_ids:
                    recall_sum += 1.0

                hits = sum(1 for rid in retrieved_ids if rid == target_id)
                precision_sum += hits / k if k > 0 else 0.0

            if total_queries > 0:
                results[f"Recall@{k}"]["score"] = recall_sum / total_queries
                results[f"Precision@{k}"]["score"] = precision_sum / total_queries
            else:
                results[f"Recall@{k}"]["score"] = 0.0
                results[f"Precision@{k}"]["score"] = 0.0

        return results

    def compute_mrr(self) -> float:
        total_queries = min(50, len(self.test_documents))
        mrr_sum = 0.0

        for doc in self.test_documents[:total_queries]:
            query_terms = self.extract_ground_truth_keywords(doc)
            if not query_terms:
                total_queries -= 1
                continue

            query = " ".join(query_terms)
            retrieved_docs = self.searcher.search(query)
            target_id = DocumentNormalizer.generate_id(doc)

            for rank, ret_doc in enumerate(retrieved_docs):
                if DocumentNormalizer.generate_id(ret_doc) == target_id:
                    mrr_sum += 1.0 / (rank + 1)
                    break

        return mrr_sum / total_queries if total_queries > 0 else 0.0

    def run_benchmark(self) -> None:
        metrics = self.evaluate([1, 5, 10])
        mrr_score = self.compute_mrr()
        ndcg_score = self.evaluate_ndcg(10)

        print("\nRetrieval Benchmark Results on TEST Data:")
        for metric, data in metrics.items():
            print(f"{metric}: {data['score']:.4f}")
        print(f"Mean Reciprocal Rank (MRR): {mrr_score:.4f}")
        print(f"Normalized Discounted Cumulative Gain (NDCG@10): {ndcg_score:.4f}")


def run_hybrid_demo() -> None:
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data" / "corpus"

    try:
        searcher = HybridSearcher(data_dir)
        searcher.initialize()

        test_query = "KRAS G12C mutation resistance mechanisms in lung cancer"
        print(f"\nRunning test query against TRAIN set: '{test_query}'")
        results = searcher.search(test_query)

        for i, res in enumerate(results):
            title = res.get("title", "Unknown")
            score = res.get("hybrid_score", 0.0)
            print(f"Rank {i + 1}: Score={score:.4f} | {title[:80]}...")

        test_file = data_dir / "test.jsonl"
        if test_file.exists():
            evaluator = RetrievalEvaluator(searcher, test_file)
            evaluator.run_benchmark()

    except Exception as e:
        print(f"Demo failed: {str(e)}")


if __name__ == "__main__":
    run_hybrid_demo()