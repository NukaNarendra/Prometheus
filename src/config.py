import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class DirectoryConfig(BaseSettings):
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    cache_dir: Path = data_dir / "cache"
    corpus_dir: Path = data_dir / "corpus"
    memory_dir: Path = data_dir / "memory"
    src_dir: Path = base_dir / "src"
    eval_dir: Path = base_dir / "eval"
    log_dir: Path = base_dir / "logs"

    def create_all(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


class ModelConfig(BaseSettings):
    mode: str = os.getenv("PROMETHEUS_MODE", "dev")
    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")

    models: Dict[str, Dict[str, Any]] = {
        "lead_agent": {
            "dev": "ai-nemotron-3-super-120b-a12b",
            "prod": "ai-nemotron-3-ultra-550b-a55b",
            "enable_thinking": True,
            "temperature": 0.1,  # Lowered to prevent <unk> token soup!
            "max_tokens": 8192,
            "reasoning_budget": 4096,
            "top_p": 0.95,
        },
        "subagent": {
            "dev": "ai-nemotron-3-super-120b-a12b",
            "prod": "ai-nemotron-3-super-120b-a12b",
            "enable_thinking": False,
            "temperature": 0.2,
            "max_tokens": 4096,
            "reasoning_budget": 0,
            "top_p": 0.95,
        },
        "scorer": {
            "dev": "ai-nemotron-3-super-120b-a12b",
            "prod": "ai-nemotron-3-super-120b-a12b",
            "enable_thinking": True,
            "temperature": 0.1,
            "max_tokens": 2048,
            "reasoning_budget": 1024,
            "top_p": 0.90,
        },
    }

    def get_model_settings(self, role: str) -> Dict[str, Any]:
        role_config = self.models.get(role, {})
        model_name = role_config.get(self.mode, role_config.get("dev"))
        return {
            "model": model_name,
            "enable_thinking": role_config.get("enable_thinking", False),
            "temperature": role_config.get("temperature", 0.1),
            "max_tokens": role_config.get("max_tokens", 8192),
            "reasoning_budget": role_config.get("reasoning_budget", 4096),
            "top_p": role_config.get("top_p", 0.95)
        }


class APIConfig(BaseSettings):
    pubmed_base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    pubmed_email: str = os.getenv("PUBMED_EMAIL", "research@prometheus-ai.org")
    pubmed_tool: str = "PrometheusDeepResearch"
    pubmed_rate_limit: int = 3

    arxiv_base_url: str = "http://export.arxiv.org/api/query"
    arxiv_rate_limit: int = 3

    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"
    semantic_scholar_api_key: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    semantic_scholar_rate_limit: int = 100


class SystemConfig(BaseSettings):
    max_concurrent_subagents: int = 5
    crag_retry_limit: int = 3
    self_rag_threshold: float = 0.65
    memory_context_window: int = 200000
    chunk_size: int = 512
    chunk_overlap: int = 50
    vector_store_path: str = "./data/chroma_db"
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")


class PromptTemplates(BaseSettings):
    lead_agent_decomposition: str = """
    You are the Lead Agent in a multi-agent pharmaceutical research system.
    Your task is to decompose the following user query into mutually exclusive, 
    collectively exhaustive sub-investigations.
    Query: {query}
    Generate exactly {max_subagents} distinct research angles.
    Return your response in strict JSON format with a 'tasks' array containing objects 
    with 'task_id', 'description', 'search_keywords', and 'expected_output_focus'.
    """

    subagent_research: str = """
    You are a specialized Subagent investigating a specific angle of a broader pharmaceutical query.
    Your specific task: {task_description}
    Here is the retrieved literature: {retrieved_context}
    Analyze the literature, extract key claims, and format your findings into a structured report.
    You must cite the source paper IDs for every claim you make.
    """

    crag_evaluator: str = """
    You are the Self-RAG Scorer. Evaluate the following generated findings against the retrieved evidence.
    Findings: {findings}
    Evidence: {evidence}
    Assign a confidence score between 0.0 and 1.0 based on factual grounding.
    If the score is below {threshold}, suggest specific queries to retrieve better evidence.
    Return JSON with 'score', 'justification', and 'retry_queries'.
    """

    synthesis_report: str = """
    You are the Lead Agent synthesizing a final report from parallel subagent investigations.
    Original Query: {query}
    Subagent Findings: {findings_aggregate}
    Citation Graph Contradictions: {contradictions}
    Synthesize a comprehensive report. Do not blend contradictions; explicitly flag them.
    Use Markdown formatting and maintain rigorous academic tone.
    """


class GraphSchemaConfig(BaseSettings):
    paper_node_label: str = "Paper"
    claim_node_label: str = "Claim"
    author_node_label: str = "Author"
    cites_rel_type: str = "CITES"
    authors_rel_type: str = "AUTHORED_BY"
    supports_rel_type: str = "SUPPORTS_CLAIM"
    contradicts_rel_type: str = "CONTRADICTS_CLAIM"
    index_fields: List[str] = ["paper_id", "doi", "title"]


class UIConfig(BaseSettings):
    theme_color: str = "#2E86C1"
    layout: str = "wide"
    sidebar_state: str = "expanded"
    max_graph_nodes: int = 150
    refresh_interval_ms: int = 2000


class EvalConfig(BaseSettings):
    baseline_model: str = "gpt-4-turbo"
    metrics: List[str] = ["recall", "precision", "mrr", "ndcg"]
    test_queries_path: str = "./eval/eval_set.json"
    output_results_path: str = "./eval/results"
    significance_alpha: float = 0.05


directories = DirectoryConfig()
models = ModelConfig()
apis = APIConfig()
system = SystemConfig()
prompts = PromptTemplates()
graph_schema = GraphSchemaConfig()
ui_config = UIConfig()
eval_config = EvalConfig()


def initialize_system() -> None:
    directories.create_all()
    if models.nvidia_api_key:
        os.environ["NVIDIA_API_KEY"] = models.nvidia_api_key


def get_db_credentials() -> Dict[str, str]:
    return {
        "uri": system.neo4j_uri,
        "user": system.neo4j_user,
        "password": system.neo4j_password
    }


def get_subagent_capacity() -> int:
    return system.max_concurrent_subagents


def validate_environment() -> bool:
    if not os.getenv("NVIDIA_API_KEY") and not models.nvidia_api_key:
        return False
    return True


if __name__ == "__main__":
    initialize_system()