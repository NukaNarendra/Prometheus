import os
import json
import asyncio
from typing import List, Dict, Any, TypeVar, Type, Callable, Awaitable
from pathlib import Path
from datetime import datetime
import aiohttp
from pydantic import BaseModel, Field, ValidationError
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import models, directories
from src.retrieval.hybrid import HybridSearcher
from src.agents.memory import WorkspaceManager

T = TypeVar('T', bound=BaseModel)


class EvaluationNetworkError(Exception):
    pass


class EvaluationParsingError(Exception):
    pass


class MetricScore(BaseModel):
    score: int = Field(..., ge=1, le=10)
    justification: str = Field(..., min_length=20)


class EvaluationResult(BaseModel):
    evidence_coverage: MetricScore
    contradiction_awareness: MetricScore
    citation_accuracy: MetricScore
    overall_quality: MetricScore
    winner: str = Field(pattern="^(single_agent|multi_agent|tie)$")


class BaselineReport(BaseModel):
    query: str
    report_content: str
    papers_retrieved: int
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class EvaluationRetryStrategy:
    def __init__(self, max_retries: int = 3, base_delay: float = 2.0) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def execute(self, func: Callable[[], Awaitable[Any]]) -> Any:
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return await func()
            except Exception as e:
                last_exception = e
                await asyncio.sleep(self.base_delay * (2 ** attempt))
        raise EvaluationNetworkError(f"API Failed after {self.max_retries} attempts: {str(last_exception)}")


class LLMClient:
    def __init__(self, role: str) -> None:
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY is required.")

        self.role_config = models.get_model_settings(role)
        self.retry_strategy = EvaluationRetryStrategy()

        self.client = ChatNVIDIA(
            model="nvidia/nemotron-3-ultra-550b-a55b",
            api_key=api_key,
            temperature=1,
            top_p=0.95,
            max_tokens=16384,
            reasoning_budget=16384,
            chat_template_kwargs={"enable_thinking": True}
        )

    def _extract_json(self, text: str) -> str:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            return text[start_idx:end_idx + 1].replace('\n', ' ')
        raise EvaluationParsingError("No JSON found in response.")

    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        async def _make_request() -> str:
            full_content = ""
            async for chunk in self.client.astream(messages):
                if chunk.content:
                    full_content += str(chunk.content)
            return full_content

        return await self.retry_strategy.execute(_make_request)

    async def generate_structured(self, system_prompt: str, user_prompt: str, schema: Type[T]) -> T:
        raw_text = await self.generate_text(system_prompt, user_prompt)
        try:
            json_str = self._extract_json(raw_text)
            parsed_dict = json.loads(json_str)
            return schema(**parsed_dict)
        except Exception as e:
            raise EvaluationParsingError(f"Parsing failed: {str(e)}")


class SingleAgentBaseline:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.searcher = HybridSearcher(base_dir / "data" / "corpus")
        self.client = LLMClient(role="lead_agent")

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        formatted = []
        for doc in documents:
            paper_id = doc.get("paper_id", "Unknown")
            title = doc.get("title", "No Title")
            abstract = doc.get("abstract", "")
            formatted.append(f"[ID: {paper_id}]\nTITLE: {title}\nABSTRACT: {abstract}\n")
        return "\n".join(formatted)

    async def generate_baseline_report(self, query: str) -> BaselineReport:
        if not self.searcher.is_initialized:
            self.searcher.initialize()

        print("    [Baseline] Retrieving monolithic document set...")
        documents = self.searcher.search(query)
        context_str = self._format_context(documents[:20])

        system_prompt = (
            "You are a monolithic RAG agent. You must answer the user's research query "
            "based solely on the provided abstracts. Write a comprehensive markdown report. "
            "Cite your claims using the [ID: ...] format provided in the text."
        )

        user_prompt = (
            f"Query: {query}\n\n"
            f"Retrieved Context:\n{context_str}"
        )

        print("    [Baseline] Generating single-agent monolithic report...")
        report_content = await self.client.generate_text(system_prompt, user_prompt)

        return BaselineReport(
            query=query,
            report_content=report_content,
            papers_retrieved=len(documents[:20])
        )


class SystemJudge:
    def __init__(self) -> None:
        self.client = LLMClient(role="lead_agent")

    def _build_prompts(self, query: str, single_report: str, multi_report: str) -> tuple[str, str]:
        system_prompt = (
            "You are an impartial academic judge. Evaluate two AI-generated research reports "
            "based on the original user query. Score them strictly from 1 to 10 on three metrics: "
            "Evidence Coverage, Contradiction Awareness, and Citation Accuracy.\n"
            "Output strictly valid JSON matching this schema:\n"
            "{\n"
            "  \"evidence_coverage\": {\"score\": int, \"justification\": \"string\"},\n"
            "  \"contradiction_awareness\": {\"score\": int, \"justification\": \"string\"},\n"
            "  \"citation_accuracy\": {\"score\": int, \"justification\": \"string\"},\n"
            "  \"overall_quality\": {\"score\": int, \"justification\": \"string\"},\n"
            "  \"winner\": \"single_agent\" or \"multi_agent\" or \"tie\"\n"
            "}"
        )

        user_prompt = (
            f"Original Query: {query}\n\n"
            f"--- REPORT A (Single-Agent RAG Baseline) ---\n{single_report}\n\n"
            f"--- REPORT B (Prometheus Multi-Agent RAG) ---\n{multi_report}\n\n"
            "Evaluate strictly based on depth of mechanism explanation, ability to isolate "
            "clinical trial data, and robust handling of biological contradictions."
        )
        return system_prompt, user_prompt

    async def evaluate_systems(self, query: str, single_report: str, multi_report: str) -> EvaluationResult:
        system_prompt, user_prompt = self._build_prompts(query, single_report, multi_report)
        return await self.client.generate_structured(system_prompt, user_prompt, EvaluationResult)


class EvaluationOrchestrator:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.workspace = WorkspaceManager(base_dir)
        self.baseline = SingleAgentBaseline(base_dir)
        self.judge = SystemJudge()
        self.results_dir = self.base_dir / "eval" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def _get_latest_multi_agent_run(self) -> tuple[str, str]:
        memory_dir = self.base_dir / "data" / "memory"
        runs = [d.name for d in memory_dir.iterdir() if d.is_dir()]
        if not runs:
            raise FileNotFoundError("No Prometheus multi-agent runs found.")

        latest_run = sorted(runs)[-1]
        report_path = memory_dir / latest_run / "final_report.md"
        plan_path = memory_dir / latest_run / "plan.json"

        with open(report_path, "r", encoding="utf-8") as f:
            report_content = f.read()

        with open(plan_path, "r", encoding="utf-8") as f:
            plan_data = json.load(f)
            query = plan_data.get("original_query", "")

        return query, report_content

    async def run_evaluation(self) -> None:
        print("\n=== Prometheus vs. Single-Agent Baseline Evaluation ===")

        print("\n1. Extracting latest Multi-Agent Artifacts...")
        query, multi_agent_report = await self._get_latest_multi_agent_run()
        print(f"   Target Query: {query}")

        print("\n2. Executing Single-Agent RAG Baseline...")
        baseline_result = await self.baseline.generate_baseline_report(query)

        baseline_path = self.results_dir / "baseline_report.md"
        with open(baseline_path, "w", encoding="utf-8") as f:
            f.write(baseline_result.report_content)

        print("\n3. Running LLM-as-a-Judge Evaluation...")
        eval_result = await self.judge.evaluate_systems(
            query=query,
            single_report=baseline_result.report_content,
            multi_report=multi_agent_report
        )

        result_path = self.results_dir / "evaluation_metrics.json"
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(eval_result.model_dump_json(indent=2))

        print("\n=== Final Evaluation Results ===")
        print(f"Evidence Coverage:      {eval_result.evidence_coverage.score}/10")
        print(f"Contradiction Handling: {eval_result.contradiction_awareness.score}/10")
        print(f"Citation Accuracy:      {eval_result.citation_accuracy.score}/10")
        print(f"Overall Quality:        {eval_result.overall_quality.score}/10")
        print(f"\nWINNER: {eval_result.winner.upper()}")
        print(f"\nJustification: {eval_result.overall_quality.justification}")
        print(f"\nDetailed metrics saved to: {result_path}")


async def main() -> None:
    base_dir = Path(__file__).parent.parent
    orchestrator = EvaluationOrchestrator(base_dir)
    await orchestrator.run_evaluation()


if __name__ == "__main__":
    asyncio.run(main())