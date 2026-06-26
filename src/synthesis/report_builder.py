import os
import json
import asyncio
from typing import List, Dict, Any, TypeVar, Type, Callable, Awaitable, Optional
from pathlib import Path
from pydantic import BaseModel
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage
import sys

class ConsoleFormatter:
    RESET = "\033[0m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"

    @classmethod
    def print_reasoning(cls, text: str) -> None:
        print(f"{cls.DIM}{cls.CYAN}{text}{cls.RESET}", end="", flush=True)

    @classmethod
    def print_content(cls, text: str) -> None:
        print(f"{cls.GREEN}{text}{cls.RESET}", end="", flush=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import models
from src.agents.memory import WorkspaceManager, FinalReport, TokenUsage
from src.synthesis.citation_graph import CitationGraphOrchestrator
from src.synthesis.contradiction import ContradictionEngine, ContradictionAnalysis

class ReportNetworkError(Exception):
    pass

class ReportRetryStrategy:
    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries

    async def execute(self, func: Callable[[], Awaitable[Any]]) -> Any:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await func()
            except Exception as e:
                last_error = e
                print(f"        [!] Synthesis Attempt {attempt + 1} Failed: {str(e)}")
                await asyncio.sleep(3.0 * (attempt + 1))
        raise ReportNetworkError(f"Report synthesis API Failed repeatedly. Last Error: {str(last_error)}")

class ReportClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.role_config = models.get_model_settings("lead_agent")
        self.retry_strategy = ReportRetryStrategy()

        self.client = ChatNVIDIA(
            model="nvidia/nemotron-3-ultra-550b-a55b",
            api_key=self.api_key,
            temperature=0.2,
            top_p=0.95,
            max_tokens=16384,
            timeout=600,
            model_kwargs={
                "reasoning_budget": 4096,
                "chat_template_kwargs": {"enable_thinking": True}
            }
        )

    async def generate_markdown(self, system_prompt: str, user_prompt: str, stream_callback: Optional[Callable[[str], None]] = None) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        async def _make_request() -> str:
            full_content = ""
            print("\nAgent is synthesizing the final report... (Streaming)\n")
            async for chunk in self.client.astream(messages):
                if chunk.additional_kwargs and "reasoning_content" in chunk.additional_kwargs:
                    reasoning_chunk = chunk.additional_kwargs["reasoning_content"]
                    if reasoning_chunk:
                        ConsoleFormatter.print_reasoning(reasoning_chunk)

                if chunk.content:
                    content_chunk = str(chunk.content)
                    ConsoleFormatter.print_content(content_chunk)
                    if stream_callback:
                        stream_callback(content_chunk)
                    full_content += content_chunk

            print("\n")  # Add newline after streaming completes

            if not full_content.strip():
                raise ReportNetworkError("Model returned an empty string. Forcing retry...")

            return full_content

        return await self.retry_strategy.execute(_make_request)

class DataAggregator:
    @staticmethod
    def format_subagent_data(findings_list: List[Any]) -> str:
        output = []
        for finding in findings_list:
            output.append(f"### Subagent Context: {finding.task_id.upper()}")
            output.append(f"Confidence Score: {finding.confidence_score:.2f}")
            output.append(f"Synthesis: {finding.raw_synthesis}")
            output.append("Key Evidence:")
            for claim in finding.key_claims:
                short_id = claim.get('paper_id', 'Unknown')[:8]
                output.append(f"- {claim.get('claim')} [Source: {short_id}]")
            output.append("\n")
        return "\n".join(output)

    @staticmethod
    def format_contradiction_data(analysis: ContradictionAnalysis) -> str:
        if not analysis.conflict_pairs:
            return "No severe scientific contradictions detected in the literature subset."

        output = ["### Detected Scientific Conflicts"]
        for idx, conflict in enumerate(analysis.conflict_pairs):
            short_a = conflict.paper_id_a[:8]
            short_b = conflict.paper_id_b[:8]
            output.append(f"Conflict {idx + 1} (Severity: {conflict.severity.upper()}):")
            output.append(f"- Paper {short_a} claims: {conflict.claim_a}")
            output.append(f"- Paper {short_b} claims: {conflict.claim_b}")
            output.append(f"- Analysis: {conflict.conflict_description}\n")
        return "\n".join(output)

class ReportBuilderOrchestrator:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.workspace = WorkspaceManager(base_dir)
        self.graph = CitationGraphOrchestrator(base_dir)
        self.contradiction = ContradictionEngine(base_dir)

        api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not api_key:
            raise ValueError("Critical Error: NVIDIA_API_KEY is missing from environment variables.")

        self.client = ReportClient(api_key)

    def _build_prompts(self, original_query: str, aggregated_data: str, conflict_data: str, graph_data: str) -> tuple[str, str]:
        system_prompt = (
            "You are the Lead Agent of the Prometheus Autonomous Research Engine. "
            "Your final task is to write a highly professional, academic-grade literature review "
            "based on the findings gathered by your parallel subagents.\n"
            "Rules:\n"
            "1. Use standard Markdown formatting with clear H1, H2, H3 headers.\n"
            "2. You MUST cite claims inline using the bracket format [Source: 12345678] matching the short 8-character source IDs provided.\n"
            "3. You MUST dedicate a specific section to 'Contradictions and Conflicting Evidence' and explicitly state where papers disagree based on the conflict data.\n"
            "4. Do not invent or hallucinate citations. Only use the ones provided in the text.\n"
            "5. Begin with a powerful Executive Summary."
        )

        user_prompt = (
            f"Original User Query: {original_query}\n\n"
            f"--- SUBAGENT AGGREGATED FINDINGS ---\n{aggregated_data}\n\n"
            f"--- CONTRADICTION ENGINE ANALYSIS ---\n{conflict_data}\n\n"
            f"--- CITATION GRAPH TOPOLOGY ---\n{graph_data}\n\n"
            "Please synthesize the final comprehensive report now."
        )
        return system_prompt, user_prompt

    async def execute_synthesis(self, run_id: str, stream_callback: Optional[Callable[[str], None]] = None) -> Path:
        print(f"[{run_id}] Loading workspace artifacts...")
        plan = await self.workspace.load_plan(run_id)
        findings = await self.workspace.load_all_findings(run_id)

        print(f"[{run_id}] Ingesting findings into Citation Graph...")
        graph_active = await self.graph.check_health()
        graph_summary_data = "{}"
        if graph_active:
            await self.graph.ingest_findings(run_id)
            graph_summary = await self.graph.generate_graph_summary()
            graph_summary_data = json.dumps(graph_summary, indent=2)
        else:
            print(f"[{run_id}] WARNING: Neo4j is offline. Bypassing graph ingestion.")

        print(f"[{run_id}] Running LLM Contradiction Engine...")
        contradiction_analysis = await self.contradiction.execute_run_analysis(run_id)

        aggregated_str = DataAggregator.format_subagent_data(findings)
        conflict_str = DataAggregator.format_contradiction_data(contradiction_analysis)

        system_prompt, user_prompt = self._build_prompts(
            plan.original_query,
            aggregated_str,
            conflict_str,
            graph_summary_data
        )

        print(f"[{run_id}] Streaming Final Executive Report Generation...")
        markdown_content = await self.client.generate_markdown(system_prompt, user_prompt, stream_callback)

        report_path = self.base_dir / "data" / "memory" / run_id / "final_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        final_report_obj = FinalReport(
            run_id=run_id,
            original_query=plan.original_query,
            executive_summary="Generated in markdown artifact.",
            detailed_findings=[f.model_dump() for f in findings],
            citation_graph_summary={"status": "active" if graph_active else "bypassed"},
            total_token_usage=TokenUsage()
        )
        await self.workspace.save_final_report(run_id, final_report_obj)

        print(f"\n[SUCCESS] Final Report saved to: {report_path}")
        return report_path

async def run_report_demo() -> None:
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        base_dir = Path(__file__).parent.parent.parent
        memory_dir = base_dir / "data" / "memory"
        runs = [d.name for d in memory_dir.iterdir() if d.is_dir()]
        if not runs:
            print("No previous runs found. Please run orchestrator.py first.")
            return
        run_id = sorted(runs)[-1]

    print(f"Targeting Run ID: {run_id}")
    builder = ReportBuilderOrchestrator(Path(__file__).parent.parent.parent)
    await builder.execute_synthesis(run_id)

if __name__ == "__main__":
    asyncio.run(run_report_demo())