import os
import json
import asyncio
from typing import List, Dict, Any, TypeVar, Type, Callable, Awaitable
from pathlib import Path
from pydantic import BaseModel, ValidationError
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import models, system
from src.agents.memory import WorkspaceManager, SubagentTask, SubagentFindings, TokenUsage
from src.retrieval.hybrid import HybridSearcher
from src.correction.scorer import SelfRAGScorer, EvidenceEvaluation


class SubagentNetworkError(Exception):
    pass


class SubagentSynthesisError(Exception):
    pass


class SynthesisOutput(BaseModel):
    key_claims: List[Dict[str, str]]
    contradictions_found: List[str]
    raw_synthesis: str


class SubagentRetryStrategy:
    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries

    async def execute(self, func: Callable[[], Awaitable[Any]]) -> Any:
        for attempt in range(self.max_retries):
            try:
                return await func()
            except Exception as e:
                await asyncio.sleep(2.0 * (attempt + 1))
        raise SubagentNetworkError("Synthesis API Failed repeatedly.")


class SynthesisClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.role_config = models.get_model_settings("subagent")
        self.retry_strategy = SubagentRetryStrategy()

        self.client = ChatNVIDIA(
            model="nvidia/nemotron-3-ultra-550b-a55b",
            api_key=self.api_key,
            temperature=1,
            top_p=0.95,
            max_tokens=16384,
            timeout=600,
            model_kwargs={
                "reasoning_budget": 4096,
                "chat_template_kwargs": {"enable_thinking": True}
            }
        )

    def _extract_json(self, text: str) -> str:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            return text[start_idx:end_idx + 1].replace('\n', ' ')
        raise SubagentSynthesisError("No JSON found in synthesis.")

    async def generate_synthesis(self, system_prompt: str, user_prompt: str) -> SynthesisOutput:
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

        raw_text = await self.retry_strategy.execute(_make_request)

        try:
            json_str = self._extract_json(raw_text)
            parsed_dict = json.loads(json_str)
            return SynthesisOutput(**parsed_dict)
        except Exception as e:
            raise SubagentSynthesisError(f"Synthesis parsing failed: {str(e)}")


class Subagent:
    def __init__(self, task: SubagentTask, run_id: str, base_dir: Path) -> None:
        self.task = task
        self.run_id = run_id
        self.base_dir = base_dir
        self.workspace = WorkspaceManager(base_dir)
        self.searcher = HybridSearcher(base_dir / "data" / "corpus")
        self.scorer = SelfRAGScorer()

        api_key = os.environ.get("NVIDIA_API_KEY", "")
        self.synthesis_client = SynthesisClient(api_key=api_key)
        self.max_crag_retries = system.crag_retry_limit

    def _format_evidence_for_synthesis(self, documents: List[Dict[str, Any]]) -> str:
        formatted = []
        for doc in documents:
            paper_id = doc.get("paper_id", "Unknown")
            title = doc.get("title", "No Title")
            abstract = doc.get("abstract", "")
            formatted.append(f"--- PAPER ID: {paper_id} ---\nTITLE: {title}\nABSTRACT: {abstract}\n")
        return "\n".join(formatted)

    def _build_synthesis_prompts(self, evidence_str: str) -> tuple[str, str]:
        system_prompt = (
            "You are a pharmaceutical research subagent. Your job is to synthesize "
            "findings from the provided scientific abstracts. You must extract key claims "
            "and explicitly flag any contradictions between papers.\n"
            "Output valid JSON ONLY:\n"
            "{\n"
            "  \"key_claims\": [{\"claim\": \"string\", \"paper_id\": \"string\"}],\n"
            "  \"contradictions_found\": [\"string detailing conflict between paper X and Y\"],\n"
            "  \"raw_synthesis\": \"string (a cohesive 2-paragraph summary)\"\n"
            "}"
        )
        user_prompt = (
            f"Task Objective: {self.task.description}\n"
            f"Focus Area: {self.task.expected_output_focus}\n\n"
            f"Retrieved Evidence:\n{evidence_str}"
        )
        return system_prompt, user_prompt

    async def execute(self) -> SubagentFindings:
        await self.workspace.update_task_status(self.run_id, self.task.task_id, "in_progress")
        if not self.searcher.is_initialized:
            self.searcher.initialize()

        current_queries = self.task.search_keywords.copy()
        best_documents: List[Dict[str, Any]] = []
        best_score = 0.0
        final_query_used = ""

        for attempt in range(self.max_crag_retries):
            query_string = " ".join(current_queries)
            documents = self.searcher.search(query_string)

            evaluation = await self.scorer.evaluate_evidence(self.task.description, documents)

            if evaluation.score > best_score:
                best_score = evaluation.score
                best_documents = documents
                final_query_used = query_string

            if evaluation.is_sufficient:
                break

            if evaluation.retry_queries:
                current_queries = evaluation.retry_queries
            else:
                break

        evidence_str = self._format_evidence_for_synthesis(best_documents)
        system_prompt, user_prompt = self._build_synthesis_prompts(evidence_str)

        try:
            synthesis = await self.synthesis_client.generate_synthesis(system_prompt, user_prompt)
            findings = SubagentFindings(
                task_id=self.task.task_id,
                query_used=final_query_used,
                papers_analyzed=len(best_documents),
                key_claims=synthesis.key_claims,
                contradictions_found=synthesis.contradictions_found,
                confidence_score=best_score,
                raw_synthesis=synthesis.raw_synthesis,
                token_usage=TokenUsage()
            )
            await self.workspace.save_subagent_findings(self.run_id, findings)
            return findings

        except Exception as e:
            await self.workspace.update_task_status(self.run_id, self.task.task_id, "failed")
            raise SubagentSynthesisError(f"Task {self.task.task_id} failed: {str(e)}")