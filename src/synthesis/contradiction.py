import os
import json
import asyncio
from typing import List, Dict, Any, TypeVar, Type, Callable, Awaitable, Set, Tuple
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import models
from src.agents.memory import WorkspaceManager

T = TypeVar('T', bound=BaseModel)


class ContradictionNetworkError(Exception):
    pass


class ContradictionParsingError(Exception):
    pass


class ConflictPair(BaseModel):
    claim_a: str
    claim_b: str
    paper_id_a: str
    paper_id_b: str
    conflict_description: str
    severity: str = Field(pattern="^(high|medium|low)$")


class ContradictionAnalysis(BaseModel):
    total_claims_analyzed: int
    conflicts_detected: int
    conflict_pairs: List[ConflictPair]
    resolution_suggestions: List[str]


class SemanticFilter:
    @staticmethod
    def get_jaccard_similarity(str1: str, str2: str) -> float:
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        if not union:
            return 0.0
        return len(intersection) / len(union)

    @classmethod
    def filter_candidates(cls, claims: List[Dict[str, str]], threshold: float = 0.15) -> List[
        Tuple[Dict[str, str], Dict[str, str]]]:
        candidates = []
        n = len(claims)
        for i in range(n):
            for j in range(i + 1, n):
                claim_i = claims[i].get("claim", "")
                claim_j = claims[j].get("claim", "")
                if not claim_i or not claim_j:
                    continue
                score = cls.get_jaccard_similarity(claim_i, claim_j)
                if score >= threshold:
                    candidates.append((claims[i], claims[j]))
        return candidates


class ContradictionRetryStrategy:
    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries

    async def execute(self, func: Callable[[], Awaitable[Any]]) -> Any:
        for attempt in range(self.max_retries):
            try:
                return await func()
            except Exception as e:
                await asyncio.sleep(2.0 * (attempt + 1))
        raise ContradictionNetworkError("Contradiction API Failed repeatedly.")


class ContradictionClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.role_config = models.get_model_settings("lead_agent")
        self.retry_strategy = ContradictionRetryStrategy()

        self.client = ChatNVIDIA(
            model=self.role_config["model"],
            api_key=self.api_key,
            temperature=self.role_config.get("temperature", 0.2),
            top_p=self.role_config.get("top_p", 0.95),
            max_tokens=self.role_config.get("max_tokens", 4096)
        )

    def _extract_json(self, text: str) -> str:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            return text[start_idx:end_idx + 1].replace('\n', ' ')
        raise ContradictionParsingError("No JSON found in response.")

    async def analyze_pairs(self, system_prompt: str, user_prompt: str) -> ContradictionAnalysis:
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
            return ContradictionAnalysis(**parsed_dict)
        except Exception as e:
            raise ContradictionParsingError(f"Contradiction parsing failed: {str(e)}")


class ContradictionEngine:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.workspace = WorkspaceManager(base_dir)
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        self.client = ContradictionClient(api_key)

    def _build_prompts(self, candidate_pairs: List[Tuple[Dict[str, str], Dict[str, str]]]) -> tuple[str, str]:
        system_prompt = (
            "You are a rigorous scientific contradiction detector. "
            "You will be given pairs of scientific claims. Your task is to identify pairs "
            "that make mutually exclusive or highly conflicting statements.\n"
            "Output valid JSON ONLY matching this schema:\n"
            "{\n"
            "  \"total_claims_analyzed\": integer,\n"
            "  \"conflicts_detected\": integer,\n"
            "  \"conflict_pairs\": [\n"
            "    {\n"
            "      \"claim_a\": \"string\",\n"
            "      \"claim_b\": \"string\",\n"
            "      \"paper_id_a\": \"string\",\n"
            "      \"paper_id_b\": \"string\",\n"
            "      \"conflict_description\": \"string\",\n"
            "      \"severity\": \"high\" | \"medium\" | \"low\"\n"
            "    }\n"
            "  ],\n"
            "  \"resolution_suggestions\": [\"string\"]\n"
            "}"
        )

        formatted_pairs = []
        for idx, (c1, c2) in enumerate(candidate_pairs):
            pair_str = (
                f"--- PAIR {idx + 1} ---\n"
                f"Claim A (Paper {c1.get('paper_id', 'Unknown')}): {c1.get('claim', '')}\n"
                f"Claim B (Paper {c2.get('paper_id', 'Unknown')}): {c2.get('claim', '')}\n"
            )
            formatted_pairs.append(pair_str)

        user_prompt = (
                "Analyze the following candidate pairs for logical or empirical contradictions. "
                "Ignore minor phrasing differences. Only flag true scientific conflicts.\n\n" +
                "\n".join(formatted_pairs)
        )
        return system_prompt, user_prompt

    async def execute_run_analysis(self, run_id: str) -> ContradictionAnalysis:
        findings_list = await self.workspace.load_all_findings(run_id)

        all_claims = []
        for finding in findings_list:
            for claim in finding.key_claims:
                if claim.get("claim") and claim.get("paper_id"):
                    all_claims.append(claim)

        if len(all_claims) < 2:
            return ContradictionAnalysis(
                total_claims_analyzed=len(all_claims),
                conflicts_detected=0,
                conflict_pairs=[],
                resolution_suggestions=["Insufficient claims to detect contradictions."]
            )

        candidate_pairs = SemanticFilter.filter_candidates(all_claims, threshold=0.10)

        if not candidate_pairs:
            return ContradictionAnalysis(
                total_claims_analyzed=len(all_claims),
                conflicts_detected=0,
                conflict_pairs=[],
                resolution_suggestions=["No semantically overlapping claims found to contrast."]
            )

        system_prompt, user_prompt = self._build_prompts(candidate_pairs[:20])

        try:
            analysis = await self.client.analyze_pairs(system_prompt, user_prompt)
            analysis.total_claims_analyzed = len(all_claims)
            return analysis
        except Exception:
            return ContradictionAnalysis(
                total_claims_analyzed=len(all_claims),
                conflicts_detected=0,
                conflict_pairs=[],
                resolution_suggestions=["Analysis engine failed to process candidate pairs."]
            )