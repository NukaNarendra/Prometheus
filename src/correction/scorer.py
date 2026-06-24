import os
import json
import re
import asyncio
from typing import List, Dict, Any, TypeVar, Type, Callable, Awaitable
from pydantic import BaseModel, Field, ValidationError
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import models, system

T = TypeVar('T', bound=BaseModel)


class ScorerNetworkError(Exception):
    pass


class ScorerParsingError(Exception):
    pass


class EvidenceEvaluation(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    justification: str = Field(..., min_length=15)
    is_sufficient: bool
    retry_queries: List[str] = Field(default_factory=list)


class ScorerRetryStrategy:
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 10.0) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def execute(self, func: Callable[[], Awaitable[Any]]) -> Any:
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return await func()
            except Exception as e:
                last_exception = e
                delay = min(self.max_delay, self.base_delay * (2 ** attempt))
                await asyncio.sleep(delay)
        raise ScorerNetworkError(
            f"Scorer API Failed after {self.max_retries} attempts. Last error: {str(last_exception)}")


class JSONExtractor:
    @staticmethod
    def clean_and_extract(text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            extracted = text[start_idx:end_idx + 1]
            extracted = re.sub(r',\s*}', '}', extracted)
            extracted = re.sub(r',\s*]', ']', extracted)
            extracted = extracted.replace('\n', ' ')
            return extracted
        raise ScorerParsingError("Failed to extract valid JSON boundaries.")


class EvidenceScorerClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.role_config = models.get_model_settings("scorer")
        self.retry_strategy = ScorerRetryStrategy()

        self.client = ChatNVIDIA(
            model=self.role_config["model"],
            api_key=self.api_key,
            temperature=self.role_config.get("temperature", 0.1),
            top_p=self.role_config.get("top_p", 0.90),
            max_tokens=self.role_config.get("max_tokens", 4096)
        )

    async def generate_evaluation(self, system_prompt: str, user_prompt: str) -> EvidenceEvaluation:
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
            json_str = JSONExtractor.clean_and_extract(raw_text)
            parsed_dict = json.loads(json_str)
            return EvidenceEvaluation(**parsed_dict)
        except Exception as e:
            raise ScorerParsingError(f"Scorer parsing failed: {str(e)} on text: {raw_text}")


class SelfRAGScorer:
    def __init__(self) -> None:
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY environment variable required for Scorer.")
        self.client = EvidenceScorerClient(api_key=api_key)
        self.threshold = system.self_rag_threshold

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        formatted_docs = []
        for i, doc in enumerate(documents):
            title = doc.get("title", "Unknown Title")
            abstract = doc.get("abstract", "No Abstract")
            paper_id = doc.get("paper_id", f"doc_{i}")
            formatted_docs.append(f"Document ID: {paper_id}\nTitle: {title}\nAbstract: {abstract}\n")
        return "\n".join(formatted_docs)

    def _build_prompts(self, task_description: str, documents: List[Dict[str, Any]]) -> tuple[str, str]:
        system_prompt = (
            "You are an objective academic evaluator. Your job is to assess whether "
            "the provided scientific abstracts contain sufficient, direct evidence to "
            "satisfy the given research task.\n"
            "You must output valid JSON matching this schema:\n"
            "{\n"
            "  \"score\": float (0.0 to 1.0),\n"
            "  \"justification\": string (explain why the evidence is strong or weak),\n"
            "  \"is_sufficient\": boolean (true if score >= 0.65, else false),\n"
            "  \"retry_queries\": [string, string] (provide 2-3 new, highly specific search terms ONLY if is_sufficient is false)\n"
            "}"
        )

        context_str = self._format_context(documents)
        user_prompt = (
            f"Task Objective:\n{task_description}\n\n"
            f"Retrieved Evidence:\n{context_str}\n\n"
            "Evaluate the relevance and density of this evidence against the objective."
        )
        return system_prompt, user_prompt

    async def evaluate_evidence(self, task_description: str, documents: List[Dict[str, Any]]) -> EvidenceEvaluation:
        if not documents:
            return EvidenceEvaluation(
                score=0.0,
                justification="No documents were retrieved.",
                is_sufficient=False,
                retry_queries=[task_description[:50]]
            )

        system_prompt, user_prompt = self._build_prompts(task_description, documents)

        try:
            evaluation = await self.client.generate_evaluation(system_prompt, user_prompt)
            if evaluation.score < self.threshold:
                evaluation.is_sufficient = False
            else:
                evaluation.is_sufficient = True
                evaluation.retry_queries = []
            return evaluation
        except ScorerParsingError:
            fallback = EvidenceEvaluation(
                score=0.5,
                justification="Fallback triggered due to parsing failure. Assuming neutral evidence.",
                is_sufficient=False,
                retry_queries=[]
            )
            return fallback