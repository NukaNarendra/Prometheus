import os
import json
import re
import asyncio
from typing import List, Dict, Any, TypeVar, Type, Callable, Awaitable, Optional
from pathlib import Path
from pydantic import BaseModel, ValidationError
import sys

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import models, prompts
from src.agents.memory import DecompositionPlan, WorkspaceManager

T = TypeVar('T', bound=BaseModel)


class LLMNetworkError(Exception):
    pass


class LLMParsingError(Exception):
    pass


class ConsoleFormatter:
    RESET = "\033[0m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BOLD = "\033[1m"

    @classmethod
    def print_reasoning(cls, text: str) -> None:
        print(f"{cls.DIM}{cls.CYAN}{text}{cls.RESET}", end="", flush=True)

    @classmethod
    def print_content(cls, text: str) -> None:
        print(f"{cls.GREEN}{text}{cls.RESET}", end="", flush=True)

    @classmethod
    def print_header(cls, text: str) -> None:
        print(f"\n{cls.BOLD}{cls.YELLOW}=== {text} ==={cls.RESET}")


class RetryStrategy:
    def __init__(self, max_retries: int = 3, base_delay: float = 2.0, max_delay: float = 15.0) -> None:
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
        raise LLMNetworkError(f"API Failed after {self.max_retries} attempts. Last error: {str(last_exception)}")


class JSONCleaner:
    @staticmethod
    def fix_common_errors(json_str: str) -> str:
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = json_str.replace('\n', ' ')
        json_str = json_str.replace('\\"', '"')
        return json_str

    @staticmethod
    def clean_markdown(text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    @staticmethod
    def extract_json_object(text: str) -> str:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx:end_idx + 1]
        raise LLMParsingError("No valid JSON boundaries found in the LLM response")


class NVIDIAAgentClient:
    def __init__(self, api_key: str, role: str = "lead_agent") -> None:
        if not api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is absolutely required.")
        self.api_key = api_key
        self.role_config = models.get_model_settings(role)
        self.retry_strategy = RetryStrategy()

        chat_kwargs = {}
        if self.role_config.get("enable_thinking"):
            chat_kwargs["enable_thinking"] = True

        self.client = ChatNVIDIA(
            model=self.role_config["model"],
            api_key=self.api_key,
            temperature=self.role_config.get("temperature", 1.0),
            top_p=self.role_config.get("top_p", 0.95),
            max_tokens=self.role_config.get("max_tokens", 16384),
            reasoning_budget=self.role_config.get("reasoning_budget", 16384),
            chat_template_kwargs=chat_kwargs
        )

    async def generate_text_stream(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        async def _make_request() -> str:
            full_content = ""
            async for chunk in self.client.astream(messages):
                if chunk.additional_kwargs and "reasoning_content" in chunk.additional_kwargs:
                    reasoning_chunk = chunk.additional_kwargs["reasoning_content"]
                    if reasoning_chunk:
                        ConsoleFormatter.print_reasoning(reasoning_chunk)

                if chunk.content:
                    content_chunk = str(chunk.content)
                    full_content += content_chunk
            return full_content

        return await self.retry_strategy.execute(_make_request)

    async def generate_structured(self, system_prompt: str, user_prompt: str, schema: Type[T]) -> T:
        raw_text = await self.generate_text_stream(system_prompt, user_prompt)
        print("\n")

        cleaned_text = JSONCleaner.clean_markdown(raw_text)
        try:
            json_str = JSONCleaner.extract_json_object(cleaned_text)
            json_str = JSONCleaner.fix_common_errors(json_str)
            parsed_dict = json.loads(json_str)
            return schema(**parsed_dict)
        except json.JSONDecodeError as e:
            raise LLMParsingError(f"JSON Decoder failed: {str(e)} \n\nRaw LLM Output:\n{raw_text}")
        except ValidationError as e:
            raise LLMParsingError(f"Output did not match Pydantic schema: {str(e)}")


class PromptFormatter:
    def __init__(self, template: str) -> None:
        self.template = template

    def format(self, **kwargs: Any) -> str:
        formatted = self.template
        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            formatted = formatted.replace(placeholder, str(value))
        return formatted


class LeadAgent:
    def __init__(self, base_dir: Path) -> None:
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        self.client = NVIDIAAgentClient(api_key=api_key, role="lead_agent")
        self.decomposition_prompt = PromptFormatter(prompts.lead_agent_decomposition)
        self.workspace = WorkspaceManager(base_dir)
        self.max_subagents = 4

    def _create_system_prompt(self) -> str:
        return (
            "You are the Lead Agent of the Prometheus Autonomous Research Engine. "
            "Your domain is advanced pharmaceutical and biomedical literature synthesis. "
            "Your mandate is orchestration. You must not answer the user's query directly. "
            "Instead, you must analyze the query and decompose it into mutually exclusive, "
            "collectively exhaustive sub-investigations. Each sub-investigation will be "
            "assigned to an independent parallel subagent. "
            "You MUST output exactly valid JSON. Do not prepend with 'Here is the plan' "
            "or append any conversational text."
        )

    def _create_user_prompt(self, query: str) -> str:
        schema_instruction = (
            "\n\nCRITICAL: Your output must strictly adhere to the following JSON schema:\n"
            "{\n"
            "  \"original_query\": \"string\",\n"
            "  \"rationale\": \"string explaining exactly why this decomposition strategy covers all angles without overlap\",\n"
            "  \"tasks\": [\n"
            "    {\n"
            "      \"task_id\": \"string (must be snake_case, e.g., 'clinical_outcomes')\",\n"
            "      \"description\": \"string (detailed instructions for the subagent)\",\n"
            "      \"search_keywords\": [\"string\", \"string\"],\n"
            "      \"expected_output_focus\": \"string (what the subagent should write to disk)\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            f"Constraint: You must generate between 2 and {self.max_subagents} tasks."
        )
        base_prompt = self.decomposition_prompt.format(query=query, max_subagents=self.max_subagents)
        return base_prompt + schema_instruction

    async def analyze_and_decompose(self, query: str) -> str:
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(query)
        run_id = await self.workspace.initialize_workspace(query)

        try:
            plan = await self.client.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=DecompositionPlan
            )
            await self.workspace.save_plan(run_id, plan)
            return run_id
        except LLMParsingError as primary_error:
            ConsoleFormatter.print_header("JSON Parsing Failed, Retrying with Correction Prompt")
            recovery_prompt = (
                "Your previous response failed JSON validation or schema compliance. "
                "You are an automated system. Output ONLY the raw JSON object starting with { and ending with }. "
                f"Error details: {str(primary_error)}"
            )
            try:
                plan = await self.client.generate_structured(
                    system_prompt=recovery_prompt,
                    user_prompt=user_prompt,
                    schema=DecompositionPlan
                )
                await self.workspace.save_plan(run_id, plan)
                return run_id
            except Exception as e:
                raise RuntimeError(f"Lead Agent completely failed to decompose after recovery attempt: {str(e)}")


async def run_lead_agent_demo() -> None:
    if not os.environ.get("NVIDIA_API_KEY"):
        print("ERROR: NVIDIA_API_KEY environment variable is missing.")
        print("Please set it in your terminal: export NVIDIA_API_KEY='your_key'")
        return

    base_dir = Path(__file__).parent.parent.parent
    agent = LeadAgent(base_dir)
    test_query = "What are the latest mechanisms for KRAS G12C inhibition resistance and what are the competing drug candidates in clinical trials?"

    ConsoleFormatter.print_header("Initializing Lead Agent Orchestrator")
    print(f"Model Engine: {agent.client.role_config['model']}")
    print(f"Target Query: '{test_query}'\n")
    print("Agent is reasoning... (streaming internal thoughts natively via Langchain)\n")

    try:
        run_id = await agent.analyze_and_decompose(test_query)
        plan = await agent.workspace.load_plan(run_id)

        ConsoleFormatter.print_header(f"RUN ID GENERATED: {run_id}")
        print("RESEARCH STRATEGY RATIONALE:")
        print(plan.rationale)
        print("=" * 80)

        for idx, task in enumerate(plan.tasks):
            print(f"\n[Subagent Task {idx + 1}] -> {task.task_id.upper()}")
            print(f"Objective: {task.description}")
            print(f"Execution Keywords: {', '.join(task.search_keywords)}")
            print(f"Artifact Focus: {task.expected_output_focus}")
            print(f"Current State: {task.status}")
            print("-" * 80)

        print(f"\nDecomposition Plan successfully committed to disk at: data/memory/{run_id}/plan.json")

    except Exception as e:
        print(f"\nLead Agent Orchestration Failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(run_lead_agent_demo())