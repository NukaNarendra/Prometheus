import os
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import system
from src.agents.memory import WorkspaceManager, DecompositionPlan, SubagentFindings
from src.agents.subagent import Subagent
from src.agents.lead_agent import LeadAgent, ConsoleFormatter


class OrchestratorExecutionError(Exception):
    pass


class SubagentOrchestrator:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.workspace = WorkspaceManager(base_dir)
        self.max_concurrent = system.max_concurrent_subagents
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

    async def _execute_single_task(self, subagent: Subagent) -> Optional[SubagentFindings]:
        async with self.semaphore:
            task_id = subagent.task.task_id.upper()
            print(f"[{task_id}] Subagent initialized. Starting CRAG retrieval loop...")
            try:
                findings = await subagent.execute()
                score = findings.confidence_score
                papers = findings.papers_analyzed
                print(f"[{task_id}] Completed. Analyzed {papers} papers. Confidence Score: {score:.2f}")
                return findings
            except Exception as e:
                print(f"[{task_id}] FAILED: {str(e)}")
                return None

    async def execute_plan(self, run_id: str) -> List[SubagentFindings]:
        try:
            plan = await self.workspace.load_plan(run_id)
        except Exception as e:
            raise OrchestratorExecutionError(f"Failed to load plan for {run_id}: {str(e)}")

        ConsoleFormatter.print_header(f"Orchestrating {len(plan.tasks)} Parallel Subagents")

        subagents = [Subagent(task, run_id, self.base_dir) for task in plan.tasks]
        tasks = [self._execute_single_task(agent) for agent in subagents]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_findings = []
        for result in results:
            if isinstance(result, SubagentFindings):
                valid_findings.append(result)
            elif isinstance(result, Exception):
                print(f"Critical Subagent Failure: {str(result)}")

        ConsoleFormatter.print_header("Parallel Execution Complete")
        print(f"Successfully aggregated {len(valid_findings)} out of {len(plan.tasks)} task findings.")
        print(f"Findings are securely persisted in: data/memory/{run_id}/subagents/")
        return valid_findings


async def run_full_pipeline_demo() -> None:
    if not os.environ.get("NVIDIA_API_KEY"):
        print("ERROR: NVIDIA_API_KEY environment variable is missing.")
        return

    base_dir = Path(__file__).parent.parent.parent

    lead = LeadAgent(base_dir)
    orchestrator = SubagentOrchestrator(base_dir)

    test_query = "What are the latest mechanisms for KRAS G12C inhibition resistance and what are the competing drug candidates in clinical trials?"

    ConsoleFormatter.print_header("Phase 1: Lead Agent Decomposition")
    run_id = await lead.analyze_and_decompose(test_query)

    ConsoleFormatter.print_header("Phase 2: Orchestrator Parallel Execution")
    await orchestrator.execute_plan(run_id)


if __name__ == "__main__":
    asyncio.run(run_full_pipeline_demo())