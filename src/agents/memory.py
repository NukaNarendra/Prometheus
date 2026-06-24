import os
import json
import asyncio
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, TypeVar, Type
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError

T = TypeVar('T', bound=BaseModel)


class MemoryAccessError(Exception):
    pass


class MemorySerializationError(Exception):
    pass


class WorkspaceInitializationError(Exception):
    pass


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class SubagentTask(BaseModel):
    task_id: str = Field(..., min_length=3)
    description: str = Field(..., min_length=20)
    search_keywords: List[str] = Field(..., min_items=1)
    expected_output_focus: str = Field(..., min_length=10)
    status: str = Field(default="pending")


class DecompositionPlan(BaseModel):
    original_query: str
    rationale: str = Field(..., min_length=20)
    tasks: List[SubagentTask] = Field(..., min_items=2, max_items=5)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class SubagentFindings(BaseModel):
    task_id: str
    query_used: str
    papers_analyzed: int
    key_claims: List[Dict[str, str]]
    contradictions_found: List[str]
    confidence_score: float
    raw_synthesis: str
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class FinalReport(BaseModel):
    run_id: str
    original_query: str
    executive_summary: str
    detailed_findings: List[Dict[str, Any]]
    citation_graph_summary: Dict[str, Any]
    total_token_usage: TokenUsage
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class FileLock:
    def __init__(self) -> None:
        self.locks: Dict[str, asyncio.Lock] = {}
        self.global_lock = asyncio.Lock()

    async def get_lock(self, file_path: str) -> asyncio.Lock:
        async with self.global_lock:
            if file_path not in self.locks:
                self.locks[file_path] = asyncio.Lock()
            return self.locks[file_path]


class AuditLogger:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "memory_audit.jsonl"
        self.lock = asyncio.Lock()

    async def log_event(self, event_type: str, run_id: str, details: Dict[str, Any]) -> None:
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "run_id": run_id,
            "details": details
        }
        async with self.lock:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")


class WorkspaceManager:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.memory_dir = self.base_dir / "data" / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.file_locker = FileLock()
        self.audit = AuditLogger(self.base_dir / "logs")

    def generate_run_id(self, query: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()[:8]
        return f"run_{timestamp}_{query_hash}"

    async def initialize_workspace(self, query: str) -> str:
        try:
            run_id = self.generate_run_id(query)
            run_dir = self.memory_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=False)

            subagents_dir = run_dir / "subagents"
            subagents_dir.mkdir(parents=True, exist_ok=False)

            await self.audit.log_event(
                event_type="workspace_created",
                run_id=run_id,
                details={"query": query, "path": str(run_dir)}
            )
            return run_id
        except FileExistsError:
            raise WorkspaceInitializationError(f"Workspace collision for run_id: {run_id}")
        except Exception as e:
            raise WorkspaceInitializationError(f"Failed to create workspace: {str(e)}")

    async def _write_json(self, file_path: Path, data: BaseModel) -> None:
        lock = await self.file_locker.get_lock(str(file_path))
        async with lock:
            try:
                json_str = data.model_dump_json(indent=2)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(json_str)
            except Exception as e:
                raise MemorySerializationError(f"Failed to write {file_path}: {str(e)}")

    async def _read_json(self, file_path: Path, schema: Type[T]) -> T:
        lock = await self.file_locker.get_lock(str(file_path))
        async with lock:
            if not file_path.exists():
                raise MemoryAccessError(f"File not found: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data_dict = json.load(f)
                return schema(**data_dict)
            except json.JSONDecodeError as e:
                raise MemorySerializationError(f"Corrupt JSON in {file_path}: {str(e)}")
            except ValidationError as e:
                raise MemorySerializationError(f"Schema mismatch in {file_path}: {str(e)}")

    async def save_plan(self, run_id: str, plan: DecompositionPlan) -> None:
        plan_file = self.memory_dir / run_id / "plan.json"
        await self._write_json(plan_file, plan)
        await self.audit.log_event("plan_saved", run_id, {"tasks_count": len(plan.tasks)})

    async def load_plan(self, run_id: str) -> DecompositionPlan:
        plan_file = self.memory_dir / run_id / "plan.json"
        return await self._read_json(plan_file, DecompositionPlan)

    async def update_task_status(self, run_id: str, task_id: str, new_status: str) -> None:
        plan = await self.load_plan(run_id)
        task_found = False
        for task in plan.tasks:
            if task.task_id == task_id:
                task.status = new_status
                task_found = True
                break

        if not task_found:
            raise MemoryAccessError(f"Task {task_id} not found in plan for {run_id}")

        await self.save_plan(run_id, plan)
        await self.audit.log_event("task_status_updated", run_id, {"task_id": task_id, "status": new_status})

    async def save_subagent_findings(self, run_id: str, findings: SubagentFindings) -> None:
        findings_file = self.memory_dir / run_id / "subagents" / f"{findings.task_id}.json"
        await self._write_json(findings_file, findings)
        await self.update_task_status(run_id, findings.task_id, "completed")
        await self.audit.log_event("findings_saved", run_id, {"task_id": findings.task_id})

    async def load_all_findings(self, run_id: str) -> List[SubagentFindings]:
        subagents_dir = self.memory_dir / run_id / "subagents"
        if not subagents_dir.exists():
            return []

        findings_list = []
        for file_path in subagents_dir.glob("*.json"):
            try:
                finding = await self._read_json(file_path, SubagentFindings)
                findings_list.append(finding)
            except Exception:
                continue

        return findings_list

    async def save_final_report(self, run_id: str, report: FinalReport) -> None:
        report_file = self.memory_dir / run_id / "final_report.json"
        await self._write_json(report_file, report)
        await self.audit.log_event("report_saved", run_id, {"summary_length": len(report.executive_summary)})

    def get_run_directory(self, run_id: str) -> Path:
        run_dir = self.memory_dir / run_id
        if not run_dir.exists():
            raise MemoryAccessError(f"Run directory not found: {run_id}")
        return run_dir