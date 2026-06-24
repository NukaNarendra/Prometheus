import streamlit as st
import json
import os
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()


class DataLoader:
    def __init__(self, base_dir: Path):
        self.memory_dir = base_dir / "data" / "memory"

    def load_json_artifact(self, run_id: str, filename: str) -> Dict[str, Any]:
        file_path = self.memory_dir / run_id / filename
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def load_subagent_findings(self, run_id: str) -> List[Dict[str, Any]]:
        subagents_dir = self.memory_dir / run_id / "subagents"
        findings = []
        if subagents_dir.exists():
            for file_path in subagents_dir.glob("*.json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    findings.append(json.load(f))
        return findings


class UIComponents:
    @staticmethod
    def render_header(title: str, subtitle: str) -> None:
        st.markdown(f"<h1 style='text-align: center;'>{title}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: gray; font-size: 1.2rem;'>{subtitle}</p>",
                    unsafe_allow_html=True)
        st.markdown("---")

    @staticmethod
    def render_metric_card(label: str, value: str) -> None:
        st.markdown(
            f"""
            <div style="padding: 1rem; border-radius: 0.5rem; background-color: #f0f2f6; border: 1px solid #e0e2e6;">
                <p style="margin: 0; color: #555; font-size: 0.9rem; font-weight: 600;">{label}</p>
                <p style="margin: 0; color: #111; font-size: 1.5rem; font-weight: 700;">{value}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


class PrometheusDashboard:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.loader = DataLoader(self.base_dir)

        # Import backend modules here to avoid event loop conflicts and pathing issues
        sys.path.append(os.path.abspath(str(self.base_dir)))
        from src.agents.lead_agent import LeadAgent
        from src.agents.orchestrator import SubagentOrchestrator
        from src.synthesis.report_builder import ReportBuilderOrchestrator

        self.LeadAgent = LeadAgent
        self.SubagentOrchestrator = SubagentOrchestrator
        self.ReportBuilderOrchestrator = ReportBuilderOrchestrator

    def setup_page(self) -> None:
        st.set_page_config(
            page_title="Prometheus | Deep Research",
            layout="wide",
            initial_sidebar_state="collapsed"
        )

    def render_orchestration_view(self, plan: Dict[str, Any]) -> None:
        st.markdown(f"**Strategic Rationale:** {plan.get('rationale', 'N/A')}")
        tasks = plan.get("tasks", [])
        if tasks:
            for idx, task in enumerate(tasks):
                st.markdown(f"- **{task.get('task_id', '').upper()}**: {task.get('description', '')}")

    def render_subagent_view(self, findings: List[Dict[str, Any]]) -> None:
        if not findings:
            return
        cols = st.columns(len(findings))
        for idx, finding in enumerate(findings):
            with cols[idx]:
                task_id = finding.get("task_id", "").upper()
                st.markdown(f"#### {task_id}")
                UIComponents.render_metric_card("Evidence Score", f"{finding.get('confidence_score', 0.0):.2f}")
                st.markdown(f"*{finding.get('papers_analyzed', 0)} papers analyzed.*")

    def execute_live_research(self, query: str) -> None:
        plan_placeholder = st.empty()
        subagent_placeholder = st.empty()
        report_placeholder = st.empty()

        async def _run_pipeline():
            try:
                # Phase 1: Planning
                with st.spinner("🧠 Phase 1: Lead Agent decomposing query and planning research strategy..."):
                    lead = self.LeadAgent(self.base_dir)
                    run_id = await lead.analyze_and_decompose(query)
                    plan_data = self.loader.load_json_artifact(run_id, "plan.json")

                with plan_placeholder.expander("📂 Phase 1 Complete: View Orchestration Plan", expanded=False):
                    self.render_orchestration_view(plan_data)

                # Phase 2: Parallel Workers
                with st.spinner("⚡ Phase 2: Parallel Subagents querying vector and keyword databases..."):
                    orchestrator = self.SubagentOrchestrator(self.base_dir)
                    await orchestrator.execute_plan(run_id)
                    findings_data = self.loader.load_subagent_findings(run_id)

                with subagent_placeholder.expander("📊 Phase 2 Complete: View Parallel Findings & Scores",
                                                   expanded=False):
                    self.render_subagent_view(findings_data)

                # Phase 3: Streaming Synthesis
                st.markdown("### 📝 Live Synthesized Report")
                streamed_text = ""

                def update_ui(chunk: str):
                    nonlocal streamed_text
                    streamed_text += chunk
                    report_placeholder.markdown(streamed_text)

                with st.spinner("✍️ Phase 3: Lead Agent synthesizing contradictions and writing final report..."):
                    builder = self.ReportBuilderOrchestrator(self.base_dir)
                    await builder.execute_synthesis(run_id, stream_callback=update_ui)

                st.success(
                    "Research Complete! The system has successfully verified citations and checked for contradictions. 🎉")

            except Exception as e:
                st.error(f"Pipeline crashed: {str(e)}")

        # CRITICAL FIX: Explicitly manage the event loop so Streamlit doesn't crash!
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run_pipeline())
        finally:
            loop.close()

    def run(self) -> None:
        self.setup_page()
        UIComponents.render_header(
            "Prometheus",
            "Autonomous Deep Research Engine for Biomedical Literature Synthesis 🧬"
        )

        query = st.chat_input(
            "Enter your research question (e.g., 'What are the resistance mechanisms to KRAS G12C inhibitors?')...")

        if query:
            st.chat_message("user").write(query)
            self.execute_live_research(query)


if __name__ == "__main__":
    dashboard = PrometheusDashboard()
    dashboard.run()