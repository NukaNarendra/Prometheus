import os
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field
from neo4j import GraphDatabase, AsyncGraphDatabase, Driver, AsyncDriver
from neo4j.exceptions import ServiceUnavailable, AuthError
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import system, graph_schema
from src.agents.memory import SubagentFindings, WorkspaceManager


class GraphConnectionError(Exception):
    pass


class GraphExecutionError(Exception):
    pass


class GraphNode(BaseModel):
    element_id: str
    labels: List[str]
    properties: Dict[str, Any]


class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any]


class Neo4jConnectionManager:
    def __init__(self) -> None:
        self.uri = system.neo4j_uri
        self.user = system.neo4j_user
        self.password = system.neo4j_password
        self.driver: Optional[AsyncDriver] = None

    async def connect(self) -> None:
        if self.driver is None:
            try:
                self.driver = AsyncGraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password)
                )
                await self.driver.verify_connectivity()
            except (ServiceUnavailable, AuthError) as e:
                raise GraphConnectionError(f"Failed to connect to Neo4j at {self.uri}: {str(e)}")

    async def close(self) -> None:
        if self.driver is not None:
            await self.driver.close()
            self.driver = None

    async def execute_write(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        if self.driver is None:
            await self.connect()
        async with self.driver.session() as session:
            try:
                result = await session.run(query, parameters or {})
                return await result.data()
            except Exception as e:
                raise GraphExecutionError(f"Write query failed: {str(e)}")

    async def execute_read(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if self.driver is None:
            await self.connect()
        async with self.driver.session() as session:
            try:
                result = await session.run(query, parameters or {})
                return await result.data()
            except Exception as e:
                raise GraphExecutionError(f"Read query failed: {str(e)}")


class GraphSchemaBuilder:
    def __init__(self, connection_manager: Neo4jConnectionManager) -> None:
        self.db = connection_manager

    async def initialize_schema(self) -> None:
        queries = [
            f"CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:{graph_schema.paper_node_label}) REQUIRE p.paper_id IS UNIQUE",
            f"CREATE CONSTRAINT claim_id_unique IF NOT EXISTS FOR (c:{graph_schema.claim_node_label}) REQUIRE c.claim_id IS UNIQUE",
            f"CREATE INDEX paper_title_idx IF NOT EXISTS FOR (p:{graph_schema.paper_node_label}) ON (p.title)"
        ]
        for query in queries:
            await self.db.execute_write(query)

    async def clear_database(self) -> None:
        query = "MATCH (n) DETACH DELETE n"
        await self.db.execute_write(query)


class CitationGraphOrchestrator:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.db = Neo4jConnectionManager()
        self.schema_builder = GraphSchemaBuilder(self.db)
        self.workspace = WorkspaceManager(base_dir)

    def _generate_claim_id(self, task_id: str, claim_text: str) -> str:
        clean_text = claim_text.strip().lower()
        text_hash = str(hash(clean_text))[-8:]
        return f"claim_{task_id}_{text_hash}"

    async def _merge_paper_node(self, paper_id: str) -> None:
        query = f"""
        MERGE (p:{graph_schema.paper_node_label} {{paper_id: $paper_id}})
        ON CREATE SET p.created_at = timestamp()
        """
        await self.db.execute_write(query, {"paper_id": paper_id})

    async def _merge_claim_node(self, claim_id: str, claim_text: str, task_id: str) -> None:
        query = f"""
        MERGE (c:{graph_schema.claim_node_label} {{claim_id: $claim_id}})
        ON CREATE SET 
            c.text = $text,
            c.task_source = $task_id,
            c.created_at = timestamp()
        """
        await self.db.execute_write(query, {
            "claim_id": claim_id,
            "text": claim_text,
            "task_id": task_id
        })

    async def _create_support_relationship(self, paper_id: str, claim_id: str) -> None:
        query = f"""
        MATCH (p:{graph_schema.paper_node_label} {{paper_id: $paper_id}})
        MATCH (c:{graph_schema.claim_node_label} {{claim_id: $claim_id}})
        MERGE (p)-[r:{graph_schema.supports_rel_type}]->(c)
        ON CREATE SET r.created_at = timestamp()
        """
        await self.db.execute_write(query, {
            "paper_id": paper_id,
            "claim_id": claim_id
        })

    async def ingest_findings(self, run_id: str) -> None:
        await self.schema_builder.initialize_schema()
        findings_list = await self.workspace.load_all_findings(run_id)

        for finding in findings_list:
            task_id = finding.task_id
            for claim_obj in finding.key_claims:
                claim_text = claim_obj.get("claim", "")
                paper_id = claim_obj.get("paper_id", "")

                if not claim_text or not paper_id:
                    continue

                claim_id = self._generate_claim_id(task_id, claim_text)

                await self._merge_paper_node(paper_id)
                await self._merge_claim_node(claim_id, claim_text, task_id)
                await self._create_support_relationship(paper_id, claim_id)

    async def get_highly_supported_claims(self, min_supports: int = 1) -> List[Dict[str, Any]]:
        query = f"""
        MATCH (p:{graph_schema.paper_node_label})-[r:{graph_schema.supports_rel_type}]->(c:{graph_schema.claim_node_label})
        WITH c, count(p) as support_count, collect(p.paper_id) as supporting_papers
        WHERE support_count >= $min_supports
        RETURN c.claim_id as claim_id, c.text as text, c.task_source as task_id, support_count, supporting_papers
        ORDER BY support_count DESC
        """
        return await self.db.execute_read(query, {"min_supports": min_supports})

    async def get_papers_by_task(self, task_id: str) -> List[Dict[str, Any]]:
        query = f"""
        MATCH (p:{graph_schema.paper_node_label})-[r:{graph_schema.supports_rel_type}]->(c:{graph_schema.claim_node_label} {{task_source: $task_id}})
        RETURN DISTINCT p.paper_id as paper_id
        """
        return await self.db.execute_read(query, {"task_id": task_id})

    async def check_health(self) -> bool:
        try:
            await self.db.connect()
            await self.db.execute_read("RETURN 1 AS test")
            return True
        except Exception:
            return False

    async def generate_graph_summary(self) -> Dict[str, Any]:
        node_count_query = "MATCH (n) RETURN count(n) as total_nodes"
        edge_count_query = "MATCH ()-[r]->() RETURN count(r) as total_edges"

        nodes = await self.db.execute_read(node_count_query)
        edges = await self.db.execute_read(edge_count_query)

        return {
            "total_nodes": nodes[0]["total_nodes"] if nodes else 0,
            "total_edges": edges[0]["total_edges"] if edges else 0,
            "status": "healthy"
        }