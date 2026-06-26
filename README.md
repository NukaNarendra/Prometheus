# 🧬 Prometheus: Autonomous Multi-Agent Research Engine

Prometheus is a frontier-grade, multi-agent Retrieval-Augmented Generation (RAG) architecture built to autonomously execute, evaluate, and synthesize deep academic research.

Designed to overcome the limitations of standard monolithic RAG systems—which often struggle with complex reasoning and hallucinate when faced with contradictory evidence—Prometheus leverages parallel subagents, hybrid retrieval, and an LLM-powered Contradiction Engine mapped via a Neo4j Citation Graph.

---

# ✨ Key Features

## 🤖 Multi-Agent Orchestration

A Lead Agent breaks down complex queries into mutually exclusive sub-tasks, deploying asynchronous Parallel Subagents to investigate each angle independently.

## 🔄 Corrective RAG (CRAG) & Self-RAG

A dedicated Evidence Scorer evaluates every claim made by the subagents. If confidence falls below the threshold, it autonomously refines search keywords and triggers a retrieval retry loop.

## 🔍 Hybrid Retrieval Pipeline

Combines dense vector similarity (ChromaDB + sentence-transformers) with sparse keyword matching (BM25Okapi) and Reciprocal Rank Fusion (RRF) for superior document recall.

## 🕸️ Graph-Based Contradiction Detection

Maps claims and paper citations to a Neo4j Graph Database. An LLM Contradiction Engine cross-examines nodes to identify, flag, and resolve conflicting scientific evidence before final synthesis.

## 📈 Automated Evaluation Engine

Includes an LLM-as-a-Judge benchmarking suite (`run_eval.py`) that mathematically evaluates the performance of the multi-agent architecture against a standard single-agent RAG baseline.

## 🎥 Live Streaming UI

A fully responsive Streamlit dashboard visualizes the agentic workflow, confidence scores, retrieval evidence, contradiction analysis, and streams the final synthesized report in real time.

---

# 🏗️ System Architecture

```text
User Query
    │
    ▼
Lead Agent (Task Decomposition)
    │
    ├── Research Angle 1
    ├── Research Angle 2
    ├── Research Angle 3
    └── Research Angle 4+
            │
            ▼
     Parallel Subagents
            │
            ▼
 Hybrid Retrieval Engine
(Dense + Sparse + RRF)
            │
            ▼
 Evidence Scorer (CRAG)
            │
     ┌──────┴──────┐
     │             │
 High Confidence  Low Confidence
     │             │
     │       Retrieval Retry Loop
     │             │
     └──────┬──────┘
            ▼
      Citation Graph
         (Neo4j)
            │
            ▼
 Contradiction Engine
            │
            ▼
   Report Synthesis Agent
            │
            ▼
 Final Research Report
```

---

# 🔬 Research Workflow

### 1. Query Decomposition

The Lead Agent analyzes the user query and creates a structured JSON research plan containing multiple independent investigation angles.

### 2. Parallel Investigation

Multiple subagents simultaneously retrieve, analyze, and summarize evidence from the local corpus.

### 3. Evidence Validation

The Evidence Scorer evaluates evidence quality, relevance, and citation support.

### 4. Corrective Retrieval

If evidence confidence is insufficient, the system automatically reformulates queries and performs additional retrieval.

### 5. Citation Graph Construction

Claims and source papers are mapped into a Neo4j knowledge graph.

### 6. Contradiction Analysis

The Contradiction Engine identifies conflicting claims, evaluates severity levels, and generates resolution summaries.

### 7. Report Generation

The synthesis module compiles validated findings into a structured, citation-backed research report.

---

# 🚀 Quick Start

## 1. Prerequisites

* Python 3.10+
* Docker
* NVIDIA API Key
* Neo4j Database

---

## 2. Clone Repository

```bash
git clone https://github.com/NukaNarendra/Prometheus.git
cd prometheus
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file:

```env
NVIDIA_API_KEY=nvapi-your-key-here

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

---

## 5. Start Neo4j

```bash
docker run \
--name neo4j-prometheus \
-p 7474:7474 \
-p 7687:7687 \
-e NEO4J_AUTH=neo4j/password \
-d neo4j:latest
```

---

## 6. Seed Research Corpus

```bash
python scripts/seed_corpus.py
```

---

## 7. Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

# 📊 Evaluation & Benchmarking

Prometheus includes a comprehensive benchmarking suite that compares:

* Multi-Agent Prometheus
* Standard Single-Agent RAG

Evaluation categories:

* Evidence Coverage
* Citation Accuracy
* Contradiction Detection
* Hallucination Reduction
* Report Completeness
* Overall Research Quality

Run evaluation:

```bash
python eval/run_eval.py
```

Results are stored in:

```text
eval/results/evaluation_metrics.json
```

Example output:

```json
{
  "multi_agent_score": 9.2,
  "single_agent_score": 7.1,
  "coverage_gain": "29.5%",
  "hallucination_reduction": "41.3%"
}
```

---

# 📂 Project Structure

```text
prometheus/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── corpus/
│   ├── memory/
│   └── chroma_db/
│
├── eval/
│   └── run_eval.py
│
├── scripts/
│   └── seed_corpus.py
│
├── src/
│   │
│   ├── agents/
│   │   ├── lead_agent.py
│   │   ├── subagent.py
│   │   └── orchestrator.py
│   │
│   ├── retrieval/
│   │   ├── hybrid_search.py
│   │   ├── vector_store.py
│   │   └── keyword_store.py
│   │
│   ├── correction/
│   │   └── evidence_scorer.py
│   │
│   ├── synthesis/
│   │   ├── contradiction_engine.py
│   │   ├── neo4j_graph.py
│   │   └── report_builder.py
│   │
│   ├── connectors/
│   │   └── data_normalizer.py
│   │
│   └── config.py
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

# 🧠 Core Technologies

| Layer            | Technology             |
| ---------------- | ---------------------- |
| LLM              | NVIDIA Nemotron        |
| Framework        | LangChain              |
| Vector DB        | ChromaDB               |
| Embeddings       | Sentence Transformers  |
| Keyword Search   | BM25Okapi              |
| Graph Database   | Neo4j                  |
| UI               | Streamlit              |
| Evaluation       | LLM-as-a-Judge         |
| Retrieval Fusion | Reciprocal Rank Fusion |

---

# 🎯 Future Roadmap

* Multi-modal research (PDFs, images, figures)
* Agent memory persistence
* Cross-paper reasoning chains
* Automated hypothesis generation
* Scientific knowledge graph expansion
* Federated document retrieval
* Research paper drafting assistant
* Autonomous literature review generation

---

# 🤝 Acknowledgements

Prometheus is inspired by cutting-edge developments in:

* Multi-Agent Systems
* Retrieval-Augmented Generation (RAG)
* Corrective RAG (CRAG)
* Self-RAG
* Knowledge Graph Reasoning
* Agentic AI Research Workflows

Built using:

* LangChain
* NVIDIA AI Endpoints
* Neo4j
* ChromaDB
* Sentence Transformers
* Streamlit

---

# 📜 License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files to deal in the Software without restriction.
