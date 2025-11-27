# 📘 Deep Agent App  
*A multi-agent, multi-model research system built with Anthropic Claude + LangGraph + DeepAgents.*

---

## 🌟 Overview

This project is a sandbox for experimenting with **agentic AI architectures** using:

- **Anthropic Claude models** (Haiku 4.5, Sonnet 4.5, Opus 4.5)  
- **LangGraph** for execution  
- **DeepAgents** for TODO-style planning  
- **Parallel multi-agent orchestration**  
- **Supervisor merging** of sub-agent results

The idea:

> **Take a user query → spawn multiple specialist agents → merge their insights → produce a high-quality research report.**

---

## 🚀 Features

### ✅ Single-Agent Fast Mode
A lightweight DeepAgent powered by Claude Sonnet 4.5:

- Shallow reasoning  
- Max 3 TODOs  
- Fast execution  
- Produces short markdown reports  

Perfect for quick research or summaries.

---

### ✅ Multi-Agent Research Mode
A team of specialist agents, each with a different role and model:

| Agent Role     | Model Used              | Strength |
|----------------|--------------------------|----------|
| Explainer      | Claude Haiku 4.5         | Fast breadth-first reasoning |
| Skeptic        | Claude Sonnet 4.5        | Balanced analysis & critique |
| Pragmatist     | Claude Opus 4.5          | Deep reasoning & real-world insight |

Each specialist generates its own markdown sub-report.

---

### ✅ Parallel Agent Execution
All agents run **simultaneously** via `ThreadPoolExecutor`.

Benefits:

- 3× speed improvement  
- Lower latency  
- Better model specialization  

---

### ✅ Supervisor Merging
A dedicated **Claude supervisor** merges all specialist reports:

- Deduplicates overlap  
- Resolves contradictions  
- Combines all perspectives  
- Produces a clean, high-level final markdown report  

Sections included:

- Overview  
- Key Insights  
- Trade-offs & Risks  
- Open Questions  
- Summary  

---

## 🧠 Architecture

               ┌──────────────────────┐
               │      User Query       │
               └──────────┬────────────┘
                          │
           ┌──────────────┼────────────────┐
           │              │                │
           ▼              ▼                ▼
   Explainer Agent   Skeptic Agent   Pragmatist Agent
  (Haiku 4.5 Model) (Sonnet 4.5)        (Opus 4.5)
           │              │                │
           └─────── Parallel Run ─────────┘
                          │
                          ▼
                Supervisor (Sonnet 4.5)
                          │
                          ▼
              Final Merged Markdown Report

---

## 📂 Output Structure

All generated files saved to:

agent_workspace/
report_explainer.md
report_skeptic.md
report_pragmatist.md
final_report.md


---

## 🛠️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
or
docker build -t deep-agent-app .

create .env file with the following keys

ANTHROPIC_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
LANGCHAIN_API_KEY=your_key_here


## running app

run_research(topic)
or
run_multi_agent_research(topic)
or
run_multi_agent_research_parallel(topic)

