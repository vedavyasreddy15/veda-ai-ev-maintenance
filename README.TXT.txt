# Veda AI: Predictive Maintenance Agent

**Author:** Vedavyas Reddy Bommineni
**Context:** M.S. Data Science Portfolio Project

## Project Overview
Veda AI is an advanced "Agentic RAG" system designed for the electric vehicle industry (specifically modeled for Lucid Motors). Instead of a standard chatbot, Veda AI acts as an autonomous engineering assistant. It uses a ReAct (Reason + Act) logic loop to translate natural language into SQL, query live vehicle telemetry, run predictive machine learning models, and output vehicle health assessments.

## The Architecture (The Radix)
This system bridges the gap between unstructured LLM reasoning and structured enterprise data pipelines.

1. **The Orchestrator (Brain):** LangChain / LlamaIndex
   - Drives the core ReAct loop.
   - Manages memory and tool selection.
2. **The Data Layer:** PostgreSQL
   - Houses over 175,000 rows of EV sensor telemetry (Battery SoH, Motor Temp, Tire Pressure).
   - Serves as the target for Text-to-SQL tool queries.
3. **The Agent's Tools (Hands):**
   - `query_vehicle_telemetry`: A Python function that takes the LLM's generated SQL, executes it against the Postgres DB via SQLAlchemy, and returns the raw sensor data as an observation.
   - `predict_failure_probability`: A Python script containing a pre-trained ML model (e.g., Random Forest) that calculates a health score based on the queried telemetry.
4. **The UI / Visualization:** - Streamlit (for the chat interface and raw data display).
   - Power BI (for enterprise dashboarding of the agent's outputs).

## Tech Stack
- **Database:** PostgreSQL (accessed via `psycopg2` and `SQLAlchemy`)
- **Data Manipulation:** `pandas`
- **Agent Framework:** LangChain (Python)
- **LLM:** OpenAI GPT-4 / Google Gemini (via API)
- **Evaluation:** Ragas (for mathematically proving low hallucination rates)

## Current Project State
- [x] Environment setup and dependency installation.
- [x] Data Ingestion Pipeline (`load_data.py`) written.
- [x] PostgreSQL database initialized and `vehicle_telemetry` table populated.
- [ ] Next Step: Initialize LangChain, connect to the PostgreSQL database, and build the Text-to-SQL tool.

## Note for AI Assistants
When assisting with this codebase, prioritize industry-standard, production-ready code. Ensure all SQL queries are parameterized or safely generated, and wrap external tool calls in `try/except` blocks to allow the ReAct loop to recover from errors gracefully.