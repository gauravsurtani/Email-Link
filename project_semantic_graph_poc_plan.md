# Semantic Email Knowledge Graph - Proof of Concept Implementation Plan

## Overview

This document outlines a step-by-step plan to build a semantic email knowledge graph PoC using GenAI and agentic systems. Each step includes recommended tools, detailed actions, and next steps for rapid, coursework-focused implementation.

---

## Step 1: Define Problem & Scope

- **Goal:** Demonstrate extraction, connection, and querying of semantic information from emails (e.g., job applications, financial data, travel, etc.)
- **Use Cases:**
  - "How many jobs did I apply to at Walmart?"
  - "What's my total credit card debt?"
  - "Show all my travel bookings in 2024."
- **Success Criteria:**
  - System answers these questions by traversing a semantic graph built from real or synthetic emails.
- **Next Actions:**
  - List 3–5 target queries for demo.
  - Gather/generate sample emails for each use case.

---

## Step 2: Data Collection & Preprocessing

- **Tools:** Python `mailbox`, [LlamaIndex Mbox Reader](https://docs.llamaindex.ai/en/stable/examples/usecases/email_data_extraction/)
- **Actions:**
  - Collect a small, diverse set of emails (10–50 per use case).
  - Parse emails to extract subject, body, sender, recipient, date.
  - Store parsed data in a simple format (CSV, JSON, or Pandas DataFrame).
- **Next Actions:**
  - Write a script to parse and save email data.
  - Validate that all required fields are extracted.

---

## Step 3: Semantic Data Extraction (Entity & Action Recognition)

- **Tools:**
  - **GenAI:** OpenAI GPT-3.5/4, Gemini, Llama2/3 via API
  - **Prompt Engineering:** Custom prompts for structured extraction
  - **spaCy:** For fallback NER
- **Actions:**
  - For each email, send content to LLM with a prompt to extract:
    - Email type (job application, statement, booking, etc.)
    - Organizations, people, products, locations
    - Monetary amounts, dates, actions
  - Parse LLM output and store as structured JSON.
- **Next Actions:**
  - Design and test extraction prompts.
  - Run extraction on sample emails and review results.

---

## Step 4: Graph Construction

- **Tools:**
  - **Database:** Neo4j (Community Edition)
  - **Visualization:** Neo4j Browser, Bloom, or LlamaIndex NeoGraph
- **Actions:**
  - For each email, create:
    - `Email` node (with metadata)
    - `Entity` nodes (Organization, Person, Product, etc.)
    - `Action` nodes (Applied, Paid, Booked, etc.)
    - Relationships: `MENTIONS`, `DESCRIBES`, `ASSOCIATED_WITH`, etc.
  - Insert nodes/edges using Cypher or LlamaIndex API.
- **Next Actions:**
  - Define the graph schema (node/edge types, properties).
  - Write scripts to populate the graph from extracted data.

---

## Step 5: Semantic Linking & Topic Clustering

- **Tools:**
  - **Embeddings:** OpenAI, HuggingFace, or LlamaIndex
  - **Clustering:** scikit-learn (KMeans, Agglomerative), LlamaIndex vector search
- **Actions:**
  - Generate embeddings for all emails.
  - Cluster emails by topic or similarity.
  - Add topic nodes and link emails to them (`SIMILAR_TO`, `BELONGS_TO_TOPIC`).
- **Next Actions:**
  - Generate and store embeddings for each email.
  - Run clustering and update the graph with topic relationships.

---

## Step 6: Agentic Query System (GenAI-Powered Q&A)

- **Tools:**
  - **LlamaIndex Agent** or **LangChain Agent**
  - **OpenAI GPT-4** (or similar) for intent recognition and Cypher query generation
  - **Neo4j Cypher** for querying
- **Actions:**
  - User asks a question in natural language.
  - Agent parses intent and generates a Cypher query.
  - Agent runs the query, fetches results, and summarizes them in natural language.
  - Optionally, show the graph visualization for the answer.
- **Next Actions:**
  - Set up LlamaIndex or LangChain agent with Neo4j integration.
  - Prepare demo queries and validate agent responses.

---

## Step 7: Demo & Presentation

- **Tools:**
  - **Neo4j Browser** for live graph exploration
  - **Streamlit or Gradio** for a simple web UI (optional)
  - **Jupyter Notebook** for interactive demo
- **Actions:**
  - Prepare a few demo queries and show the system answering them.
  - Visualize the graph and highlight semantic links.
  - Optionally, show LLM prompts and outputs for transparency.
- **Next Actions:**
  - Script a demo session with 3–5 queries.
  - Prepare screenshots or a video walkthrough.

---

## Step 8: Validation & Iteration

- **Actions:**
  - Test with additional queries and edge cases.
  - Refine prompts and extraction logic as needed.
  - Document limitations and lessons learned.
- **Next Actions:**
  - Collect feedback from peers/instructors.
  - Iterate on weak points and finalize documentation.

---

## References

- [LlamaIndex Email Data Extraction Example](https://docs.llamaindex.ai/en/stable/examples/usecases/email_data_extraction/)
- [Building a Knowledge Graph: Step-by-Step Guide](https://smythos.com/ai-agents/ai-tutorials/building-a-knowledge-graph/)
- [Proof of Concept Best Practices](https://niftypm.com/blog/proof-of-concept/)

---

## Progress Checklist

- [ ] Step 1: Define Problem & Scope
- [ ] Step 2: Data Collection & Preprocessing
- [ ] Step 3: Semantic Data Extraction
- [ ] Step 4: Graph Construction
- [ ] Step 5: Semantic Linking & Topic Clustering
- [ ] Step 6: Agentic Query System
- [ ] Step 7: Demo & Presentation
- [ ] Step 8: Validation & Iteration
