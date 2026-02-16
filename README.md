# Text-to-SQL AI Agent

<div align="center">
    <em>A local-first, LLM-powered agent for translating natural language into executable SQL queries.</em>
</div>

---

## 📖 Overview

This repository contains the implementation of a Text-to-SQL AI agent. The primary objective is to bridge the gap between human language and relational databases, allowing users to intuitively query datasets (such as the included Chinook database) without needing to write complex SQL syntax. 

The project leverages local deployment frameworks to run advanced large language models securely and efficiently on local hardware.


## 🧠 Architecture & Models

The agent is designed to run locally, ensuring data privacy and reducing API latency overhead. It relies on the **Ollama** framework for local model execution. 

Supported and evaluated models include:
* **Qwen** (Local deployment configured via `qwen_local/`)
* Other open source models are also under consideration and may selected in the future.

## 📂 Repository Structure

* `qwen_local/` - Configurations and scripts for local Qwen model deployment.
* `sql_agent.ipynb` & `sql_agent_v2.ipynb` - Core implementation notebooks detailing the agent's logic and iterative improvements.
* `ollama_test.ipynb` - Sandbox environment for testing local Ollama endpoints and evaluating model responses.
* `Chinook_Sqlite.sqlite` & `sql_agent_database.db` - Sample relational databases utilized for querying, testing, and validation.

## 🚀 Quick Start

To explore the agent, ensure you have a local instance of Ollama running. 

1. Clone the repository to your local machine.
2. Ensure you have the required models pulled via Ollama (e.g., `ollama run qwen` or `ollama run deepseek`).
3. Open and execute the cells within `sql_agent_v2.ipynb` to initialize the agent and begin querying the included SQLite databases. 

*(Note: Exhaustive test steps and environment setups are omitted here for brevity. Please refer to the inline documentation within the Jupyter notebooks for specific execution flows.)*

## 🔮 Future Updates

* **Retrieval-Augmented Generation (RAG):** Future iterations will integrate RAG pipelines to dynamically retrieve and inject relevant database schema context and metadata into the prompt. This will significantly enhance the model's accuracy on massive, enterprise-scale databases by filtering the global schema $\mathcal{S}$ down to a highly relevant subset $\mathcal{S}_{RAG} \subset \mathcal{S}$ prior to generation:
    
$$
\hat{Y} = \arg\max_{Y} P_{\theta}(Y \mid U, \mathcal{S}_{RAG})
$$
