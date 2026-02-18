"""
sql_agent_v3.py
---------------
Importable Python module for the LangGraph SQL agent.

This file is the single source of truth for all agent logic.
- sql_agent_v3.ipynb  imports from here (thin interactive wrapper)
- benchmark_chinook.py imports from here (automated evaluation)

Configuration via environment variables (all have local fallbacks):
  SQL_AGENT_DB_PATH    Path to the SQLite database file
  SQL_AGENT_CHROMA_DIR Directory for ChromaDB persistence
  SQL_AGENT_MODEL      Ollama model name
  OLLAMA_HOST          Ollama server URL

Docker example:
  ENV SQL_AGENT_DB_PATH=/data/Chinook_Sqlite.sqlite
  ENV SQL_AGENT_CHROMA_DIR=/data/chroma_sql_rag
  ENV OLLAMA_HOST=http://ollama:11434
"""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional, TypedDict

import chromadb
import ollama
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langgraph.graph import END, START, StateGraph

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH: str = os.environ.get(
    "SQL_AGENT_DB_PATH",
    "D:/SQL_agent/Chinook_Sqlite.sqlite",   # local Windows fallback
)

CHROMA_DIR: str = os.environ.get(
    "SQL_AGENT_CHROMA_DIR",
    str(Path(__file__).parent / "chroma_sql_rag"),
)

MODEL_NAME: str = os.environ.get("SQL_AGENT_MODEL", "qwen2.5-coder:7b")

COLLECTION_NAME = "schema_docs"
MAX_TRIES = 2

# Ollama client — reads OLLAMA_HOST automatically.
# Locally defaults to http://localhost:11434.
# In Docker set OLLAMA_HOST=http://ollama:11434 to reach the sidecar.
_ollama_client = ollama.Client(
    host=os.environ.get("OLLAMA_HOST", "http://localhost:11434")
)

# ─────────────────────────────────────────────────────────────────────────────
# RAG — schema indexing & retrieval
# ─────────────────────────────────────────────────────────────────────────────

_embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def schema_to_docs(schema_text: str) -> list[dict]:
    """Split a schema string into one document per CREATE TABLE block."""
    docs: list[dict] = []
    blocks = re.split(r";\s*\n", schema_text.strip())
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if re.search(r"create\s+table", b, re.IGNORECASE):
            m = re.search(
                r"create\s+table\s+(?:if\s+not\s+exists\s+)?([^\s(]+)", b, re.IGNORECASE
            )
            table = m.group(1) if m else "unknown"
            docs.append({"id": f"table::{table}", "text": b + ";", "meta": {"table": table}})
    return docs


def build_or_load_chroma(
    schema_text: str, force_rebuild: bool = False
) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if force_rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    col = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=_embedding_fn
    )
    if col.count() == 0:
        docs = schema_to_docs(schema_text)
        col.add(
            ids=[d["id"] for d in docs],
            documents=[d["text"] for d in docs],
            metadatas=[d["meta"] for d in docs],
        )
        print(f"✅ Chroma populated with {len(docs)} schema docs.")
    else:
        print(f"✅ Chroma collection already has {col.count()} docs.")
    return col


def retrieve_schema_context(col: chromadb.Collection, question: str, k: int = 6) -> str:
    """k=6 covers wider multi-table joins without blowing up the prompt."""
    res = col.query(query_texts=[question], n_results=min(k, col.count()))
    docs = res["documents"][0] if res and res.get("documents") else []
    return "\n\n".join(docs)


# ─────────────────────────────────────────────────────────────────────────────
# Schema extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_database_schema(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
    schema = "".join(f"{row[0]};\n" for row in cursor.fetchall() if row[0])
    conn.close()
    return schema


# ─────────────────────────────────────────────────────────────────────────────
# SQL generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_sql(question: str, schema: str, rag_context: str = "") -> str:
    system_prompt = f"""You are an expert SQLite SQL assistant.

Hard constraints:
- Produce a SINGLE read-only query: SELECT (optionally WITH / EXPLAIN).
- DO NOT use INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE/PRAGMA/ATTACH/DETACH/VACUUM.
- Output ONLY the SQL query — no markdown fences, no explanation.

Relevant schema context (retrieved):
{rag_context}

Full schema (fallback reference):
{schema}
"""
    response = _ollama_client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    sql = response["message"]["content"].strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


# ─────────────────────────────────────────────────────────────────────────────
# SQL execution (read-only connection)
# ─────────────────────────────────────────────────────────────────────────────

def connect_readonly(db_path: str) -> sqlite3.Connection:
    """URI mode with mode=ro — SQLite itself refuses writes at the engine level."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def execute_sql(db_path: str, sql: str) -> tuple[list[str], list[tuple]]:
    conn = connect_readonly(db_path)
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description] if cur.description else []
    conn.close()
    return cols, rows


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph agent
# ─────────────────────────────────────────────────────────────────────────────

class SQLState(TypedDict):
    question:    str
    schema:      str
    rag_context: str
    sql:         str
    result:      Any
    error:       Optional[str]
    tries:       int


BLOCKED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|PRAGMA|ATTACH|DETACH|VACUUM)\b",
    re.IGNORECASE,
)

_chroma_collection: chromadb.Collection | None = None


# ── nodes ────────────────────────────────────────────────────────────────────

def node_load_schema(state: SQLState) -> SQLState:
    state["schema"] = get_database_schema(DB_PATH)
    return state


def node_build_rag_index(state: SQLState) -> SQLState:
    global _chroma_collection
    if _chroma_collection is None:
        _chroma_collection = build_or_load_chroma(state["schema"])
    return state


def node_retrieve_rag(state: SQLState) -> SQLState:
    global _chroma_collection
    state["rag_context"] = retrieve_schema_context(_chroma_collection, state["question"])
    return state


def node_generate_sql(state: SQLState) -> SQLState:
    q = state["question"]
    if state.get("error"):
        q = (
            f"{q}\n\nThe previous SQL failed with this error:\n"
            f"{state['error']}\nFix the SQL."
        )
    state["sql"] = generate_sql(q, state["schema"], rag_context=state.get("rag_context", ""))
    state["error"] = None
    return state


def node_security_check(state: SQLState) -> SQLState:
    """Flags disallowed keywords. route_after_security will short-circuit to END."""
    if BLOCKED.search(state["sql"]):
        state["error"] = "Blocked: query contains a disallowed keyword."
        state["result"] = None
    return state


def node_execute_sql(state: SQLState) -> SQLState:
    try:
        cols, rows = execute_sql(DB_PATH, state["sql"])
        state["result"] = {"columns": cols, "rows": rows}
        state["error"] = None
    except Exception as e:
        state["result"] = None
        state["error"] = str(e)
    return state


def node_inc_tries(state: SQLState) -> SQLState:
    state["tries"] += 1
    return state


# ── routers ──────────────────────────────────────────────────────────────────

def route_after_security(state: SQLState) -> str:
    """Short-circuit to END when the security check flagged the query.
    This was the critical bug in v2: the old code had an unconditional edge
    sec_check → exec_sql, so blocked queries still reached the database.
    """
    return "blocked" if state.get("error") else "execute"


def route_after_execute(state: SQLState) -> str:
    if state["error"] is None:
        return "done"
    if state["tries"] >= MAX_TRIES:
        return "done"
    return "retry"


# ── graph ────────────────────────────────────────────────────────────────────

def build_sql_graph():
    g = StateGraph(SQLState)

    g.add_node("load_schema",  node_load_schema)
    g.add_node("build_rag",    node_build_rag_index)
    g.add_node("retrieve_rag", node_retrieve_rag)
    g.add_node("gen_sql",      node_generate_sql)
    g.add_node("sec_check",    node_security_check)
    g.add_node("exec_sql",     node_execute_sql)
    g.add_node("inc_tries",    node_inc_tries)

    g.add_edge(START,          "load_schema")
    g.add_edge("load_schema",  "build_rag")
    g.add_edge("build_rag",    "retrieve_rag")
    g.add_edge("retrieve_rag", "gen_sql")
    g.add_edge("gen_sql",      "sec_check")

    # ✅ FIX: conditional branch — blocked queries never reach exec_sql
    g.add_conditional_edges(
        "sec_check",
        route_after_security,
        {"blocked": END, "execute": "exec_sql"},
    )

    g.add_conditional_edges(
        "exec_sql",
        route_after_execute,
        {"retry": "inc_tries", "done": END},
    )
    g.add_edge("inc_tries", "gen_sql")

    return g.compile()


# Build the graph once at import time so both the notebook and benchmark
# share the same compiled app without rebuilding it.
app = build_sql_graph()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_sql_agent(question: str) -> SQLState:
    """Main entrypoint — call this from the notebook or benchmark."""
    initial_state: SQLState = {
        "question":    question,
        "schema":      "",
        "rag_context": "",
        "sql":         "",
        "result":      None,
        "error":       None,
        "tries":       0,
    }
    return app.invoke(initial_state)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point (python sql_agent_v3.py "your question here")
# ─────────────────────────────────────────────────────────────────────────────

def main(question: Optional[str] = None) -> None:
    import sys
    q = question or (sys.argv[1] if len(sys.argv) > 1 else None) or \
        "Show me the top 5 customers who spent the most money, including their email."

    print(f"Question: {q}\n")
    out = run_sql_agent(q)

    print("-" * 60)
    print("Generated SQL:")
    print(out["sql"])
    print("-" * 60)

    if out["error"]:
        print(f"❌ Error after retries: {out['error']}")
        return

    result = out["result"] or {"columns": [], "rows": []}
    print(result["columns"])
    for row in result["rows"]:
        print(row)


if __name__ == "__main__":
    main()
