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
    str(Path(__file__).parent / "Chinook_Sqlite.sqlite"),
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
    schema_text: str, force_rebuild: bool = False #easier for large deployments to control when the index is built, rather than always building at startup
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
"""
def generate_sql(question: str, schema: str, rag_context: str = "") -> str:
    system_prompt = fYou are an expert SQLite SQL assistant.

Hard constraints:
- Produce a SINGLE read-only query: SELECT (optionally WITH / EXPLAIN).
- DO NOT use INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE/PRAGMA/ATTACH/DETACH/VACUUM.
- Output ONLY the SQL query — no markdown fences, no explanation.

Relevant schema context (retrieved):
{rag_context}

Full schema (fallback reference):
{schema}

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
"""


def _summarise_result(cols: list[str], rows: list[tuple], max_sample: int = 3) -> str:
    """Compact one-line summary of a query result for conversation context."""
    if not rows:
        return "0 rows returned"
    sample_lines = [", ".join(str(v) for v in row) for row in rows[:max_sample]]
    tail = f" … (+{len(rows) - max_sample} more rows)" if len(rows) > max_sample else ""
    return f"{len(rows)} row(s) | columns: {cols} | sample: {sample_lines}{tail}"


def generate_sql(
    question: str,
    schema: str,
    rag_context: str = "",
    history: list[dict] | None = None,
) -> str:
    system_prompt = f"""You are an expert SQL assistant for a SQLite database.

Hard constraints:
- Produce a SINGLE read-only query: SELECT (optionally WITH / EXPLAIN).
- DO NOT use INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE/PRAGMA/ATTACH/DETACH/VACUUM.
- Output ONLY the SQL query — no markdown fences, no explanation.
- Use ONLY the tables and columns that appear in the schema below. Never invent or assume tables/columns not listed.

Column selection rules:
- Always include the primary key column(s) of the queried table(s) in your SELECT list.
- When the question asks to "list" or "show" items, include identifying columns (IDs, names) alongside the requested data.
- When joining tables, include the key columns that establish the join so results are traceable.
- For aggregation queries (COUNT, SUM, AVG, etc.), use a descriptive alias for the result column.
  Example: COUNT(*) AS record_count, SUM(amount) AS total_amount, AVG(price) AS avg_price.
- When the question mentions "ordered by" a column, include that column in the SELECT list.
- For GROUP BY queries, include all grouping columns in the SELECT list.
- ROUND numeric aggregations to 2 decimal places unless the question specifies otherwise.

Table selection rules:
- Use ONLY the tables defined in the schema below. Do not reference any table not listed there.
- Infer table and column names from the schema DDL; do not rely on assumptions about naming conventions.

Relevant schema context (retrieved):
{rag_context}

Full schema (fallback reference):
{schema}
"""
    # Build the message list: system prompt + conversation history + current question.
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    for turn in (history or []):
        messages.append({"role": "user", "content": turn["question"]})
        if turn.get("sql"):
            assistant_content = turn["sql"]
            if turn.get("result_summary"):
                assistant_content += f"\n-- Result: {turn['result_summary']}"
        else:
            assistant_content = f"-- Could not answer: {turn.get('error', 'unknown error')}"
        messages.append({"role": "assistant", "content": assistant_content})

    messages.append({"role": "user", "content": question})

    response = _ollama_client.chat(model=MODEL_NAME, messages=messages)
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
    empty_retry: bool   # True after an empty-result retry has been attempted
    history:     list   # Prior turns passed in from SQLSession [{question, sql, result_summary, error}]


# Checks generated SQL for any blocked DML/DDL keyword.
BLOCKED_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|PRAGMA|ATTACH|DETACH|VACUUM)\b",
    re.IGNORECASE,
)

# Checks the user's question for actual SQL injection patterns (paired command + target).
# Deliberately does NOT match isolated words like "delete", "drop", "update" in natural language.
INJECTION_PATTERN = re.compile(
    r"\b(DROP\s+TABLE|DELETE\s+FROM|INSERT\s+INTO|UPDATE\s+\w+\s+SET|ALTER\s+TABLE"
    r"|CREATE\s+(?:TABLE|DATABASE|INDEX|VIEW)|TRUNCATE\s+TABLE)\b",
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
    state["sql"] = generate_sql(
        q,
        state["schema"],
        rag_context=state.get("rag_context", ""),
        history=state.get("history", []),
    )
    state["error"] = None
    return state


def node_security_check(state: SQLState) -> SQLState:
    """Two-tier security check.

    1. Generated SQL: reject any blocked DML/DDL keyword (strict).
    2. User question: only reject if it contains an actual SQL injection pattern
       (e.g. "DROP TABLE …", "DELETE FROM …"), not isolated natural-language words
       like "delete", "drop", or "update".
    """
    if BLOCKED_SQL.search(state["sql"]):
        state["error"] = "Blocked: generated SQL contains a disallowed operation."
        state["result"] = None
    elif INJECTION_PATTERN.search(state["question"]):
        state["error"] = "Blocked: question contains a disallowed SQL operation."
        state["result"] = None
    return state


def node_execute_sql(state: SQLState) -> SQLState:
    try:
        cols, rows = execute_sql(DB_PATH, state["sql"])
        state["result"] = {"columns": cols, "rows": rows}
        state["error"] = None
    except Exception as e:
        err_msg = str(e)
        err_lower = err_msg.lower()
        # Convert schema-mismatch errors into user-friendly messages so the
        # caller gets a clear explanation instead of a raw SQLite exception.
        if "no such table" in err_lower:
            m = re.search(r"no such table:\s*(\S+)", err_msg, re.IGNORECASE)
            name = m.group(1) if m else "unknown"
            state["error"] = (
                f"This question references a table ({name}) that does not exist "
                "in this database. Cannot answer with the available schema."
            )
        elif "no such column" in err_lower:
            m = re.search(r"no such column:\s*(\S+)", err_msg, re.IGNORECASE)
            name = m.group(1) if m else "unknown"
            state["error"] = (
                f"This question references a field ({name}) that does not exist "
                "in this database. Cannot answer with the available schema."
            )
        else:
            state["error"] = err_msg
        state["result"] = None
    return state


def node_inc_tries(state: SQLState) -> SQLState:
    state["tries"] += 1
    return state


def node_handle_empty_result(state: SQLState) -> SQLState:
    """Augment the question with empty-result guidance before the retry."""
    state["question"] = (
        f"{state['question']}\n\n"
        "Note: the previous query executed without error but returned 0 rows, "
        "which seems unexpected. Review your JOIN conditions, WHERE filters, and "
        "string comparisons (case-sensitivity, exact vs. partial match). "
        "Generate a corrected query."
    )
    state["empty_retry"] = True
    state["error"] = None
    return state


# ── routers ──────────────────────────────────────────────────────────────────

def route_after_security(state: SQLState) -> str:
    """Short-circuit to END when the security check flagged the query.
    This was the critical bug in v2: the old code had an unconditional edge
    sec_check → exec_sql, so blocked queries still reached the database.
    """
    return "blocked" if state.get("error") else "execute"


def route_after_execute(state: SQLState) -> str:
    error = state.get("error")

    if error is not None:
        if state["tries"] >= MAX_TRIES:
            return "done"
        # Schema errors (missing table/column) can never be fixed by retrying —
        # the schema won't change between attempts.
        err_lower = error.lower()
        if "does not exist in this database" in err_lower:
            return "done"
        return "retry"

    # Semantic validation: if a non-aggregate query unexpectedly returns 0 rows,
    # retry once with explicit guidance to review the query logic.
    if not state.get("empty_retry", False):
        rows = (state.get("result") or {}).get("rows", [])
        sql_upper = state.get("sql", "").upper()
        is_aggregate = any(kw in sql_upper for kw in ("COUNT(", "SUM(", "AVG(", "MIN(", "MAX("))
        if len(rows) == 0 and not is_aggregate:
            return "empty_retry"

    return "done"


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
    g.add_node("handle_empty", node_handle_empty_result)

    g.add_edge(START,          "load_schema")
    g.add_edge("load_schema",  "build_rag")
    g.add_edge("build_rag",    "retrieve_rag")
    g.add_edge("retrieve_rag", "gen_sql")
    g.add_edge("gen_sql",      "sec_check")

    # Blocked queries short-circuit to END; legitimate queries reach exec_sql.
    g.add_conditional_edges(
        "sec_check",
        route_after_security,
        {"blocked": END, "execute": "exec_sql"},
    )

    # Three outcomes after execution:
    #   retry       — SQL error that may be fixable (retried via inc_tries → gen_sql)
    #   empty_retry — success but 0 rows; retry once with guidance
    #   done        — success, graceful schema error, or max retries reached
    g.add_conditional_edges(
        "exec_sql",
        route_after_execute,
        {"retry": "inc_tries", "empty_retry": "handle_empty", "done": END},
    )
    g.add_edge("inc_tries",    "gen_sql")
    g.add_edge("handle_empty", "inc_tries")

    return g.compile()


# Build the graph once at import time so both the notebook and benchmark
# share the same compiled app without rebuilding it.
app = build_sql_graph()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_sql_agent(
    question: str,
    history: list[dict] | None = None,
) -> SQLState:
    """Main entrypoint — call this from the notebook, benchmark, or SQLSession."""
    initial_state: SQLState = {
        "question":    question,
        "schema":      "",
        "rag_context": "",
        "sql":         "",
        "result":      None,
        "error":       None,
        "tries":       0,
        "empty_retry": False,
        "history":     history or [],
    }
    return app.invoke(initial_state)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-turn session
# ─────────────────────────────────────────────────────────────────────────────

class SQLSession:
    """Stateful multi-turn session — keeps a rolling window of the last
    MAX_HISTORY conversation rounds and passes them as context to each new query.

    Usage:
        session = SQLSession()
        r1 = session.ask("Top 5 customers by total spend?")
        r2 = session.ask("Which of those are from Brazil?")   # references r1 context
        r3 = session.ask("Show their invoice dates too.")     # references r2 context
        session.reset()  # start fresh
    """

    MAX_HISTORY: int = 3

    def __init__(self) -> None:
        self.history: list[dict] = []

    def ask(self, question: str) -> SQLState:
        """Run a question with the current conversation history as context."""
        state = run_sql_agent(question, history=self.history)

        # Build a compact summary of the result to store in history.
        result = state.get("result")
        if isinstance(result, dict) and result.get("rows") is not None:
            summary = _summarise_result(
                result.get("columns", []),
                result.get("rows", []),
            )
        else:
            summary = None

        turn = {
            "question":       question,
            "sql":            state.get("sql", ""),
            "result_summary": summary,
            "error":          state.get("error"),
        }
        # Append and keep only the most recent MAX_HISTORY turns.
        self.history = (self.history + [turn])[-self.MAX_HISTORY:]
        return state

    def reset(self) -> None:
        """Clear all conversation history."""
        self.history = []

    def __repr__(self) -> str:
        return f"SQLSession(rounds={len(self.history)}/{self.MAX_HISTORY})"


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
