"""
Simple CLI to run a single question through the agent graph.

Usage examples:
  python run_query.py --question "List 5 customers" --show-rows 5
  python run_query.py --question "Total sales by country" --json

Notes:
- This assumes Ollama is reachable via OLLAMA_HOST (env var).
- DB and Chroma paths are controlled by SQL_AGENT_DB_PATH and SQL_AGENT_CHROMA_DIR.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from sql_agent_v3 import build_sql_graph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True, help="Natural language question to answer with SQL.")
    ap.add_argument("--max-retries", type=int, default=2, help="Max retries inside the graph (default: 2).")
    ap.add_argument("--show-rows", type=int, default=10, help="How many result rows to print (default: 10).")
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = ap.parse_args()

    app = build_sql_graph(max_retries=args.max_retries)

    state: Dict[str, Any] = {"question": args.question}
    out: Dict[str, Any] = app.invoke(state)

    # Normalize outputs
    sql = out.get("sql", "")
    error = out.get("error", "")
    rows = out.get("rows") or []
    cols = out.get("cols") or []

    if args.json:
        print(json.dumps({"sql": sql, "error": error, "cols": cols, "rows": rows}, ensure_ascii=False, indent=2))
        return

    print("=== SQL ===")
    print(sql or "(empty)")
    if error:
        print("\n=== ERROR ===")
        print(error)
        return

    print("\n=== RESULT (first rows) ===")
    if not rows:
        print("(no rows)")
        return

    # Print a simple table without extra deps
    head = rows[: max(0, args.show_rows)]
    if cols:
        print(" | ".join(map(str, cols)))
        print("-" * max(20, len(" | ".join(map(str, cols)))))
    for r in head:
        print(" | ".join(map(str, r)))


if __name__ == "__main__":
    main()
