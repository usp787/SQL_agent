"""
benchmark_chinook_eval_jsonl.py
--------------------------------
JSONL-based Chinook benchmark runner for sql_agent_v3.py

Features
- Tier A/B/C/D benchmark loading from JSONL
- Evaluation mode (gold SQL execution + exact result comparison)
- Robustness cases with expected behaviors:
    * success
    * graceful_error
    * blocked
- Outputs:
    * per-case run log JSONL
    * summary JSON
    * summary CSV (one-row aggregate)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Ensure local imports work when run from any directory
HERE = Path(__file__).parent.resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import sql_agent_v3 as agent  # type: ignore


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {i}: {e}") from e
            cases.append(obj)
    return cases


def execute_sql_direct(db_path: str, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description] if cur.description else []
    conn.close()
    return cols, rows


def _normalize_value(v: Any) -> Any:
    # Robust exact-ish comparison for numeric values from SQLite / Python
    if isinstance(v, float):
        # Round to avoid tiny binary artifacts while still being strict enough
        return round(v, 6)
    return v


def normalize_result(cols: list[Any], rows: list[tuple[Any, ...]]) -> dict[str, Any]:
    norm_cols = [str(c) for c in cols]
    norm_rows = [[_normalize_value(v) for v in row] for row in rows]
    return {"columns": norm_cols, "rows": norm_rows}


def infer_failure_reason(case: dict[str, Any], state: dict[str, Any] | None, exception: str | None) -> str:
    if exception:
        return "runner_exception"
    if state is None:
        return "no_state"
    err = str(state.get("error") or "")
    q = case.get("question", "")
    sql = str(state.get("sql") or "")
    if "Blocked" in err or "blocked" in err:
        if case.get("expect_behavior") == "blocked":
            return "blocked_as_expected"
        return "unexpected_block"
    if case.get("expect_behavior") == "graceful_error":
        if err:
            return "graceful_error_as_expected"
        return "unexpected_success_on_nonexistent"
    if err:
        low = err.lower()
        if "no such table" in low:
            return "no_such_table"
        if "no such column" in low:
            return "no_such_column"
        if "syntax" in low:
            return "sql_syntax_error"
        if "ambiguous" in low:
            return "ambiguous_column_error"
        return "exec_error_other"
    # success path but maybe wrong
    if not sql.strip():
        return "empty_sql"
    return "result_mismatch_or_success"


def evaluate_case(case: dict[str, Any], db_path: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    state: dict[str, Any] | None = None
    exception_msg = None

    record: dict[str, Any] = {
        "id": case["id"],
        "tier": case["tier"],
        "category": case["category"],
        "question": case["question"],
        "expect_behavior": case.get("expect_behavior", "success"),
        "tags": case.get("tags", []),
        "notes": case.get("notes", ""),
        # runtime fields
        "latency_s": None,
        "tries": None,
        "agent_error": None,
        "generated_sql": None,
        "executed": False,
        "blocked": False,
        "graceful_error": False,
        "gold_sql_executed": False,
        "gold_error": None,
        "actual_result": None,
        "gold_result": None,
        "exact_columns_match": None,
        "exact_rows_match": None,
        "exact_result_match": None,
        "pass_case": False,
        "failure_reason": None,
        "exception": None,
    }

    try:
        state = agent.run_sql_agent(case["question"])
    except Exception as exc:
        exception_msg = f"{type(exc).__name__}: {exc}"
        record["exception"] = exception_msg
    finally:
        record["latency_s"] = round(time.perf_counter() - t0, 4)

    if state is not None:
        record["tries"] = state.get("tries")
        record["agent_error"] = state.get("error")
        record["generated_sql"] = state.get("sql")

        err = str(state.get("error") or "")
        record["blocked"] = ("blocked" in err.lower())
        record["graceful_error"] = bool(err) and not record["blocked"]

        result = state.get("result")
        if isinstance(result, dict) and "rows" in result:
            record["executed"] = (state.get("error") is None)
            cols = result.get("columns") or []
            rows = result.get("rows") or []
            # Ensure JSON-serializable normalized copy
            record["actual_result"] = normalize_result(cols, [tuple(r) for r in rows])
        else:
            record["executed"] = (state.get("error") is None)

    # Evaluate against expected behavior
    expected = case.get("expect_behavior", "success")
    if expected == "blocked":
        record["pass_case"] = bool(record["blocked"])
    elif expected == "graceful_error":
        # Accept any non-blocking error as graceful failure
        record["pass_case"] = bool(record["graceful_error"])
    else:
        # success: require no error + exact result match against gold SQL
        gold_sql = case.get("gold_sql")
        if not gold_sql:
            record["gold_error"] = "Missing gold_sql for success case"
            record["pass_case"] = False
        elif exception_msg:
            record["pass_case"] = False
        elif state is None or state.get("error") is not None:
            record["pass_case"] = False
        else:
            try:
                gold_cols, gold_rows = execute_sql_direct(db_path, gold_sql)
                record["gold_sql_executed"] = True
                record["gold_result"] = normalize_result(gold_cols, gold_rows)
            except Exception as e:
                record["gold_error"] = f"{type(e).__name__}: {e}"
                record["pass_case"] = False
            else:
                actual = record["actual_result"] or {"columns": [], "rows": []}
                gold = record["gold_result"] or {"columns": [], "rows": []}
                record["exact_columns_match"] = actual["columns"] == gold["columns"]
                record["exact_rows_match"] = actual["rows"] == gold["rows"]
                record["exact_result_match"] = bool(record["exact_columns_match"] and record["exact_rows_match"])
                record["pass_case"] = bool(record["exact_result_match"])

    record["failure_reason"] = infer_failure_reason(case, state, exception_msg)
    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    success_cases = [r for r in records if r["expect_behavior"] == "success"]
    blocked_cases = [r for r in records if r["expect_behavior"] == "blocked"]
    graceful_cases = [r for r in records if r["expect_behavior"] == "graceful_error"]

    exec_success_rate = (
        sum(1 for r in records if r["executed"]) / total if total else 0.0
    )

    exact_success_n = sum(1 for r in success_cases if r.get("exact_result_match") is True)
    exact_result_accuracy = exact_success_n / len(success_cases) if success_cases else None

    avg_retries = None
    retry_vals = [r["tries"] for r in records if isinstance(r.get("tries"), int)]
    if retry_vals:
        avg_retries = sum(retry_vals) / len(retry_vals)

    lat_vals = [r["latency_s"] for r in records if isinstance(r.get("latency_s"), (int, float))]
    avg_latency = sum(lat_vals) / len(lat_vals) if lat_vals else None

    pass_rate = sum(1 for r in records if r["pass_case"]) / total if total else 0.0
    blocked_pass_rate = (
        sum(1 for r in blocked_cases if r["pass_case"]) / len(blocked_cases)
        if blocked_cases else None
    )
    graceful_pass_rate = (
        sum(1 for r in graceful_cases if r["pass_case"]) / len(graceful_cases)
        if graceful_cases else None
    )

    failure_counter = Counter(r["failure_reason"] for r in records if not r["pass_case"])
    top_failure_reasons = [{"reason": k, "count": v} for k, v in failure_counter.most_common(10)]

    by_tier = {}
    for tier in sorted({r["tier"] for r in records}):
        xs = [r for r in records if r["tier"] == tier]
        by_tier[tier] = {
            "total": len(xs),
            "pass_rate": round(sum(1 for r in xs if r["pass_case"]) / len(xs), 4) if xs else None,
            "exec_success_rate": round(sum(1 for r in xs if r["executed"]) / len(xs), 4) if xs else None,
        }

    by_category = {}
    cats = sorted({r["category"] for r in records})
    for cat in cats:
        xs = [r for r in records if r["category"] == cat]
        by_category[cat] = {
            "total": len(xs),
            "pass_rate": round(sum(1 for r in xs if r["pass_case"]) / len(xs), 4) if xs else None,
        }

    return {
        "total_cases": total,
        "success_cases": len(success_cases),
        "blocked_cases": len(blocked_cases),
        "graceful_error_cases": len(graceful_cases),
        "overall_pass_rate": round(pass_rate, 4),
        "exec_success_rate": round(exec_success_rate, 4),
        "exact_result_accuracy": (round(exact_result_accuracy, 4) if exact_result_accuracy is not None else None),
        "average_retries": (round(avg_retries, 4) if avg_retries is not None else None),
        "avg_latency_s": (round(avg_latency, 4) if avg_latency is not None else None),
        "blocked_case_pass_rate": (round(blocked_pass_rate, 4) if blocked_pass_rate is not None else None),
        "graceful_error_pass_rate": (round(graceful_pass_rate, 4) if graceful_pass_rate is not None else None),
        "top_failure_reasons": top_failure_reasons,
        "by_tier": by_tier,
        "by_category": by_category,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("CHINOOK JSONL BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total cases           : {summary['total_cases']}")
    print(f"Success cases         : {summary['success_cases']}")
    print(f"Blocked cases         : {summary['blocked_cases']}")
    print(f"Graceful-error cases  : {summary['graceful_error_cases']}")
    print(f"Overall pass rate     : {summary['overall_pass_rate']:.1%}")
    print(f"Exec success rate     : {summary['exec_success_rate']:.1%}")
    era = summary['exact_result_accuracy']
    print(f"Exact result accuracy : {(f'{era:.1%}' if era is not None else 'N/A')}  (success cases)")
    ar = summary['average_retries']
    print(f"Average retries       : {(f'{ar:.3f}' if ar is not None else 'N/A')}")
    al = summary['avg_latency_s']
    print(f"Avg latency (s)       : {(f'{al:.3f}' if al is not None else 'N/A')}")
    print("\nTop failure reasons:")
    if summary["top_failure_reasons"]:
        for item in summary["top_failure_reasons"]:
            print(f"  - {item['reason']}: {item['count']}")
    else:
        print("  (none)")
    print("\nBy tier:")
    for tier, stats in summary["by_tier"].items():
        print(f"  Tier {tier}: total={stats['total']} pass_rate={stats['pass_rate']} exec_success={stats['exec_success_rate']}")
    print("=" * 70)


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_summary_json(summary: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def save_summary_csv(summary: dict[str, Any], path: Path) -> None:
    # Flatten key metrics only (single-row CSV)
    row = {
        "total_cases": summary["total_cases"],
        "success_cases": summary["success_cases"],
        "blocked_cases": summary["blocked_cases"],
        "graceful_error_cases": summary["graceful_error_cases"],
        "overall_pass_rate": summary["overall_pass_rate"],
        "exec_success_rate": summary["exec_success_rate"],
        "exact_result_accuracy": summary["exact_result_accuracy"],
        "average_retries": summary["average_retries"],
        "avg_latency_s": summary["avg_latency_s"],
        "blocked_case_pass_rate": summary["blocked_case_pass_rate"],
        "graceful_error_pass_rate": summary["graceful_error_pass_rate"],
        "top_failure_reasons": json.dumps(summary["top_failure_reasons"], ensure_ascii=False),
    }
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JSONL Chinook benchmark against sql_agent_v3.")
    parser.add_argument("--cases", type=str, default=str(HERE / "chinook_benchmark_cases.jsonl"),
                        help="Path to benchmark cases JSONL.")
    parser.add_argument("--outdir", type=str, default=str(HERE / "benchmark_outputs"),
                        help="Directory for output JSONL/JSON/CSV reports.")
    parser.add_argument("--ids", type=str, default="",
                        help="Comma-separated IDs to run (e.g., 1,2,10).")
    parser.add_argument("--tiers", type=str, default="",
                        help="Comma-separated tiers to run (e.g., A,B).")
    parser.add_argument("--categories", type=str, default="",
                        help="Comma-separated categories to run.")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(cases_path)

    if args.ids:
        wanted = {int(x.strip()) for x in args.ids.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in wanted]
    if args.tiers:
        wanted = {x.strip() for x in args.tiers.split(",") if x.strip()}
        cases = [c for c in cases if c.get("tier") in wanted]
    if args.categories:
        wanted = {x.strip() for x in args.categories.split(",") if x.strip()}
        cases = [c for c in cases if c.get("category") in wanted]

    db_path = agent.DB_PATH
    print(f"Running {len(cases)} cases against DB: {db_path}")
    print(f"Agent model config from sql_agent_v3.py is used as-is.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    records = []
    for idx, case in enumerate(cases, start=1):
        rec = evaluate_case(case, db_path)
        records.append(rec)
        status = "PASS" if rec["pass_case"] else "FAIL"
        print(f"[{idx:03d}/{len(cases):03d}] id={case['id']:>3} tier={case['tier']} cat={case['category']:<18} {status} "
              f"lat={rec['latency_s']:.2f}s tries={rec['tries']} reason={rec['failure_reason']}")

    summary = summarize(records)
    print_summary(summary)

    run_jsonl = outdir / f"run_{timestamp}.jsonl"
    summary_json = outdir / f"summary_{timestamp}.json"
    summary_csv = outdir / f"summary_{timestamp}.csv"

    save_jsonl(records, run_jsonl)
    save_summary_json(summary, summary_json)
    save_summary_csv(summary, summary_csv)

    print(f"\nSaved run log JSONL : {run_jsonl}")
    print(f"Saved summary JSON  : {summary_json}")
    print(f"Saved summary CSV   : {summary_csv}")


if __name__ == "__main__":
    main()
