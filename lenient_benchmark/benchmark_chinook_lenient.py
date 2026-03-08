"""
benchmark_chinook_v2.py
-----------------------
Comprehensive Chinook benchmark runner for sql_agent_v3.

Loads questions from chinook_benchmark_v2.json and evaluates the agent using
execution-accuracy matching against gold SQL results.

Outputs:
  - Console summary with per-tier and per-category breakdown
  - JSONL detailed results  (--out-jsonl results.jsonl)
  - CSV  detailed results   (--out-csv   results.csv)
  - JSON summary report     (--out-summary summary.json)

Eval modes:
  exec_match          — predicted result set == gold result set (set comparison)
  exec_match_rounded  — same, but numeric values rounded to 2 decimals
  row_count           — only checks that row count matches expected_rows
  exec_success        — query ran without error (for ambiguous questions)
  expect_blocked      — agent must block (security test)
  expect_graceful_fail — agent should fail gracefully (nonexistent field)

Usage:
  C:/Users/usp78/AppData/Local/Programs/Python/Python311/python.exe d:/SQL_agent/Claude_solution/benchmark_chinook_lenient.py --out-csv results.csv --out-jsonl results.jsonl --out-summary summary.json
  python benchmark_chinook_lenient.py --tier A --tier B
  python benchmark_chinook_lenient.py --ids C01,C02,D11
  python benchmark_chinook_lenient.py --out-csv results.csv --out-jsonl results.jsonl --out-summary summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ── Ensure sql_agent_v3 is importable ────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()
_ROOT = _HERE.parent
for _p in (_ROOT, _HERE):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

from sql_agent_v3 import DB_PATH, MODEL_NAME, MAX_TRIES, run_sql_agent  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# Load benchmark questions
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_FILE = _HERE / "chinook_benchmark_v2.json"


def load_cases(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Filter out comment/section marker objects
    return [c for c in raw if "id" in c]


# ─────────────────────────────────────────────────────────────────────────────
# Gold SQL execution
# ─────────────────────────────────────────────────────────────────────────────

def exec_gold(sql: str) -> list[tuple]:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    conn.close()
    return rows


def normalize_row(row: tuple, round_nums: bool = False) -> tuple:
    out = []
    for v in row:
        if isinstance(v, float) and round_nums:
            out.append(round(v, 2))
        elif isinstance(v, str):
            out.append(v.strip())
        else:
            out.append(v)
    return tuple(out)


def normalize_result_set(rows: list[tuple], round_nums: bool = False) -> set[tuple]:
    return {normalize_row(r, round_nums) for r in rows}


# ─────────────────────────────────────────────────────────────────────────────
# Single-case evaluator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_case(case: dict) -> dict:
    cid = case["id"]
    mode = case["eval_mode"]
    result = {
        "id": cid,
        "tier": case["tier"],
        "category": case["category"],
        "question": case["question"],
        "eval_mode": mode,
        "gold_sql": case.get("gold_sql"),
        "predicted_sql": "",
        "passed": False,
        "exec_success": False,
        "blocked": False,
        "error": "",
        "retries": 0,
        "latency_s": 0.0,
        "actual_rows": 0,
        "expected_rows": case.get("expected_rows"),
        "failure_reason": "",
    }

    t0 = time.perf_counter()
    try:
        state = run_sql_agent(case["question"])
    except Exception as e:
        result["latency_s"] = round(time.perf_counter() - t0, 3)
        result["error"] = str(e)
        result["failure_reason"] = "agent_exception"
        return result
    result["latency_s"] = round(time.perf_counter() - t0, 3)

    result["predicted_sql"] = state.get("sql", "")
    result["retries"] = state.get("tries", 0)
    agent_err = state.get("error")
    agent_result = state.get("result")

    # ── Security tests: expect_blocked ────────────────────────────────────
    if mode == "expect_blocked":
        result["blocked"] = bool(agent_err)
        result["passed"] = bool(agent_err)
        if not result["passed"]:
            result["failure_reason"] = "security_not_blocked"
        return result

    # ── Graceful fail tests ───────────────────────────────────────────────
    if mode == "expect_graceful_fail":
        # Pass if agent returned an error OR returned empty results
        has_err = bool(agent_err)
        empty = agent_result is None or len(agent_result.get("rows", [])) == 0
        result["passed"] = has_err or empty
        result["exec_success"] = not has_err and agent_result is not None
        if not result["passed"]:
            result["failure_reason"] = "should_have_failed_gracefully"
        return result

    # ── Normal execution ──────────────────────────────────────────────────
    if agent_err:
        result["error"] = agent_err
        result["failure_reason"] = "exec_error"
        return result

    result["exec_success"] = True
    pred_rows = agent_result.get("rows", []) if agent_result else []
    result["actual_rows"] = len(pred_rows)

    # ── exec_success: just needs to run ───────────────────────────────────
    if mode == "exec_success":
        result["passed"] = True
        return result

    # ── row_count: check count only ───────────────────────────────────────
    if mode == "row_count":
        expected = case.get("expected_rows", 0)
        result["passed"] = len(pred_rows) == expected
        if not result["passed"]:
            result["failure_reason"] = f"row_count_{len(pred_rows)}_vs_{expected}"
        return result

    # ── exec_match / exec_match_rounded ───────────────────────────────────
    gold_sql = case.get("gold_sql")
    if not gold_sql:
        result["failure_reason"] = "no_gold_sql"
        return result

    round_nums = mode == "exec_match_rounded"
    try:
        gold_rows = exec_gold(gold_sql)
    except Exception as e:
        result["failure_reason"] = f"gold_sql_error: {e}"
        return result

    gold_set = normalize_result_set(gold_rows, round_nums)
    pred_set = normalize_result_set([tuple(r) for r in pred_rows], round_nums)

    result["passed"] = gold_set == pred_set
    if not result["passed"]:
        # Check if it's a row count issue or value mismatch
        if len(gold_set) != len(pred_set):
            result["failure_reason"] = f"row_count_{len(pred_set)}_vs_{len(gold_set)}"
        else:
            result["failure_reason"] = "value_mismatch"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    exec_ok = sum(1 for r in results if r["exec_success"])
    blocked_cases = [r for r in results if r["eval_mode"] == "expect_blocked"]
    blocked_ok = sum(1 for r in blocked_cases if r["passed"])
    normal = [r for r in results if r["eval_mode"] not in ("expect_blocked", "expect_graceful_fail")]
    fp_cases = [r for r in results if r["category"] == "security_false_positive"]
    fp_ok = sum(1 for r in fp_cases if r["exec_success"])

    avg_retries = sum(r["retries"] for r in normal) / len(normal) if normal else 0
    avg_latency = sum(r["latency_s"] for r in results) / total if total else 0

    # Failure reasons
    failures = [r for r in results if not r["passed"]]
    reason_counts = Counter(r["failure_reason"] for r in failures if r["failure_reason"])

    # Per-tier
    tiers = sorted(set(r["tier"] for r in results))
    tier_stats = {}
    for t in tiers:
        t_results = [r for r in results if r["tier"] == t]
        t_passed = sum(1 for r in t_results if r["passed"])
        tier_stats[t] = {"total": len(t_results), "passed": t_passed,
                         "rate": t_passed / len(t_results) if t_results else 0}

    # Per-category
    cats = sorted(set(r["category"] for r in results))
    cat_stats = {}
    for c in cats:
        c_results = [r for r in results if r["category"] == c]
        c_passed = sum(1 for r in c_results if r["passed"])
        cat_stats[c] = {"total": len(c_results), "passed": c_passed,
                        "rate": c_passed / len(c_results) if c_results else 0}

    # ── Print ─────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  CHINOOK BENCHMARK v2 — SUMMARY")
    print("═" * 70)
    print(f"  Model              : {MODEL_NAME}")
    print(f"  Max retries        : {MAX_TRIES}")
    print(f"  Database           : {DB_PATH}")
    print(f"  Total cases        : {total}")
    print(f"  Overall pass rate  : {passed}/{total} ({100*passed/total:.1f}%)")
    print(f"  Exec success rate  : {exec_ok}/{len(normal)} normal cases ({100*exec_ok/len(normal):.1f}%)" if normal else "")
    if blocked_cases:
        print(f"  Security block rate: {blocked_ok}/{len(blocked_cases)} ({100*blocked_ok/len(blocked_cases):.1f}%)")
    if fp_cases:
        print(f"  False positive rate: {len(fp_cases)-fp_ok}/{len(fp_cases)} blocked incorrectly")
    print(f"  Avg retries/case   : {avg_retries:.2f}")
    print(f"  Avg latency        : {avg_latency:.1f}s")

    print("\n  Per Tier:")
    for t in tiers:
        s = tier_stats[t]
        print(f"    Tier {t}: {s['passed']}/{s['total']} ({100*s['rate']:.1f}%)")

    print("\n  Per Category:")
    for c in cats:
        s = cat_stats[c]
        print(f"    {c:<25s} {s['passed']}/{s['total']} ({100*s['rate']:.1f}%)")

    if reason_counts:
        print("\n  Top Failure Reasons:")
        for reason, cnt in reason_counts.most_common(10):
            print(f"    {reason:<40s} {cnt}")

    print("═" * 70)

    # Build summary dict
    summary = {
        "model": MODEL_NAME,
        "max_retries": MAX_TRIES,
        "db_path": DB_PATH,
        "total_cases": total,
        "passed": passed,
        "pass_rate": round(100 * passed / total, 2) if total else 0,
        "exec_success_rate": round(100 * exec_ok / len(normal), 2) if normal else 0,
        "security_block_rate": round(100 * blocked_ok / len(blocked_cases), 2) if blocked_cases else None,
        "avg_retries": round(avg_retries, 2),
        "avg_latency_s": round(avg_latency, 2),
        "tier_stats": tier_stats,
        "category_stats": cat_stats,
        "top_failure_reasons": dict(reason_counts.most_common(10)),
    }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# File output
# ─────────────────────────────────────────────────────────────────────────────

def save_jsonl(results: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  JSONL saved to {path}")


def save_csv(results: list[dict], path: str):
    fields = [
        "id", "tier", "category", "question", "eval_mode",
        "passed", "exec_success", "blocked", "error",
        "predicted_sql", "gold_sql", "retries", "latency_s",
        "actual_rows", "expected_rows", "failure_reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"  CSV  saved to {path}")


def save_summary(summary: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary saved to {path}")

def _resolve_out(path_str: str | None) -> str | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = _HERE / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chinook Benchmark v2 Runner")
    parser.add_argument("--benchmark", default=str(BENCHMARK_FILE),
                        help="Path to benchmark JSON file")
    parser.add_argument("--tier", action="append", default=None,
                        help="Filter by tier (A/B/C/D). Can repeat: --tier A --tier B")
    parser.add_argument("--category", action="append", default=None,
                        help="Filter by category. Can repeat.")
    parser.add_argument("--ids", default=None,
                        help="Comma-separated case IDs (e.g. A01,B05,D11)")
    parser.add_argument("--out-jsonl", default=None, help="Output JSONL file path")
    parser.add_argument("--out-csv", default=None, help="Output CSV file path")
    parser.add_argument("--out-summary", default=None, help="Output summary JSON path")
    args = parser.parse_args()
    args.out_jsonl = _resolve_out(args.out_jsonl)
    args.out_csv = _resolve_out(args.out_csv)
    args.out_summary = _resolve_out(args.out_summary)

    # Load
    cases = load_cases(Path(args.benchmark))
    print(f"Loaded {len(cases)} cases from {args.benchmark}")

    # Filter
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    if args.tier:
        tiers = {t.upper() for t in args.tier}
        cases = [c for c in cases if c["tier"] in tiers]
    if args.category:
        cats = set(args.category)
        cases = [c for c in cases if c["category"] in cats]

    if not cases:
        print("No cases selected. Check your filters.")
        return

    print(f"Running {len(cases)} cases | Model: {MODEL_NAME} | DB: {DB_PATH}\n")

    # Run
    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i:>3}/{len(cases)}] {case['id']:>4} ({case['tier']}/{case['category']:<20}) "
              f"{case['question'][:55]}...")
        r = evaluate_case(case)

        icon = "✅" if r["passed"] else ("🔒" if r["blocked"] else "❌")
        extra = f" | {r['failure_reason']}" if r["failure_reason"] and not r["passed"] else ""
        print(f"        {icon} {r['latency_s']:.1f}s retries={r['retries']}{extra}")
        results.append(r)

    # Report
    summary = print_report(results)

    # Save outputs
    if args.out_jsonl:
        save_jsonl(results, args.out_jsonl)
    if args.out_csv:
        save_csv(results, args.out_csv)
    if args.out_summary:
        save_summary(summary, args.out_summary)

    if not any([args.out_jsonl, args.out_csv, args.out_summary]):
        print("\n  Tip: use --out-csv, --out-jsonl, --out-summary to save results.")


if __name__ == "__main__":
    main()
