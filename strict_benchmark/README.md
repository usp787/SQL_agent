# Chinook JSONL Benchmark Package

Files:
- `chinook_benchmark_cases.jsonl` — 101 benchmark cases (Tier A/B/C/D)
- `benchmark_chinook_eval_jsonl.py` — evaluation runner for `sql_agent_v3.py`

## Tiers
- Tier A: basic single-table / aggregates / top-k
- Tier B: joins / group by + having / date filters / multi-condition / distinct
- Tier C: nested subqueries / for-each-X-find-Y / ratios / multi-hop joins
- Tier D: robustness (ambiguous, synonym-heavy, column mismatch, nonexistent fields, dangerous wording)

## Outputs
The runner writes:
- per-case run JSONL
- summary JSON
- summary CSV

## Example
```bash
python benchmark_chinook_eval_jsonl.py --cases chinook_benchmark_cases.jsonl
python benchmark_chinook_eval_jsonl.py --tiers A,B
python benchmark_chinook_eval_jsonl.py --categories security,dangerous_wording
```
