Optional / Nice-to-Have
Expand benchmark dataset — Add cases from other databases (Northwind, AdventureWorks) to measure generalization beyond Chinook
RAG quality improvements — Index column descriptions and sample values alongside DDL to improve retrieval for ambiguous column names
Result caching — Cache SQL for identical or near-identical questions (embedding similarity threshold) to reduce latency for repeated queries
Web UI — Add a minimal FastAPI + simple frontend so non-technical users can interact without CLI
Schema change detection — Auto-invalidate and rebuild the ChromaDB index when the database schema changes (currently requires manual force_rebuild=True)
Benchmark CI integration — Run the strict benchmark as a GitHub Actions job on every PR to catch accuracy regressions before merge