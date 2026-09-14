"""Phase 2: incumbent (hosted LLM) evaluation on the Phase 2 LegalBench tasks.

This package is the ONLY place in legalbench_map that talks to a paid API.
Phase 1 (cli.py, core/, candidates/) must never import phase2 or litellm;
phase2 may import core (read-only data loading) but never the reverse.
"""
