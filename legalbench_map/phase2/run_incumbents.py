"""Incumbent runner: score hosted LLMs on the Phase 2 LegalBench tasks.

THE ONLY PHASE 2 COMPONENT THAT SPENDS MONEY.

Produces FILE B of the Phase 2 data contract:

    <results_dir>/incumbent/<task>__<model_key>.csv
    columns (exact order): task,model_key,item_idx,text_sha256,gold,raw_output,
                           pred,parse_ok,in_tokens,out_tokens,cached_tokens,
                           cost_usd,latency_s,ts

One row per item_idx of the TEST split (item_idx = 0-based position exactly as
returned by core.data.load_task_data). Rows are appended and flushed one at a
time, so a crash loses at most the in-flight call; on startup existing rows are
read and completed (task, model_key, item_idx) are skipped. A row whose
raw_output is '<error: Name>' (a permanently failed call) is NOT completed: on
startup such rows are dropped from the file and their items retried. Within a task,
identical text (same text_sha256) is sent to the API ONCE and the result is
fanned out to every item_idx that shares it.

Prompting: data/prompts/<task>.txt with the literal '{{text}}' replaced by the
item text, sent as ONE user message, temperature=0. Only the response's
'content' field is parsed -- never reasoning_content or any thinking field.

Scoring: pred = phase2.legalbench_eval.normalize(raw_output) (the verbatim
LegalBench exact-match normalizer). parse_ok = 1 iff pred is one of the task's
class labels; otherwise pred is STILL the normalized raw string, never coerced.

Budget guard: --max-cost-usd (default 5.0). Before every call, the sum of
cost_usd over every existing row in this run's (tasks x models) files plus this
run's spend (completed + reserved-in-flight estimates) plus the estimated cost
of the next call must not exceed the cap; if it would, no further calls are
started, in-flight calls are drained (their rows are written), the total is
printed, and the process exits 3.

Usage:
    python phase2/run_incumbents.py --dry-run
    python phase2/run_incumbents.py --limit 3 --tasks opp115_data_retention
    python phase2/run_incumbents.py            # full run (3 tasks x 3 models)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
DOWNSHIFT_SRC = "/Users/vasyl/zadumai/src"
DOTENV_PATH = "/Users/vasyl/zadumai/.env"
for _p in (str(ROOT), DOWNSHIFT_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.data import load_registry, load_task_data  # noqa: E402
from phase2.legalbench_eval import normalize  # noqa: E402

logger = logging.getLogger("legalbench_map.phase2.run_incumbents")

DEFAULT_TASKS = [
    "opp115_data_retention",
    "supply_chain_disclosure_best_practice_audits",
    "learned_hands_consumer",
]
DEFAULT_MODELS = ["glm-5.3-flash", "gpt-oss-120b-fireworks", "qwen3.7-plus"]

COLUMNS = [
    "task", "model_key", "item_idx", "text_sha256", "gold", "raw_output", "pred",
    "parse_ok", "in_tokens", "out_tokens", "cached_tokens", "cost_usd", "latency_s", "ts",
]

EXIT_OK = 0
EXIT_FATAL = 2
EXIT_BUDGET = 3

MAX_TRIES = 6
BACKOFF_CAP_S = 60.0
CALL_TIMEOUT_S = 180


# --------------------------------------------------------------------------
# Pure helpers (unit-tested without network)
# --------------------------------------------------------------------------
def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_instruction_for(classes) -> str:
    """One generic output-format line derived ONLY from the task's own label
    set -- never hand-written per task. Exists because modern chat/thinking
    models do not continue a 2023 completion-style few-shot template with a
    bare label: the pilot showed GLM-5.3 answering '**Label: Yes**' plus a
    rationale and gpt-oss echoing 'Label: No', both scored wrong by
    LegalBench's exact-match rule. This is output-format specification, not
    prompt optimization: it names no label semantics and is identical in
    shape for every task."""
    labels = [str(c).strip().capitalize() for c in classes]
    if len(labels) <= 1:
        joined = labels[0] if labels else ""
    elif len(labels) == 2:
        joined = f"{labels[0]} or {labels[1]}"
    else:
        joined = ", ".join(labels[:-1]) + f", or {labels[-1]}"
    return f"Answer with exactly one word: {joined}."


def build_prompt(template: str, text: str, instruction: str | None = None) -> str:
    """Substitute {{text}}; if an instruction is given, prepend it as the
    first line of the message (blank line before the verbatim template)."""
    if "{{text}}" not in template:
        raise ValueError("prompt template has no '{{text}}' placeholder")
    body = template.replace("{{text}}", text)
    return f"{instruction}\n\n{body}" if instruction else body


def parse_pred(raw_output: str, classes) -> tuple[str, int]:
    """LegalBench-normalize the raw content; parse_ok=1 iff it is a class label.
    On a miss, pred is still the normalized raw string (never coerced)."""
    pred = normalize(raw_output if raw_output is not None else "", stem=False)
    return pred, int(pred in set(classes))


def _cached_rate(spec) -> float:
    pc = getattr(spec, "price_cached", None)
    if pc is None or pc < 0:
        return float(spec.price_in)
    return float(pc)


def compute_cost(spec, in_tokens: int, out_tokens: int, cached_tokens: int = 0) -> float:
    """cost = in*price_in + out*price_out (per 1M), except cached prompt tokens
    (a subset of in_tokens, as reported by usage.prompt_tokens_details) are
    priced at price_cached when the registry has one."""
    cached = max(0, min(int(cached_tokens or 0), int(in_tokens or 0)))
    fresh = int(in_tokens or 0) - cached
    return (
        fresh * float(spec.price_in) / 1e6
        + cached * _cached_rate(spec) / 1e6
        + int(out_tokens or 0) * float(spec.price_out) / 1e6
    )


def estimate_call_cost(spec, prompt_tokens: int, max_tokens: int) -> float:
    """Pre-call upper-bound estimate used by the budget guard and --dry-run:
    every prompt token at the fresh rate, max_tokens of output."""
    return prompt_tokens * float(spec.price_in) / 1e6 + max_tokens * float(spec.price_out) / 1e6


def _get(obj: Any, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def extract_content(response: Any) -> str:
    """The response's 'content' string, verbatim ('' if None). Never looks at
    reasoning_content / thinking fields."""
    choices = _get(response, "choices") or []
    if not choices:
        return ""
    message = _get(choices[0], "message")
    content = _get(message, "content")
    if content is None:
        return ""
    return str(content)


def extract_usage(response: Any) -> tuple[int, int, int]:
    usage = _get(response, "usage")
    in_tok = int(_get(usage, "prompt_tokens", 0) or 0)
    out_tok = int(_get(usage, "completion_tokens", 0) or 0)
    details = _get(usage, "prompt_tokens_details")
    cached = int(_get(details, "cached_tokens", 0) or 0)
    return in_tok, out_tok, cached


def _status_code(exc: BaseException):
    for attr in ("status_code", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(exc, "response", None)
    v = getattr(resp, "status_code", None)
    return v if isinstance(v, int) else None


def is_retryable(exc: BaseException) -> bool:
    """429 / 5xx / transport-level failures get exponential backoff."""
    code = _status_code(exc)
    if code is not None:
        return code == 429 or code >= 500
    name = type(exc).__name__
    return name in {
        "RateLimitError", "ServiceUnavailableError", "InternalServerError",
        "APIConnectionError", "Timeout", "APITimeoutError", "APIError",
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# API call with retry
# --------------------------------------------------------------------------
def call_model(
    completion_fn: Callable[..., Any],
    spec,
    prompt: str,
    max_tokens: int,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    extra_request: dict | None = None,
) -> dict:
    """One item's API call. Returns a dict with raw_output, in/out/cached
    tokens, latency_s, ts, error (ExcName or ''). Never raises.
    extra_request (e.g. {"extra_body": {"reasoning_effort": "low"}}) is merged
    over spec.extra_body."""
    rng = rng or random.Random()
    extra = dict(getattr(spec, "extra_body", None) or {})
    extra.update(extra_request or {})
    t0 = time.perf_counter()
    last_exc: BaseException | None = None
    for attempt in range(MAX_TRIES):
        try:
            response = completion_fn(
                model=spec.call_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
                timeout=CALL_TIMEOUT_S,
                **extra,
            )
            in_tok, out_tok, cached = extract_usage(response)
            return {
                "raw_output": extract_content(response),
                "in_tokens": in_tok,
                "out_tokens": out_tok,
                "cached_tokens": cached,
                "latency_s": time.perf_counter() - t0,
                "ts": now_iso(),
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001 - the batch must never crash
            last_exc = exc
            if is_retryable(exc) and attempt < MAX_TRIES - 1:
                delay = min(BACKOFF_CAP_S, 2.0 ** attempt) * (0.5 + rng.random())
                logger.warning(
                    "retryable %s on %s (attempt %d/%d, status=%s); sleeping %.1fs",
                    type(exc).__name__, spec.call_id, attempt + 1, MAX_TRIES,
                    _status_code(exc), delay,
                )
                sleep_fn(delay)
                continue
            break
    name = type(last_exc).__name__ if last_exc is not None else "UnknownError"
    logger.error("call failed permanently on %s: %s: %s", spec.call_id, name,
                 str(last_exc)[:200].replace("\n", " "))
    return {
        "raw_output": f"<error: {name}>",
        "in_tokens": 0,
        "out_tokens": 0,
        "cached_tokens": 0,
        "latency_s": time.perf_counter() - t0,
        "ts": now_iso(),
        "error": name,
    }


# --------------------------------------------------------------------------
# FILE B I/O
# --------------------------------------------------------------------------
def incumbent_path(results_dir: Path, task: str, model_key: str) -> Path:
    return Path(results_dir) / "incumbent" / f"{task}__{model_key}.csv"


def read_existing_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        if list(reader.fieldnames) != COLUMNS:
            raise ValueError(f"{path}: header {reader.fieldnames} != contract columns {COLUMNS}")
        rows = [r for r in reader if r.get("item_idx", "") != ""]
    return rows


def is_error_row(row: dict) -> bool:
    """A row written for a permanently failed call (raw_output '<error: Name>',
    pred '', cost 0). It is NOT a completed item: resume must retry it."""
    return str(row.get("raw_output", "")).startswith("<error:")


def compact_error_rows(path: Path, existing: list[dict]) -> tuple[list[dict], int]:
    """Drop API-error rows from an existing FILE B so the retried items get a
    fresh row without breaking the one-row-per-item_idx invariant. Rewrites
    the file atomically (tmp + os.replace) only when there is something to
    drop. Returns (kept rows, number dropped)."""
    good = [r for r in existing if not is_error_row(r)]
    n_err = len(existing) - len(good)
    if n_err:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(good)
        os.replace(tmp, path)
    return good, n_err


def sum_existing_cost(results_dir: Path, tasks, model_keys) -> float:
    total = 0.0
    for task in tasks:
        for mk in model_keys:
            for r in read_existing_rows(incumbent_path(results_dir, task, mk)):
                try:
                    total += float(r.get("cost_usd") or 0.0)
                except ValueError:
                    pass
    return total


class RowWriter:
    """Append-only CSV writer with a header-on-create and flush-per-row."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        need_header = not path.exists() or path.stat().st_size == 0
        self._f = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=COLUMNS, extrasaction="ignore")
        if need_header:
            self._w.writeheader()
            self._f.flush()
        self._lock = threading.Lock()

    def write(self, row: dict) -> None:
        with self._lock:
            self._w.writerow(row)
            self._f.flush()

    def close(self) -> None:
        with self._lock:
            self._f.close()


# --------------------------------------------------------------------------
# Tokenizer (dry-run + budget estimates)
# --------------------------------------------------------------------------
_ENC = None


def count_tokens(text: str) -> int:
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("o200k_base")
    return len(_ENC.encode(text, disallowed_special=()))


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
class BudgetExceeded(Exception):
    def __init__(self, projected: float, cap: float):
        super().__init__(f"projected spend ${projected:.4f} would exceed --max-cost-usd {cap:.4f}")
        self.projected = projected
        self.cap = cap


class Budget:
    """Thread-safe spend ledger: base (rows already on disk for this run's
    tasks x models) + completed this run + reserved estimates for in-flight."""

    def __init__(self, base: float, cap: float):
        self.base = base
        self.cap = cap
        self.spent = 0.0
        self.reserved = 0.0
        self._lock = threading.Lock()

    @property
    def total(self) -> float:
        return self.base + self.spent + self.reserved

    def reserve(self, estimate: float) -> None:
        with self._lock:
            projected = self.base + self.spent + self.reserved + estimate
            if projected > self.cap:
                raise BudgetExceeded(projected, self.cap)
            self.reserved += estimate

    def settle(self, estimate: float, actual: float) -> None:
        with self._lock:
            self.reserved -= estimate
            self.spent += actual


class PairStats:
    def __init__(self, task: str, model_key: str, n_total: int, n_done_before: int):
        self.task, self.model_key = task, model_key
        self.n_total = n_total
        self.n_done = n_done_before
        self.n_new = 0
        self.parse_ok = 0
        self.in_tok = self.out_tok = self.cached = 0
        self.cost = 0.0
        self.errors = 0

    def add(self, row: dict) -> None:
        self.n_done += 1
        self.n_new += 1
        self.parse_ok += int(row["parse_ok"])
        self.in_tok += int(row["in_tokens"])
        self.out_tok += int(row["out_tokens"])
        self.cached += int(row["cached_tokens"])
        self.cost += float(row["cost_usd"])
        if str(row["raw_output"]).startswith("<error:"):
            self.errors += 1

    def log(self, cumulative_cost: float) -> None:
        n = max(1, self.n_new)
        logger.info(
            "task=%s model=%s done=%d/%d new=%d parse_ok=%.1f%% errors=%d "
            "mean_in=%.0f mean_out=%.1f mean_cached=%.0f pair_cost=$%.4f cum_cost=$%.4f",
            self.task, self.model_key, self.n_done, self.n_total, self.n_new,
            100.0 * self.parse_ok / n, self.errors,
            self.in_tok / n, self.out_tok / n, self.cached / n, self.cost, cumulative_cost,
        )


def _load_prompt_template(prompts_dir: Path, task: str) -> str:
    path = prompts_dir / f"{task}.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _select_entries(registry_entries, tasks):
    by_name = {e.get("task"): e for e in registry_entries}
    missing = [t for t in tasks if t not in by_name]
    if missing:
        raise KeyError(f"task(s) not in registry: {missing}")
    return [by_name[t] for t in tasks]


def run_pair(
    *,
    task_data,
    template: str,
    spec,
    model_key: str,
    results_dir: Path,
    limit: int | None,
    max_tokens: int,
    concurrency: int,
    budget: Budget,
    completion_fn,
    sleep_fn=time.sleep,
    seed: int = 0,
    extra_request: dict | None = None,
    instruction: str | None = None,
) -> PairStats:
    task = task_data.spec.task
    classes = task_data.spec.classes
    X, y = task_data.X_test, task_data.y_test
    n_items = len(X) if limit is None else min(limit, len(X))
    shas = [text_sha256(X[i]) for i in range(n_items)]

    path = incumbent_path(results_dir, task, model_key)
    existing, n_err = compact_error_rows(path, read_existing_rows(path))
    if n_err:
        logger.warning("task=%s model=%s: %d API-error row(s) (<error: ...>) dropped from %s; "
                       "those items will be retried", task, model_key, n_err, path)
    done: dict[int, dict] = {}
    for r in existing:
        if r["task"] != task or r["model_key"] != model_key:
            continue
        idx = int(r["item_idx"])
        if idx < n_items and r["text_sha256"] != shas[idx]:
            raise ValueError(
                f"{path}: item_idx={idx} text_sha256 on disk != current data; "
                "the dataset or item order changed -- refusing to resume"
            )
        done[idx] = r
    stats = PairStats(task, model_key, n_items, sum(1 for i in done if i < n_items))
    writer = RowWriter(path)

    # Group undone items by text_sha256 (dedupe API calls within a task).
    groups: dict[str, list[int]] = {}
    for i in range(n_items):
        if i in done:
            continue
        groups.setdefault(shas[i], []).append(i)

    # Resume shortcut: an undone item whose text already has a scored row in
    # this file gets a fan-out copy instead of a new call.
    by_sha_done = {}
    for i, r in done.items():
        by_sha_done.setdefault(r["text_sha256"], r)
    for sha in list(groups):
        src = by_sha_done.get(sha)
        if src is None:
            continue
        for i in groups.pop(sha):
            row = dict(src)
            row["item_idx"] = i
            row["gold"] = y[i]
            writer.write(row)
            done[i] = row
            stats.add(row)
            logger.info("task=%s model=%s item_idx=%d fanned out from existing row (same text)",
                        task, model_key, i)

    def make_rows(sha: str, idxs: list[int], result: dict) -> list[dict]:
        pred, ok = parse_pred(result["raw_output"], classes)
        if result["error"]:
            pred, ok = "", 0
        cost = compute_cost(spec, result["in_tokens"], result["out_tokens"], result["cached_tokens"])
        rows = []
        for i in idxs:
            rows.append({
                "task": task,
                "model_key": model_key,
                "item_idx": i,
                "text_sha256": sha,
                "gold": y[i],
                "raw_output": result["raw_output"],
                "pred": pred,
                "parse_ok": ok,
                "in_tokens": result["in_tokens"],
                "out_tokens": result["out_tokens"],
                "cached_tokens": result["cached_tokens"],
                "cost_usd": f"{cost:.10f}",
                "latency_s": f"{result['latency_s']:.3f}",
                "ts": result["ts"],
            })
        return rows

    work = list(groups.items())  # deterministic order: first item_idx ascending
    work.sort(key=lambda kv: kv[1][0])
    rng = random.Random(seed)
    budget_hit: BudgetExceeded | None = None
    pending = {}
    log_every = 25
    since_log = 0

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        while work or pending:
            while work and len(pending) < max(1, concurrency) and budget_hit is None:
                sha, idxs = work[0]
                prompt = build_prompt(template, X[idxs[0]], instruction)
                est = estimate_call_cost(spec, count_tokens(prompt), max_tokens)
                try:
                    budget.reserve(est)
                except BudgetExceeded as e:
                    budget_hit = e
                    break
                work.pop(0)
                fut = pool.submit(
                    call_model, completion_fn, spec, prompt, max_tokens,
                    sleep_fn=sleep_fn, rng=random.Random(rng.random()),
                    extra_request=extra_request,
                )
                pending[fut] = (sha, idxs, est)
            if budget_hit is not None and not pending:
                break
            if not pending:
                continue
            finished, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in finished:
                sha, idxs, est = pending.pop(fut)
                result = fut.result()
                rows = make_rows(sha, idxs, result)
                actual = float(rows[0]["cost_usd"])
                budget.settle(est, actual)
                for row in rows:
                    writer.write(row)
                    done[row["item_idx"]] = row
                    stats.add(row)
                since_log += len(rows)
                if since_log >= log_every:
                    stats.log(budget.total)
                    since_log = 0
    writer.close()
    stats.log(budget.total)
    if budget_hit is not None:
        raise budget_hit
    return stats


def dry_run(*, entries, model_specs: dict, prompts_dir: Path, results_dir: Path,
            limit, max_tokens, task_data_loader, out=None, format_instruction: bool = False) -> float:
    """Render every prompt, count tokens (tiktoken o200k_base), estimate cost
    per (task, model) from registry prices. ZERO API calls."""
    out = out if out is not None else sys.stdout
    rows = []
    grand = 0.0
    for entry in entries:
        td = task_data_loader(entry)
        task = td.spec.task
        template = _load_prompt_template(prompts_dir, task)
        n_items = len(td.X_test) if limit is None else min(limit, len(td.X_test))
        uniq: dict[str, int] = {}
        for i in range(n_items):
            sha = text_sha256(td.X_test[i])
            if sha not in uniq:
                uniq[sha] = count_tokens(build_prompt(
                    template, td.X_test[i],
                    format_instruction_for(td.spec.classes) if format_instruction else None))
        n_calls = len(uniq)
        tok_in = sum(uniq.values())
        for mk, spec in model_specs.items():
            done = sum(
                1 for r in read_existing_rows(incumbent_path(results_dir, task, mk))
                if int(r["item_idx"]) < n_items
            )
            cost = tok_in * spec.price_in / 1e6 + n_calls * max_tokens * spec.price_out / 1e6
            grand += cost
            rows.append((task, mk, n_items, n_calls, done, tok_in, tok_in / max(1, n_calls),
                         n_calls * max_tokens, cost))
    hdr = f"{'task':<46} {'model':<24} {'items':>6} {'calls':>6} {'done':>5} {'in_tok':>9} {'in/call':>8} {'max_out':>8} {'est_usd':>9}"
    print(hdr, file=out)
    print("-" * len(hdr), file=out)
    for task, mk, n_items, n_calls, done, tok_in, per, max_out, cost in rows:
        print(f"{task:<46} {mk:<24} {n_items:>6} {n_calls:>6} {done:>5} {tok_in:>9} {per:>8.0f} {max_out:>8} {cost:>9.4f}", file=out)
    print("-" * len(hdr), file=out)
    print(f"{'GRAND TOTAL (upper bound: all prompt tokens fresh, max_tokens output per call)':<100} {grand:>9.4f}", file=out)
    return grand


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 2 incumbent runner (spends money; see --dry-run)")
    p.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS))
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    p.add_argument("--limit", type=int, default=None, help="only item_idx < N per task")
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument("--max-cost-usd", type=float, default=5.0)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--results-dir", type=str, default=str(ROOT / "results"))
    p.add_argument("--seed", type=int, default=0, help="ignored by the API; recorded in the run log")
    p.add_argument("--format-instruction", action="store_true", default=False,
                   help="prepend one generic line, 'Answer with exactly one word: <labels>.', built from the "
                        "task's own label set (see format_instruction_for). Off = the verbatim 2023 template.")
    p.add_argument("--reasoning-effort", type=str, default=None,
                   help="if set, sent as extra_body={'reasoning_effort': VALUE} (Fireworks: low|medium|high; "
                        "'none' is rejected by thinking-only models such as GLM-5.3). Default: not sent.")
    p.add_argument("--registry", type=str, default=str(ROOT / "data" / "task_registry.json"))
    p.add_argument("--prompts-dir", type=str, default=str(ROOT / "data" / "prompts"))
    p.add_argument("--log-level", type=str, default="INFO")
    return p


def _check_env_keys(specs: dict) -> list[str]:
    missing = []
    for mk, spec in specs.items():
        key = getattr(spec, "env_key", "") or ""
        if not key:
            continue
        val = os.environ.get(key)
        if not val:
            missing.append(f"{mk} needs env var {key} (not set)")
        else:
            logger.info("model=%s env key %s present (len=%d)", mk, key, len(val))
    return missing


def main(argv=None, *, completion_fn=None, task_data_loader=None, by_key=None,
         sleep_fn=time.sleep) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    for noisy in ("LiteLLM", "litellm", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    results_dir = Path(args.results_dir)
    prompts_dir = Path(args.prompts_dir)

    if by_key is None:
        from downshift.models import BY_KEY as by_key  # noqa: N813
    unknown = [m for m in model_keys if m not in by_key]
    if unknown:
        print(f"FATAL: unknown model key(s) {unknown}; known: {sorted(by_key)[:20]}...", file=sys.stderr)
        return EXIT_FATAL
    specs = {m: by_key[m] for m in model_keys}

    try:
        entries = _select_entries(load_registry(args.registry), tasks)
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: {e!r}", file=sys.stderr)
        return EXIT_FATAL
    for t in tasks:
        if not (prompts_dir / f"{t}.txt").exists():
            print(f"FATAL: no prompt file for task {t} in {prompts_dir}", file=sys.stderr)
            return EXIT_FATAL

    loader = task_data_loader or load_task_data

    if args.dry_run:
        logger.info("DRY RUN: zero API calls will be made")
        dry_run(entries=entries, model_specs=specs, prompts_dir=prompts_dir,
                results_dir=results_dir, limit=args.limit, max_tokens=args.max_tokens,
                task_data_loader=loader, format_instruction=args.format_instruction)
        return EXIT_OK

    # Live path: key + client.
    try:
        from dotenv import load_dotenv
        load_dotenv(DOTENV_PATH, override=False)
    except Exception:  # noqa: BLE001
        pass
    missing = _check_env_keys(specs)
    if missing:
        for m in missing:
            print(f"FATAL: {m}", file=sys.stderr)
        return EXIT_FATAL
    if completion_fn is None:
        import litellm
        litellm.suppress_debug_info = True
        litellm.telemetry = False
        completion_fn = litellm.completion

    extra_request = {"extra_body": {"reasoning_effort": args.reasoning_effort}} if args.reasoning_effort else None
    if extra_request:
        logger.info("every call will carry extra_body=%s", extra_request["extra_body"])

    base_cost = sum_existing_cost(results_dir, tasks, model_keys)
    budget = Budget(base=base_cost, cap=args.max_cost_usd)
    logger.info("budget: existing rows for this run's %d task(s) x %d model(s) sum to $%.4f; cap=$%.2f; seed=%d",
                len(tasks), len(model_keys), base_cost, args.max_cost_usd, args.seed)

    run_log = results_dir / "incumbent" / "run_log.jsonl"
    run_log.parent.mkdir(parents=True, exist_ok=True)
    started = now_iso()

    exit_code = EXIT_OK
    budget_err: BudgetExceeded | None = None
    for entry in entries:
        td = loader(entry)
        task = td.spec.task
        template = _load_prompt_template(prompts_dir, task)
        instruction = format_instruction_for(td.spec.classes) if args.format_instruction else None
        if instruction:
            logger.info("task=%s format instruction: %r", task, instruction)
        logger.info("task=%s n_test=%d dropped_test=%d n_unique_text=%d",
                    task, len(td.X_test), td.dropped_test, len({text_sha256(x) for x in td.X_test}))
        for mk, spec in specs.items():
            try:
                run_pair(
                    task_data=td, template=template, spec=spec, model_key=mk,
                    results_dir=results_dir, limit=args.limit, max_tokens=args.max_tokens,
                    concurrency=args.concurrency, budget=budget, completion_fn=completion_fn,
                    sleep_fn=sleep_fn, seed=args.seed, extra_request=extra_request,
                    instruction=instruction,
                )
            except BudgetExceeded as e:
                budget_err = e
                break
        if budget_err is not None:
            break

    total = budget.base + budget.spent
    with open(run_log, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "started": started, "finished": now_iso(), "tasks": tasks, "models": model_keys,
            "limit": args.limit, "max_tokens": args.max_tokens, "seed": args.seed,
            "concurrency": args.concurrency, "max_cost_usd": args.max_cost_usd,
            "reasoning_effort": args.reasoning_effort,
            "format_instruction": bool(args.format_instruction),
            "spent_this_run_usd": round(budget.spent, 8), "total_on_disk_usd": round(total, 8),
            "budget_exceeded": budget_err is not None,
        }) + "\n")

    if budget_err is not None:
        print(
            f"BUDGET GUARD: {budget_err}. Spent this run ${budget.spent:.4f}; "
            f"total across this run's files ${total:.4f}. Exiting 3.",
            file=sys.stderr,
        )
        exit_code = EXIT_BUDGET
    logger.info("finished: spent this run $%.4f; total across this run's (tasks x models) files $%.4f",
                budget.spent, total)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
