"""Phase 2 incumbent runner tests. No network, no API key: litellm.completion
is replaced by an injected fake via main(..., completion_fn=...)."""
from __future__ import annotations

import csv
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2 import legalbench_eval, run_incumbents as ri  # noqa: E402
from tests.helpers import make_synthetic_task_data  # noqa: E402

TRICKY = ["Yes.", " yes", "A: Yes", "YES\n", "No, it does not.", "Answer: No", "", "  Label: yes  ", "no!", 42]


# --------------------------------------------------------------------------
# normalize() is a faithful port of data/legalbench_evaluation.py
# --------------------------------------------------------------------------
def _load_official_eval():
    """Import data/legalbench_evaluation.py by path. That file does
    `from nltk.stem.porter import *` at module top and nltk is not installed
    (we install nothing), so a stub module is injected for the import only;
    the stem=False path under test never touches it."""
    if "nltk" not in sys.modules:
        nltk = types.ModuleType("nltk")
        stem = types.ModuleType("nltk.stem")
        porter = types.ModuleType("nltk.stem.porter")

        class PorterStemmer:  # pragma: no cover - must never be called here
            def stem(self, _):
                raise AssertionError("stub PorterStemmer called: exact-match path must use stem=False")

        porter.PorterStemmer = PorterStemmer
        porter.__all__ = ["PorterStemmer"]
        nltk.stem = stem
        stem.porter = porter
        sys.modules["nltk"] = nltk
        sys.modules["nltk.stem"] = stem
        sys.modules["nltk.stem.porter"] = porter
    path = ROOT / "data" / "legalbench_evaluation.py"
    spec = importlib.util.spec_from_file_location("legalbench_official_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_normalize_matches_official_file():
    official = _load_official_eval()
    for s in TRICKY:
        assert legalbench_eval.normalize(s, stem=False) == official.normalize(s, stem=False), repr(s)
        assert legalbench_eval.normalize(s) == official.normalize(s, stem=False), repr(s)


def test_official_exact_match_path_uses_stem_false():
    """The official evaluate_exact_match_balanced_accuracy must call normalize(stem=False);
    the stub PorterStemmer raises if any stemming happens."""
    official = _load_official_eval()
    gens = ["Yes.", "no", "Answer: Yes", "No"]
    answers = ["Yes", "No", "Yes", "No"]
    a = official.evaluate_exact_match_balanced_accuracy(gens, answers)
    b = legalbench_eval.balanced_accuracy_exact_match(gens, answers)
    assert a == b
    assert a == pytest.approx(0.75)  # "answer yes" is wrong under exact match


# --------------------------------------------------------------------------
# parse_ok semantics
# --------------------------------------------------------------------------
def test_parse_ok_semantics():
    classes = ["no", "yes"]
    assert ri.parse_pred("Yes.", classes) == ("yes", 1)
    assert ri.parse_pred("  NO\n", classes) == ("no", 1)
    pred, ok = ri.parse_pred("Answer: Yes", classes)
    assert ok == 0 and pred == "answer yes"  # normalized raw string, NOT coerced to 'yes'
    pred, ok = ri.parse_pred("", classes)
    assert ok == 0 and pred == ""
    pred, ok = ri.parse_pred(None, classes)
    assert ok == 0 and pred == ""


# --------------------------------------------------------------------------
# cost arithmetic
# --------------------------------------------------------------------------
def _fake_spec(price_in=1.0, price_out=2.0, price_cached=-1.0, env_key="", call_id="fake/model", extra_body=None):
    return SimpleNamespace(call_id=call_id, env_key=env_key, price_in=price_in, price_out=price_out,
                           price_cached=price_cached, extra_body=extra_body, think=None, label="fake")


def test_cost_arithmetic():
    spec = _fake_spec(price_in=0.15, price_out=0.5, price_cached=-1.0)
    assert ri.compute_cost(spec, 1000, 10, 0) == pytest.approx(1000 * 0.15 / 1e6 + 10 * 0.5 / 1e6, abs=1e-12)
    # no cached price on file (-1 or None): cached tokens billed at price_in
    assert ri.compute_cost(spec, 1000, 10, 400) == pytest.approx(1000 * 0.15 / 1e6 + 10 * 0.5 / 1e6, abs=1e-12)
    spec_none = _fake_spec(price_in=0.15, price_out=0.5, price_cached=None)
    assert ri.compute_cost(spec_none, 1000, 10, 400) == pytest.approx(1000 * 0.15 / 1e6 + 10 * 0.5 / 1e6, abs=1e-12)
    # cached price on file: cached subset of prompt tokens at price_cached
    spec_c = _fake_spec(price_in=0.15, price_out=0.5, price_cached=0.03)
    expected = 600 * 0.15 / 1e6 + 400 * 0.03 / 1e6 + 10 * 0.5 / 1e6
    assert ri.compute_cost(spec_c, 1000, 10, 400) == pytest.approx(expected, abs=1e-12)
    # cached can never exceed prompt tokens
    assert ri.compute_cost(spec_c, 100, 0, 500) == pytest.approx(100 * 0.03 / 1e6, abs=1e-12)


def test_extract_usage_and_content_ignore_reasoning():
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=" Yes\n", reasoning_content="thinking..."))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=5)),
    )
    assert ri.extract_content(resp) == " Yes\n"
    assert ri.extract_usage(resp) == (12, 3, 5)
    resp2 = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
                            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=0, prompt_tokens_details=None))
    assert ri.extract_content(resp2) == ""
    assert ri.extract_usage(resp2) == (7, 0, 0)


# --------------------------------------------------------------------------
# end-to-end on a synthetic task with a fake completion
# --------------------------------------------------------------------------
class FakeCompletion:
    def __init__(self, answer="yes", in_tokens=100, out_tokens=2, cached=0, fail_on=None):
        self.calls = 0
        self.prompts = []
        self.answer, self.in_tokens, self.out_tokens, self.cached = answer, in_tokens, out_tokens, cached
        self.fail_on = fail_on or {}

    def __call__(self, *, model, messages, temperature, max_tokens, **kw):
        self.calls += 1
        self.last_kwargs = kw
        assert temperature == 0 and len(messages) == 1 and messages[0]["role"] == "user"
        self.prompts.append(messages[0]["content"])
        exc = self.fail_on.get(self.calls)
        if exc is not None:
            raise exc
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.answer, reasoning_content="ignored"))],
            usage=SimpleNamespace(prompt_tokens=self.in_tokens, completion_tokens=self.out_tokens,
                                  prompt_tokens_details=SimpleNamespace(cached_tokens=self.cached)),
        )


def _setup(tmp_path, n_test=8, duplicate_pair=False):
    """Registry + prompt + synthetic loader for a 2-class task named synthetic_task."""
    td = make_synthetic_task_data(n_train=4, n_test=n_test, n_classes=2)
    td.spec.classes = ["no", "yes"]
    td.y_test = ["yes" if i % 2 == 0 else "no" for i in range(n_test)]
    if duplicate_pair:
        td.X_test[1] = td.X_test[0]  # identical text, distinct item_idx
    entry = {"task": "synthetic_task", "hf_config": "synthetic", "input_columns": ["text"],
             "class_distribution": {"yes": 1, "no": 1}, "metric": "balanced_accuracy"}
    reg = tmp_path / "registry.json"
    reg.write_text(__import__("json").dumps([entry]))
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "synthetic_task.txt").write_text("Is it?\n\nClause: foo\nLabel: Yes\n\nClause: {{text}}\nLabel:")
    results = tmp_path / "results"

    def loader(e):
        assert e["task"] == "synthetic_task"
        return td

    return td, reg, prompts, results, loader


def _base_argv(reg, prompts, results, **extra):
    argv = ["--tasks", "synthetic_task", "--models", "fake", "--registry", str(reg),
            "--prompts-dir", str(prompts), "--results-dir", str(results), "--concurrency", "2"]
    for k, v in extra.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv


def _read(results, model="fake"):
    p = results / "incumbent" / f"synthetic_task__{model}.csv"
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        assert list(r.fieldnames) == ri.COLUMNS
        return list(r)


def test_resumability(tmp_path, monkeypatch):
    td, reg, prompts, results, loader = _setup(tmp_path, n_test=8)
    monkeypatch.setenv("PHASE2_TEST_KEY", "x")
    fake = FakeCompletion(answer="Yes.")
    spec = _fake_spec(env_key="PHASE2_TEST_KEY")
    rc = ri.main(_base_argv(reg, prompts, results, limit=5), completion_fn=fake, task_data_loader=loader,
                 by_key={"fake": spec}, sleep_fn=lambda s: None)
    assert rc == 0
    assert fake.calls == 5
    rows = _read(results)
    assert sorted(int(r["item_idx"]) for r in rows) == [0, 1, 2, 3, 4]
    assert all(r["pred"] == "yes" and r["parse_ok"] == "1" for r in rows)
    assert all(r["text_sha256"] == ri.text_sha256(td.X_test[int(r["item_idx"])]) for r in rows)
    assert "{{text}}" not in fake.prompts[0] and fake.prompts[0].endswith("Label:")

    rc = ri.main(_base_argv(reg, prompts, results, limit=5), completion_fn=fake, task_data_loader=loader,
                 by_key={"fake": spec}, sleep_fn=lambda s: None)
    assert rc == 0
    assert fake.calls == 5  # zero new calls
    rows2 = _read(results)
    assert len(rows2) == 5 and rows2 == rows  # unchanged, no duplicates

    # extending the limit only calls the new items
    rc = ri.main(_base_argv(reg, prompts, results, limit=8), completion_fn=fake, task_data_loader=loader,
                 by_key={"fake": spec}, sleep_fn=lambda s: None)
    assert rc == 0 and fake.calls == 8
    assert sorted(int(r["item_idx"]) for r in _read(results)) == list(range(8))


def test_dedupe_identical_text(tmp_path, monkeypatch):
    td, reg, prompts, results, loader = _setup(tmp_path, n_test=4, duplicate_pair=True)
    monkeypatch.setenv("PHASE2_TEST_KEY", "x")
    fake = FakeCompletion(answer="no")
    spec = _fake_spec(env_key="PHASE2_TEST_KEY")
    rc = ri.main(_base_argv(reg, prompts, results), completion_fn=fake, task_data_loader=loader,
                 by_key={"fake": spec}, sleep_fn=lambda s: None)
    assert rc == 0
    assert fake.calls == 3  # 4 items, 2 share text -> 3 calls
    rows = _read(results)
    assert len(rows) == 4
    by_idx = {int(r["item_idx"]): r for r in rows}
    assert by_idx[0]["text_sha256"] == by_idx[1]["text_sha256"]
    assert by_idx[0]["raw_output"] == by_idx[1]["raw_output"] == "no"
    assert by_idx[0]["gold"] == "yes" and by_idx[1]["gold"] == "no"  # gold is per item, not per text


def test_budget_guard_exits_3(tmp_path, monkeypatch):
    td, reg, prompts, results, loader = _setup(tmp_path, n_test=4)
    monkeypatch.setenv("PHASE2_TEST_KEY", "x")
    # Billed usage: 100 prompt tokens * $6000/1M = $0.60 per call. The pre-call
    # estimate uses tiktoken on the real (short, ~26-token) prompt, ~$0.16.
    # Cap $0.70: first call fits (0 + 0.16), second projects 0.60 + 0.16 > 0.70.
    fake = FakeCompletion(answer="yes", in_tokens=100, out_tokens=1)
    spec = _fake_spec(price_in=6000.0, price_out=0.0, env_key="PHASE2_TEST_KEY")
    rc = ri.main(_base_argv(reg, prompts, results, max_cost_usd=0.7, concurrency=1), completion_fn=fake,
                 task_data_loader=loader, by_key={"fake": spec}, sleep_fn=lambda s: None)
    assert rc == 3
    assert fake.calls == 1
    rows = _read(results)
    assert len(rows) == 1 and float(rows[0]["cost_usd"]) == pytest.approx(0.6)
    # a re-run with the same cap refuses immediately: existing rows count toward the total
    rc = ri.main(_base_argv(reg, prompts, results, max_cost_usd=0.7, concurrency=1), completion_fn=fake,
                 task_data_loader=loader, by_key={"fake": spec}, sleep_fn=lambda s: None)
    assert rc == 3 and fake.calls == 1 and len(_read(results)) == 1


def test_dry_run_makes_zero_calls(tmp_path, capsys):
    td, reg, prompts, results, loader = _setup(tmp_path, n_test=6, duplicate_pair=True)
    fake = FakeCompletion()
    spec = _fake_spec(price_in=1.0, price_out=2.0)
    rc = ri.main(_base_argv(reg, prompts, results, dry_run=None)[:-1] + ["--dry-run"], completion_fn=fake,
                 task_data_loader=loader, by_key={"fake": spec}, sleep_fn=lambda s: None)
    assert rc == 0
    assert fake.calls == 0
    out = capsys.readouterr().out
    assert "synthetic_task" in out and "fake" in out and "GRAND TOTAL" in out
    line = [l for l in out.splitlines() if l.startswith("synthetic_task")][0]
    cols = line.split()
    assert cols[2] == "6" and cols[3] == "5"  # 6 items, 5 unique texts -> 5 calls
    assert not (results / "incumbent").exists()


def test_error_rows_and_retry(tmp_path, monkeypatch):
    td, reg, prompts, results, loader = _setup(tmp_path, n_test=3)
    monkeypatch.setenv("PHASE2_TEST_KEY", "x")

    class Boom(Exception):
        pass

    class Flaky(Exception):
        status_code = 503

    fake = FakeCompletion(answer="yes", fail_on={1: Boom("nope"), 2: Flaky("busy")})
    spec = _fake_spec(env_key="PHASE2_TEST_KEY")
    slept = []
    rc = ri.main(_base_argv(reg, prompts, results, concurrency=1), completion_fn=fake, task_data_loader=loader,
                 by_key={"fake": spec}, sleep_fn=slept.append)
    assert rc == 0
    rows = sorted(_read(results), key=lambda r: int(r["item_idx"]))
    assert rows[0]["raw_output"] == "<error: Boom>" and rows[0]["pred"] == "" and rows[0]["parse_ok"] == "0"
    assert rows[0]["cost_usd"].startswith("0.0000000000")
    assert rows[1]["raw_output"] == "yes" and rows[2]["raw_output"] == "yes"  # 503 retried once, then ok
    assert len(slept) == 1 and 0 < slept[0] <= 60
    assert fake.calls == 4  # 3 items + 1 retry


def test_phase1_never_imports_phase2_or_litellm():
    bad = []
    for p in [ROOT / "cli.py", *(ROOT / "core").glob("*.py"), *(ROOT / "candidates").glob("*.py")]:
        src = p.read_text()
        for needle in ("phase2", "litellm"):
            if f"import {needle}" in src or f"from {needle}" in src:
                bad.append((str(p), needle))
    assert bad == []


def test_extra_body_and_reasoning_effort_forwarded(tmp_path, monkeypatch):
    td, reg, prompts, results, loader = _setup(tmp_path, n_test=2)
    monkeypatch.setenv("PHASE2_TEST_KEY", "x")
    fake = FakeCompletion(answer="yes")
    spec = _fake_spec(env_key="PHASE2_TEST_KEY", extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    rc = ri.main(_base_argv(reg, prompts, results, reasoning_effort="low", max_tokens=512), completion_fn=fake,
                 task_data_loader=loader, by_key={"fake": spec}, sleep_fn=lambda s: None)
    assert rc == 0 and fake.calls == 2
    assert fake.last_kwargs["chat_template_kwargs"] == {"enable_thinking": False}
    assert fake.last_kwargs["extra_body"] == {"reasoning_effort": "low"}
    assert fake.last_kwargs["timeout"] == ri.CALL_TIMEOUT_S


def test_error_rows_are_retried_on_resume(tmp_path, monkeypatch):
    """A persisted '<error: Name>' row is not a completed item: the next run must
    drop it, retry exactly that item, and leave one row per item_idx."""
    td, reg, prompts, results, loader = _setup(tmp_path, n_test=4)
    monkeypatch.setenv("PHASE2_TEST_KEY", "x")

    class Boom(Exception):
        pass

    spec = _fake_spec(env_key="PHASE2_TEST_KEY")
    fake = FakeCompletion(answer="yes", fail_on={1: Boom("nope"), 3: Boom("nope again")})
    rc = ri.main(_base_argv(reg, prompts, results, concurrency=1), completion_fn=fake, task_data_loader=loader,
                 by_key={"fake": spec}, sleep_fn=lambda s: None)
    assert rc == 0 and fake.calls == 4
    rows = {int(r["item_idx"]): r for r in _read(results)}
    assert len(rows) == 4
    err_idx = sorted(i for i, r in rows.items() if r["raw_output"].startswith("<error:"))
    assert len(err_idx) == 2

    fake2 = FakeCompletion(answer="no")
    rc = ri.main(_base_argv(reg, prompts, results, concurrency=1), completion_fn=fake2, task_data_loader=loader,
                 by_key={"fake": spec}, sleep_fn=lambda s: None)
    assert rc == 0
    assert fake2.calls == 2  # exactly the two failed items, nothing else
    rows2 = _read(results)
    assert sorted(int(r["item_idx"]) for r in rows2) == [0, 1, 2, 3]  # no duplicates
    by_idx = {int(r["item_idx"]): r for r in rows2}
    for i in err_idx:
        assert by_idx[i]["raw_output"] == "no" and by_idx[i]["parse_ok"] == "1"
    for i in set(range(4)) - set(err_idx):
        assert by_idx[i] == rows[i]  # untouched rows are byte-for-byte the same
    assert not any(r["raw_output"].startswith("<error:") for r in rows2)
    assert not (results / "incumbent" / "synthetic_task__fake.csv.tmp").exists()

    # third run: nothing to do
    fake3 = FakeCompletion(answer="yes")
    rc = ri.main(_base_argv(reg, prompts, results, concurrency=1), completion_fn=fake3, task_data_loader=loader,
                 by_key={"fake": spec}, sleep_fn=lambda s: None)
    assert rc == 0 and fake3.calls == 0 and _read(results) == rows2


def test_error_row_is_never_fanned_out_to_duplicate_text(tmp_path, monkeypatch):
    td, reg, prompts, results, loader = _setup(tmp_path, n_test=3, duplicate_pair=True)
    monkeypatch.setenv("PHASE2_TEST_KEY", "x")

    class Boom(Exception):
        pass

    spec = _fake_spec(env_key="PHASE2_TEST_KEY")
    fake = FakeCompletion(answer="yes", fail_on={1: Boom("nope")})  # first call = shared text of items 0 and 1
    rc = ri.main(_base_argv(reg, prompts, results, concurrency=1), completion_fn=fake, task_data_loader=loader,
                 by_key={"fake": spec}, sleep_fn=lambda s: None)
    assert rc == 0 and fake.calls == 2
    rows = {int(r["item_idx"]): r for r in _read(results)}
    assert rows[0]["raw_output"].startswith("<error:") and rows[1]["raw_output"].startswith("<error:")
    fake2 = FakeCompletion(answer="no")
    rc = ri.main(_base_argv(reg, prompts, results, concurrency=1), completion_fn=fake2, task_data_loader=loader,
                 by_key={"fake": spec}, sleep_fn=lambda s: None)
    assert rc == 0 and fake2.calls == 1  # one call, fanned out to both items
    rows2 = {int(r["item_idx"]): r for r in _read(results)}
    assert sorted(rows2) == [0, 1, 2]
    assert rows2[0]["raw_output"] == rows2[1]["raw_output"] == "no" and rows2[2] == rows[2]
