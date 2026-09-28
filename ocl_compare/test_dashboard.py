"""Dashboard generation. Offline, no network, no browser.

Each test pins a failure the first version actually had.
"""
from __future__ import annotations

import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dashboard as D                                     # noqa: E402


def _embedded(path):
    html = open(path, encoding="utf-8").read()
    js = html.split("<script>", 1)[1]
    return json.loads(re.search(r"^const DATA = (.*?);\nconst META", js,
                                re.S | re.M).group(1))


@pytest.fixture
def two_shapes(tmp_path):
    """The two artifact layouts the pipeline actually writes."""
    prof = {"n": 100, "n_classes": 3, "class_counts": {"0": 40, "1": 30, "2": 30},
            "majority_class": 0, "majority_baseline": 0.4, "imbalance_ratio": 1.3,
            "duplicate_rows": 0, "duplicate_share": 0.0,
            "resolution_half_width": 0.05, "headline_metric": "accuracy",
            "notes": []}
    cand = {"ranking": [{"name": "m1", "balanced": 0.7}, {"name": "m2", "balanced": 0.6}],
            "best": "m1"}
    multi = {"tasks": [{"task": "alpha", "trained_on": "gold", "profile": prof,
                        "candidates": cand, "free_arm_shipped": "m1"}]}
    single = {"task": "solo", "trained_on": "gold", "profile": prof,
              "candidates": cand, "free_arm_shipped": "m1", "expert": None}
    (tmp_path / "pipeline.json").write_text(json.dumps(multi))
    (tmp_path / "solo_run.json").write_text(json.dumps(single))
    return tmp_path


def test_single_task_artifacts_are_not_dropped(two_shapes, tmp_path):
    """banking77 writes the task dict at the top level rather than under
    "tasks", and the first version of collect() looked only for the list --
    so the dataset the whole run existed to measure was silently absent."""
    out = str(tmp_path / "d.html")
    D.build(str(two_shapes), out)
    tasks = {t["task"] for t in _embedded(out)}
    assert tasks == {"alpha", "solo"}


def test_reconstruction_and_dashboard_artifacts_are_skipped(tmp_path):
    (tmp_path / "reconstruct.json").write_text(json.dumps({"tasks": [{"x": 1}]}))
    (tmp_path / "label_matched_llm.json").write_text(json.dumps({"tasks": {"a": []}}))
    (tmp_path / "pipeline.json").write_text(json.dumps({"tasks": [{
        "task": "keep", "profile": {"n": 9, "n_classes": 2, "class_counts": {},
        "majority_class": 0, "majority_baseline": 0.5, "imbalance_ratio": 1.0,
        "duplicate_rows": 0, "duplicate_share": 0.0, "resolution_half_width": 0.1,
        "headline_metric": "accuracy", "notes": []},
        "candidates": {"ranking": []}, "free_arm_shipped": "m"}]}))
    out = str(tmp_path / "d.html")
    D.build(str(tmp_path), out)
    assert {t["task"] for t in _embedded(out)} == {"keep"}


def test_page_is_self_contained(two_shapes, tmp_path):
    """It has to open from file://, where fetch() of a sibling is blocked and a
    CDN may be unreachable. Everything inline or the page is a blank frame."""
    out = str(tmp_path / "d.html")
    D.build(str(two_shapes), out)
    html = open(out, encoding="utf-8").read()
    assert "fetch(" not in html
    assert not re.search(r"<(script|link)[^>]+(src|href)\s*=\s*[\"']https?:", html)


def test_empty_results_dir_fails_loudly(tmp_path):
    with pytest.raises(SystemExit):
        D.build(str(tmp_path), str(tmp_path / "d.html"))


# --- layout contract -------------------------------------------------------
# The dashboard grew panel by panel and ended up rendering a different shape per
# dataset: some tabs had a deferral dial, some did not, some had a per-class
# chart, some did not, and the chart's columns silently meant different things
# depending on which artifact fed it. These pin the layout instead.

SECTIONS = ["Verdict", "Dataset", "Where the wins are", "Confusion matrix —",
            "Gates", "Leaderboard", "Per-class competence —", "Deferral dial",
            "Per-class competence, as numbers"]


def test_layout_is_one_fixed_set_of_sections():
    """Seven sections, same order, every dataset. A section with nothing to show
    says why rather than disappearing -- panels that come and go make two runs
    impossible to compare and hide 'measured and empty' behind 'not measurable'."""
    src = open(os.path.join(HERE, "dashboard.py"), encoding="utf-8").read()
    assert ("[verdict(t),dataset(t),confusionPanel(t),matp,gates(t),leaderboard(t),\n"
            "   chart,dial(t),classTable(t)]") in src
    for name in SECTIONS:
        assert name in src, name
    # no conditional filtering of the section list
    assert ".filter(Boolean).forEach(x=>main.appendChild(x))" not in src


def test_all_eight_gates_render_with_an_explicit_status():
    """Printing only the gates that fired made a missing G8 row mean either
    'passed' or 'could not run'."""
    src = open(os.path.join(HERE, "dashboard.py"), encoding="utf-8").read()
    for g in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"):
        assert f'["{g}",' in src, g
    assert 'const colour = {pass:"good", fail:"bad", fired:"warn"' in src


def test_the_expert_is_never_a_candidate_column():
    """class_router.board includes the LLM as a member while competence_board
    does not, so the chart's columns meant different things on different tabs --
    and on imdb it could mark the LLM best in a class while the verdict said
    ship bge-large. Free candidates are columns; the LLM is a marker."""
    src = open(os.path.join(HERE, "dashboard.py"), encoding="utf-8").read()
    assert "const free = all.filter(n=>!isExpert(n));" in src
    assert "expert: all.find(isExpert) || null" in src
    # winner highlighting ranges over the free names only
    assert "const names=b.free" in src


def test_one_ranking_not_three(tmp_path):
    """Candidate pool, constructions and DESlib each sorted a slice, so the
    highest number on the page could sit three panels below what looked like
    the winner. One table now, with the statistics that were in the others."""
    src = open(os.path.join(HERE, "dashboard.py"), encoding="utf-8").read()
    assert "function candidates(" not in src
    assert "function deslib(" not in src
    assert "Leaderboard — every construction that could ship" in src
    for col in ("95% CI", "family", "vs best base"):
        assert col in src


def test_sections_with_no_data_explain_themselves(tmp_path):
    prof = {"n": 80, "n_classes": 3, "class_counts": {}, "majority_class": 0,
            "majority_baseline": 0.4, "imbalance_ratio": 1.2, "duplicate_rows": 0,
            "duplicate_share": 0.0, "resolution_half_width": 0.06,
            "headline_metric": "accuracy", "notes": []}
    art = {"task": "noexp", "profile": prof, "free_arm_shipped": "m1", "expert": None,
           "candidates": {"ranking": [{"name": "m1", "balanced": 0.7}]}}
    (tmp_path / "pipeline.json").write_text(json.dumps({"tasks": [art]}))
    out = str(tmp_path / "d.html")
    D.build(str(tmp_path), out)
    page = open(out, encoding="utf-8").read()
    assert "Not measurable on this dataset" in page
    assert "needs per-item LLM predictions" in page


def _art(task, **kw):
    prof = {"n": 100, "n_classes": kw.pop("k", 3), "class_counts": {},
            "majority_class": 0, "majority_baseline": 0.4, "imbalance_ratio": 1.2,
            "duplicate_rows": 0, "duplicate_share": 0.0,
            "resolution_half_width": 0.05, "headline_metric": "accuracy", "notes": []}
    return {"task": task, "profile": prof, "free_arm_shipped": "m1",
            "candidates": {"ranking": [{"name": "m1", "balanced": 0.7}]}, **kw}


def test_variants_are_separated_from_datasets(tmp_path):
    """The flat tab row read "isear 7c main", "isear 7c ablation_nonli",
    "isear 7c llmlabels" -- three tabs for one dataset, labelled by file stem.
    Ablations and --train-on llm are experiments ON a dataset, not datasets."""
    (tmp_path / "pipeline.json").write_text(json.dumps({"tasks": [_art("isear")]}))
    (tmp_path / "pipeline_ablation_nonli.json").write_text(
        json.dumps({"tasks": [_art("isear")]}))
    (tmp_path / "pipeline_llmlabels.json").write_text(
        json.dumps({"tasks": [_art("isear")]}))
    out = str(tmp_path / "d.html")
    D.build(str(tmp_path), out)
    data = _embedded(out)
    canon = [t for t in data if t["canonical"]]
    assert len(canon) == 1 and canon[0]["variant"] is None
    assert {t["variant"] for t in data if not t["canonical"]} == {
        "without NLI", "trained on LLM labels"}


def test_a_superseded_partial_run_is_labelled_not_promoted(tmp_path):
    """pipeline_fever_imdb is an older run that a later full run replaced, and it
    appeared on the tab bar as if it were its own dataset."""
    import time as _t
    old = tmp_path / "pipeline_fever_imdb.json"
    old.write_text(json.dumps({"tasks": [_art("fever")]}))
    os.utime(old, (1, 1))
    new = tmp_path / "pipeline.json"
    new.write_text(json.dumps({"tasks": [_art("fever")]}))
    os.utime(new, (_t.time(), _t.time()))
    out = str(tmp_path / "d.html")
    D.build(str(tmp_path), out)
    data = _embedded(out)
    assert sum(1 for t in data if t["canonical"]) == 1
    assert [t["variant"] for t in data if not t["canonical"]] == [
        "superseded by a later run"]


def test_variant_labels_are_words_not_file_stems(tmp_path):
    src = open(os.path.join(HERE, "dashboard.py"), encoding="utf-8").read()
    assert '"ablation_nonli": "without NLI"' in src
    assert '"llmlabels": "trained on LLM labels"' in src
    # and the nav puts them behind a toggle rather than in the main row
    assert 'variants.style.display="none"' in src
    assert "variant runs (" in src


def test_the_oracle_is_not_listed_as_something_that_could_ship(tmp_path):
    """It reads the answer, then picks whichever pool member got that item
    right. Ranking it beside things you can build put an unshippable row at the
    top of a table headed "every construction that could ship" -- which reads as
    a recommendation to ship it."""
    art = _art("orc", deslib={
        "available": True, "baseline": "m1", "baseline_balanced": 0.7, "verdict": "v",
        "methods": {
            "DESlib LCA": {"balanced": 0.72, "delta_vs_best_single": 0.02,
                           "ci": [0.01, 0.03], "p": 0.01,
                           "significant_after_holm": True,
                           "clears_unpaired_floor": True},
            "DESlib Oracle (ceiling)": {"balanced": 0.95,
                                        "delta_vs_best_single": 0.25,
                                        "ci": [0.2, 0.3], "p": 0.0,
                                        "significant_after_holm": True,
                                        "clears_unpaired_floor": True}}})
    (tmp_path / "pipeline.json").write_text(json.dumps({"tasks": [art]}))
    out = str(tmp_path / "d.html")
    D.build(str(tmp_path), out)
    src = open(os.path.join(HERE, "dashboard.py"), encoding="utf-8").read()
    # pulled out before the rows are built, and explained in plain words
    assert "if(/Oracle/.test(k)){ oracle={name:k, ...m}; return; }" in src
    assert "you cannot build it" in src
    assert "size the prize" in src


def test_a_higher_scoring_row_explains_why_it_does_not_ship(tmp_path):
    """"Why is the top row not what ships" has to be answered where the question
    is asked, in the table, not three sections away."""
    src = open(os.path.join(HERE, "dashboard.py"), encoding="utf-8").read()
    assert "Why is the top row not what ships?" in src
    assert "smaller than the measurement error" in src


def test_confusion_describes_the_configuration_the_verdict_quotes(tmp_path):
    """A confusion matrix for a different arm than the headline number is worse
    than no matrix: it looks like an explanation of a score it does not belong
    to. The engine builds it from the nested operating point's own predictions."""
    eng = open(ROOT_SRC, encoding="utf-8").read()
    assert 'conf_block = CP.confusion(np.asarray(bp.pop("pred")), gold, label_names)' in eng
    assert '"pred": final,' in open(ROOT_CP, encoding="utf-8").read()


def test_the_paired_table_is_2x2_at_every_class_count(tmp_path):
    """A k x k class grid says what one model is wrong ABOUT; it does not say
    what a cascade can do, and at 77 classes it is 5,929 cells nobody reads. The
    panel is the paired table instead -- free right/wrong against the other arm
    -- which is the same shape at 2 classes and at 77."""
    src = open(os.path.join(HERE, "dashboard.py"), encoding="utf-8").read()
    assert "2x2, not k x k" in src
    for cell in ("both_right", "free_only_right", "other_only_right", "neither_right"):
        assert cell in src, cell
    # the class-by-class detail stays in the artifact, not on the panel
    assert "full\n      +`${c.classes.length}x${c.classes.length} class matrix is in the artifact" in src \
        or "class matrix is in the artifact" in src


ROOT_SRC = os.path.join(os.path.dirname(HERE), "src", "downshift", "engine.py")
ROOT_CP = os.path.join(os.path.dirname(HERE), "src", "downshift", "classpipe.py")


def test_the_paired_table_shows_its_margins(tmp_path):
    """Without row and column totals the four cells read as four unrelated
    percentages, and "free right & LLM right = 60.1%" looks like a claim that
    both arms score 60%. It is the overlap: the cells partition the dataset, and
    the margins are the two arms' own accuracies, which is what connects this
    panel to every other number on the page."""
    src = open(os.path.join(HERE, "dashboard.py"), encoding="utf-8").read()
    assert "const rowFreeRight=(r.both_right+r.free_only_right)/r.n;" in src
    assert "const colOtherRight=(r.both_right+r.other_only_right)/r.n;" in src
    assert "free arm accuracy" in src
    assert "the four sum to 100%" in src


def test_the_panel_is_the_2x2_and_nothing_else(tmp_path):
    """TP / FP / FN / TN is what "confusion matrix" means, and it is the whole
    panel. The by-class list and the k x k grid were extra views that answered a
    question nobody asked here -- the per-class detail already has two sections
    of its own further down."""
    src = open(os.path.join(HERE, "dashboard.py"), encoding="utf-8").read()
    for cell in ("true positive", "false alarm", "true negative",
                 "missed — should have been caught"):
        assert cell in src, cell
    assert "precision ${f4(prec)}" in src and "recall ${f4(rec)}" in src
    for gone in ("matMode", "function summary(", "function full(", "mm-sum", "mm-full"):
        assert gone not in src, f"{gone} survived"


def test_the_2x2_says_which_class_is_positive(tmp_path):
    """A single 2x2 over a multiclass problem has to pick what "positive" means.
    One-vs-rest with a class selector, defaulting to the positive class when the
    dataset is binary."""
    src = open(os.path.join(HERE, "dashboard.py"), encoding="utf-8").read()
    assert "function ovr(ci){" in src
    assert 'if(posClass===null && k===2) posClass = 1;' in src
    assert "all classes (micro)" in src


def test_the_micro_average_admits_it_is_degenerate(tmp_path):
    """Micro-averaged one-vs-rest over single-label multiclass gives FP == FN and
    precision == recall == accuracy. Standard, and confusing if unexplained --
    the panel says why two numbers are always identical instead of leaving the
    reader to wonder."""
    src = open(os.path.join(HERE, "dashboard.py"), encoding="utf-8").read()
    assert "degenerate" in src
    assert "so FP equals FN" in src


def test_counts_come_with_their_share(tmp_path):
    """A count and its percentage together, and the percentage is of the row so
    the TP cell reads as recall."""
    src = open(os.path.join(HERE, "dashboard.py"), encoding="utf-8").read()
    assert "(${(100*v/d).toFixed(1)}%)" in src
    assert "the TP cell is recall and the TN cell is specificity" in src


def test_the_dial_plots_every_rule_in_the_artifact(tmp_path):
    """engine.render() already carries this rule with the reason beside it: a
    rule that is in the search but not in the table is a rule nobody checks. The
    dashboard had drifted to a hardcoded three of five, so committee -- the
    winning rule on imdb and hatespeech -- was never drawn, the resolution band
    was a max over the drawn subset, and the "ship this" ring landed on a curve
    that did not exist."""
    src = open(os.path.join(HERE, "dashboard.py"), encoding="utf-8").read()
    assert 'const rules=["confidence","class_aware","per_class_conf"]' not in src
    assert "const rules=Object.keys(d.points[0]).filter" in src


def test_conformal_and_coverage_reach_the_payload(tmp_path):
    """The headline measurement of the newest commit was being dropped by _slim,
    so the only artifact anyone opens could not show that the guarantee fails out
    of fold."""
    src = open(os.path.join(HERE, "dashboard.py"), encoding="utf-8").read()
    assert '"conformal"' in src and '"guaranteed_coverage"' in src


# --------------------------------------------------------------------------
# clinc150: the confirmation run for the class-count hypothesis
# --------------------------------------------------------------------------
def test_clinc150_is_registered_and_contiguous():
    """Every check here is a mistake the Dataset contract exists to catch, and
    the oos drop is exactly the shape that produces them: removing a class from
    the middle leaves a gap at the old index 42, and every per-class array in
    the pipeline indexes by position."""
    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import adapters                                          # noqa: F401
    from downshift.dataset import load, available
    assert "clinc150" in available()
    d = load("clinc150")                                     # validate() runs here
    assert d.n == 22500 and d.n_classes == 150
    counts = np.bincount(d.gold)
    assert counts.min() == counts.max() == 150, (
        f"clinc150 is supposed to be exactly balanced; got {counts.min()}"
        f"..{counts.max()} per class")
    assert sorted(d.label_names) == list(range(150))
    assert not d.has_expert, "no per-item LLM stream exists for clinc150"
    assert "oos" not in {v.strip() for v in d.label_names.values()}, (
        "oos is the open-set reject bucket, not a 151st intent")
