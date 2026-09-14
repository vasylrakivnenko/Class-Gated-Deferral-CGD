"""Dataset loading and the split discipline the whole result depends on.

The split is the methodology. Everything else is bookkeeping.

GEPA is an *optimizer*, and optimizers overfit whatever you point them at. It
reads the train set to propose instructions and scores candidates on the val
set to pick survivors. Both are therefore contaminated -- a number measured on
either one is a training-set number wearing a disguise. Only a third split the
optimizer never touches can support the sentence the product actually sells:
"this cheaper model clears your bar."

We use Financial PhraseBank's `sentences_allagree` subset: the 2,264 sentences
where *every* human annotator assigned the same label. That is the "human-
verified holdout" the brief demands, and it is why this dataset was chosen over
SST-2 (binary, saturated) or synthetic labels (measures agreement with the
teacher's mistakes, which is the exact failure we are trying to avoid).

Two honesty notes that belong in any writeup of these results:

1. `allagree` is the *easy* subset by construction. Unanimous human agreement
   selects for unambiguous sentences, so accuracies here run higher than the
   same model would score in production on messy text. It is the right choice
   for a *trustworthy* holdout and the wrong choice for estimating production
   accuracy. Both things are true.

2. The class balance is 13% negative / 61% neutral / 25% positive. Always
   answering "neutral" scores 61.4%. Any model below that line is worse than a
   constant, which is why `majority_baseline` exists and why it is drawn on the
   chart. A three-class task with a 61% floor flatters weak models badly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import hashlib

import numpy as np

DATASET_REPO = "ghbacct/financial-phrasebank-all-agree-classification"
LABELS = ("negative", "neutral", "positive")


@dataclass(frozen=True)
class TaskSpec:
    """Everything needed to turn a HuggingFace dataset into a benchmark task.

    Adding a dataset is meant to be a data change, not a code change. The
    split/dedupe/stratify machinery below is identical for every task; only
    these fields differ. Datasets are ADDED, never swapped -- a result is only
    interpretable next to the task it was measured on, so each task keeps its
    own results directory.
    """

    key: str                      # short id; also the results subdirectory
    label: str                    # display name
    repo: str                     # HuggingFace repo id
    text_column: str
    label_column: str
    description: str              # seed instruction handed to the DSPy signature
    config: str = ""              # HF config name, "" if the repo has none
    splits: tuple[str, ...] = ()  # HF splits to concatenate; () means all of them
    notes: str = ""

    # ── generative-task fields ──
    # A task is "classification" (pick one of a fixed label set, scored by
    # equality) or "generative" (free-form / structured output, scored by a
    # named function in scoring.SCORERS). The distinction drives three things:
    # the DSPy signature (Literal[labels] vs a free output field), the metric,
    # and whether the TF-IDF / encoder baselines run at all -- on a generative
    # task they cannot produce the output type, so they are reported as
    # not-applicable rather than as a misleading 0%.
    kind: str = "classification"
    scorer: str = "exact_match"
    output_desc: str = ""         # what the model should produce
    required_keys: tuple = ()     # for JSON scorers: keys the schema must carry
    gold_column: str = ""         # generative: column holding the reference answer

    @property
    def is_generative(self) -> bool:
        return self.kind == "generative"

    def load(self, n_train: int = 200, n_val: int = 200, n_test: int = 250,
             seed: int = 0) -> "Task":
        return load_task(self, n_train, n_val, n_test, seed)


TASKS: dict[str, TaskSpec] = {
    "financial_phrasebank": TaskSpec(
        key="financial_phrasebank",
        label="Financial PhraseBank (all-agree)",
        repo=DATASET_REPO,
        text_column="text",
        label_column="label",
        splits=("train", "test"),
        description=("Classify the sentiment of a financial news sentence from the "
                     "point of view of an investor reading it."),
        notes=("3 classes, ~120 chars per input, 61.2% majority floor. Saturated: "
               "TF-IDF already scores 88.8% and eight LLMs sit within 4 points of "
               "each other, so it no longer discriminates at the top end."),
    ),
    "banking77": TaskSpec(
        key="banking77",
        label="Banking77 (intent classification)",
        # NOT PolyAI/banking77 -- that repo is still script-based and raises
        # "Dataset scripts are no longer supported" on datasets 5.x, the same
        # trap financial_phrasebank set. This mirror is parquet-native.
        repo="legacy-datasets/banking77",
        text_column="text",
        label_column="label",
        splits=("train", "test"),
        description=("Classify a customer's online-banking query into the single "
                     "intent that best matches what they are asking about."),
        notes=("77 classes, ~54 chars per input, 1.3% majority floor. Difficulty is "
               "strongly data-dependent: on the same 250 held-out items TF-IDF scores "
               "35.1% given 100 training rows, 85.3% given 2,000 (~26 per class) and "
               "92.0% given all 12,701, passing every prompted LLM between 1,000 and "
               "2,000 rows. Prompted LLMs get zero "
               "training examples and must pick from 77 deliberately confusable intents "
               "listed in the prompt, so this measures the thing Financial PhraseBank "
               "could not -- how much labelled data a trained classifier needs before it "
               "beats a prompted model."),
    ),
}


def get_task(key: str) -> TaskSpec:
    if key not in TASKS:
        raise KeyError(f"unknown task {key!r}; known: {sorted(TASKS)}")
    return TASKS[key]


@dataclass
class Task:
    """A classification task, split for optimizer-safe evaluation."""

    name: str
    labels: tuple[str, ...]
    train: list = field(repr=False)      # GEPA reflective mutation -- contaminated
    val: list = field(repr=False)        # GEPA Pareto selection    -- contaminated
    test: list = field(repr=False)       # never seen by any optimizer
    pool: list = field(repr=False)       # extra labels for encoder fine-tuning
    input_field: str = "sentence"

    def __repr__(self) -> str:
        return (f"Task({self.name}: train={len(self.train)} val={len(self.val)} "
                f"test={len(self.test)} pool={len(self.pool)} labels={list(self.labels)})")

    @property
    def majority_baseline(self) -> tuple[str, float]:
        """Accuracy of always predicting the most common test label."""
        gold = [ex.label for ex in self.test]
        counts = {lab: gold.count(lab) for lab in self.labels}
        best = max(counts, key=counts.__getitem__)
        return best, counts[best] / len(gold)

    def class_balance(self, split: str = "test") -> dict[str, float]:
        rows = getattr(self, split)
        return {lab: sum(ex.label == lab for ex in rows) / len(rows) for lab in self.labels}

    def demands(self, instruction: str | None = None, profile: str = "direct",
                sample: int = 0) -> "TaskDemands":
        """What this task asks of a deployment, MEASURED from the loaded data.

        Measured rather than declared, because a declared number is a number
        that goes stale. Both quantities here move on their own: the output
        floor follows the longest label (add a class and it changes), and the
        prompt length follows the instruction, which GEPA rewrites mid-run --
        the seed prompt renders at 239 tokens on Financial PhraseBank and the
        optimized rows on disk measure 570-790. So `demands()` takes the
        instruction it should measure, and a caller screening the OPTIMIZED
        stage must pass the optimized instruction rather than reuse this.

        Both figures come from the same ChatAdapter that the run will use, so
        the envelope -- field headers, the `Literal[...]` label enumeration, the
        trailing format reminder -- is counted rather than guessed. That
        envelope is most of it: 218 of 238 tokens on Financial PhraseBank.

        The token count is tiktoken's `o200k_base`, not each model's own
        tokenizer. Cross-checked against the providers' counters on the runs
        already on disk it lands within 3% (239 vs 245 on fp, 1,937 vs 1,923 on
        banking77) -- close enough to gate a 128,000-token window on, and the
        gate is deliberately one-sided anyway: it only ever refuses a model the
        arithmetic says cannot fit.
        """
        from .metric import count_tokens
        from .program import build_program
        from dspy.adapters.chat_adapter import ChatAdapter

        text = instruction if instruction is not None else getattr(
            self.spec, "description", "") if hasattr(self, "spec") else ""
        program = build_program(profile, self.labels, text,
                                getattr(self, "spec", None))
        signature = next(iter(program.named_predictors()))[1].signature
        adapter = ChatAdapter()

        # The output floor: the longest label wearing the full reply envelope.
        # A model capped below this cannot answer at all -- it truncates mid
        # field and the row reads as a parse failure rather than a budget that
        # was never big enough.
        longest = max(self.labels, key=len) if self.labels else ""
        out_tokens = count_tokens(
            adapter.format_assistant_message_content(signature, {"label": longest}))

        rows = self.test[:sample] if sample else self.test
        lengths = []
        for example in rows:
            inputs = {name: getattr(example, name, "") or ""
                      for name in signature.input_fields}
            lengths.append(count_tokens("\n".join(
                f"{m.get('role', '')}: {m.get('content', '')}"
                for m in adapter.format(signature, [], inputs))))
        lengths.sort()

        def pct(p: float) -> int:
            return lengths[min(len(lengths) - 1, int(p * (len(lengths) - 1)))]

        return TaskDemands(
            max_output_tokens=out_tokens,
            p95_prompt_tokens=pct(0.95),
            max_prompt_tokens=lengths[-1],
            mean_prompt_tokens=round(sum(lengths) / len(lengths)),
            n_labels=len(self.labels),
            n_measured=len(lengths),
            profile=profile,
        )


@dataclass(frozen=True)
class TaskDemands:
    """The arithmetic a deployment must satisfy to serve a task at all.

    Deliberately only the DEDUCTIVE part. Everything here is a count, so a
    model failing one of these checks is not predicted to do badly -- it cannot
    produce a well-formed answer, and spending on it buys a corrupted row.
    Whether a model that fits is any GOOD is a separate, empirical question
    this object says nothing about.
    """

    max_output_tokens: int
    p95_prompt_tokens: int
    max_prompt_tokens: int
    mean_prompt_tokens: int
    n_labels: int
    n_measured: int
    profile: str = "direct"

    def __str__(self) -> str:
        return (f"needs {self.max_output_tokens} output tokens; prompt p95 "
                f"{self.p95_prompt_tokens:,} (mean {self.mean_prompt_tokens:,}, max "
                f"{self.max_prompt_tokens:,}) over {self.n_measured} items, "
                f"{self.n_labels} labels, {self.profile} profile")

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def split_fingerprint(examples) -> str:
    """A content hash of an ORDERED split, for verifying paired comparisons.

    Every McNemar and non-inferiority number in this project compares two
    `correct` vectors POSITIONALLY, so they are only valid if index i means the
    same item in both. The only guard was `len(task.test) == n_test`, which is
    far too weak: the split is drawn from per-class lists built in SOURCE ROW
    ORDER, and `_stratified_take` picks by per-class COUNT, so a reordered
    upstream dataset with the same seed and same sizes yields DIFFERENT items.

    Reproduced: swapping two source rows changed 5 of 30 test items while
    leaving the length identical AND the gold-label sequence identical -- so
    neither the existing assert nor a gold-sequence check could catch it, and
    the paired statistics would have been silently computed across two
    different test sets. `load_dataset` pins no revision, so an upstream
    refresh is enough to trigger it.

    Hashes text and label together, in order, so it detects reordering as well
    as substitution.
    """
    h = hashlib.sha256()
    for ex in examples:
        h.update(str(getattr(ex, "sentence", "")).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(getattr(ex, "label", "")).encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()[:32]


def _stratified_take(rng: np.random.Generator, by_label: dict[str, list], n: int,
                     labels: tuple[str, ...]) -> list:
    """Draw `n` items preserving class proportions, consuming from `by_label`.

    Stratifying matters more than usual here: the minority class is 13% of the
    data, so an unstratified 200-item test set would swing between roughly 18
    and 36 negative examples run to run. Per-class accuracy would then move for
    reasons that have nothing to do with the model.

    Per-class rounding can miss `n` in EITHER direction, and only the short
    direction used to be handled. `round(n * share)` summed over the classes
    overshoots whenever the per-class quota rounds up: at 29 classes
    round(250/29) = 9 gives 261, at 151 classes round(250/151) = 2 gives 302.
    Both current tasks happen to round down (3 classes -> 249, 77 -> 231) so the
    top-up always fired and the overshoot never showed -- except by one item,
    which is why banking77's test split was 251 and Financial PhraseBank's train
    split was 201. Asking for 250 and being handed 302 would be worse than
    untidy: the caller pays per item, and `tests_invariants` asserts the size.

    Trimmed items go BACK to `by_label`, never to the void. The caller draws
    test, then val, then train from the same shrinking pool, so an item dropped
    on the floor here would vanish from every split and from `pool` as well.
    """
    total = sum(len(v) for v in by_label.values())
    taken: dict[str, list] = {}
    for lab in labels:
        share = len(by_label[lab]) / total
        k = min(int(round(n * share)), len(by_label[lab]))
        idx = rng.choice(len(by_label[lab]), size=k, replace=False)
        taken[lab] = [by_label[lab][i] for i in sorted(idx, reverse=True)]
        for i in sorted(idx, reverse=True):
            by_label[lab].pop(i)

    # Give back from whichever class holds most, the mirror of the top-up below:
    # that is the class best able to spare one without distorting its share.
    while sum(len(v) for v in taken.values()) > n:
        lab = max(labels, key=lambda l: len(taken[l]))
        by_label[lab].append(taken[lab].pop())

    take: list = [item for lab in labels for item in taken[lab]]
    # Rounding can leave us a few short; top up from whichever class has most left.
    while len(take) < n and any(by_label.values()):
        lab = max(labels, key=lambda l: len(by_label[l]))
        if not by_label[lab]:
            break
        take.append(by_label[lab].pop())
    rng.shuffle(take)
    return take


def load_task(spec: TaskSpec, n_train: int = 200, n_val: int = 200,
              n_test: int = 250, seed: int = 0) -> Task:
    """Load any registered task and split it for optimizer-safe evaluation.

    Same discipline for every dataset: deduplicate, carve out the test set
    FIRST, stratify every split by class. Handles both int ClassLabel columns
    (where the label names live in the feature) and plain string label columns.
    """
    from datasets import concatenate_datasets, load_dataset
    import dspy

    # The loader has no generative branch. It derives the label set from the
    # label column's DISTINCT VALUES over the concatenated splits -- which, for
    # a generative task, is the set of ANSWERS, including the held-out ones.
    # Those then get rendered into the prompt as a Literal[...] enumeration by
    # build_signature, i.e. the test answers are handed to the model. Three of
    # the four build_program call sites also drop `spec`, so the generative
    # signature that does exist is never reached from a real run.
    #
    # Refusing is the only honest option until the loader, the runner and the
    # optimizer are wired for it: a silent leak produces a beautiful accuracy
    # number that means nothing.
    if getattr(spec, "is_generative", False):
        raise NotImplementedError(
            f"task {spec.key!r} is kind={spec.kind!r}, and load_task only implements "
            f"classification. Running it here would derive the Literal label set from "
            f"every answer in the dataset -- including held-out test answers -- and put "
            f"them in the prompt. See scoring.py / build_generative_signature for the "
            f"pieces that exist; the loader and experiment runner are not wired up.")

    ds = load_dataset(spec.repo, spec.config) if spec.config else load_dataset(spec.repo)
    wanted = spec.splits or tuple(ds.keys())
    full = concatenate_datasets([ds[s] for s in wanted if s in ds])

    feat = full.features[spec.label_column]
    names = tuple(feat.names) if hasattr(feat, "names") else tuple(
        sorted({str(v) for v in full[spec.label_column]}))
    to_name = (lambda v: names[v]) if hasattr(feat, "names") else (lambda v: str(v))

    # Deduplicate before splitting: an exact duplicate straddling train and test
    # silently turns a held-out item into a memorised one.
    seen: set[str] = set()
    by_label: dict[str, list[tuple[str, str]]] = {lab: [] for lab in names}
    n_dupes = 0
    for row in full:
        text = str(row[spec.text_column]).strip()
        if not text or text in seen:
            n_dupes += text in seen
            continue
        seen.add(text)
        lab = to_name(row[spec.label_column])
        by_label.setdefault(lab, []).append((text, lab))

    rng = np.random.default_rng(seed)
    working = {lab: list(by_label.get(lab, [])) for lab in names}

    def to_examples(pairs):
        return [dspy.Example(sentence=t, label=l).with_inputs("sentence") for t, l in pairs]

    test = to_examples(_stratified_take(rng, working, n_test, names))
    val = to_examples(_stratified_take(rng, working, n_val, names))
    train = to_examples(_stratified_take(rng, working, n_train, names))
    pool = to_examples([item for lab in names for item in working[lab]])

    task = Task(name=spec.key, labels=names, train=train, val=val, test=test, pool=pool)
    task.n_duplicates_dropped = n_dupes
    task.spec = spec
    return task


def load_financial_phrasebank(n_train: int = 200, n_val: int = 200, n_test: int = 250,
                              seed: int = 0) -> Task:
    """Load Financial PhraseBank (all-agree) and split it for optimizer-safe eval.

    Split sizes are deliberate:
      test  250 -- large enough that the 95% Wilson half-width is ~4.5% at 85%
                   accuracy. Below ~150 the interval gets so wide that no two
                   candidate models can be told apart, which would make the
                   chart decorative. See stats.resolution_warning.
      val   200 -- GEPA's Pareto frontier needs enough items to rank candidates
                   without latching onto a handful of lucky examples.
      train 200 -- GEPA samples reflective minibatches (default 3) from here.
      pool  rest -- reserved for encoder fine-tuning, which needs far more
                   labels than prompt optimization does.

    The order matters: test is carved out *first*, before anything else can
    touch it.
    """
    from datasets import concatenate_datasets, load_dataset
    import dspy

    ds = load_dataset(DATASET_REPO)
    full = concatenate_datasets([ds["train"], ds["test"]])
    names = tuple(full.features["label"].names)  # ('negative', 'neutral', 'positive')

    # Deduplicate before splitting. The raw set has 5 exact-duplicate sentences;
    # left in, one can land in train and its twin in test, which silently turns a
    # held-out item into a memorised one. Small here, fatal as a habit.
    seen: set[str] = set()
    by_label: dict[str, list[tuple[str, str]]] = {lab: [] for lab in names}
    n_dupes = 0
    for row in full:
        text = row["text"].strip()
        if text in seen:
            n_dupes += 1
            continue
        seen.add(text)
        by_label[names[row["label"]]].append((text, names[row["label"]]))

    rng = np.random.default_rng(seed)
    working = {lab: list(by_label[lab]) for lab in names}

    def to_examples(pairs):
        return [dspy.Example(sentence=t, label=l).with_inputs("sentence") for t, l in pairs]

    test = to_examples(_stratified_take(rng, working, n_test, names))
    val = to_examples(_stratified_take(rng, working, n_val, names))
    train = to_examples(_stratified_take(rng, working, n_train, names))
    pool = to_examples([item for lab in names for item in working[lab]])

    task = Task(name="financial_phrasebank_allagree", labels=names,
                train=train, val=val, test=test, pool=pool)
    task.n_duplicates_dropped = n_dupes
    return task

def verify_or_record_split(payload: dict, task, path: str) -> None:
    """Refuse to add rows to a run whose split cannot be shown to match.

    Pairing is positional. If the stored split and the freshly rebuilt one are
    not the same items in the same order, every McNemar / non-inferiority
    number computed across them is invalid -- and the previous check (test-set
    LENGTH only) could not tell the difference. See data.split_fingerprint.

    Runs written before fingerprints existed have none. Those get the
    fingerprint recorded and a loud note, because "unverified" and "verified"
    must not read the same: the rows already on disk were all measured inside
    one dataset load, which is good evidence but not proof.
    """
    from downshift.data import split_fingerprint
    fresh = split_fingerprint(task.test)
    stored = (payload.get("task") or {}).get("split_fingerprint")
    if stored is None:
        payload.setdefault("task", {})["split_fingerprint"] = fresh
        print(f"  NOTE: {path} predates split fingerprinting. Recording {fresh} now.\n"
              f"        Rows already in this file were paired on an UNVERIFIED split "
              f"(same seed and size, but item identity was never checked).")
        return
    if stored != fresh:
        raise SystemExit(
            f"REFUSING to write {path}: the rebuilt test split does not match the one "
            f"these rows were measured on.\n  stored {stored}\n  rebuilt {fresh}\n"
            f"Same seed and size are NOT sufficient -- the split is drawn from per-class "
            f"lists in source-row order, so an upstream dataset change reorders it. Any "
            f"paired comparison across these two splits would be invalid.")
