"""Sources, adapted into the one shape the pipeline ingests.

Every dataset-specific fact lives in this file: where the bytes come from, how
labels are numbered, what each class is called in words. The pipeline reads none
of it -- it is handed a `Dataset` and never learns which loader produced it.

Adding a dataset is adding a function here. It is not an edit to the pipeline,
and it is certainly not the thing `run_banking77.py` used to do, which was to
reach into the pipeline and mutate its label table before calling it.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from downshift.dataset import Dataset, register              # noqa: E402

# Label wording is a property of the dataset, so it lives with the dataset. It
# is used verbatim as the NLI hypothesis, taken from each source's own
# vocabulary where that is descriptive and from the task description where the
# vocabulary is a bare yes/no. Recorded here so it is auditable -- not tuned
# against results.
OCL_LABELS = {
    "imdb": {0: "negative sentiment", 1: "positive sentiment"},
    "hatespeech": {0: "no hate speech", 1: "hate speech"},
    "fever": {0: "a false claim", 1: "a true claim"},
}


def _ocl(task: str) -> Dataset:
    import reconstruct as R
    streams, pins = R.load_streams()
    s = streams[task]
    names = OCL_LABELS.get(task) or {v: k for k, v in R.ISEAR_TO_ID.items()}
    return Dataset(
        name=task,
        texts=np.asarray(s["text"], dtype=object),
        gold=np.asarray(s["gold"], dtype=int),
        label_names=names,
        expert=np.asarray(s["gpt3.5"]["pred"], dtype=int),
        source=f"Online Cascade Learning (Nie et al., ICML 2024), stream {task!r}; "
               f"repo pinned at {R.OCL_COMMIT}",
        notes=["teacher is GPT-3.5 Turbo, as published",
               "every LLM cell of the paper's Table 1 is re-derived by reconstruct.py"],
    )


for _t in ("imdb", "hatespeech", "isear", "fever"):
    register(_t)(lambda t=_t: _ocl(t))


@register("banking77")
def banking77() -> Dataset:
    """77 banking intents over 13,083 short queries.

    Train and test are pooled and re-split by the pipeline's own grouped CV, so
    every item is predicted out-of-fold exactly once. No per-item LLM stream
    exists for the full split -- `runs/banking77/` holds 251 paired test items
    against 20 models, which at 77 classes is ~3 per class and a resolution floor
    near +/-0.033 -- so no expert is supplied and the cascade half reports itself
    as not measurable.
    """
    from datasets import load_dataset
    hf = "legacy-datasets/banking77"
    d = load_dataset(hf)
    names = d["train"].features["label"].names
    return Dataset(
        name="banking77",
        texts=np.array(list(d["train"]["text"]) + list(d["test"]["text"]), dtype=object),
        gold=np.array(list(d["train"]["label"]) + list(d["test"]["label"]), dtype=int),
        label_names={i: n.replace("_", " ") for i, n in enumerate(names)},
        expert=None,
        source=f"HuggingFace {hf} (Casanueva et al. 2020), train+test pooled",
        notes=["no per-item LLM stream for the full split, so the cascade is skipped",
               "first dataset here to trigger label_shortlist(): 65,415 pair-scoring "
               "passes instead of 1,007,391"],
    )


@register("clinc150")
def clinc150() -> Dataset:
    """150 assistant intents over 22,500 short utterances, 150 items per class.

    The confirmation run for the class-count hypothesis. banking77 is the
    current high-water mark at k=77, and the question CLINC150 answers is
    whether what was seen there is a property of having many classes or a
    property of banking77.

    It is the cleanest instrument available for that: exactly balanced, so the
    resolution floor is set by 150 items per class rather than by a rare class
    (+/-0.0066 on banking77 against +/-0.043 on the only multi-class LegalBench
    task of comparable size). Balance also removes the confound that makes a
    per-class rule look good for the wrong reason -- on a skewed dataset a rule
    that merely notices the minority classes gains balanced accuracy without
    knowing anything about classes.

    `oos` is dropped. It is 1,350 of the 23,850 rows and it is not an intent:
    it is the out-of-scope REJECT bucket, a different task (open-set detection)
    wearing a class label. Keeping it would put one class at 9x the size of
    every other, re-introducing exactly the imbalance this dataset was chosen
    to avoid, and would let a model score on "is this in scope at all" while the
    question here is which of 150 intents it is. Dropping it renumbers the
    labels, which is done here because the pipeline requires 0..k-1 contiguous
    and a gap at the old index 42 would break every per-class array.

    All three splits are pooled: the pipeline re-splits with its own grouped CV,
    so every item is predicted out of fold exactly once and the upstream split
    boundary would only shrink the data.

    No per-item LLM stream exists, so -- as with banking77 -- no expert is
    supplied and the cascade half reports itself as not measurable rather than
    being silently skipped.
    """
    from datasets import load_dataset
    hf, cfg = "clinc/clinc_oos", "plus"
    d = load_dataset(hf, cfg)
    names = d["train"].features["intent"].names
    oos = names.index("oos")
    texts, gold = [], []
    for split in ("train", "validation", "test"):
        for t, y in zip(d[split]["text"], d[split]["intent"]):
            if y != oos:
                texts.append(t)
                gold.append(y)
    # contiguous 0..149, preserving the upstream order of the intents we kept
    remap = {old: new for new, old in
             enumerate(i for i in range(len(names)) if i != oos)}
    return Dataset(
        name="clinc150",
        texts=np.asarray(texts, dtype=object),
        gold=np.asarray([remap[y] for y in gold], dtype=int),
        label_names={remap[i]: n.replace("_", " ")
                     for i, n in enumerate(names) if i != oos},
        expert=None,
        source=f"HuggingFace {hf} config {cfg!r} (Larson et al., EMNLP 2019), "
               f"train+validation+test pooled, out-of-scope rows removed",
        notes=["no per-item LLM stream, so the cascade half is skipped",
               "oos dropped: 1,350 open-set-reject rows, not a 151st intent",
               "exactly 150 items per class, so the resolution floor is not set "
               "by a rare class"],
    )
