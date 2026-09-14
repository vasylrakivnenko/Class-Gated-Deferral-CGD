"""DynaSent Round 2 loader — offline, license-clean (CC BY 4.0), no HF loading script.

Primary path: official GitHub zip (17 MB, one download, cached locally).
Fallback:     HF parquet mirror HelloWorld2307/dynasent (r1+r2 test concatenated).

Test/dev splits are the human-verified gold: every item was labeled by 5
crowdworkers and kept only if >=4 of 5 agreed. Perfectly balanced 240/240/240.
"""
import json, os, random, ssl, subprocess, urllib.request, zipfile
from pathlib import Path

ZIP_URL = "https://github.com/cgpotts/dynasent/raw/main/dynasent-v1.1.zip"
CACHE = Path(os.environ.get("DYNASENT_DIR", Path.home() / ".cache" / "dynasent"))
LABELS = ("negative", "neutral", "positive")


def _ensure_zip() -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    zp = CACHE / "dynasent-v1.1.zip"
    if not zp.exists() or zp.stat().st_size < 1_000_000:
        try:
            ctx = None
            try:
                import certifi
                ctx = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                pass
            with urllib.request.urlopen(ZIP_URL, timeout=180, context=ctx) as r:
                zp.write_bytes(r.read())
        except Exception:
            # python.org builds on macOS often have no CA bundle -> use curl
            subprocess.run(["curl", "-sSfL", "-o", str(zp), ZIP_URL], check=True)
    return zp


def load_split(round_: str = "r2", split: str = "test"):
    """round_ in {'r1','r2'}; split in {'train','dev','test'}.
    Returns list of {'text', 'label', 'n_agree'} with ternary labels only."""
    stem = {"r1": "round01-yelp", "r2": "round02-dynabench"}[round_]
    name = f"dynasent-v1.1/dynasent-v1.1-{stem}-{split}.jsonl"
    with zipfile.ZipFile(_ensure_zip()) as z:
        raw = z.read(name).decode("utf-8")
    out = []
    for line in raw.strip().split("\n"):
        d = json.loads(line)
        if d["gold_label"] not in LABELS:      # drops 'mixed' and None (train only)
            continue
        ld = d["label_distribution"]
        out.append({"text": d["sentence"],
                    "label": d["gold_label"],
                    "n_agree": max(len(v) for v in ld.values())})
    return out


def demo_sets(n_holdout=150, n_train=200, seed=0):
    """Human-verified holdout from r2 TEST; GEPA train/val from r2 TRAIN (>=4/5 only)."""
    rng = random.Random(seed)
    test = load_split("r2", "test")            # 720, all >=4/5, balanced
    per = n_holdout // 3
    by = {l: [r for r in test if r["label"] == l] for l in LABELS}
    holdout = [r for l in LABELS for r in rng.sample(by[l], per)]
    rng.shuffle(holdout)

    train_pool = [r for r in load_split("r2", "train") if r["n_agree"] >= 4]
    byt = {l: [r for r in train_pool if r["label"] == l] for l in LABELS}
    k = min(n_train // 3, min(len(v) for v in byt.values()))
    train = [r for l in LABELS for r in rng.sample(byt[l], k)]
    rng.shuffle(train)
    return train, holdout


if __name__ == "__main__":
    import collections
    tr, ho = demo_sets()
    print("train", len(tr), collections.Counter(r["label"] for r in tr))
    print("holdout", len(ho), collections.Counter(r["label"] for r in ho))
    print("holdout min agreement:", min(r["n_agree"] for r in ho), "/5")
    print("example:", ho[0])
