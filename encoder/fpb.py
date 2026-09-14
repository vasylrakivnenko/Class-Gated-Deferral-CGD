import numpy as np, time
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
for repo, cfg in [("gtfintechlab/financial_phrasebank_sentences_allagree", "5768"),
                  ("takala/financial_phrasebank", "sentences_allagree")]:
    try:
        d = load_dataset(repo, cfg) if cfg else load_dataset(repo)
        print("OK", repo, {k: (len(v), v.column_names) for k, v in d.items()}, flush=True)
        sp = d["train"]
        tcol = "sentence" if "sentence" in sp.column_names else ("text" if "text" in sp.column_names else sp.column_names[0])
        lcol = "label" if "label" in sp.column_names else sp.column_names[-1]
        X = list(sp[tcol]); yraw = list(sp[lcol])
        uniq = sorted(set(yraw)); m = {u: i for i, u in enumerate(uniq)}
        y = np.array([m[v] for v in yraw])
        print("  textcol", tcol, "labelcol", lcol, "labels", uniq, "n", len(X), flush=True)
        pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True),
                             LogisticRegression(max_iter=3000, C=4.0))
        t0=time.time()
        s = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="accuracy")
        f1 = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="f1_macro")
        print(f"  TFIDF+LR 5-fold CV acc={s.mean():.4f}+-{s.std():.4f} macroF1={f1.mean():.4f} total_fit={time.time()-t0:.1f}s", flush=True)
        break
    except Exception as e:
        print("FAIL", repo, type(e).__name__, str(e)[:150], flush=True)
