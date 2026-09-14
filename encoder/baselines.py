import time, numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

def run(name, tr_txt, tr_y, te_txt, te_y):
    t0 = time.time()
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000),
        LogisticRegression(max_iter=2000, C=4.0),
    )
    pipe.fit(tr_txt, tr_y)
    fit_s = time.time() - t0
    p = pipe.predict(te_txt)
    print(f"[{name}] TFIDF+LR  n_train={len(tr_txt)} n_test={len(te_txt)} "
          f"acc={accuracy_score(te_y,p):.4f} macroF1={f1_score(te_y,p,average='macro'):.4f} fit={fit_s:.1f}s", flush=True)
    # learning curve
    for n in [100, 250, 500, 1000, 2000]:
        if n > len(tr_txt): break
        idx = np.random.RandomState(0).permutation(len(tr_txt))[:n]
        pp = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True),
                           LogisticRegression(max_iter=2000, C=4.0))
        pp.fit([tr_txt[i] for i in idx], [tr_y[i] for i in idx])
        q = pp.predict(te_txt)
        print(f"    n={n:5d} acc={accuracy_score(te_y,q):.4f} macroF1={f1_score(te_y,q,average='macro'):.4f}", flush=True)

try:
    d = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    run("tweet_eval/sentiment", d["train"]["text"], d["train"]["label"], d["test"]["text"], d["test"]["label"])
except Exception as e:
    print("tweet_eval FAILED", type(e).__name__, e, flush=True)

try:
    d = load_dataset("takala/financial_phrasebank", "sentences_allagree", trust_remote_code=True)["train"]
    d = d.train_test_split(test_size=0.2, seed=42)
    run("financial_phrasebank/allagree", d["train"]["sentence"], d["train"]["label"], d["test"]["sentence"], d["test"]["label"])
except Exception as e:
    print("financial_phrasebank FAILED", type(e).__name__, e, flush=True)

try:
    d = load_dataset("stanfordnlp/sst2")
    run("sst2(val)", d["train"]["sentence"], d["train"]["label"], d["validation"]["sentence"], d["validation"]["label"])
except Exception as e:
    print("sst2 FAILED", type(e).__name__, e, flush=True)
