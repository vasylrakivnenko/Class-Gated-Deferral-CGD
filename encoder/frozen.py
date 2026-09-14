import os, time, numpy as np
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

d = load_dataset("cardiffnlp/tweet_eval", "sentiment")
rs = np.random.RandomState(0)
full_tr_txt = list(d["train"]["text"]); full_tr_y = np.array(d["train"]["label"])
te = d["test"].select(range(2000))
te_txt, te_y = list(te["text"]), np.array(te["label"])

for mname in ["sentence-transformers/all-MiniLM-L6-v2", "intfloat/multilingual-e5-small"]:
    m = SentenceTransformer(mname, device="mps")
    t0 = time.time()
    sub = rs.permutation(len(full_tr_txt))[:8000]
    tr_txt = [full_tr_txt[i] for i in sub]; tr_y = full_tr_y[sub]
    Xtr = m.encode(tr_txt, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    Xte = m.encode(te_txt, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    emb_s = time.time() - t0
    print(f"\n[{mname}] dim={Xtr.shape[1]} embed_time_for_{len(tr_txt)+len(te_txt)}_texts={emb_s:.1f}s "
          f"({(len(tr_txt)+len(te_txt))/emb_s:.0f} texts/sec on MPS)", flush=True)
    for n in [100, 250, 500, 1000, 2000, 4000, 8000]:
        clf = LogisticRegression(max_iter=3000, C=4.0).fit(Xtr[:n], tr_y[:n])
        p = clf.predict(Xte)
        print(f"    n={n:5d} acc={accuracy_score(te_y,p):.4f} macroF1={f1_score(te_y,p,average='macro'):.4f} "
              f"macroRec={recall_score(te_y,p,average='macro'):.4f}", flush=True)
