import os, time, numpy as np, collections
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["HF_DATASETS_OFFLINE"]="1"
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score

d = load_dataset("cardiffnlp/tweet_eval","sentiment")
full_txt=list(d["train"]["text"]); full_y=np.array(d["train"]["label"])
# EXACTLY the indices bench2.py used for ModernBERT
idx2000 = np.random.RandomState(0).permutation(len(full_txt))[:2000]
tr_txt=[full_txt[i] for i in idx2000]; tr_y=full_y[idx2000]
te=d["test"].select(range(2000)); te_txt=list(te["text"]); te_y=np.array(te["label"])
full_te_txt=list(d["test"]["text"]); full_te_y=np.array(d["test"]["label"])

def rep(tag,y,p):
    print(f"  {tag:44s} acc={accuracy_score(y,p):.4f} macroF1={f1_score(y,p,average='macro'):.4f} macroRec={recall_score(y,p,average='macro'):.4f}",flush=True)

print("MAJORITY baseline test[:2000] acc=", max(collections.Counter(te_y).values())/2000)
print("MAJORITY baseline full test  acc=", max(collections.Counter(full_te_y).values())/len(full_te_y))

# TF-IDF on the SAME 2000 train rows, both eval sets
pp=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True),LogisticRegression(max_iter=2000,C=4.0)).fit(tr_txt,tr_y)
print("TFIDF n=2000:")
rep("eval=test[:2000]",te_y,pp.predict(te_txt))
rep("eval=FULL test(12284)",full_te_y,pp.predict(full_te_txt))

for mname in ["intfloat/multilingual-e5-small"]:
    m=SentenceTransformer(mname,device="mps")
    for pref in ["", "query: "]:
        t0=time.time()
        Xtr=m.encode([pref+t for t in tr_txt],batch_size=128,show_progress_bar=False,normalize_embeddings=True)
        Xte=m.encode([pref+t for t in te_txt],batch_size=128,show_progress_bar=False,normalize_embeddings=True)
        et=time.time()-t0
        clf=LogisticRegression(max_iter=3000,C=4.0).fit(Xtr,tr_y)
        print(f"{mname} prefix={pref!r} embed={et:.1f}s dim={Xtr.shape[1]}")
        rep("eval=test[:2000]",te_y,clf.predict(Xte))
