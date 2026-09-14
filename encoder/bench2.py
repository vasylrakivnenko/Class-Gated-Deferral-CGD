import os, time, sys
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

MODEL = sys.argv[1] if len(sys.argv) > 1 else "answerdotai/ModernBERT-base"
NTRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
DEV = "mps"
SEQ, BS, EPOCHS, LR = 128, 16, 3, 5e-5

d = load_dataset("cardiffnlp/tweet_eval", "sentiment")
rs = np.random.RandomState(0)
tr_idx = rs.permutation(len(d["train"]))[:NTRAIN]
tr_txt = [d["train"][int(i)]["text"] for i in tr_idx]
tr_y = [d["train"][int(i)]["label"] for i in tr_idx]
te = d["test"].select(range(2000))
te_txt, te_y = list(te["text"]), list(te["label"])

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3, attn_implementation="sdpa")
print(f"LOAD_SEC {time.time()-t0:.1f}  PARAMS {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
      f"MAXPOS {model.config.max_position_embeddings} VOCAB {model.config.vocab_size}", flush=True)
model.to(DEV)

def encode(txts):
    e = tok(txts, truncation=True, max_length=SEQ, padding="max_length", return_tensors="pt")
    return e["input_ids"], e["attention_mask"]
ids, am = encode(tr_txt); y = torch.tensor(tr_y)
tids, tam = encode(te_txt); ty = np.array(te_y)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
n = len(tr_txt); spe = n // BS
model.train(); torch.mps.synchronize(); t0 = time.time(); step = 0; t_warm = None
for ep in range(EPOCHS):
    perm = torch.randperm(n)
    for i in range(spe):
        idx = perm[i*BS:(i+1)*BS]
        out = model(input_ids=ids[idx].to(DEV), attention_mask=am[idx].to(DEV), labels=y[idx].to(DEV))
        out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True); step += 1
        if step == 10: torch.mps.synchronize(); t_warm = time.time()
torch.mps.synchronize()
train_sec = time.time() - t0
print(f"TRAIN n={n} epochs={EPOCHS} bs={BS} seq={SEQ} steps={step} "
      f"TOTAL_SEC {train_sec:.1f} SEC_PER_STEP {(time.time()-t_warm)/(step-10):.3f} "
      f"EX_PER_SEC {n*EPOCHS/train_sec:.1f}", flush=True)
print(f"MPS_ALLOC_GB {torch.mps.current_allocated_memory()/1e9:.2f} "
      f"DRIVER_GB {torch.mps.driver_allocated_memory()/1e9:.2f}", flush=True)

model.eval(); preds = []
with torch.no_grad():
    for i in range(0, len(te_txt), 64):
        lo = model(input_ids=tids[i:i+64].to(DEV), attention_mask=tam[i:i+64].to(DEV)).logits
        preds.append(lo.argmax(-1).cpu().numpy())
p = np.concatenate(preds)
print(f"EVAL acc={accuracy_score(ty,p):.4f} macroF1={f1_score(ty,p,average='macro'):.4f} "
      f"macroRecall={f1_score(ty,p,average='macro'):.4f}", flush=True)
from sklearn.metrics import recall_score
print(f"EVAL macroRecall_true={recall_score(ty,p,average='macro'):.4f}", flush=True)

def bench(dev, bs, reps=20):
    m = model.to(dev); m.eval()
    a, b = tids[:bs].to(dev), tam[:bs].to(dev)
    with torch.no_grad():
        for _ in range(5): m(input_ids=a, attention_mask=b)
        if dev == "mps": torch.mps.synchronize()
        t = time.time()
        for _ in range(reps): m(input_ids=a, attention_mask=b)
        if dev == "mps": torch.mps.synchronize()
        el = time.time() - t
    return el/reps*1000, bs*reps/el
for dev in ["mps", "cpu"]:
    for bs in [1, 32]:
        ms, tps = bench(dev, bs)
        print(f"INFER {dev} bs={bs} seq={SEQ}: {ms:.1f} ms/batch, {tps:.0f} texts/sec", flush=True)
