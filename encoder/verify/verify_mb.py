import os,time,sys
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1"); os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["HF_DATASETS_OFFLINE"]="1"
import torch,numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score,f1_score,recall_score
MODEL="answerdotai/ModernBERT-base"; DEV="mps"; SEQ,BS,EPOCHS=128,16,3
d=load_dataset("cardiffnlp/tweet_eval","sentiment")
idx=np.random.RandomState(0).permutation(len(d["train"]))[:2000]
tr=d["train"].select(idx); tr_txt=list(tr["text"]); tr_y=list(tr["label"])
te=d["test"].select(range(2000)); te_txt=list(te["text"]); te_y=np.array(te["label"])
tok=AutoTokenizer.from_pretrained(MODEL)
def enc(t):
    e=tok(t,truncation=True,max_length=SEQ,padding="max_length",return_tensors="pt"); return e["input_ids"],e["attention_mask"]
ids,am=enc(tr_txt); y=torch.tensor(tr_y); tids,tam=enc(te_txt)
for LR in [5e-5,3e-5,2e-5]:
    torch.manual_seed(42)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=3,attn_implementation="sdpa",dtype=torch.float32).to(DEV)
    print("PARAMS %.1fM"%(sum(p.numel() for p in model.parameters())/1e6),flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=LR); n=len(tr_txt); spe=n//BS
    model.train(); torch.mps.synchronize(); t0=time.time()
    for ep in range(EPOCHS):
        perm=torch.randperm(n)
        for i in range(spe):
            b=perm[i*BS:(i+1)*BS]
            out=model(input_ids=ids[b].to(DEV),attention_mask=am[b].to(DEV),labels=y[b].to(DEV))
            out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.mps.synchronize(); ts=time.time()-t0
    model.eval(); pr=[]
    with torch.no_grad():
        for i in range(0,len(te_txt),64):
            pr.append(model(input_ids=tids[i:i+64].to(DEV),attention_mask=tam[i:i+64].to(DEV)).logits.argmax(-1).cpu().numpy())
    p=np.concatenate(pr)
    print(f"LR={LR:g} TRAIN_SEC={ts:.1f} MPS_ALLOC_GB={torch.mps.current_allocated_memory()/1e9:.2f} DRIVER_GB={torch.mps.driver_allocated_memory()/1e9:.2f} acc={accuracy_score(te_y,p):.4f} macroF1={f1_score(te_y,p,average='macro'):.4f} macroRec={recall_score(te_y,p,average='macro'):.4f}",flush=True)
    del model,opt; torch.mps.empty_cache()
