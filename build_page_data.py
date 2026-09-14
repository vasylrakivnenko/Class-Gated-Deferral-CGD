"""Regenerate the embedded chart data in model_pricing.html for ALL tasks.

Emits {tasks: {key: {...}}} so the page can switch datasets without a reload.
"""
import json, re, glob, os, sys
sys.path.insert(0, "src")

# The per-classification cost derivation lives with the fields it mirrors, so
# every re-charting script agrees with the pipeline that wrote the run.
from downshift.evaluate import row_cost_per_1k_items, row_n_calls
from downshift.program import record_prompt_changed

CLASSICAL={"majority","tfidf-logreg","frozen-embed"}; ENCODER={"finetuned-encoder","modernbert"}
fam=lambda k:"classical" if k in CLASSICAL else "encoder" if k in ENCODER else "llm"
LOCAL=CLASSICAL|ENCODER



tasks={}
LOCAL_FIT = {"tfidf-logreg", "frozen-embed", "finetuned-encoder", "modernbert"}

for path in sorted(glob.glob("runs/*/results.json")):
    key=os.path.basename(os.path.dirname(path))
    d=json.load(open(path)); res=d['results']; comp=d.get('comparisons',{}); task=d['task']
    opts=d.get('optimizations',{})
    rows=[]
    for k,r in res.items():
        base=k.replace("+gepa","")
        linked=None
        if k.endswith("+gepa") and base in res:
            b=res[base]; linked={"cost":round(row_cost_per_1k_items(b),8),"acc":b["accuracy"]["point"]}
        c=comp.get(k,{}); ni=c.get("non_inferiority",{}); mc=c.get("mcnemar",{})
        # A row measured under a superseded scorer or otherwise not comparable
        # says so in its own tooltip. Silently ranking it beside rows measured
        # differently is how a chart tells a lie it was never asked to tell.
        note=" ".join(r.get("scorer_note","").split())[:300] or (
            " ".join(r.get("instruction","").split())[:200] if k in LOCAL else "")
        rows.append({"key":k,"label":r["label"],"family":fam(base),
          "cost":round(row_cost_per_1k_items(r),8),"cost_call":round(r["cost_per_1k_calls"],8),
          # Only the optimized half of a pair can answer "did GEPA change the
          # prompt". A stock row shares the base key, so asking there would tag
          # it with its own optimization's verdict -- a flag about a run that
          # row is not.
          "calls":row_n_calls(r),
          "changed":record_prompt_changed(opts.get(base)) if k.endswith("+gepa") else None,
          "acc":r["accuracy"]["point"],
          "lo":r["accuracy"]["lo"],"hi":r["accuracy"]["hi"],
          "tin":round(r["mean_in_tokens"],1),"tout":round(r["mean_out_tokens"],1),
          "tcached":round(r.get("mean_cached_tokens",0),1),
          "secs":round(r["wall_seconds"]),"n":r["n"],
          # What wall_seconds MEANS. Encoder rows store FIT time (1,347s of GPU
          # training on banking77); LLM rows store the duration of the whole
          # concurrent batch. The page rendered both under one "Wall time"
          # label, inviting a comparison no reader could make correctly.
          # Default "batch" for rows written before the field existed -- true
          # for every LLM row; the encoder/classical rows are re-derived below.
          "basis":r.get("timing_basis") or ("fit" if k in LOCAL_FIT else "batch"),
          # `passes` is 31 uncorrected one-sided 5% non-inferiority tests. This
          # says whether the same verdict survives a Holm correction over that
          # family, so a green PASS that only exists because nobody corrected
          # for multiplicity is visible instead of implied.
          "passes_holm":c.get("passes_holm_corrected"),
          "perclass":{a:round(b,3) for a,b in list(r["per_class_accuracy"].items())[:6]},
          "nclasses":len(r["per_class_accuracy"]),
          "linked":linked,"verdict":ni.get("verdict",""),"passes":ni.get("passes"),
          "diff":ni.get("diff"),"p":mc.get("p_value"),"isref":k==d["reference_model"],"note":note})
    rows.sort(key=lambda r:(r["cost"],-r["acc"]))
    tasks[key]={"key":key,"label":task.get("label",key),"rows":rows,
      "majority":task["majority_accuracy"],"bar":d["accuracy_bar"],
      "ntest":task["n_test"],"ref":d["reference_model"],
      "nclasses":len(task.get("labels",[])),"resolution":task["resolution_note"]}
    # Printed per task because both bugs are per-row facts: a run with no
    # retried items and no unchanged prompt is one where neither fix moves
    # anything, and that is worth being able to see at a glance.
    retried=[r for r in rows if r["calls"]>r["n"]]
    noimp=[r for r in rows if r["linked"] and r["changed"] is False]
    worst=max(retried,key=lambda r:r["calls"]/r["n"],default=None)
    print(f"  {key:<22} {len(rows)} rows, {len(task.get('labels',[]))} classes")
    print(f"    bug 1: {len(retried)} rows billed for more calls than items"
          + (f" (worst {worst['key']} {worst['calls']}/{worst['n']} calls,"
             f" ${worst['cost_call']:.5f}/1k calls -> ${worst['cost']:.5f}/1k items)" if worst else ""))
    print(f"    bug 2: {len(noimp)} GEPA pair(s) kept the baseline prompt"
          + (f" ({', '.join(r['key'] for r in noimp)})" if noimp else ""))

data={"tasks":tasks,"default":"financial_phrasebank"}
html=open('model_pricing.html').read()
payload='const CHART = '+json.dumps(data,separators=(',',':'))+';\n'
new,n=re.subn(r'const CHART = \{.*?\};\n', lambda m: payload, html, count=1, flags=re.S)
# Assert the splice HAPPENED, not that the bytes moved: rebuilding after a
# no-op change produces an identical payload, and "nothing to do" is not a
# failure. (It was asserted as one, so a second run in a row crashed.)
assert n==1 and new.count('const CHART = ')==1, f"spliced {n} sites"
open('model_pricing.html','w').write(new)
print(f"embedded {len(tasks)} tasks")
