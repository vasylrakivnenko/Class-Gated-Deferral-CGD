import json, sys, os
sys.path.insert(0,"src")
import numpy as np
from downshift import stats
task, new_ref = sys.argv[1], sys.argv[2]
p=f"runs/{task}/results.json"; d=json.load(open(p)); r=d["results"]; cfg=d["config"]
assert new_ref in r, f"{new_ref} not evaluated on {task}"
ref=r[new_ref]; margin=cfg["non_inferiority_margin"]
comp,raw={},{}
for k,v in r.items():
    if k==new_ref or v["n"]!=ref["n"]: continue
    a,b=np.array(v["correct"],bool),np.array(ref["correct"],bool)
    mc,ni=stats.mcnemar_test(a,b),stats.non_inferiority_test(a,b,margin=margin)
    comp[k]={"mcnemar":mc.to_dict(),"non_inferiority":ni.to_dict()}; raw[k]=mc.p_value
for (k,pv),rej in zip(raw.items(), stats.holm_bonferroni(list(raw.values()))):
    comp[k]["holm_significant"]=bool(rej)
d["comparisons"]=comp; d["reference_model"]=new_ref
json.dump(d,open(p,"w"),indent=2,default=str)
print(f"{task}: reference -> {ref['label']} ({ref['accuracy']['point']:.1%}, ${ref['cost_per_1k_calls']:.4f}/1k)")
