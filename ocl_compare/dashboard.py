"""One dashboard, regenerated from whatever artifacts are on disk.

Point it at `results/` and it picks up every pipeline run there -- the four OCL
streams, banking77, and anything added later -- with no per-dataset code. A new
dataset shows up as a new tab the next time this runs.

Self-contained on purpose: the data is inlined rather than fetched, because a
page opened from file:// cannot fetch() its siblings, and a dashboard that needs
a web server to look at is a dashboard nobody opens. No CDN either, so it works
with the network off.

Sections adapt to what a run actually contains. A dataset with no per-item LLM
predictions has no cascade dial and no G8, and those panels disappear instead of
rendering an empty frame.
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _provenance as P                                   # noqa: E402

SKIP = ("reconstruct", "label_matched", "dashboard")

# A run is either the current answer for its dataset, or a variant of it. The
# first version put all of them in one flat tab row labelled by file stem, so
# "isear 7c main", "isear 7c ablation_nonli", "isear 7c llmlabels" sat side by
# side as if they were three datasets. They are one dataset and two experiments
# on it.
VARIANTS = {
    "ablation_nonli": "without NLI",
    "ablation_norerank": "without reranker",
    "llmlabels": "trained on LLM labels",
}


def collect(results_dir: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        base = os.path.basename(path)[:-5]
        if any(base.startswith(s) for s in SKIP):
            continue
        try:
            d = json.load(open(path))
        except Exception:
            continue
        # Multi-task runs emit {"tasks": [...]}; a single-dataset run emits the
        # task dict at the top level. banking77 is the second shape and was
        # dropped entirely by the first version of this, which only looked for
        # the first.
        tasks = d.get("tasks")
        if not isinstance(tasks, list):
            tasks = [d] if isinstance(d.get("profile"), dict) else []
        for t in tasks:
            if not isinstance(t, dict) or "profile" not in t:
                continue
            out.append({"run": base, "trained_on": t.get("trained_on", "gold"),
                        "mtime": os.path.getmtime(path), **t})
    # Canonical = the newest run of a dataset that is not a named experiment.
    # Everything else is a variant, including an older partial run that a later
    # full run has replaced -- which is how pipeline_fever_imdb ended up on the
    # tab bar as if it were a separate dataset.
    for t in out:
        suffix = t["run"].replace("pipeline_", "").replace("pipeline", "")
        t["_suffix"] = suffix
        t["variant"] = VARIANTS.get(suffix)
    by_task = {}
    for t in out:
        by_task.setdefault(t["task"], []).append(t)
    for task, group in by_task.items():
        plain = [t for t in group if t["variant"] is None]
        plain.sort(key=lambda t: -t["mtime"])
        for i, t in enumerate(plain):
            t["variant"] = None if i == 0 else "superseded by a later run"
    for t in out:
        t["canonical"] = t["variant"] is None
        t.pop("_suffix", None)
    out.sort(key=lambda t: (not t["canonical"], -t["profile"]["n_classes"], t["task"]))
    return out


def _slim(t: dict) -> dict:
    """Only what the page draws. The full artifact carries bootstrap draws and
    per-fold detail that would make the page tens of megabytes."""
    keep = ("task", "run", "variant", "canonical", "trained_on", "profile",
            "confusion", "conformal", "guaranteed_coverage",
            "margin_coverage", "candidates",
            "G3_roc_dominance", "G4_pool_diversity", "G5_class_advantage",
            "G7_base_rate_dominance", "G8_per_class_headroom", "deslib",
            "class_aware_layer", "free_arm_shipped", "recommendation",
            "versus_expert", "competence_board", "label_names", "budget_split")
    s = {k: t[k] for k in keep if k in t}
    d = t.get("deferral_curve")
    if d:
        s["deferral_curve"] = {
            "free_alone": d.get("free_alone"), "expert_alone": d.get("expert_alone"),
            "points": d.get("points", []),
            "cheapest_equivalent": d.get("cheapest_equivalent"),
            "best_point": d.get("best_point"),
            "best_point_optimistic": d.get("best_point_optimistic"),
            "selection_bias": d.get("selection_bias")}
    cr = t.get("class_router")
    if cr:
        s["class_router"] = {k: cr[k] for k in
                             ("balanced", "expert_share", "classes_to_expert",
                              "stable", "routable_rows", "board")
                             if k in cr}
    return s


TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>downshift pipeline — @@TITLE@@</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--ink:#e6e9ef;--dim:#8b93a7;
--faint:#6f7789;--ok:#4ea1ff;--good:#3ecf8e;--bad:#ff6b6b;--warn:#f2c14e;
--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{padding:20px 28px 0}
h1{font-size:19px;margin:0 0 2px;font-weight:600}
.sub{color:var(--dim);font-size:12.5px}
nav{padding:16px 28px 0}
.navrow{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px}
.navrow.variants{padding-left:10px;border-left:2px solid var(--line);margin-left:2px}
nav button.toggle{font-size:12px;color:var(--faint);padding:5px 11px}
nav button{background:var(--panel);color:var(--dim);border:1px solid var(--line);
border-radius:7px;padding:7px 13px;font-size:13px;cursor:pointer;font-family:inherit}
nav button:hover{color:var(--ink)}
nav button[aria-selected=true]{background:#1e2430;color:var(--ink);border-color:#3a4356}
nav button .n{color:var(--faint);font-size:11px;margin-left:6px}
main{padding:18px 28px 60px;display:grid;gap:16px;
grid-template-columns:repeat(6,minmax(0,1fr));align-items:start}
section{grid-column:span 3}
section{background:var(--panel);border:1px solid var(--line);border-radius:11px;
padding:16px 18px;min-width:0}
section.wide{grid-column:1/-1}
.cm{border-collapse:collapse;font-variant-numeric:tabular-nums}
.cm td,.cm th{border:none;padding:0;text-align:center}
.cm .lab{color:var(--dim);font-size:10px;padding:0 4px;text-align:right;white-space:nowrap;
max-width:96px;overflow:hidden;text-overflow:ellipsis}
.cm .diag{outline:1px solid rgba(62,207,142,.55);outline-offset:-1px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:var(--dim);
margin:0 0 12px;font-weight:600;display:flex;justify-content:space-between;gap:12px}
h2 .step{color:var(--faint);font-weight:400;letter-spacing:.04em}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:right;padding:4px 7px;border-bottom:1px solid var(--line);
font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
th{color:var(--dim);font-weight:500;font-size:11.5px;text-transform:uppercase;letter-spacing:.05em}
tr:last-child td{border-bottom:none}
tr.rule td{border-top:1px solid #3a4356}
.mono{font-family:var(--mono)}
.good{color:var(--good)} .bad{color:var(--bad)} .warn{color:var(--warn)} .dim{color:var(--dim)}
.pill{display:inline-block;padding:1px 7px;border-radius:99px;font-size:11px;
border:1px solid currentColor;opacity:.92;white-space:nowrap}
.big{font-size:27px;font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 14px;font-size:13px}
.kv div:nth-child(odd){color:var(--dim)}
.gate{display:grid;grid-template-columns:30px 88px 1fr;gap:10px;align-items:baseline;
padding:6px 0;border-bottom:1px solid var(--line)}
.gate:last-child{border-bottom:none}
.gate .tag{font:600 11.5px var(--mono)}
.gate .st{font-size:11px}
.gate .msg{font-size:12.5px;color:var(--dim);line-height:1.45}
.empty{color:var(--faint);font-size:12.5px;line-height:1.55;padding:10px 0 4px;
border-left:2px solid var(--line);padding-left:12px}
svg{display:block;width:100%;height:auto}
.heat td{padding:2px 4px;font-size:11.5px;border:none}
.heat th{font-size:10.5px;padding:3px 4px}
.heat .cell{text-align:center;border-radius:3px;font-variant-numeric:tabular-nums}
.scroll{max-height:440px;overflow:auto}
.hscroll{overflow-x:auto}
.note{font-size:12px;color:var(--dim);margin-top:10px;line-height:1.55}
.note button{background:var(--panel);color:var(--dim);border:1px solid var(--line);
border-radius:6px;padding:4px 10px;font-size:12px;cursor:pointer;font-family:inherit;margin-right:5px}
.note button[aria-selected=true]{background:#1e2430;color:var(--ink);border-color:#3a4356}
footer{padding:0 28px 40px;color:var(--faint);font-size:11.5px}
@media(max-width:1200px){main{grid-template-columns:repeat(2,minmax(0,1fr))}
  section{grid-column:span 1}}
@media(max-width:760px){main{grid-template-columns:1fr}}
</style>
<header>
  <h1>downshift &mdash; classification pipeline</h1>
  <div class="sub">@@SUBTITLE@@</div>
</header>
<nav id="nav"></nav>
<main id="main"></main>
<footer id="foot"></footer>
<script>
const DATA = @@DATA@@;
const META = @@META@@;

/* ======================================================================== *
 * Every dataset renders the SAME seven sections in the same order. A section
 * with nothing to show says why, rather than vanishing -- panels that appear
 * and disappear between tabs make two runs impossible to compare, and hide the
 * difference between "measured and empty" and "not measurable here".
 * ======================================================================== */

const f4 = x => (x==null || Number.isNaN(x)) ? "&mdash;" : x.toFixed(4);
const pc = x => (x==null) ? "&mdash;" : (100*x).toFixed(1) + "%";
const sgn = x => (x==null) ? "&mdash;" :
  `<span class="${x>0?'good':x<0?'bad':'dim'}">${x>=0?'+':''}${x.toFixed(4)}</span>`;
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const el = (tag,cls,html)=>{const e=document.createElement(tag);
  if(cls)e.className=cls; if(html!=null)e.innerHTML=html; return e;};
function sect(step,title,wide){
  const s=el("section", wide?"wide":null);
  s.appendChild(el("h2",null,`<span>${title}</span><span class="step">${step}</span>`));
  return s;
}
const empty = msg => el("div","empty", msg);

/* ---- shared accessors -------------------------------------------------- *
 * One definition of "the pool" for the whole page. The board comes from two
 * places -- competence_board on a run with no expert, class_router.board on a
 * run with one -- and the second includes the LLM as a member. Plotting that
 * as a candidate column changed what a column meant between tabs and let the
 * chart mark the LLM as winner of a class while the verdict said ship a free
 * model. Free candidates are columns; the expert is a separate overlay.       */
function boardOf(t0){
  const t=t0;
  const raw = t.competence_board || (t.class_router && t.class_router.board);
  if(!raw || !raw.shrunk) return null;
  const all = Object.keys(raw.shrunk);
  const isExpert = n => /EXPERT/i.test(n);
  const free = all.filter(n=>!isExpert(n));
  if(!free.length) return null;
  // Two different expert numbers exist and only one of them decides anything.
  // `EXPERT(own class)` is the LLM's precision on ITS OWN predictions;
  // `routable_rows` is its accuracy on the items the FREE arm called c, which is
  // what the router consults and can point the other way. On FEVER the own-class
  // row reads 0.749/0.877 -- "the LLM is worse at class 0" -- while the routable
  // row reads 0.847/0.758, so the LLM is better on both and the rule escalates
  // broadly. Plotting the first next to the free columns invited exactly that
  // misreading, so the chart plots the second.
  const rr = (t0.class_router||{}).routable_rows || null;
  return {shrunk:raw.shrunk, support:raw.support||{}, free,
          expert: all.find(isExpert) || null,
          routableExpert: rr ? rr["EXPERT | free says c"] : null,
          routableFree: rr ? rr["FREE winner | it says c"] : null,
          classes: Object.keys(raw.shrunk[free[0]]).sort((a,b)=>+a-+b)};
}
function constructions(t){
  const l=t.class_aware_layer||{}; const out={};
  Object.entries(l).forEach(([k,v])=>{
    if(v && typeof v==="object" && typeof v.balanced==="number") out[k]=v;});
  return out;
}
const hasExpert = t => t.expert !== null && !!t.deferral_curve;

/* ---- 1. verdict -------------------------------------------------------- */
function verdict(t){
  const s=sect("1","Verdict");
  const d=t.deferral_curve, bp=d&&d.best_point;
  const rank=(t.candidates&&t.candidates.ranking)||[];
  s.appendChild(el("div","big", bp? f4(bp.balanced) : f4(rank.length?rank[0].balanced:null)));
  s.appendChild(el("div","sub", bp
    ? `balanced accuracy at ${pc(bp.expert_share)} of traffic sent to the LLM`
    : "balanced accuracy, free arm alone &mdash; no LLM predictions for this dataset"));
  const kv=el("div","kv"); kv.style.marginTop="12px";
  const add=(k,v)=>{kv.appendChild(el("div",null,k));kv.appendChild(el("div",null,v));};
  add("ship", `<span class="mono">${esc(t.free_arm_shipped||"—")}</span>`);
  add("LLM share", bp? pc(bp.expert_share) : "0% (none available)");
  if(d) add("reference", `free alone ${f4(d.free_alone)} &middot; LLM alone ${f4(d.expert_alone)}`);
  if(bp) add("rule stability", bp.stable
    ? '<span class="good">same rule in every fold</span>'
    : '<span class="warn">unstable across folds</span>');
  if(d&&d.selection_bias!=null) add("selection bias", sgn(d.selection_bias));
  s.appendChild(kv);
  return s;
}

/* ---- 2. dataset -------------------------------------------------------- */
function dataset(t){
  const s=sect("2","Dataset"), p=t.profile;
  const kv=el("div","kv");
  const add=(k,v)=>{kv.appendChild(el("div",null,k));kv.appendChild(el("div",null,v));};
  add("items", p.n.toLocaleString());
  add("classes", p.n_classes);
  add("imbalance", "1:"+ (p.imbalance_ratio||1).toFixed(2));
  add("constant predictor", f4(p.majority_baseline));
  add("headline metric", esc(p.headline_metric));
  add("resolves", "&plusmn;"+p.resolution_half_width.toFixed(4));
  add("duplicate rows", p.duplicate_rows
    ? `${p.duplicate_rows} (${(100*p.duplicate_share).toFixed(2)}%), grouped in CV`
    : "none");
  add("trained on", t.trained_on==="llm"
    ? '<span class="warn">the LLM\'s own annotations</span>' : "gold labels");
  s.appendChild(kv);
  return s;
}

/* ---- 3. confusion matrix ----------------------------------------------- *
 * For the configuration the Verdict quotes, not for the free arm and not for
 * the LLM. Balanced accuracy says how much is wrong; this says what it is wrong
 * ABOUT, which is what decides whether an error matters.                      */
function confusionPanel(t){
  const s=sect("3","Where the wins are");
  const c=t.confusion, r=c&&c.routing;
  if(!r){ s.appendChild(empty(
    "No paired table in this artifact. It needs a second arm to compare the "+
    "free model against — the LLM, or the pool's own ceiling.")); return s; }
  const other=r.other_name||"the LLM";
  /* 2x2, not k x k. A class confusion says what one model is wrong ABOUT; this
     says what a cascade can do about it, and it is the same shape at 2 classes
     and at 77 where the class grid is 5,929 unreadable cells. */
  /* Each cell says what the RIGHT call is for those items, and -- when the
     shipped rule's own decisions are available -- what it actually did with
     them. "already fine" said nothing; the question a reader has is whether to
     pay for this item and whether the rule agreed. */
  const cell=(key,lab,tone)=>{
    const n=r[key], sh=n/r.n, did=(r.sent||{})[key];
    const bg = tone==="win" ? `rgba(62,207,142,${(0.10+0.55*Math.min(1,sh*2.5)).toFixed(3)})`
             : tone==="lose"? `rgba(255,107,107,${(0.10+0.55*Math.min(1,sh*2.5)).toFixed(3)})`
             : `rgba(139,147,167,${(0.06+0.22*Math.min(1,sh*2)).toFixed(3)})`;
    const good = tone==="win", bad = tone==="lose";
    let didLine="";
    if(did){
      // on the prize cell more sent is better; on the at-risk cell less is
      const ok = good ? did.share_sent>=0.5 : bad ? did.share_sent<0.5 : null;
      const colr = ok===null ? "var(--faint)" : ok ? "var(--good)" : "var(--bad)";
      didLine=`<div style="font-size:10.5px;margin-top:5px;padding-top:4px;`
        +`border-top:1px solid rgba(255,255,255,.08);color:${colr}">`
        +`rule sent <b>${pc(did.share_sent)}</b></div>`;
    }
    return `<td style="background:${bg};padding:9px 7px;border:1px solid var(--line);`
      +`text-align:center;line-height:1.3">`
      +`<div style="font-size:17px;font-weight:600">${pc(sh)}</div>`
      +`<div class="dim" style="font-size:10.5px">${n.toLocaleString()} items</div>`
      +`<div style="font-size:10.5px;margin-top:3px;color:`
      +`${good?"var(--good)":bad?"var(--bad)":"var(--faint)"}">${lab}</div>${didLine}</td>`;
  };
  /* Margins, because without them the cells look like four unrelated
     percentages. Each cell is a share of ALL items and the four sum to 100%; the
     row and column totals are then the two arms' own accuracies, which is what
     ties this panel to every other number on the page. */
  const marg=(v,lab)=>`<td style="padding:6px 7px;text-align:center;`
    +`border:1px solid var(--line);background:rgba(255,255,255,.02)">`
    +`<div style="font-size:13px;font-weight:600">${pc(v)}</div>`
    +`<div class="dim" style="font-size:9.5px;line-height:1.25">${lab}</div></td>`;
  const rowFreeRight=(r.both_right+r.free_only_right)/r.n;
  const rowFreeWrong=(r.other_only_right+r.neither_right)/r.n;
  const colOtherRight=(r.both_right+r.other_only_right)/r.n;
  const colOtherWrong=(r.free_only_right+r.neither_right)/r.n;
  const tb=el("table");
  tb.style.tableLayout="fixed";
  tb.innerHTML =
    `<tr><th style="width:70px"></th><th style="text-align:center">${esc(other)} right</th>`
    +`<th style="text-align:center">${esc(other)} wrong</th>`
    +`<th style="text-align:center;width:74px">any</th></tr>`
    +`<tr><th style="text-align:left">free right</th>`
    + cell("both_right","don't pay — free already gets it","flat")
    + cell("free_only_right", r.risk_cell_is_structural
        ? "empty by construction" : "keep free — a call breaks it",
        r.risk_cell_is_structural ? "flat" : "lose")
    + marg(rowFreeRight, "free arm accuracy") + `</tr>`
    +`<tr><th style="text-align:left">free wrong</th>`
    + cell("other_only_right", r.risk_cell_is_structural
        ? "a better pool member gets these" : "pay — only the call fixes it", "win")
    + cell("neither_right","neither arm gets these","flat")
    + marg(rowFreeWrong, "free arm errors") + `</tr>`
    +`<tr><th style="text-align:left">any</th>`
    + marg(colOtherRight, `${esc(other)} accuracy`)
    + marg(colOtherWrong, `${esc(other)} errors`)
    + marg(1, "all items") + `</tr>`;
  s.appendChild(tb);
  const span=r.oracle_accuracy-r.free_accuracy;
  let note=`<span class="dim">Every cell is a share of all `
    +`${r.n.toLocaleString()} items and the four sum to 100% — each item lands in `
    +`exactly one. So the top-left is not "both score 60%", it is the `
    +`${pc((r.both_right)/r.n)} of items <i>both arms answer correctly</i>. The `
    +`margins are the two arms' own accuracies.</span><br>`
    +`<b style="color:var(--ink)">A perfect router would score `
    +`${f4(r.oracle_accuracy)}</b>, against ${f4(r.free_accuracy)} for the free arm `
    +`alone and ${f4(r.other_accuracy)} for ${esc(other)} alone. `
    +`The most any rule can add is <b>${pc(r.headroom)}</b> and the most it can `
    +`throw away is <b>${pc(r.at_risk)}</b>.`;
  if(r.captured!=null && span>1e-12)
    note+=`<br>The shipped configuration takes <b style="color:var(--good)">`
      +`${pc(r.captured)}</b> of that headroom (${f4(c.accuracy)} raw accuracy).`;
  else if(r.no_rule_reason)
    note+=`<br><b style="color:var(--ink)">Nothing is routed here</b> — `
      +`${esc(r.no_rule_reason)}. The table is the ceiling a router WOULD have, `
      +`not a score for one.`;
  if(r.risk_cell_is_structural)
    note+=`<br><span class="dim">The top-right cell is empty by construction, not by `
      +`merit: the pool ceiling is right whenever any member is, so there are no items `
      +`the free arm gets right and it does not.</span>`;
  if(r.sent)
    note+=`<br><span class="dim">"rule sent" is what the shipped dial actually did `
      +`with each cell. It cannot tell the cells apart — it only sees confidence — `
      +`so the numbers it wants are high on the green cell and low on the red one, `
      +`and the gap between those two is where the capture figure comes from.</span>`;
  if(c.top_confusions&&c.top_confusions.length)
    note+=`<br><span class="dim">Largest single error: ${esc(c.top_confusions[0].label)}, `
      +`${c.top_confusions[0].n.toLocaleString()} items, `
      +`${pc(c.top_confusions[0].share_of_true)} of that class. The full `
      +`${c.classes.length}x${c.classes.length} class matrix is in the artifact.</span>`;
  s.appendChild(el("div","note", note));
  return s;
}

/* ---- 4. correct vs wrong ------------------------------------------------ *
 * What the shipped thing gets right and wrong, in counts and shares, overall
 * and per class. Section 3 is the cascade's own 2x2 (who gets it right);
 * this is the plain one. The full class x class grid is a toggle away -- it is
 * the right object at 2 classes and 5,929 unreadable cells at 77, so it is not
 * what the panel leads with.                                                  */
let posClass = null;         // which class counts as positive; null = micro over all

function matrixPanel(t){
  const c=t.confusion;
  const s=sect("4","Confusion matrix — " + (c&&c.configuration ? esc(c.configuration) : "shipped"));
  if(!c||!c.matrix){ s.appendChild(empty(
    "No confusion matrix in this artifact. It is emitted for whatever the "+
    "Verdict quotes, so a run predating it will show this.")); return s; }
  const L=t.label_names||{}, cls=c.classes, k=cls.length;
  const nm=i=>String(L[cls[i]]!=null?L[cls[i]]:cls[i]);
  const total=c.matrix.flat().reduce((a,b)=>a+b,0);
  const right=c.matrix.reduce((a,row,i)=>a+row[i],0);
  const wrong=total-right;

  /* One-vs-rest, because a single 2x2 over a multiclass problem has to pick what
     "positive" means. Pick a class and it is the ordinary TP/FP/FN/TN table with
     precision and recall that differ; pick "all classes" and it is the
     micro-average, which is standard and degenerate -- FP equals FN and both
     margins collapse to accuracy. The panel says so rather than letting the
     reader wonder why two numbers are always identical. */
  if(posClass===null && k===2) posClass = 1;         // binary: the positive class
  if(posClass!==null && posClass>=k) posClass = null;
  const ctl=el("div","note");
  const opts = cls.map((_,i)=>`<option value="${i}"${posClass===i?" selected":""}>`
      +`${esc(nm(i))}</option>`).join("");
  ctl.innerHTML=`<span id="poswrap">positive class: `
    +`<select id="posclass" style="background:var(--panel);color:var(--ink);`
    +`border:1px solid var(--line);border-radius:6px;padding:3px 6px;font:inherit;`
    +`font-size:12px">${opts}<option value="all"${posClass===null?" selected":""}>`
    +`all classes (micro)</option></select></span>`;
  s.appendChild(ctl);
  const host=el("div"); host.id="mathost"; s.appendChild(host);

  /* TP/FP/FN/TN straight off the stored k x k grid: no rerun, and it cannot
     disagree with the other views because it is the same numbers. */
  function ovr(ci){
    if(ci===null){
      let tp=0,fp=0,fn=0,tn=0;
      cls.forEach((_,i)=>{
        const t=c.matrix[i][i];
        const colSum=c.matrix.reduce((a,r)=>a+r[i],0);
        const rowSum=c.matrix[i].reduce((a,v)=>a+v,0);
        tp+=t; fp+=colSum-t; fn+=rowSum-t; tn+=total-t-(colSum-t)-(rowSum-t);
      });
      return {tp,fp,fn,tn,micro:true};
    }
    const t=c.matrix[ci][ci];
    const colSum=c.matrix.reduce((a,r)=>a+r[ci],0);
    const rowSum=c.matrix[ci].reduce((a,v)=>a+v,0);
    return {tp:t, fp:colSum-t, fn:rowSum-t,
            tn:total-t-(colSum-t)-(rowSum-t), micro:false};
  }

  function binary(){
    const m=ovr(posClass);
    const name = posClass===null ? "the true class" : nm(posClass);
    const pos=m.tp+m.fn, neg=m.fp+m.tn;
    const q=(v,d)=>d? `<span class="dim">(${(100*v/d).toFixed(1)}%)</span>` : "";
    const box=(v,tag,lab,tone,d)=>`<td style="padding:11px 9px;text-align:center;`
      +`border:1px solid var(--line);background:rgba(`
      +`${tone==="ok"?"62,207,142":"255,107,107"},`
      +`${(0.09+0.30*(d? v/d : 0)).toFixed(3)})">`
      +`<div class="dim" style="font-size:10px;letter-spacing:.06em">${tag}</div>`
      +`<div style="font-size:21px;font-weight:600;line-height:1.15">${v.toLocaleString()}</div>`
      +`<div style="font-size:12px">${q(v,d)}</div>`
      +`<div style="font-size:10.5px;margin-top:3px;color:`
      +`${tone==="ok"?"var(--good)":"var(--bad)"}">${lab}</div></td>`;
    const prec = m.tp+m.fp ? m.tp/(m.tp+m.fp) : NaN;
    const rec  = m.tp+m.fn ? m.tp/(m.tp+m.fn) : NaN;
    const f1   = (prec+rec) ? 2*prec*rec/(prec+rec) : NaN;
    const spec = neg ? m.tn/neg : NaN;
    let h=`<table style="table-layout:fixed">`
      +`<tr><th style="width:96px"></th>`
      +`<th style="text-align:center">predicted <b style="color:var(--ink)">${esc(name)}</b></th>`
      +`<th style="text-align:center">predicted other</th>`
      +`<th style="text-align:center;width:78px">total</th></tr>`
      +`<tr><th style="text-align:left">actually ${esc(name)}</th>`
      + box(m.tp,"TP","true positive","ok",pos)
      + box(m.fn,"FN","missed — should have been caught","bad",pos)
      + `<td style="text-align:center" class="dim">${pos.toLocaleString()}</td></tr>`
      +`<tr><th style="text-align:left">actually other</th>`
      + box(m.fp,"FP","false alarm","bad",neg)
      + box(m.tn,"TN","true negative","ok",neg)
      + `<td style="text-align:center" class="dim">${neg.toLocaleString()}</td></tr>`
      +`</table>`;
    h+=`<div class="note"><b style="color:var(--ink)">precision ${f4(prec)}</b> `
      +`&middot; <b style="color:var(--ink)">recall ${f4(rec)}</b> &middot; F1 ${f4(f1)} `
      +`&middot; specificity ${f4(spec)}<br><span class="dim">Percentages are of the `
      +`row, so the TP cell is recall and the TN cell is specificity.`
      + (m.micro
         ? ` This is the micro-average over all ${k} classes, and it is degenerate `
           +`by construction: every wrong item is one FP for the class it was given `
           +`and one FN for the class it belonged to, so FP equals FN and precision, `
           +`recall and F1 all collapse to accuracy. Pick a single class above for a `
           +`table where they differ.`
         : ` One-vs-rest: "${esc(name)}" against every other class.`)
      +`</span></div>`;
    host.innerHTML=h;
  }

  s._paint=()=>{
    const sel=document.getElementById("posclass");
    if(sel) sel.value = posClass===null ? "all" : String(posClass);
    binary();
  };
  s._paint();
  s.appendChild(el("div","note",
    `<span class="dim">Overall accuracy ${f4(c.accuracy)}, balanced ${f4(c.balanced)} — `
    +`the gap between those two is how unevenly the errors fall across classes.</span>`));
  return s;
}

/* ---- 5. gates ---------------------------------------------------------- *
 * All eight, always, with an explicit status. A gate that is absent because it
 * could not run reads differently from a gate that passed, and the earlier
 * version printed only the ones that fired -- so "no G8 row" meant either.     */
function gates(t){
  const s=sect("5","Gates","wide");
  const p=t.profile, exp=hasExpert(t);
  const V = k => (t[k]&&t[k].verdict) || null;
  const fromVerdict = (v, na) => {
    if(!v) return ["n/a", na||"not computed for this run"];
    if(/FAIL/.test(v)) return ["fail", v];
    if(/n\/a/i.test(v)) return ["n/a", v];
    return ["pass", v];
  };
  const dup = p.duplicate_rows;
  const rows = [
    ["G1","majority baseline", p.majority_baseline>=0.65
      ? ["fired", `a constant predictor scores ${f4(p.majority_baseline)} — headline metric forced to balanced accuracy`]
      : ["pass", `a constant predictor scores ${f4(p.majority_baseline)}, below the 0.65 bar — raw accuracy still ranks systems`]],
    ["G2","duplicate texts", dup
      ? ["fired", `${dup} rows (${(100*p.duplicate_share).toFixed(2)}%) share text with another; CV is grouped so copies cannot straddle a fold`]
      : ["pass", "no repeated texts, so no copy can straddle a fold"]],
    ["G3","ROC dominance", fromVerdict(V("G3_roc_dominance"), "binary-only check; this dataset is multiclass")],
    ["G4","pool diversity", fromVerdict(V("G4_pool_diversity"))],
    ["G5","per-class advantage", exp ? fromVerdict(V("G5_class_advantage"))
      : ["n/a","compares the expert against the free arm per class; needs per-item LLM predictions"]],
    ["G6","resolution floor", ["active",
      `differences below &plusmn;${p.resolution_half_width.toFixed(4)} are not differences on this dataset; every margin on this page is checked against it`]],
    ["G7","base-rate dominance", fromVerdict(V("G7_base_rate_dominance"))],
    ["G8","per-class headroom", exp ? fromVerdict(V("G8_per_class_headroom"))
      : ["n/a","oracle ceiling for a per-class threshold; needs per-item LLM predictions"]],
  ];
  const colour = {pass:"good", fail:"bad", fired:"warn", active:"dim", "n/a":"dim"};
  rows.forEach(([tag,name,[st,msg]])=>{
    const g=el("div","gate");
    g.appendChild(el("div","tag "+colour[st], tag));
    g.appendChild(el("div","st "+colour[st], st));
    g.appendChild(el("div","msg", `<b style="color:var(--ink);font-weight:500">${esc(name)}</b> &mdash; ${esc(msg)}`));
    s.appendChild(g);
  });
  return s;
}

/* ---- 4. leaderboard ---------------------------------------------------- *
 * One ranking. There used to be three -- candidate pool, constructions, DESlib
 * -- each sorting a slice, so the highest number on the page could sit three
 * panels below something that looked like the winner.                         */
function leaderboard(t){
  const s=sect("6","Leaderboard — every construction that could ship","wide");
  const rows=[];
  const base=(t.candidates&&t.candidates.ranking)||[];
  const bestBase = base.length? Math.max(...base.map(x=>x.balanced)) : null;
  base.forEach(x=>rows.push({name:x.name, fam:"base model", bal:x.balanced}));
  Object.entries(constructions(t)).forEach(([k,m])=>rows.push({
    name:k, fam:"construction", bal:m.balanced, ci:m.ci, p:m.p,
    holm:m.significant_after_holm, floor:true}));
  const ds=t.deslib; let oracle=null;
  if(ds&&ds.available) Object.entries(ds.methods).forEach(([k,m])=>{
    // The Oracle is not a construction: it reads the answer, then picks whichever
    // pool member got that item right. Ranking it with things you can build put
    // an unshippable row at the top of a table headed "could ship".
    if(/Oracle/.test(k)){ oracle={name:k, ...m}; return; }
    rows.push({name:k, fam:"DESlib", bal:m.balanced, ci:m.ci, p:m.p,
               holm:m.significant_after_holm, floor:m.clears_unpaired_floor});
  });
  if(!rows.length){ s.appendChild(empty("No candidate scores in this artifact.")); return s; }
  rows.sort((a,b)=>b.bal-a.bal);
  const shipped=t.free_arm_shipped, dsShip=(t.recommendation||{}).deslib_ships;
  const tb=el("table");
  tb.innerHTML="<tr><th>construction</th><th>family</th><th>balanced</th>"+
    "<th>vs best base</th><th>95% CI</th><th>p</th><th>status</th></tr>";
  rows.forEach(r=>{
    const d = bestBase==null? null : r.bal-bestBase;
    let flag="";
    if(r.name===dsShip||r.name===shipped) flag='<span class="pill good">shipped</span>';
    else if(d>0 && r.holm && r.floor===false)
      flag='<span class="pill warn">wins the paired test, inside the resolution floor</span>';
    else if(d>0 && r.holm) flag='<span class="pill good">beats baseline (Holm)</span>';
    else if(d>0) flag='<span class="pill dim">not significant</span>';
    tb.appendChild(el("tr", null,
      `<td class="mono">${esc(r.name)}</td>`+
      `<td class="dim">${r.fam}</td><td>${f4(r.bal)}</td>`+
      `<td>${d==null?"&mdash;":(Math.abs(d)<1e-12?'<span class="dim">baseline</span>':sgn(d))}</td>`+
      `<td class="dim">${r.ci?`[${r.ci[0].toFixed(4)}, ${r.ci[1].toFixed(4)}]`:"&mdash;"}</td>`+
      `<td class="dim">${r.p!=null?r.p.toFixed(3):"&mdash;"}</td><td>${flag}</td>`));
  });
  s.appendChild(tb);
  const top=rows[0];
  if(top && top.name!==shipped && top.name!==dsShip)
    s.appendChild(el("div","note",
      `<b style="color:var(--ink)">Why is the top row not what ships?</b> `+
      `<span class="mono">${esc(top.name)}</span> scores ${f4(top.bal)}, `+
      `${sgn(top.bal-bestBase)} over <span class="mono">${esc(shipped)}</span> — but this `+
      `dataset can only resolve differences of &plusmn;${t.profile.resolution_half_width.toFixed(4)}, `+
      `so that gap is smaller than the measurement error. Shipping it would be acting on noise.`));
  if(oracle) s.appendChild(el("div","empty",
    `<b style="color:var(--ink)">Ceiling: ${f4(oracle.balanced)}</b> — what you would score if, `+
    `for every single item, you always picked whichever model in the pool happened to get it `+
    `right. It is not in the table above because you cannot build it: choosing correctly every `+
    `time requires already knowing the answer. It is here to size the prize — `+
    `<b>${sgn(oracle.delta_vs_best_single)}</b> beyond the best single model is sitting in this `+
    `pool, and nothing tested reaches it.`));
  return s;
}

/* ---- 5. per-class chart ------------------------------------------------ */
const BARC = ["#4ea1ff","#3ecf8e","#f2c14e","#c98bff","#ff8f6b","#68d8e3"];
let barMode = "dev", barScope = "top";
const TOP_FRAC = 0.20;

function classChart(t){
  const s=sect("7","Per-class competence — P(correct | this model predicts c)","wide");
  const b=boardOf(t);
  if(!b){ s.appendChild(empty(
    "No competence board in this artifact. It is produced whenever the candidate "+
    "pool runs, so an older run predating the board will show this.")); return s; }
  const names=b.free, labels=t.label_names||{};
  const wins0={}; names.forEach(n=>wins0[n]=0);
  b.classes.forEach(c=>{let bn=names[0];
    names.forEach(n=>{if(b.shrunk[n][c]>b.shrunk[bn][c]) bn=n;}); wins0[bn]++;});
  const live=names.filter(n=>wins0[n]>0), dead=names.filter(n=>!wins0[n]);
  const rank = live.length>1 ? live : names;
  const rows=b.classes.map(c=>{
    const v=names.map(n=>b.shrunk[n][c]);
    const lv=rank.map(n=>b.shrunk[n][c]);
    const mean=v.reduce((a,z)=>a+z,0)/v.length;
    return {c, v, mean, spread:Math.max(...lv)-Math.min(...lv),
            exp: b.routableExpert ? b.routableExpert[c]
                 : (b.expert? b.shrunk[b.expert][c] : null),
            expFree: b.routableFree ? b.routableFree[c] : null,
            win:v.indexOf(Math.max(...v))};
  }).sort((x,y)=>y.spread-x.spread);

  const zoomable = rows.length>=10;
  const nTop = Math.max(1, Math.ceil(rows.length*TOP_FRAC));
  const total = rows.reduce((a,r)=>a+r.spread,0);
  const topSum = rows.slice(0,nTop).reduce((a,r)=>a+r.spread,0);

  const ctl=el("div","note");
  ctl.innerHTML=(zoomable
      ? `<button id="bs-top">top ${nTop} by spread</button> `+
        `<button id="bs-all">all ${rows.length}</button>`+
        `<span style="margin:0 12px;opacity:.35">|</span>` : "")+
    `<button id="bm-dev">deviation from class mean</button> `+
    `<button id="bm-abs">absolute</button>`;
  s.appendChild(ctl);
  const host=el("div","hscroll"); host.id="barhost"; s.appendChild(host);

  function paint(){
    const dev=barMode==="dev";
    const shown=(zoomable&&barScope==="top")? rows.slice(0,nTop) : rows;
    const zoomed=shown.length<rows.length;
    const G=names.length;
    const bw = zoomed ? Math.max(6, Math.min(26,(1150-70)/shown.length/(G+0.8)))
                      : Math.max(5, Math.min(16, 640/(G*shown.length)*6));
    const gap=Math.max(8,bw*0.8), gw=G*bw+gap;
    const W=Math.round(Math.max(760, shown.length*gw+70));
    const H=zoomed?330:300, T=zoomed?26:12, B=zoomed?116:96, L=52;
    const vals=shown.flatMap(r=>r.v.map(x=>dev? x-r.mean : x));
    if(b.expert) shown.forEach(r=>{ if(r.exp!=null) vals.push(dev? r.exp-r.mean : r.exp); });
    let lo,hi;
    if(dev){ const m=Math.max(...vals.map(Math.abs)); lo=-m; hi=m; }
    else { lo=Math.max(0,Math.min(...vals)-0.05); hi=Math.min(1,Math.max(...vals)+0.02); }
    const pad=(hi-lo)*0.08||0.01; lo-=pad; hi+=pad;
    const ys=v=>T+(1-(v-lo)/(hi-lo))*(H-T-B);
    let g=`<svg viewBox="0 0 ${W} ${H}" style="min-width:${W}px">`;
    for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4,y=ys(v);
      g+=`<line x1="${L}" x2="${W-8}" y1="${y}" y2="${y}" stroke="#262b36"/>`
       + `<text x="${L-7}" y="${y+4}" fill="#8b93a7" font-size="10.5" text-anchor="end">`
       + `${dev?(v>=0?"+":"")+v.toFixed(3):v.toFixed(2)}</text>`;}
    if(dev){const y0=ys(0);
      g+=`<line x1="${L}" x2="${W-8}" y1="${y0}" y2="${y0}" stroke="#4a5266" stroke-width="1.5"/>`;}
    const base=dev?ys(0):ys(lo);
    shown.forEach((r,ri)=>{
      const x0=L+ri*gw+gap/2;
      r.v.forEach((val,gi)=>{
        const v=dev? val-r.mean : val, y=ys(v), x=x0+gi*bw;
        const top=Math.min(y,base), h=Math.max(1,Math.abs(base-y));
        const winner=gi===r.win;
        g+=`<rect x="${x.toFixed(1)}" y="${top.toFixed(1)}" width="${(bw-1).toFixed(1)}" `
         + `height="${h.toFixed(1)}" fill="${BARC[gi%BARC.length]}" `
         + `opacity="${winner?0.98:0.22}" rx="1"><title>`
         + `${esc(labels[r.c]!=null?labels[r.c]:r.c)} — ${esc(names[gi])}: ${val.toFixed(3)}`
         + `${winner?" — best free candidate":""}</title></rect>`;
      });
      /* The routing decision is a PAIR, and showing half of it invites the
         reading that sank both FEVER and HateSpeech: "tfidf beats the LLM on
         class 1, so why escalate?" -- comparing tfidf's accuracy on tfidf's own
         hate calls against the LLM's on the LLM's. What decides is the free
         WINNER and the LLM scored on the SAME items: 0.382 against 0.639. So
         both ends are drawn, joined, and the join is green when escalating that
         class pays. The columns behind them are context, not the comparison. */
      if(r.exp!=null){
        const cx=x0+G*bw/2-bw/2;
        const ye=ys(dev? r.exp-r.mean : r.exp);
        if(r.expFree!=null){
          const yf=ys(dev? r.expFree-r.mean : r.expFree);
          const gainful=r.exp>r.expFree;
          g+=`<line x1="${cx}" x2="${cx}" y1="${yf}" y2="${ye}" `
           + `stroke="${gainful?"#3ecf8e":"#4a5266"}" stroke-width="2" opacity=".85"/>`
           + `<circle cx="${cx}" cy="${yf}" r="3.4" fill="#e6e9ef"><title>`
           + `${esc(labels[r.c]!=null?labels[r.c]:r.c)} — free winner, on the items it `
           + `called this class: ${r.expFree.toFixed(3)}</title></circle>`;
        }
        g+=`<path d="M ${cx} ${ye-5} L ${cx+5} ${ye} L ${cx} ${ye+5} L ${cx-5} ${ye} Z" `
         + `fill="#ff6b6b" fill-opacity=".9" stroke="#ff6b6b" stroke-width="1.4"><title>`
         + `${esc(labels[r.c]!=null?labels[r.c]:r.c)} — LLM, on the items the free arm `
         + `called this class: ${r.exp.toFixed(3)}`
         + `${r.expFree!=null?` (free winner on the same items: ${r.expFree.toFixed(3)}`
            +`, so the LLM ${r.exp>r.expFree?"gains":"loses"} `
            +`${Math.abs(r.exp-r.expFree).toFixed(3)} here)`:""}`
         + `</title></path>`;
      }
      const lab=String(labels[r.c]!=null?labels[r.c]:r.c);
      const cx=x0+G*bw/2, cut=zoomed?34:22;
      g+=`<text x="${cx}" y="${H-B+14}" fill="${zoomed?"#c7cede":"#8b93a7"}" `
       + `font-size="${zoomed?11.5:10}" transform="rotate(-42 ${cx} ${H-B+14})" `
       + `text-anchor="end">${esc(lab.length>cut?lab.slice(0,cut-1)+"…":lab)}</text>`;
      if(zoomed) g+=`<text x="${cx}" y="14" fill="#6f7789" font-size="9.5" `
       + `text-anchor="middle">${r.spread.toFixed(3)}</text>`;
    });
    host.innerHTML=g+`</svg>`;
  }
  paint();
  let legend=names.map((n,i)=>`<span style="color:${BARC[i%BARC.length]}">&#9632;</span> `
    +esc(n.split("+")[0])).join(" &nbsp; ");
  if(rows.some(r=>r.exp!=null)){
    legend += b.routableFree
      ? ` &nbsp; <span style="color:#e6e9ef">&#9679;</span>&#8202;&#8212;&#8202;`
        +`<span style="color:#ff6b6b">&#9670;</span> <b style="color:var(--ink)">the routing `
        +`decision</b>: free winner &rarr; LLM, both on the items the free arm called that `
        +`class. <span class="dim">Green join = escalating that class pays. This pair is `
        +`what the router compares — not a candidate column against the LLM's accuracy on `
        +`its own predictions, which can point the other way.</span>`
      : ` &nbsp; <span style="color:#ff6b6b">&#9670;</span> LLM `
        +`<span class="dim">(reference only — not a free candidate)</span>`;
  }
  let tail=" &nbsp;&mdash;&nbsp; dimmed columns lost that class; ordered by spread, widest first.";
  if(zoomable) tail+=`<br>The top ${nTop} of ${rows.length} classes carry `
    +`<b>${(100*topSum/total).toFixed(0)}%</b> of all the between-candidate spread. `
    +`The rest is where every candidate agrees, so no selection rule can act there.`;
  if(dead.length&&live.length>1) tail+=`<br>Ranked on the spread among the `
    +`${live.length} candidates that win at least one class; `
    +`<span class="mono">${dead.map(n=>esc(n.split("+")[0])).join(", ")}</span> win none, `
    +`and counting them would rank "where that candidate is worst" ahead of "where the choice matters".`;
  s.appendChild(el("div","note",legend+tail));
  s._paint=paint; s._zoomable=zoomable;
  return s;
}

/* ---- 6. deferral dial -------------------------------------------------- */
function dial(t){
  const s=sect("8","Deferral dial — accuracy against LLM spend","wide");
  const d=t.deferral_curve;
  if(!d||!d.points||!d.points.length){
    s.appendChild(empty(
      "Not measurable on this dataset: the cascade needs the LLM's prediction on "+
      "every item, and no such stream exists here. The free arm answers 100% of "+
      "traffic, which is the 0% point of a dial that has no other points."));
    return s;
  }
  /* Derived from the data, never a hardcoded list. engine.render() carries the
     same rule with the reason beside it -- "a rule that is in the search but not
     in the table is a rule nobody checks" -- and this panel had drifted to three
     of five, so committee (the winning rule on imdb and hatespeech) was absent,
     the resolution band was anchored to a max over the drawn subset, and the
     "ship this" ring sat on coordinates belonging to no drawn curve. */
  const SKIP=new Set(["expert_share_target","gain_per_10pct_spend"]);
  const rules=Object.keys(d.points[0]).filter(r=>!SKIP.has(r)
      && d.points[0][r] && typeof d.points[0][r].balanced==="number");
  const PALETTE=["#4ea1ff","#3ecf8e","#f2c14e","#c98bff","#ff8f6b","#68d8e3","#9bd35a"];
  const col=Object.fromEntries(rules.map((r,i)=>[r,PALETTE[i%PALETTE.length]]));
  const W=980,H=320,L=56,R=16,T=14,B=34;
  const xs=v=>L+v*(W-L-R), all=[];
  rules.forEach(r=>d.points.forEach(p=>all.push(p[r].balanced)));
  all.push(d.free_alone,d.expert_alone);
  let lo=Math.min(...all),hi=Math.max(...all);
  const pad=(hi-lo)*0.16||0.01; lo-=pad; hi+=pad;
  const ys=v=>T+(1-(v-lo)/(hi-lo))*(H-T-B);
  let g=`<svg viewBox="0 0 ${W} ${H}">`;
  for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4,y=ys(v);
    g+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="#262b36"/>`
     + `<text x="${L-8}" y="${y+4}" fill="#8b93a7" font-size="11" text-anchor="end">${v.toFixed(3)}</text>`;}
  [0,.25,.5,.75,1].forEach(v=>{g+=`<text x="${xs(v)}" y="${H-12}" fill="#8b93a7" `
    +`font-size="11" text-anchor="middle">${(v*100)|0}%</text>`;});
  rules.forEach(r=>{
    const pts=d.points.map(p=>[xs(p[r].expert_share),ys(p[r].balanced)]);
    g+=`<polyline fill="none" stroke="${col[r]}" stroke-width="2" points="${pts.map(p=>p.join(",")).join(" ")}"/>`;
    pts.forEach((p,i)=>{g+=`<circle cx="${p[0]}" cy="${p[1]}" r="2.6" fill="${col[r]}">`
      +`<title>${r} at ${pc(d.points[i][r].expert_share)}: ${f4(d.points[i][r].balanced)}</title></circle>`;});
  });
  // Everything inside one resolution floor of the peak is the same answer as the
  // peak. Without the band the eye reads the argmax as "the" operating point and
  // pays for spend that buys nothing measurable.
  const floor=t.profile.resolution_half_width;
  const peak=Math.max(...rules.flatMap(r=>d.points.map(p=>p[r].balanced)));
  const yTop=ys(Math.min(hi,peak)), yBot=ys(peak-floor);
  g+=`<rect x="${L}" y="${yTop}" width="${W-R-L}" height="${Math.max(1,yBot-yTop)}" `
   + `fill="#3ecf8e" opacity="0.07"/>`
   + `<text x="${W-R-4}" y="${yBot-4}" fill="#3ecf8e" font-size="10" opacity=".8" `
   + `text-anchor="end">within &plusmn;${floor.toFixed(4)} of the best — same answer</text>`;
  if(d.best_point){const x=xs(d.best_point.expert_share),y=ys(d.best_point.balanced);
    g+=`<circle cx="${x}" cy="${y}" r="7" fill="none" stroke="#fff" stroke-width="2"/>`
     + `<text x="${x}" y="${y-14}" fill="#e6e9ef" font-size="11.5" text-anchor="middle">`
     + `nested ${d.best_point.balanced.toFixed(4)}</text>`;}
  const ce=d.cheapest_equivalent;
  if(ce&&ce.found&&d.best_point&&ce.expert_share < d.best_point.expert_share-1e-9){
    const x=xs(ce.expert_share), y=ys(ce.balanced);
    g+=`<circle cx="${x}" cy="${y}" r="7" fill="none" stroke="#3ecf8e" stroke-width="2"/>`
     + `<text x="${x}" y="${y+20}" fill="#3ecf8e" font-size="11.5" text-anchor="middle">`
     + `same for ${pc(ce.expert_share)}</text>`;
  }
  s.appendChild(el("div",null,g+`</svg>`));
  let note=rules.map(r=>`<span style="color:${col[r]}">&#9632;</span> ${r}`).join(" &nbsp; ")
    +" &nbsp;&mdash;&nbsp; the white ring is the nested operating point: rule and budget "
    +"chosen inside each fold's training data, then applied once to its held-out items.";
  if(ce&&ce.found&&d.best_point&&ce.expert_share < d.best_point.expert_share-1e-9){
    note+=`<br><b style="color:var(--good)">The green ring is the one to ship.</b> `
      +`${pc(ce.expert_share)} of traffic scores ${f4(ce.balanced)} — `
      +`${(d.best_point.balanced-ce.balanced).toFixed(4)} below the peak, inside the `
      +`&plusmn;${floor.toFixed(4)} this dataset can resolve. The extra `
      +`${pc(d.best_point.expert_share-ce.expert_share)} of LLM spend buys a difference `
      +`the data cannot measure.`;
  }
  s.appendChild(el("div","note", note));
  s.appendChild(promisePanel(t));
  return s;
}

/* ---- the promise panel, inside section 8 -------------------------------
 * The dial answers "we have budget for x%, what do we get". This answers the
 * question a buyer actually asks: "what can you promise, and what does it
 * cost". Two ways to set the bar, side by side, because the comparison is the
 * finding -- an absolute bar is set by someone's intuition and is wrong in both
 * directions at once, while an anchored bar is set by the model being replaced
 * and therefore cannot ask for the impossible.                              */
function promisePanel(t){
  const abs=t.guaranteed_coverage||[], rel=t.margin_coverage||[];
  const box=el("div");
  if(!abs.length && !rel.length) return box;
  box.appendChild(el("h3",null,"What can be promised, and what it costs"));
  const tb=el("table");
  let h=`<tr><th>promise</th><th style="text-align:right">LLM spend</th>`
       +`<th style="text-align:right">balanced</th>`
       +`<th style="text-align:right">classes refused</th><th>kept?</th></tr>`;
  abs.forEach(g=>{
    const ok=g.guarantee_holds;
    h+=`<tr><td>every class at <b>${g.bar.toFixed(2)}</b> precision</td>`
      +`<td style="text-align:right">${pc(g.expert_share)}</td>`
      +`<td style="text-align:right">${f4(g.balanced)}</td>`
      +`<td style="text-align:right">${(g.vacuous_classes||[]).length}</td>`
      +`<td class="${ok?'good':'bad'}">${ok?'held':'BROKEN'}</td></tr>`;
  });
  rel.forEach(g=>{
    const ok=g.promise_holds;
    h+=`<tr><td>no class more than <b>${pc(g.margin)}</b> below the LLM`
      +` on that class</td>`
      +`<td style="text-align:right">${pc(g.expert_share)}</td>`
      +`<td style="text-align:right">${f4(g.balanced)}</td>`
      +`<td style="text-align:right">${(g.deferred_classes||[]).length}</td>`
      +`<td class="${ok?'good':'bad'}">${ok?'held':'BROKEN'}</td></tr>`;
  });
  tb.innerHTML=h;
  box.appendChild(tb);
  // The comparison, stated rather than left to the reader. Picking the two rows
  // that make the same-strength claim keeps this from being a cherry-pick: the
  // strictest anchored promise there is (margin 0, "match the LLM class by
  // class") against the absolute bar, on the same data.
  const tight=rel.find(g=>g.margin===0);
  const mid=abs.find(g=>g.bar===0.90);
  if(tight&&mid){
    const d=mid.expert_share-tight.expert_share;
    box.appendChild(el("div","note",
      `Matching the LLM class by class costs <b>${pc(tight.expert_share)}</b> of `
      +`traffic and scores ${f4(tight.balanced)}. Demanding a flat 0.90 instead `
      +`costs <b>${pc(mid.expert_share)}</b> and scores ${f4(mid.balanced)}`
      +(mid.guarantee_holds?"":", and still does not hold")+`. `
      +(d>0.01
        ? `The extra ${pc(d)} buys nothing the anchored promise did not already `
          +`give: a flat bar cannot tell a class the free arm is genuinely weak `
          +`on from one the LLM is equally weak on, so it pays for both.`
        : `The two land in the same place here, which is what happens when the `
          +`LLM's per-class precision is already near the flat bar.`)));
  }
  return box;
}

/* ---- 7. per-class table ------------------------------------------------ */
function classTable(t){
  const s=sect("9","Per-class competence, as numbers","wide");
  const b=boardOf(t);
  if(!b){ s.appendChild(empty("No competence board in this artifact.")); return s; }
  const names=b.free.concat(b.expert?[b.expert]:[]);
  const labels=t.label_names||{};
  const rows=b.classes.map(c=>{
    const v=b.free.map(n=>b.shrunk[n][c]);
    return {c, v:names.map(n=>b.shrunk[n][c]),
            spread:Math.max(...v)-Math.min(...v),
            n:Math.max(...b.free.map(n=>(b.support[n]||{})[c]||0)),
            bestFree:b.free[v.indexOf(Math.max(...v))]};
  }).sort((x,y)=>y.spread-x.spread);
  const wrap=el("div","scroll"), tb=el("table","heat");
  tb.innerHTML="<tr><th>class</th><th>n</th>"+
    names.map(n=>`<th${/EXPERT/i.test(n)?' style="color:#ff6b6b"':''}>`
      +`${esc(/EXPERT/i.test(n)?"LLM":n.split("+")[0])}</th>`).join("")+
    "<th>spread</th><th>best free</th></tr>";
  rows.forEach(r=>{
    /* Shaded against THIS ROW's own range, not a fixed [0.4, 1.0] ramp.
       A global ramp is unreadable exactly where the table matters most: on
       CLINC150 nearly every cell is 0.85-1.00, so 0.864 and 0.940 came out
       0.07 apart in alpha and the best model in a row was indistinguishable
       from the third-best by eye. Per-row contrast makes the winner obvious at
       any absolute level, and the number is still printed for the absolute
       reading. The winning free cell is also outlined, so the answer does not
       depend on discriminating two blues at all. */
    const fv=r.v.filter((_,i)=>!/EXPERT/i.test(names[i]));
    const rlo=Math.min(...fv), rhi=Math.max(...fv), span=(rhi-rlo)||1;
    const cells=r.v.map((v,i)=>{
      const isExp=/EXPERT/i.test(names[i]);
      const a=Math.max(0,Math.min(1,(v-rlo)/span));
      const bg=isExp?`rgba(255,107,107,${(0.08+0.35*Math.max(0,Math.min(1,(v-0.4)/0.6))).toFixed(3)})`
                    :`rgba(78,161,255,${(0.06+0.62*a).toFixed(3)})`;
      const win=!isExp && v===rhi;
      return `<td class="cell" style="background:${bg}`
        +(win?`;box-shadow:inset 0 0 0 1.5px #4ea1ff;font-weight:600`:"")
        +`">${v.toFixed(3)}</td>`;
    }).join("");
    tb.appendChild(el("tr",null,
      `<td>${esc(labels[r.c]!=null?labels[r.c]:r.c)}</td><td class="dim">${r.n}</td>`
      +cells+`<td>${r.spread.toFixed(3)}</td>`
      +`<td class="mono dim">${esc(r.bestFree.split("+")[0])}</td>`));
  });
  wrap.appendChild(tb); s.appendChild(wrap);
  s.appendChild(el("div","note",
    "Shading is relative to each row's own best and worst free candidate, so the "+
    "outlined cell is the winner for that class whatever the absolute level. "+
    "Rows are sorted by spread: the top of the table is where the pool disagrees "+
    "most, which is the only place a per-class rule has anything to work with."));
  if(b.expert) s.appendChild(el("div","note",
    "The LLM column is its precision on <i>its own</i> predictions, shown for "+
    "reference. It is not what the router consults: that is P(LLM correct | the "+
    "<i>free</i> arm said c), which can differ in sign — on ISEAR class 1 the two "+
    "are 0.560 and 0.750."));
  return s;
}

/* ---- render ------------------------------------------------------------ */
let cur=0;
function draw(){
  const t=DATA[cur], main=document.getElementById("main");
  main.innerHTML="";
  const chart=classChart(t), matp=matrixPanel(t);
  [verdict(t),dataset(t),confusionPanel(t),matp,gates(t),leaderboard(t),
   chart,dial(t),classTable(t)]
    .forEach(x=>main.appendChild(x));
  const wire=(ids,set)=>{ids.forEach(([id,v])=>{const b=document.getElementById(id);
    if(b) b.onclick=()=>set(v);});};
  if(matp._paint){
    const sel=document.getElementById("posclass");
    if(sel) sel.onchange=()=>{const v=sel.value;
      posClass = v==="all" ? null : Number(v); matp._paint();};
  }
  if(chart._paint){
    const setMode=m=>{barMode=m; chart._paint();
      [["bm-dev","dev"],["bm-abs","abs"]].forEach(([id,v])=>{
        const b=document.getElementById(id); if(b) b.setAttribute("aria-selected", m===v);});};
    const setScope=v=>{barScope=v; chart._paint();
      [["bs-top","top"],["bs-all","all"]].forEach(([id,x])=>{
        const b=document.getElementById(id); if(b) b.setAttribute("aria-selected", v===x);});};
    wire([["bm-dev","dev"],["bm-abs","abs"]], setMode);
    wire([["bs-top","top"],["bs-all","all"]], setScope);
    setMode(barMode); if(chart._zoomable) setScope(barScope);
  }
}
/* Datasets on the top row. Variants -- ablations, --train-on llm, a superseded
   partial run -- are experiments ON a dataset, not datasets, and sat in the same
   flat row labelled by file stem until now. They are one click away. */
function nav(){
  const n=document.getElementById("nav");
  const buttons=[];
  const mk=(t,i)=>{
    const b=el("button",null,
      `${esc(t.task)}<span class="n">${t.profile.n_classes}c</span>`+
      (t.variant?`<span class="n">${esc(t.variant)}</span>`:""));
    b.onclick=()=>{cur=i; select(); draw();};
    buttons.push([b,i]);
    return b;
  };
  const select=()=>buttons.forEach(([b,i])=>b.setAttribute("aria-selected", i===cur));

  const primary=el("div","navrow"), variants=el("div","navrow variants");
  DATA.forEach((t,i)=>(t.canonical?primary:variants).appendChild(mk(t,i)));
  n.appendChild(primary);

  const nVar=DATA.filter(t=>!t.canonical).length;
  if(nVar){
    let open=false;
    variants.style.display="none";
    const tog=el("button","toggle",`variant runs (${nVar})`);
    tog.onclick=()=>{open=!open;
      variants.style.display=open?"flex":"none";
      tog.setAttribute("aria-selected",open);
      tog.innerHTML=`${open?"hide ":""}variant runs (${nVar})`;};
    const row=el("div","navrow"); row.appendChild(tog);
    n.appendChild(row); n.appendChild(variants);
  }
  select();
}
document.getElementById("foot").innerHTML =
  `generated ${esc(META.generated)} from ${DATA.length} run(s) in ${esc(META.dir)} `+
  `&middot; every dataset renders the same seven sections &middot; `+
  `<span class="mono">python ocl_compare/dashboard.py</span>`;
nav(); draw();
</script>
"""


def build(results_dir: str, out: str) -> str:
    tasks = collect(results_dir)
    if not tasks:
        raise SystemExit(f"no pipeline artifacts found in {results_dir}")
    slim = [_slim(t) for t in tasks]
    subs = {
        "@@TITLE@@": ", ".join(sorted({t["task"] for t in tasks})),
        "@@SUBTITLE@@": html.escape(
            f"{sum(1 for t in tasks if t['canonical'])} datasets · "
            + " · ".join(f"{t['task']} ({t['profile']['n_classes']}c, "
                         f"n={t['profile']['n']:,})"
                         for t in tasks if t["canonical"])
            + (f" · plus {sum(1 for t in tasks if not t['canonical'])} variant runs"
               if any(not t["canonical"] for t in tasks) else "")),
        "@@DATA@@": json.dumps(slim, separators=(",", ":"), default=float),
        "@@META@@": json.dumps({"generated": time.strftime("%Y-%m-%d %H:%M"),
                                "dir": os.path.relpath(results_dir)}),
    }
    page = TEMPLATE
    for k, v in subs.items():
        page = page.replace(k, v)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=P.RESULTS)
    ap.add_argument("--out", default=os.path.join(P.RESULTS, "dashboard.html"))
    a = ap.parse_args()
    path = build(a.results, a.out)
    n = len(collect(a.results))
    print(f"wrote {path} ({os.path.getsize(path)/1024:.0f} KB, {n} runs)")


if __name__ == "__main__":
    main()
