# Online Cascade Learning — comparison, and the pipeline it produced

> Nie, Ding, Hu, Jermaine, Chaudhuri. **Online Cascade Learning for Efficient Inference
> over Streams.** ICML 2024, PMLR 235:38071–38090.
> [paper](https://proceedings.mlr.press/v235/nie24a.html) ·
> [code](https://github.com/Flitternie/online_cascade_learning)

Two things live here. The **reconstruction** (`reconstruct.py`) rebuilds OCL's four
evaluation streams and proves they are the ones the paper measured. The **pipeline**
(`run.py` over `../src/downshift/engine.py`) is what the comparison turned
into: a gated sweep that decides, per dataset, which free model to ship and how much
traffic to send to an LLM.

```bash
python ocl_compare/run.py                            # every registered dataset
python ocl_compare/run.py --list                     # what is registered
python ocl_compare/run.py --datasets banking77       # one of them
python ocl_compare/run.py --train-on llm             # OCL's no-gold setting
python ocl_compare/dashboard.py                      # one page over every run
python ocl_compare/reconstruct.py                    # ~8s warm; asserts Table 1
python -m pytest tests/ ocl_compare/ -q
```

## The pipeline is fixed; datasets are adapted to it

The pipeline grew around the four OCL streams, so its label table, its loader and
its provenance pins were reachable from inside the analysis. Adding banking77
then needed a runner that reached back in and mutated the pipeline's own state
before calling it -- `RP.LABEL_NAMES["banking77"] = names`. That is a pipeline
that is not fixed: every new dataset edits the thing that is meant to be
constant.

| file | holds |
|---|---|
| `src/downshift/dataset.py` | the contract: `Dataset`, `validate()`, the registry |
| `src/downshift/engine.py` | the pipeline. `Dataset` in, artifact out |
| `ocl_compare/adapters.py` | every dataset-specific fact, one function per source |
| `ocl_compare/run.py` | the only CLI; resolves names through the registry |

A dataset is `name, texts, gold, label_names, expert?, source, notes`. Adding one
is a function in `adapters.py`. The engine is edited when a gate is wrong, a
statistic is wrong, or a candidate joins the pool for everyone -- never to
accommodate a source, and a test enforces that by rejecting any dataset name
reachable from engine *code* (comments recording a measurement are fine and
wanted).

`validate()` runs at the boundary and rejects the mistakes that have already
reached a result in this project: texts and labels of different lengths after a
filter touched one array; label ids that are not contiguous from zero, which
every per-class array silently mis-indexes; a label-name map missing a class,
which quietly weakens the NLI candidate rather than failing; and an expert array
predicting a class that never appears in gold.

`run_pipeline.py` and `run_banking77.py` are gone -- their content moved, and
leaving runners that monkey-patch the pipeline would undo the point. The refactor
was verified field-by-field against the pre-refactor artifacts: **0 differences
at 1e-12 across all five datasets.**

## Dashboard

`python ocl_compare/dashboard.py` writes `results/dashboard.html` from whatever
artifacts are on disk. A new dataset becomes a new tab next time it runs; there
is no per-dataset code in it.

**Five dataset tabs, and the eight variant runs behind a toggle.** A run is
either the current answer for its dataset or an experiment on it, and the first
version put all thirteen in one flat row labelled by file stem -- so `isear 7c
main`, `isear 7c ablation_nonli` and `isear 7c llmlabels` sat side by side as if
they were three datasets. Ablations, `--train-on llm`, and an older partial run
that a later full run replaced are now collapsed, and named in words:

```
banking77 77c   isear 7c   fever 2c   hatespeech 2c   imdb 2c
[ variant runs (8) ]
    isear · without NLI            fever · superseded by a later run
    isear · without reranker       imdb  · superseded by a later run
    isear · trained on LLM labels  ...
```

`pipeline_fever_imdb.json` is kept rather than deleted -- it is a provenance
record with its own source pins -- but it is labelled for what it is instead of
appearing to be a dataset.

**Every dataset renders the same seven sections in the same order.** That is the
contract, and it is enforced by a test:

| | section | with nothing to show |
|---|---|---|
| 1 | Verdict | &mdash; |
| 2 | Dataset | &mdash; |
| 3 | Gates | all eight, always, each with a status |
| 4 | Leaderboard | &mdash; |
| 5 | Per-class competence (chart) | says the board is absent from the artifact |
| 6 | Deferral dial | says why the cascade is not measurable here |
| 7 | Per-class competence (numbers) | as above |

The first version grew panel by panel and rendered a different shape per
dataset: some tabs had a dial, some did not; some had a per-class chart, some did
not. Two runs could not be compared, and a missing section meant either "measured
and empty" or "not measurable" with no way to tell. Sections now always appear
and explain their own absence -- banking77's dial reads *"the cascade needs the
LLM's prediction on every item, and no such stream exists here."*

Three things the rewrite fixed, all the same underlying mistake of letting the
data decide the shape:

- **Gates printed only when they fired**, so no G8 row meant either "passed" or
  "could not run". All eight now render with an explicit status: pass, fail,
  fired, active, or n/a with the reason.
- **The chart's columns meant different things on different tabs.** The board
  comes from `competence_board` on a run with no expert and from
  `class_router.board` on a run with one -- and the second includes the LLM as a
  member. On imdb that put `EXPERT(own class)` in the candidate legend and let
  the chart mark the LLM best in a class while the verdict said ship bge-large.
  Free candidates are now columns; the LLM is a red diamond marker, excluded
  from winner highlighting, labelled *reference only*.
- **Three panels each ranked a slice** -- candidate pool, constructions, DESlib
  -- so the highest number on the page could sit three panels below what looked
  like the winner. One Leaderboard now, carrying the confidence intervals and
  p-values that were in the others.

The Oracle is **not** in the leaderboard. It reads the answer and then picks
whichever pool member got that item right, so it cannot be built -- choosing
correctly every time requires already knowing the answer. Ranking it beside
things you can build put an unshippable row at the top of a table headed "every
construction that could ship", which reads as a recommendation. It sits below
the table instead, phrased as what it is: *"what you would score if, for every
item, you always picked whichever model happened to get it right ... +0.0375
beyond the best single model is sitting in this pool, and nothing tested reaches
it."*

And when the top row is not what ships, the table says why on the spot: on
banking77, *"DESlib LCA scores 0.9289, +0.0058 over bge-large — but this dataset
can only resolve differences of ±0.0066, so that gap is smaller than the
measurement error. Shipping it would be acting on noise."*

Panel 5 keeps two toggles: scope (top 20% by spread / all) and view (deviation
from class mean / absolute), described below.

## Headline

**With gold labels, a cascade beats both pure strategies on all four datasets**, by
margins that clear each dataset's resolution floor. Operating points chosen by nested
CV, so these are the procedure's numbers rather than the best of 16 configurations.

| dataset | n | classes | free alone | GPT-3.5 | **best hybrid** | LLM share | ship |
|---|---:|---:|---:|---:|---:|---:|---|
| IMDB | 25,000 | 2 | 0.9420 | 0.9415 | **0.9508** | 10% | bge-large |
| HateSpeech | 10,703 | 2 (1:7.95) | 0.8460 | 0.8331 | **0.8642** | 20% | bge-large |
| ISEAR | 7,666 | 7 | 0.6975 | 0.7036 | **0.7131** | 33% | stacking |
| FEVER | 6,512 | 2 | 0.7349 | 0.7997 | **0.8183** | 75% | bge-large |

Balanced accuracy throughout — a constant predictor scores 0.8883 on HateSpeech, so raw
accuracy ranks nothing there.

**Without gold labels it evaporates.** `--train-on llm` fits every learned component on
GPT-3.5's own annotations, reproducing OCL's setting. The best operating point then
gains **+0.0000, +0.0005, +0.0003, +0.0022** over simply calling the LLM, against floors
of 0.0102, 0.0057, 0.0062, 0.0032. G6 flags all four. Gold labels are the load-bearing
ingredient, not the routing.

## Against OCL's published numbers, at a matched label budget

`label_matched.py`. The comparison the earlier draft could not make, and it
changes the answer.

**The confound.** `N` in online cascade learning is the maximum number of LLM
calls -- and because a call both answers a query and returns a training label,
it is simultaneously the size of the training set. Our cross-validated pipeline
trains on 80% of the data at every dial position, so at their low-budget cells
we held 1.9x to 15.4x more labels. "Higher on 12 of 12" measured that, not the
method.

Here one number buys both, exactly as it does for them:

    N calls = m warm-up calls + (N - m) escalations
    training labels = the m warm-up answers, and nothing else

Labels are the LLM's own annotations, never gold, because that is what a call
returns and what OCL trains on. `m` is swept and chosen on five held-out
streams, then scored on five others, so the warm-up split is not picked by
looking at the score it produces (selection bias runs +0.0000 to +0.0082 and is
printed per cell). The escalation threshold comes from cross-validated warm-up
confidences, never in-sample ones.

| dataset | N | budget | ours | OCL | delta | vs their best baseline |
|---|---:|---:|---:|---:|---:|---:|
| IMDB | 1300 | 5.2% | 0.9426 | 0.8795 | **+0.0631** | +0.0753 |
| IMDB | 3800 | 15.2% | 0.9460 | 0.9248 | **+0.0212** | +0.0580 |
| IMDB | 5200 | 20.8% | 0.9453 | 0.9301 | **+0.0152** | +0.0458 |
| ISEAR | 1200 | 15.7% | 0.6390 | 0.6078 | **+0.0312** | +0.0734 |
| ISEAR | 1500 | 19.6% | 0.6463 | 0.6534 | −0.0071 | +0.0421 |
| ISEAR | 2700 | 35.2% | 0.6682 | 0.6975 | −0.0293 | +0.0504 |
| FEVER | 700 | 10.7% | 0.6805 | 0.6195 | **+0.0610** | +0.0636 |
| FEVER | 2000 | 30.7% | 0.7237 | 0.7186 | **+0.0051** | +0.0259 |
| FEVER | 2800 | 43.0% | 0.7464 | 0.7849 | −0.0385 | −0.0203 |

HateSpeech is scored separately because a constant "no" predictor gets 0.8883
there -- above every system in that column of Table 1, theirs included -- so
raw accuracy ranks nothing. OCL publishes accuracy *and* positive recall on this
dataset alone, which is enough to reconstruct balanced accuracy for both sides:

| N | ours recall | OCL recall | ours balanced | OCL balanced | delta |
|---:|---:|---:|---:|---:|---:|
| 600 | 0.7689 | 0.8236 | 0.7775 | 0.8253 | −0.0478 |
| 2700 | 0.8381 | 0.7720 | 0.8128 | 0.8179 | −0.0051 |
| 4900 | 0.8425 | 0.8103 | 0.8255 | 0.8229 | **+0.0026** |

On raw accuracy that row reads 0/3 and −0.0604 at N=2700. On the metric that
ranks anything there it is 1/3 and −0.0051. Both are printed; the second is the
one to quote.

### The settled answer

**7 of 12 cells, on the metric each dataset warrants.** Not 12 of 12. Roughly
half the earlier margin was the label budget.

And what remains is not noise, it is a crossover:

| | low budget | high budget |
|---|---|---|
| IMDB | +0.0631 @5.2% | +0.0152 @20.8% |
| ISEAR | +0.0312 @15.7% | −0.0293 @35.2% |
| FEVER | +0.0610 @10.7% | −0.0385 @43.0% |

**A static cascade wins below roughly 20-30% of the stream; the online one wins
above it.** The mechanism is the handicap written into the protocol rather than
argued away: OCL keeps learning from every call across the whole stream, while
we train once on the warm-up and never again. With few calls there is nothing to
accumulate and spending the budget well is what matters; with many, their
continual updating compounds. HateSpeech runs the other way (−0.0478, −0.0051,
+0.0026) for the same reason seen from the other side -- at N=600 a 5.6% budget
holds only ~67 positive examples, too few to learn the minority class at all.

Against **Online Ensemble Learning**, the strongest baseline in their own table,
the same runs win 9 of 12.

With gold labels instead of the LLM's -- a counterfactual, not a comparison,
since no one gets gold for free -- it is 9 of 12, and HateSpeech flips from 1/3
to 2/3. That gap is the price of training on a teacher's mistakes.

Two limits that remain and cannot be closed from published numbers: the
comparison is unpaired (their point estimates, no per-item predictions, so no
McNemar), and our encoder is 335M against their 110M BERT-base.

**The class-aware method specifically does not beat OCL.** It lost on HateSpeech;
class-aware *selection* lost to the best single model everywhere it was tried; its dial
rule was unstable on ISEAR. What won on ISEAR was stacking, which is not class-aware.

## The competence board

One table answers the operational question directly: for every candidate *and*
the paid expert, how often is it right **when it predicts class c**? Each
candidate's own claim then competes for the item, and whole classes are handed
to the expert where it is reliably better. The output is a class list, which is
the form the decision actually takes -- none of them, four of seven, or all.

| dataset | classes | handed to the LLM | LLM share | board | dial, nested | stable |
|---|---:|---|---:|---:|---:|---|
| IMDB | 2 | `[0]` | 49.6% | 0.9485 | 0.9508 @10% | 5/5 |
| HateSpeech | 2 | `[0]` | 60.4% | 0.8482 | 0.8642 @20% | 4/5 |
| ISEAR | 7 | `[0, 1, 3, 5]` | 58.3% | 0.7132 | 0.7131 @32.9% | 5/5 |
| FEVER | 2 | `[0, 1]` (everything) | 100% | 0.7997 | 0.8183 @75% | 5/5 |

Four findings, none of them flattering, all of them the point:

- **Whole-class hand-over is coarse, and the cost shows up as spend.** It loses
  to the budgeted confidence dial on three of four and ties on ISEAR -- while
  paying 49.6%/60.4%/58.3%/100% of traffic against the dial's 10%/20%/32.9%/75%.
  It cannot express "the worst tenth of class 0", so it buys whole classes. The
  board's value is that you can read *why* off the table, not that it scores
  higher.
- **Arbitration among free candidates never beat the best single model** -- it
  tied on IMDB and ISEAR and lost on HateSpeech (0.8397 vs 0.8460) and FEVER
  (0.7270 vs 0.7349). Combination wins, selection loses, again.
- **Stage 1 has a class-difficulty trap.** Comparing "model A's 0.777 for class
  0" against "model B's 0.703 for class 1" conflates model quality with class
  difficulty, so the item goes to whoever named the easier class. Measured on
  FEVER: 0.7001 against the best single model's 0.7349. Centring each class
  column on its cross-candidate mean recovers 2.7 points.
- **Stage 2 has the same trap one level down**, and it is the more dangerous of
  the two because the number it reports looks right. Ranking classes by paired
  accuracy gain optimises accuracy *inside* the escalated bucket: on HateSpeech
  the expert is +25.7 points on items the free arm calls class 1 (0.639 against
  0.382), so a gain-ranked router takes that class -- and loses 1.2 points of
  balanced accuracy. Choosing greedily on the headline metric instead swings it
  by 2.07 points and hands over the **majority** class rather than the minority
  one. Both directions are stable under their own objective; only one of them
  is answering the question asked.

One row in the table describes the expert but cannot route to it. Its own-class
precision needs its prediction, which costs the call you are deciding whether to
make. The routable row is `P(expert correct | the FREE winner said c)`, and the
two disagree in sign: on ISEAR class 1 the expert's own-class precision is 0.560
against bge-large's 0.698, while on items the free arm calls class 1 it scores
0.750 against 0.699. Both rows are printed; only the second decides.

## banking77: what changes at 77 classes

`run.py --datasets banking77`. Every per-class result above was measured at 2 to 7 classes,
which is the regime where a per-class method has the least to differentiate
between. banking77 is 13,083 short queries over 77 intents, a 1.3% majority
floor, and the first dataset here to trigger `label_shortlist()` -- 65,415
pair-scoring passes instead of 1,007,391, a 15.4x cut, on its first real outing.

No per-item LLM predictions exist for the full split, so the cascade half is
skipped and the pool half runs in full. (`runs/banking77/` holds 251 paired test
items against 20 LLMs, but at 77 classes that is ~3 per class and a ±0.033
floor; G6 would refuse to read anything into it.)

**The ordering flips.**

| construction | banking77, k=77 | OCL datasets, k≤7 |
|---|---:|---|
| DESlib LCA | **+0.0058** p=0.000 ✓Holm | −0.0057 … −0.0368, 4/4 lose |
| DESlib KNORA-E | **+0.0031** p=0.027 ✓Holm | −0.0053 … −0.0397, 4/4 lose |
| DESlib OLA | −0.0140 ✓Holm | 4/4 lose |
| our DCS-LCA variant | −0.0170 ✓Holm | loses wherever tried |
| stacking | −0.0019, p=0.286 | **+0.0097 on ISEAR** ✓Holm |
| best single (bge-large) | 0.9232 | ships on all four |
| Oracle ceiling | 0.9607 (+0.0375) | +0.0314 … +0.1971 |

Selection starts working and combination stops. At few classes the only thing
that ever converted pool headroom into accuracy was stacking; at 77 classes
stacking ties and two of DESlib's three selectors win.

Three things keep this a hint rather than a result:

- **+0.58 points is below banking77's own ±0.0066 unpaired floor.** The paired
  bootstrap resolves it -- that is what pairing is for, and its CI excludes zero
  -- but both numbers are printed so the tension is visible. One dataset, half a
  point.
- **LCA captures 15% of the Oracle gap** (+0.0058 of +0.0375). The first
  positive selection result in the project still leaves 85% on the table.
- **Our rule loses at both ends.** "Selection can work at high k" is a fact
  about DESlib's LCA, not about the per-class rule this project proposed.

### The per-class table

`competence_board` needs no expert -- it is a property of the pool -- and now
prints on the no-expert path, transposed so 77 intents are rows. Sorted by the
spread between the best and worst candidate on each class, because a class where
every candidate scores the same is one no selection rule can act on.

```
class                              n       tfidf bge-small bge-large    nli   spread  best
card acceptance                  135       0.809     0.869     0.895  0.519    0.376  bge-large
card delivery estimate           178       0.827     0.896     0.919  0.669    0.250  bge-large
transfer not received by reci    203       0.865     0.783     0.828  0.670    0.195  tfidf
transfer timing                  231       0.822     0.726     0.765  0.635    0.187  tfidf
declined card payment            225       0.862     0.797     0.785  0.705    0.156  tfidf
...
passcode forgotten               150       0.970     0.977     0.961  0.966    0.016  bge-small

classes owned: bge-large 56, tfidf 11, bge-small 10, nli 0
spread: median 0.101, max 0.376, min 0.016
excluding nli (owns 0 classes): median 0.029, max 0.095   <- what selection can use
```

**That last line is the one that matters, and it is a 3.5x correction to the
one above it.** NLI wins none of the 77 classes and is uniformly the worst, so
it widens every spread while offering nothing to select. Among the three
competitive candidates the median spread is 2.9 points, not 10.1. TF-IDF beats
bge-large on 13 of 77 classes at a mean margin of +0.029 and a maximum of
+0.076 -- thin, but real, and concentrated in classes with lexical giveaways
("transfer timing", "declined card payment", "apple pay or google pay").

So the reason selection works at 77 classes and not at 7 is not that the
per-class differences are larger. They are small either way. It is that 77
classes offer 13 chances to be right instead of one or two, and LCA converts
that into +0.58 points.

The pipeline reports the win and then declines to ship it, in those words:

    -> DESlib beats the best single model with DESlib LCA, DESlib KNORA-E
       not shipped: ... beat the baseline on a paired test but stay inside
       the unpaired resolution floor

That rule is new here. A construction that is named a winner and then silently
not shipped is the bug that once printed "stacking wins" three lines above a
comparison of bge-large; DESlib is now a candidate for the free arm on the same
terms as our own layer -- beat the best single model, survive Holm, **and** clear
the floor.

CLINC150, at 150 classes, is the confirmation run and needs no new code.

## DESlib, on our own pool

`deslib_selection()`. Dynamic classifier selection is the classic form of the
idea this project started from -- per input, trust the model most reliable in
that region -- so the library's own LCA, OLA and KNORA-E run on the same pool,
the same folds and the same data, tested against the best single candidate with
a paired bootstrap and Holm correction. 12 of 12 comparisons lose:

| dataset | best single | LCA | OLA | KNORA-E | Oracle ceiling |
|---|---:|---:|---:|---:|---:|
| IMDB | 0.9420 | −0.0057 | −0.0257 | −0.0053 | +0.0314 |
| HateSpeech | 0.8460 | −0.0292 | −0.0583 | −0.0397 | +0.1109 |
| ISEAR | 0.6879 | −0.0368 | −0.0416 | −0.0196 | +0.1393 |
| FEVER | 0.7349 | −0.0132 | −0.0275 | −0.0140 | +0.1971 |

Every loss survives Holm. Our own per-class rule lost to these same
implementations earlier, so the ordering is: best single model > DESlib > our
per-class selection. Selection loses; the question is only by how much.

**The Oracle column is the point.** +0.03 to +0.20 of headroom exists in these
pools -- on FEVER the Oracle is 0.9320 against a 0.7349 best single -- and no
selection rule tested here, ours or the library's, reaches any of it. That gap
is the standing invitation, and it is why the answer is "combine, do not
select": stacking is the only construction in this repo that ever converted pool
headroom into accuracy.

Two implementation notes, because both were bugs first.

- **The pool is frozen out-of-fold predictions, not refitted models.** DESlib
  needs a DSEL for competence estimation; carving it out of the training fold
  leaves the pool fitted on 2/3 of what the baseline saw, and the first version
  of this comparison reported that handicap as a result. Freezing makes DSEL an
  index set rather than a data cut, and the pool's predictions are identical to
  the candidates the rest of the pipeline scores. A test pins the baseline to
  the candidate's own number.
- **Refitting into one shared feature matrix was rejected**, not skipped: it
  would need TF-IDF reduced and the NLI scorer reshaped, so DESlib would be
  selecting among different models than the ones we ship. Instead `X` carries
  the row index and a companion kNN measures neighbourhoods in encoder space.

`pip install -e '.[deslib]'`. The pin is deliberate -- 0.3.7 predates
scikit-learn 1.7 removing `BaseEstimator._validate_data`, which `classpipe`
shims rather than patching the library, so the comparison is against DESlib's
own code.

## Board + dial: a negative result, and the gate that found it

The two rules compose into one: give each predicted class its own confidence
threshold, and split the budget across classes by marginal value per call. The
value of escalating item `i` is

    v(i) = ( expert_right[c] - free_right(conf_i, c) ) * W[c]

with `expert_right[c]` the board's shrunk P(expert correct | the free arm says
c), `free_right(q, c)` a per-class isotonic reliability curve shrunk toward one
global curve, and `W[c]` the mean balanced-accuracy weight of an item predicted
`c`. Both terms are per class and both are shrunk.

**It does not work on any of these four datasets, and the reason is that there
is nothing to win.** G8 fits *both* allocation rules on the held-out fold itself
-- an oracle neither can be built as -- so the gap is the ceiling with
estimation error removed:

| dataset | raw oracle gap | shuffled classes buy | real | floor | G8 |
|---|---:|---:|---:|---:|---|
| IMDB | +0.0005 | −0.0001 | +0.0006 | 0.0032 | FAIL |
| HateSpeech | −0.0000 | −0.0018 | +0.0018 | 0.0102 | FAIL |
| ISEAR | +0.0090 | **+0.0043** | +0.0047 | 0.0057 | FAIL |
| FEVER | −0.0079 | −0.0137 | +0.0058 | 0.0062 | FAIL |

The null column is the whole story. A per-class oracle fits K thresholds on the
fold it is scored on while the global oracle fits one, so part of any gap is the
wider search winning on noise. The null keeps every threshold and every budget
and destroys only the class structure, by permuting which group each item is in.
On ISEAR it buys +0.0043 of the +0.0090 -- **the one dataset that looked like a
win was about half search width** -- and the +0.0047 left over is inside the
±0.0057 that dataset resolves.

With G8 gating (recomputed inside each fold's training data, because the oracle
form reads held-out labels), `per_class_conf` is admitted in 0 of 5 folds on all
four datasets and the pipeline returns exactly its two-rule numbers: IMDB 0.9508
stable, HateSpeech 0.8642 stable, FEVER 0.8183 stable, ISEAR 0.7131 unstable.
The rule stays in the code and on the printed dial for inspection; it is simply
never selected. That is the intended outcome of a gate.

### Everything retracted along the way, and what replaced it

Recorded rather than quietly edited, because the pattern is the finding: five
claims, every one from tuning an estimator before measuring whether the quantity
it estimates has a ceiling above zero.

- **"It overfits, and the class count sets how badly"** (−0.0146 at K=20). No --
  the oracle gap on that synthetic is +0.0021 at K=7 and −0.0073 at K=20, so
  there was no headroom to overfit toward. What the number showed is a chunked
  allocator losing to a global sort on structureless data.
- **"Both rules it replaces are corner cases."** True only at 0% and 100%.
  Allocation moves in chunks of 2% of the training fold, so at intermediate
  budgets it cannot reproduce a global sort -- which is why the K=20 oracle is
  *worse* than a global threshold. The test asserting this checked only the
  endpoints and has been rewritten.
- **"When it helps is predictable from the spend split."** False. HateSpeech has
  the most uneven split of the four (0.5% of class 0 against 84.4% of class 1)
  and the smallest raw gap. The split does not predict it.
- **"ISEAR is the one dataset with real headroom."** Half of it was search
  width, and the remainder does not clear the floor.
- **The gate itself carried the bias it was built to catch.** Before the null,
  train-fold G8 admitted the rule on 5 of 5 IMDB folds against a full-data
  verdict of FAIL, because an oracle measured in a small fold is optimistic in
  proportion to how many parameters it fits. G4 had already solved this shape
  with an independence null; reusing that was the fix.

Two dead ends kept so they are not retried:

- Shrinking the *realised* per-class value toward the pooled value is a no-op
  for an arithmetic reason: with equal chunk sizes, `(sum + a*g)/(chunk + a)` is
  monotone in `sum`, so the argmax cannot move.
- Making the free-reliability curve global to "reduce variance" removes exactly
  the class-dependence the rule exists to exploit. Inside one confidence band,
  P(free correct) runs 0.388 to 0.810 across seven classes, because the
  posterior moves with each class's base rate even when the likelihood does not.

One cost worth recording: per-class isotonic curves depend on the training fold,
not the budget, and refitting them per budget point took the four-dataset run
from 164s to over half an hour. They are cached per fold, content-addressed
rather than index-addressed -- the same fold indices can be handed a different
free arm by an ablation or by `--train-on llm`.

## The deferral rules, and the two that were measured off the dial

```python
RULES = ("confidence", "margin", "committee", "class_aware", "per_class_conf",
         "precision_floor")
```

**`precision_floor` is the only rule that is not handed a budget.** Every other
rule is told how much traffic to spend and asked where; this one is told what to
promise -- *no class falls more than `m` below the LLM's own precision on that
class* -- and the spend is whatever keeping the promise costs. Its swept
parameter is therefore a margin, not a share, which is why `PARAM_KIND` exists
and why it has to bypass the `frac <= 0` / `frac >= 1` guards in `_allocate`:
margin 0 is the strictest setting there is, and the budget guards would read it
as "escalate nothing".

It is the Class Precision Margin of Aperstein & Apartsin, *CalexNet*
(arXiv:2509.08318), carried over from early-exit CNNs. The borrow is the
anchoring, not the machinery: this project already had a per-class precision bar
in `guaranteed_coverage`, and an **absolute** bar is wrong in both directions at
once.

HateSpeech is the clean demonstration. The free arm scores 0.8460 there against
the LLM's 0.8331 -- it is the *better* model -- yet a flat 0.90 bar hands 22.1%
of traffic to the LLM, drops the result to 0.8302, and still does not hold,
because a flat bar cannot tell a class the free arm is weak on from one the LLM
is equally weak on. Anchored at margin 0 -- "match the LLM class by class" --
the same data keeps 100% of traffic free, scores 0.8460, and the promise holds.
An anchored bar also cannot ask for the impossible, so no class is refused
merely for being hard.

`margin` (p1 - p2) and `committee` (how many pool members disagree, tie-broken by
confidence) are the two policies **Cache & Distil** uses, so until they were
added the dial was being compared against a literature baseline it did not
contain. Two notes on them:

- **committee earns its place.** It is what `cheapest_equivalent` picks on IMDB
  and HateSpeech, and it halves the recommended spend again -- both drop to 5% of
  traffic (0.9502 and 0.8552).
- **margin is confidence on a binary task**, necessarily: `p1 - p2 = 2*max - 1`
  is a monotone transform, so the ranking and the escalation set cannot differ.
  The dial prints that when it happens rather than showing two identical columns
  as if they were two agreeing pieces of evidence. Only ISEAR and banking77
  exercise it.

**`conformal` and `router` were on the dial for one full run and came off.** Both
are implemented and tested; neither is in the search:

| | why |
|---|---|
| `conformal` | a near-duplicate of confidence as a *routing* signal -- it won HateSpeech's nested pick 5/5 and scored 0.0005 less than confidence would have. Its value is the coverage guarantee, below, which needs no seat on the dial. |
| `router` | `P(free right) - P(expert right)` from a gradient-boosted pair. Lost to plain confidence at every budget on HateSpeech (0.8483 against 0.8538 at 5%), and where it won a nested pick it destabilised it. |

A third idea was measured and left out the same way: **knowledge distillation
into the cheap arm**, the dominant lever in CalexNet's own ablation. The LLM
cannot be the teacher here -- OCL ships parsed labels, not logprobs -- so the
test used the arm we actually ship, bge-large+logreg, teaching tfidf+logreg,
with teacher probabilities from an inner CV inside each training fold (in-sample
teacher outputs are near-one-hot; the pipeline's own out-of-fold probabilities
would leak the outer test fold into the student's targets).

| dataset | hard-label student | best soft-target student | delta | floor |
|---|---|---|---|---|
| isear | 0.6062 | 0.6100 | +0.0038 | ±0.0057 |
| hatespeech | 0.7789 | 0.7873 | +0.0084 | ±0.0102 |
| fever | 0.6829 | 0.6828 | −0.0001 | ±0.0062 |

Never outside the floor, negative once, and it recovers at most a seventh of the
0.052–0.080 the teacher leads by. The reason is structural: CalexNet's branch
reads the backbone's *own* feature map, so "this is a 3, and otherwise a 5" is a
sentence the student can express. Ours reads sparse n-grams while the teacher
reads a 1024-d embedding, and that geometry has no coordinates in the student's
space. `soft_target_fit` stays in `classpipe.py`, tested, for the case it was
built for and that we cannot yet run: a teacher that emits a distribution.

### Two models measured as candidates and not added

Neither is referenced anywhere in the code. They are recorded here so the same
two ideas are not re-run on the same evidence.

**GLiNER2 205M** (`fastino/gliner2-base-v1`), as a sixth pool member, on ISEAR
-- deliberately ISEAR, because its ceiling gap of 0.1393 is the largest we have
and a member that cannot help there cannot help anywhere.

| | without GLiNER2 | with GLiNER2 |
|---|---|---|
| oracle ceiling | 0.8272 | **0.8503 (+0.0231)** |
| G4 diversity_ratio | 0.492 | 0.546 |
| DESlib LCA | −0.0368 | −0.0391 |
| DESlib KNORA-E | −0.0196 | −0.0245 |
| selection_dcs_lca | −0.0237 | −0.0276 |
| stacking | +0.0097 | +0.0116 |

It is the first candidate ever to move the ceiling by a resolvable amount
(+0.0231 against a ±0.0057 floor), and none of it converts. Every method that
*selects* a member got worse, because selection has to decide when to trust the
weakest model in the pool (0.5955) and cannot. Stacking, which *weights* rather
than selects, gained +0.0019 -- inside the floor. Two facts worth keeping: its
encoder is `microsoft/deberta-v3-base`, the same backbone as the NLI candidate,
so it is a second head on a transformer already in the pool rather than a new
architecture; and **zero-shot it scores 0.5953 against a trained tfidf+logreg's
0.6011**, a gap at the resolution floor with no labels at all. That last number
is a cold-start result, not a pool result, and nothing here measures it.

**TinyLlama-1.1B-Chat**, as the *expert* the high-k datasets lack.

| | TinyLlama (scored) | free arm | real expert | chance |
|---|---|---|---|---|
| isear (k=7) | 0.3893 | 0.6879 | 0.7036 | 0.1429 |
| banking77 (k=77) | 0.2623 | 0.9232 | none exists | 0.0130 |

At 28% of the free arm on banking77 it cannot be an expert: escalating to it
lowers accuracy on every item routed, and the resulting cascade's correct answer
is "escalate nothing", which HateSpeech already demonstrates. Two method notes.
*Prompted*, it is unusable -- 94% of replies unparseable, because a 1.1B chat
model echoes the input rather than holding a label-list instruction; the numbers
above come from constrained scoring, which is its best case, and the isear
control (0.3893 against a 0.1429 chance floor) is what proves the harness sound
rather than the model weak. And *length-normalised* scoring collapses to near
chance (0.1821 / 0.1558): score a label set by raw logprob sum, not per-token
mean.

Neither moved a score by more than its dataset's resolution floor, while
together they took the run from 165s to 324s and cost stability: FEVER's
selection bias went +0.0009 to +0.0037, and IMDB lost a stable 10% operating
point for budgets scattered over 0.2-0.5. That is the cost of a wider search
stated as a measurement rather than a worry.

The router result is worth keeping as a finding rather than an absence. Every
other rule is a *proxy* for "the expert will do better here"; the router
estimates it directly, with gradient boosting rather than the two logistic
regressions the research brief specified, precisely because the signal is an
interaction a linear model cannot represent. It still loses -- so the earlier
LegalBench negative was not "too weak a learner". It has to separate two cells of
the routing table from a noisy target, on the 20-34% of items where the arms
disagree at all.

## Conformal coverage, and why Mondrian

The per-class guarantee was the one broken thing in this project:
`guaranteed_coverage()` used an empirical threshold that did not survive out of
fold, realising 0.866 against a 0.90 bar on HateSpeech. Split conformal derives
the threshold instead -- score every calibration item by `1 - p(true label)`,
take the quantile matching the error rate you asked for, with the `(n+1)/n`
correction that makes it finite-sample rather than asymptotic.

Its guarantee is **marginal**, though, and that is exactly how a rare class gets
sacrificed to hold a headline number:

| dataset | split: worst class | under target | Mondrian: worst class | under target |
|---|---:|---|---:|---|
| **ISEAR** (7 classes) | 0.8686 | **5 of 7** | 0.8966 | **1 of 7** |
| HateSpeech | 0.8992 | 1 of 2 | 0.9002 | **0 of 2** |
| banking77 | 0.8589 | 1 of 2 | 0.9000 | 1 of 2 |
| FEVER | 0.8973 | 1 of 2 | 0.8993 | 1 of 2 |

ISEAR is the case that makes the point. Split conformal's *marginal* coverage
reads 0.8994 against a 0.90 target -- it looks fine -- while five of seven
classes miss, the worst at 0.8686. Mondrian calibrates a threshold per class and
takes that to one of seven. Testing candidate label `c` against class `c`'s own
threshold is what keeps it runnable: it conditions on the label being
*considered*, never on the true label, which is the thing the decision is for.

## The cheapest operating point that is the same answer

The dial reports its argmax, and for a long time nothing asked whether something
cheaper was just as good. `cheapest_equivalent()` now reports the lowest LLM
spend within one resolution floor of the peak, and on all four datasets that is
roughly **half the traffic**:

| dataset | peak | cheapest equivalent | gives up | floor | spend saved |
|---|---|---|---:|---:|---:|
| IMDB | 0.9508 @10% | 0.9486 @**5%** | 0.0022 | ±0.0032 | 5 pts |
| HateSpeech | 0.8642 @20% | 0.8602 @**10%** | 0.0040 | ±0.0102 | 10 pts |
| ISEAR | 0.7131 @33% | 0.7142 @**20%** | 0.0039 | ±0.0057 | 13 pts |
| FEVER | 0.8192 @75% | 0.8151 @**50%** | 0.0041 | ±0.0062 | 25 pts |

This is G6 applied to operating points. The gate already refuses to call a
sub-floor gap a difference when comparing systems; there was no reason budgets
got a pass. The dial shades the band within one floor of the peak and rings the
cheapest point inside it.

## Where the wins are: the 2x2, not the class grid

A k x k class confusion says what one model is wrong *about*. It does not say
what a cascade can do about it, and at 77 classes it is 5,929 cells nobody
reads. The panel is the paired table instead -- the same shape at 2 classes and
at 77 -- with each cell saying what the right call is for those items and what
the shipped rule actually did with them:

|  | LLM right | LLM wrong |
|---|---|---|
| **free right** | don't pay — free already gets it | keep free — a call breaks it |
| **free wrong** | **pay — only the call fixes it** | neither arm gets these |

Each cell is a share of **all** items and the four sum to 100% -- every item
lands in exactly one -- so the top-left is not "both arms score 60%", it is the
share both answer correctly. The row and column totals are then the two arms'
own accuracies, which is what ties the panel to the rest of the page. ISEAR:

```
              LLM right        LLM wrong        any
free right    60.1%  4,608     9.6%    739     69.7%  <- free arm accuracy
free wrong    10.2%    785    20.0%  1,534     30.3%
any           70.3%           29.7%           100.0%
                ^ LLM accuracy
```

Read that way it also says how much room routing has at all: the two arms agree
on 80.1% of ISEAR and disagree on 19.8%, which is the entire playing field --
10.2% winnable against 9.6% breakable. The halves nearly cancel, and ISEAR
duly captures only 15% of its headroom.

Only one cell is the prize, and one is a trap. On FEVER: 1,311 items (20.1%) are
winnable and 888 (13.6%) are broken by escalating, so **a perfect router scores
0.9361** against 0.7348 for the free arm alone. The shipped dial reaches 0.8183,
which is **41% of the available headroom** -- a number that did not exist before,
because the ceiling and the achieved score were reported separately and nobody
divided them.

The `rule sent` figure under each cell is the diagnosis rather than the ceiling.
The dial cannot tell the cells apart -- it only sees confidence -- so what it
wants is a high share on the green cell and a low one on the red, and the gap
between those two is exactly where the capture figure comes from. If they were
equal, routing would be worth nothing at any budget.

Datasets with no teacher get the same table against the pool's own Oracle, so
banking77 reads as *what selection could win* rather than showing an empty
panel. The full class matrix, per-class recall and precision, and the ranked
confusions are all still in the artifact; the panel quotes one line of it (the
largest single error) and leaves the rest there.

## TP / FP / FN / TN for the shipped configuration

Section 3 says what a cascade *could* win. Section 4 is the ordinary 2x2 for
what actually ships. A single 2x2 over a multiclass problem has to pick what
"positive" means, so it is one-vs-rest with a class selector -- binary datasets
default to their positive class:

```
FEVER, positive = a true claim
                          predicted true claim     predicted other      total
actually a true claim     TP 2,597 (79.8%)         FN   656 (20.2%)     3,253
actually other            FP   527 (16.2%)         TN 2,732 (83.8%)     3,259
precision 0.8313 · recall 0.7983 · F1 0.8145 · specificity 0.8383
```

Percentages are of the row, so the TP cell is recall and the TN cell is
specificity. Pick any class on a multiclass set and it reads the same way --
ISEAR with `shame` as positive is precision 0.6707 against recall 0.5109, which
says the model is reasonably careful when it *does* say shame and misses half of
them, a distinction nothing else on the page makes.

There is also an **all classes (micro)** option, and it is degenerate by
construction rather than by accident: every wrong item contributes one FP to the
class it was given and one FN to the class it belonged to, so FP equals FN and
precision, recall and F1 all collapse to accuracy (ISEAR: 0.7130 three times
over). It is the standard micro-average and it is worth having, but the panel
says why the three numbers are identical instead of leaving it to be puzzled
over.

The panel is that table and the class selector, nothing else. A by-class list
and a full k x k grid lived here briefly and were removed: per-class detail
already has two sections of its own further down, and this one answers a
different question. The k x k matrix stays in the artifact -- it is what the 2x2
is computed from, so the two cannot disagree.

## Reading the per-class chart: the pair, not the columns

Two questions look identical and are not:

- *how often is this model right when it says class c* — the columns
- *on the items the FREE arm called class c, who is right more* — the decision

They can point opposite ways, and when they do the columns are the misleading
one. HateSpeech, predicted class "hate":

| | |
|---|---|
| tfidf on its own hate calls | 0.434 |
| LLM on its own hate calls | 0.390 |
| **free winner on the items it called hate** | **0.382** |
| **LLM on those same items** | **0.639** |

Read the first two and the LLM looks worse and escalation looks pointless. Read
the second two — the pair the router actually consults — and the LLM gains 25.7
points there. The difference is the item set: tfidf labels hate on different
things, and the free arm's misses sit inside its own hate bucket where the LLM
recovers them.

So the chart draws the decision as a joined pair: a white dot for the free
winner, a red diamond for the LLM, both on the same items, joined green when
escalating that class pays. FEVER shows two green joins, HateSpeech one. The
candidate columns behind them are context.

## The gates

Each one exists because it caught a mistake that had already been made by hand.

| gate | checks | fired on |
|---|---|---|
| G1 | a constant predictor beats the models → force balanced accuracy | HateSpeech (0.8883) |
| G2 | duplicate texts must not straddle CV folds | 3 of 4 datasets |
| G3 | pool is a total ROC order → nothing to select between | (opened once NLI joined) |
| G4 | oracle vs an independence null → near-identical members | IMDB (92.3% agreement) |

G4's null is saturated and the ratio, not the null, is the number to read. An
oracle over five members at ~0.7 accuracy is right about 1 − 0.3⁵ ≈ 0.998 of the
time even when the members are independent, so `oracle_if_independent` sits at
0.97–1.00 on every dataset and the gap to it says almost nothing. Making the
permutation class-stratified was a real fix to a real bug and moved the null by
0.011 at most (ISEAR 0.9824 → 0.9710); it did not, and could not, cure the
saturation. The usable statistic is `diversity_ratio` = (oracle − best single) /
(null − best single), the share of the *achievable* headroom the pool actually
spans:

| dataset | oracle | null | diversity_ratio | mean pairwise disagreement |
|---|---|---|---|---|
| banking77 | 0.9607 | 0.9996 | 0.491 | 12.5% |
| fever | 0.9320 | 0.9906 | 0.771 | 28.2% |
| hatespeech | 0.9568 | 0.9985 | 0.727 | 15.0% |
| imdb | 0.9734 | 0.9996 | 0.545 | 7.7% |
| isear | 0.8272 | 0.9710 | 0.492 | 33.9% |
| G5 | per-class advantage flat → selection layer has no signal | FEVER, IMDB |
| G6 | difference smaller than the dataset can resolve | all four `--train-on llm` |
| G7 | P(correct \| predicts c) below a coin flip → rules drift to the majority | HateSpeech (40.7%) |
| G8 | oracle per-class vs oracle global, against a shuffled-class null | all four |

The class-aware layer is fitted only when the gates open — once in four, on ISEAR, where
stacking beat the best single model by +0.0097 (p=0.004, survives Holm) and DCS-LCA lost
by 0.0237. **Combination wins, selection loses**, on every pool tested here and against
DESlib's own LCA/OLA/KNORA-E.

## Reconstruction

The repo ships LLM annotation streams but not the `*_preprocessed.csv` its configs read;
those are in a Google Drive archive linked from the README, with no revision and no
published checksum. So alignment is demonstrated, not assumed: every LLM cell of Table 1
comes back for both teachers.

| task | rows | GPT-3.5 | Llama-2-70B |
|---|---:|---|---|
| IMDB | 25,000 | 94.15 ✓ | 93.33 ✓ |
| HateSpeech | 10,703 | 83.34 ✓ / recall 83.28 ✓ | 77.81 ✓ / recall 82.19 ✓ |
| ISEAR | 7,666 | 70.34 ✓ | 68.23 ✓ |
| FEVER | 6,512 | 79.98 ✓ | 77.15 ✓ |

A misaligned stream scores at chance: before the authors' archive was found, rebuilding
ISEAR from public sources scored 0.1542 against a 0.1556 shuffled control. Sources are
pinned by sha256, the GitHub side by commit `285e03fc98afabdf7768558d33ee7b5f662b5f19`.
Two details measured rather than assumed: the paper **truncates** on ISEAR and rounds
elsewhere, and two ISEAR annotations are malformed and resolve by first-substring-match.

## Candidate pool

`tfidf+logreg`, `bge-small`, `bge-large`, `zero-shot NLI`, and `constant-majority`.

- **bge-base was dropped**: small/base/large is one family at three sizes, so it came out
  ROC-ordered by capacity every time and G3 could only ever fail.
- **NLI was added** because it scores (text, label-sentence) pairs jointly — the only
  mechanism in the pool that does. It is load-bearing: ablation drops the ISEAR stacking
  win from +0.0112 to +0.0046 (p=0.124, fails Holm).
- **The reranker was removed on measurement**, not suspicion: last of the real candidates
  on all three datasets it saw (0.4181 / 0.5210 / 0.6562), and worth 0.0015 to the
  stacking win. It raised raw disagreement while contributing almost nothing, which is
  why G4's disagreement proxy is weaker than it looks. Still one `--with-reranker` away.
- **Fine-tuning was considered and rejected** for the default pool: a frozen encoder plus
  a linear head retrains in seconds on CPU, while fine-tuning needs a GPU job on every
  data shift. A gate you run less often is a worse gate.

## Twenty-five algorithms, one AutoML run, and what they bought

The pool's members were all chosen by hand, so two questions were open: is a better
single arm sitting in a family we never tried, and would per-class routing work if the
pool were bigger? Both were answered by brute force — every classifier family scikit-learn
offers, plus LightGBM and XGBoost, on the pipeline's own grouped folds, out-of-fold, with
the same `class_weight="balanced"`. FLAML ran alongside as real AutoML. Everything below
ran in an isolated venv on frozen arrays; the project environment was never touched.

**A better single arm exists, and it is not a neural network.** Counting only models that
beat the shipped `logreg` by more than the dataset's resolution floor:

| | floor | shipped `logreg` | best in the zoo | beat it, resolvably |
|---|---|---|---|---|
| ISEAR | 0.0057 | 0.6879 | `svc-rbf` 0.6948 (+0.0069) | 1 of 24 |
| HateSpeech | 0.0102 | 0.8460 | `ridge` 0.8542 (+0.0082) | 0 of 24 |
| banking77 | 0.0066 | 0.9232 | `knn-5` **0.9416** (+0.0184) | 7 of 24 |

banking77 is where it matters: `knn-5`, `linear-svc`, `svc-rbf`, `knn-15`, `extra-trees`,
`random-forest` and `sgd-huber` all clear the floor over the shipped arm. The three
gradient-boosting libraries — the families AutoML frameworks are built around — land
*below* it on all three datasets (banking77 `lightgbm` 0.9049, `hist-gb` 0.9025 after
1,427s, `xgboost` 0.8848) after twenty to forty times the compute of a one-second linear
model. Dense sentence embeddings are not a tabular problem.

**Real AutoML lost to a one-line default, and lost on search space rather than search.**
FLAML at 300s/fold on ISEAR returned **0.6872**, below `LogisticRegression` at 0.6879 and
well below `svc-rbf` at 0.6948, after 26 minutes. It picked `sgd` on all five folds. Its
default classification space is `lgbm, rf, xgboost, extra_tree, xgb_limitdepth, sgd, lrl1`
— no kernel SVM, no kNN, no ridge, no LDA, and `lrl1` is L1-penalised only, so not even
plain logistic regression. Every algorithm that won here is outside it. Registered as
custom learners, FLAML does find `svc_rbf` on every fold and reaches 0.6990 — +0.0042
over untuned `svc-rbf`, which is *below* the 0.0057 floor. Practically all of the gain is
the algorithm; almost none of it is the tuning. (Caveat on that last figure: in the run
that produced it the custom learner's `gamma` never reached the estimator, so only `C` was
searched. Re-running with `gamma` live is outstanding, and since `gamma` is the more
important RBF knob the +0.0042 is a lower bound on what tuning can buy.)

**Routing over a bigger pool is strictly worse, and the harm scales with pool size.** The
pool filled best-first, routed by per-(model, predicted class) precision:

| N | 1 | 2 | 4 | 8 | 12 | 25 |
|---|---|---|---|---|---|---|
| ISEAR | **0.6948** | 0.6904 | 0.6871 | 0.6753 | 0.6537 | 0.6037 |
| HateSpeech | **0.8542** | 0.8463 | 0.8315 | 0.8056 | 0.6704 | 0.5078 |

Monotone, both datasets, unchanged as the minimum support per cell moves from 20 to 400.
Every candidate added is another chance for a per-class precision estimate to be fooled by
a lucky high-precision sliver, and at 25 candidates the winner's curse dominates
completely. This is the "why not just route the classes tfidf wins?" question answered at
full scale: the answer is that selection noise grows faster than the competence signal.

So the zoo's payoff is a better *member*, never a better *combination* — which is the
gates' position, arrived at independently.

### kNN is the one addition that earned a re-run

`knn-cos` was measured as a pool member on all six datasets, against the arm the pipeline
would actually ship, with the paired class-stratified bootstrap:

| | shipped arm | with `knn-cos` | delta | p |
|---|---|---|---|---|
| banking77 | `bge-large+logreg` 0.9232 | **`knn-cos` 0.9363** | +0.0131 | 0.0000 |
| HateSpeech | `bge-large+logreg` 0.8460 | **`stacking` 0.8587** | +0.0127 | 0.0000 |
| ISEAR | `stacking` 0.6975 | **`stacking` 0.7057** | +0.0082 | 0.0010 |
| FEVER | `stacking` 0.7640 | `stacking` 0.7629 | −0.0011 | 0.137 |
| IMDB | `stacking` 0.9457 | `knn-cos` 0.9458 | +0.0000 | 0.961 |
| CLINC150 | `bge-large+logreg` 0.9766 | unchanged | 0.0000 | 1.000 |

Holm over the six leaves the top three, which are exactly the three above their floors.
On HateSpeech it changes the verdict rather than a number: the pipeline currently ships a
bare `bge-large+logreg` because no construction wins, and with `knn-cos` in the pool
stacking wins and ships.

CLINC150 gaining nothing at k=150 while banking77 gains at k=77 rules out class count as
the mechanism; it is headroom, the same conclusion CLINC150 forced on the oracle gap.
`k=15` was fixed before any of this ran and never tuned, so there is no selection-on-test
— and also no optimisation: the zoo later found `knn-5` better at 0.9416.

A kNN win is the result most likely to be near-duplicate leakage, so it was attacked
before being believed. The **first attempt was wrong**: connected components over
"cosine ≥ τ" chains, and on banking77 at τ=0.95 the largest component swallowed 111 of the
199 items in *activate my card* while at τ=0.90 one component held 4,683 items, 36% of the
corpus. Holding a group like that out deletes half an intent instead of controlling for
leakage — both arms dropped, which was the tell. Redone with within-class complete linkage,
which cannot chain (largest cluster 19), the win survives: **+0.0139** at τ=0.95 with 6,223
items in multi-item groups, +0.0082 at τ=0.90. And it is not retrieval — kNN's margin is
*largest* where the nearest training neighbour is far (+0.0354 in the [0.80, 0.90) band)
and smallest where it is close (+0.0082 in [0.95, 0.98)).

Not yet acted on: adding `knn-cos` changes every dataset's numbers and the dashboard, so
it is a pool change, not a patch.

### Measured and not added: PyTorch MLPs

A ReLU head on TF-IDF reproduces the linear head it replaces — ISEAR +0.0008, and it
*loses* on HateSpeech (−0.0278) and FEVER (−0.0350). On embeddings it ties (±0.006 on three
datasets). It pays only where there is headroom and many classes: banking77 +0.0111
(resolvable, p=0.0000) for `tfidf⊕bge-large+mlp`, and CLINC150 only +0.0031 against a
0.0033 floor — significant at p=0.0005 and still smaller than the dataset can measure,
which is the case G6 exists for. `knn-5` beats the best MLP on banking77 (0.9416 vs 0.9342)
at a thousandth of the training cost, so the MLP has no configuration in which it is the
right answer here.

## Cost controls

- `pair_scoring_plan()` budgets in token-units (pairs × truncated length), not pairs.
  IMDB is why: 25,000 two-class documents is only 50,000 pairs but ran **8.3 hours**, and
  both pair scorers then lost to the frozen bi-encoder. It is now skipped there, and the
  full four-dataset run takes about three minutes.
- `label_shortlist()` is retrieve-then-rerank for large label sets: the bi-encoder picks
  the top 5, so CLINC150 would need ~113k passes instead of ~3.4M. Not exercised by these
  four datasets (k ≤ 7).
- `eliminate()` is successive halving (Jamieson & Talwalkar 2016; Li et al. 2017) with
  class-stratified rungs and **conservative** elimination — a candidate is dropped only
  if its interval lies entirely below the leader's. Textbook halving keeps the top half
  by point estimate, which on a 20% rung of HateSpeech (~239 minority items, ±2.5 points)
  is the noise-driven decision the gates exist to prevent.

## Credit

Dynamic classifier selection is classic — LCA, OLA, KNORA-E, and DESlib implement them;
our per-class confidence rule is a variant of LCA, not an invention, and it lost to
DESlib's implementations here. What is not in that literature is combining a per-class
**guarantee** with LLM deferral, and that half is unfinished: the empirical precision
threshold does not hold out-of-fold on rare classes (HateSpeech realized 0.866 against a
0.90 bar, 0.818 against 0.95), which points at class-conditional (Mondrian) conformal
prediction rather than empirical thresholds.

## Limits

- Four datasets from one paper. The gates are calibrated on a sample of four.
- No licence on either the OCL repo or the Drive archive; both are pinned, nothing is
  vendored. Teachers are 2023 models.
- ISEAR's dial rule is **unstable**: four folds pick confidence at 30%, one picks
  class-aware at 50%. "About 30% to the LLM, rule unresolved" is the defensible reading.
- IMDB's G4 failure is partly self-inflicted — the cost guard skipped the pair scorers,
  leaving a pool of near-identical encoders.
- Nothing here is paired against OCL, so no significance test against them is possible.
