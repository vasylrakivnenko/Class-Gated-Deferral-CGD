"""Measurement probes for the response-selection heads H5 / H7 / H4.

WHY THIS PACKAGE EXISTS
-----------------------
DECISIONS D2b and D7 record, twice, that skeleton (H5), template (H7) and values
(H4) are "not yet measured on any axis". Every measured cell in this project
covers only the dialogue-state heads (nextstep, intent, action). H7 nevertheless
carries the highest loss weight in the config (``model.loss_weights.template:
2.0``), so the least-measured head is the most heavily optimised one.

This package is the measurement protocol for those three heads. It is a PROBE,
not part of the shipped pipeline: nothing in ``src/reflex/`` imports it, and it
never writes into ``outputs/compile``.

WHAT IS IN HERE
---------------
* :mod:`probes.featurizer_api` -- the interface this harness requires from the
  recency-tagging featurizer (owned by a different agent), plus a conformance
  check that runs with no corpus.
* :mod:`probes.response_labels` -- how a gold skeleton / gold template / gold
  value is derived per agent turn, and the coverage audit that says what
  fraction of turns even have one.
* :mod:`probes.response_metrics` -- the metric per head, its label-blind
  constant baseline, the near-duplicate tiers for H7, and conversation-clustered
  bootstrap CIs.
* :mod:`probes.run_response_probe` -- the CLI. ``validate`` reproduces a known
  D7 cell before anything else is trusted; ``measure`` runs the arms.

READ BEFORE CHANGING ANYTHING HERE
----------------------------------
D5 (constant-predictor guard): every headline metric is reported beside what a
label-blind constant scores, and a metric a constant wins is DROPPED, not
caveated. D7 (controls that can fail): ``nat->shuf`` alone is vacuous; the
experiment is ``shuf->shuf``, refit on scrambled data. D6 (a baseline is only a
baseline WITH its configuration attached): every number this harness emits
carries its window, its tagging, its denominator and its row set.
"""

__all__ = ["featurizer_api", "response_labels", "response_metrics"]
