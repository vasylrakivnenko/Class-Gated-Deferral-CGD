# Long-context ModernBERT fine-tune on supply_chain_disclosure_best_practice_audits -- ABANDONED (hardware/tooling limit)

Motivation: TF-IDF (80.9%) beat every neural cheap model on this task because the
documents are long (p50 510 tokens, p90 1,115, max 4,924) and every embedder/
fine-tune candidate truncated at 512 tokens (only 50% of docs fully visible;
bge-large at 512 scored 70.2%). ModernBERT supports 8k context; the question was
whether seeing the whole document closes the gap. Raising the cap to 2048 covers
97.4% of documents.

What was tried on an Apple M4 Pro (MPS backend, torch 2.14):

| config | outcome |
|---|---|
| 512 tokens, batch 8 (baseline, learned_hands) | 30 fits in 72 min; 84.5% vs embed_small 84.3% (+0.2, ns) |
| 2048 tokens, batch 4 | MPS out of memory at 29.8 GiB of 30.2 GiB (global-attention 2048x2048 score matrices, no flash-attention on MPS) |
| 2048 tokens, batch 2, dynamic padding | fit 1 still running at 40 min; stack samples ~50% Metal graph *compilation* -- every distinct padded length compiles new graphs across 22 layers, fwd+bwd |
| 2048 tokens, batch 2, lengths bucketed to multiples of 256 (<=8 shapes) | fit 1 still >40 min, still ~50% compilation. Killed. |

Conclusion: on this hardware, 2048-token fine-tuning of ModernBERT-base costs
>40 min per fold-fit (10 fits for one repeat, 30 for the project's 3-repeat
standard), dominated by MPS graph compilation that does not amortize within a
fit. This is a tooling/hardware limit, not a model result: the long-context
question is untested, not answered. It would be a ~10-minute job on a CUDA
GPU with flash-attention. The env knobs added for the attempt
(FINETUNE_MAX_LENGTH, FINETUNE_TRAIN_BATCH_SIZE, FINETUNE_EVAL_BATCH_SIZE,
FINETUNE_LEN_BUCKET) default to the original behaviour and are kept.
