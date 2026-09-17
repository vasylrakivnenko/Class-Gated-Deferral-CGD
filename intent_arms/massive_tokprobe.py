"""Tokenizer fertility probe: evidence for the encoder prediction that does NOT
depend on training anything.

The pre-registered claim is that jhu-clsp/ettin-encoder-68m, an ENGLISH encoder,
will collapse on non-Latin scripts because its vocabulary has no subwords for
them, so each character falls back to a byte/UNK piece. Fertility (subword
pieces per whitespace-free character, and pieces per utterance) measures that
directly, and the UNK rate measures the worst case. Run before reading the
fine-tuning results so the mechanism is not retro-fitted to the outcome.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

REPO, REV = "AmazonScience/massive", "refs/convert/parquet"
S = "/private/tmp/claude-501/-Users-vasyl-zadumai/12e40f00-be9f-4f43-9226-7a491b668aef/scratchpad"
LOCALES = ["en-US", "de-DE", "es-ES", "fr-FR", "ru-RU", "zh-CN", "ja-JP",
           "ko-KR", "ar-SA", "hi-IN", "th-TH", "sw-KE", "am-ET"]
MODELS = {"ettin_en": "jhu-clsp/ettin-encoder-68m",
          "e5_multi": "intfloat/multilingual-e5-small"}


def utts(loc, part="test"):
    p = hf_hub_download(REPO, f"{loc}/{part}/0000.parquet",
                        repo_type="dataset", revision=REV)
    return pq.read_table(p, columns=["utt"]).to_pydict()["utt"]


def main():
    toks = {k: AutoTokenizer.from_pretrained(v) for k, v in MODELS.items()}
    for k, t in toks.items():
        print(f"{k}: vocab {t.vocab_size:,}  unk={t.unk_token!r}")
    out = {}
    print(f"\n{'locale':8} " + " ".join(
        f"{k+' piece/ch':>18} {k+' unk%':>13}" for k in MODELS))
    for loc in LOCALES:
        xs = utts(loc)
        nchar = float(np.mean([len(x.replace(' ', '')) for x in xs]))
        row = {"mean_chars_nospace": nchar}
        for k, t in toks.items():
            ids = t(list(xs), add_special_tokens=False)["input_ids"]
            npiece = float(np.mean([len(i) for i in ids]))
            unk = t.unk_token_id
            n_unk = sum(i.count(unk) for i in ids) if unk is not None else 0
            n_tot = sum(len(i) for i in ids)
            row[k] = {"pieces_per_utt": npiece,
                      "pieces_per_char": npiece / nchar,
                      "unk_share": n_unk / max(n_tot, 1)}
        out[loc] = row
        print(f"{loc:8} " + " ".join(
            f"{row[k]['pieces_per_char']:>18.3f} {row[k]['unk_share']:>12.2%}"
            for k in MODELS))
    json.dump(out, open(f"{S}/massive_tokprobe.json", "w"), indent=1)
    print(f"\nwrote {S}/massive_tokprobe.json")


if __name__ == "__main__":
    main()
