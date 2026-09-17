"""Why the word-only TF-IDF half collapses, per locale -- and the correction to
a prediction this arm got WRONG.

Pre-registered prediction (in massive_arm.py): the word half collapses on the
three no-whitespace scripts zh-CN, ja-JP, th-TH, and the char_wb half carries
them. Measured outcome:

  zh-CN  word 0.1459 vs char 0.7599   -61.4 pts   <- predicted, confirmed
  ja-JP  word 0.1715 vs char 0.8013   -63.0 pts   <- predicted, confirmed
  th-TH  word 0.7095 vs char 0.8090   -10.0 pts   <- predicted, MUCH weaker
  hi-IN  word 0.5797 vs char 0.8211   -24.1 pts   <- NOT predicted at all

So there are TWO separate mechanisms, not one, and the prediction conflated
them. This script measures both.

MECHANISM 1 -- no whitespace. sklearn splits on whitespace, so for zh-CN the
whole utterance is a single token (1.05 whitespace tokens/utterance). Nothing
generalises and the word half is near-useless.

MECHANISM 2 -- sklearn's DEFAULT token_pattern r"(?u)\\b\\w\\w+\\b" does not
match Unicode combining marks. Devanagari vowel signs / virama are categories
Mn/Mc, so a Hindi word is shredded into the consonant runs between its marks:
    'शुक्रवार को सुबह नौ बजे मुझे जगा दो'  ->  ['रव', 'बह', 'बज', 'जग']
That is why hi-IN loses 24 points with plenty of whitespace available, and it
also explains why th-TH loses only 10: MASSIVE's Thai carries artificial
spacing (3.35 whitespace tokens/utterance; the paper notes Thai spacing is
optional and that models learn from the artificial spacing around slots), so
Thai is hurt mainly by mechanism 2, not mechanism 1.

The practical consequence is a config bug, not a language property: the default
token_pattern is wrong for abugida scripts, and it is invisible if you only ever
look at the union score.
"""
from __future__ import annotations

import json
import re
import unicodedata as ud

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

S = "/private/tmp/claude-501/-Users-vasyl-zadumai/12e40f00-be9f-4f43-9226-7a491b668aef/scratchpad"
REPO, REV = "AmazonScience/massive", "refs/convert/parquet"
LOCALES = ["en-US", "de-DE", "es-ES", "fr-FR", "ru-RU", "zh-CN", "ja-JP",
           "ko-KR", "ar-SA", "hi-IN", "th-TH", "sw-KE", "am-ET"]
SKLEARN_DEFAULT = re.compile(r"(?u)\b\w\w+\b")


def get(loc, part="train"):
    p = hf_hub_download(REPO, f"{loc}/{part}/0000.parquet",
                        repo_type="dataset", revision=REV)
    return pq.read_table(p, columns=["utt"]).to_pydict()["utt"]


def main():
    out = {}
    print(f"{'locale':8} {'ws tok':>7} {'skl tok':>8} {'ws rejected':>12} "
          f"{'mark chars':>11} {'empty utts':>11}")
    for loc in LOCALES:
        xs = get(loc)
        ws = np.array([len(x.split()) for x in xs], float)
        skl = np.array([len(SKLEARN_DEFAULT.findall(x)) for x in xs], float)
        rej = np.array([sum(1 for t in x.split()
                            if not SKLEARN_DEFAULT.fullmatch(t)) for x in xs], float)
        # share of characters that are Unicode combining marks (Mn/Mc/Me)
        nm = nt = 0
        for x in xs:
            for c in x:
                if not c.isspace():
                    nt += 1
                    if ud.category(c) in ("Mn", "Mc", "Me"):
                        nm += 1
        empty = float((skl == 0).mean())
        out[loc] = {"mean_whitespace_tokens": float(ws.mean()),
                    "mean_sklearn_tokens": float(skl.mean()),
                    "mean_whitespace_tokens_rejected": float(rej.mean()),
                    "share_whitespace_tokens_rejected": float(rej.sum() / ws.sum()),
                    "combining_mark_char_share": nm / nt,
                    "share_utts_with_zero_sklearn_tokens": empty}
        print(f"{loc:8} {ws.mean():>7.2f} {skl.mean():>8.2f} "
              f"{rej.sum()/ws.sum():>11.1%} {nm/nt:>10.1%} {empty:>10.1%}")

    print("\nhi-IN examples under sklearn's default token_pattern:")
    for x in get("hi-IN")[:5]:
        print(f"  {x!r}\n    -> {SKLEARN_DEFAULT.findall(x)}")

    json.dump(out, open(f"{S}/massive_wordprobe.json", "w"), indent=1)
    print(f"\nwrote {S}/massive_wordprobe.json")


if __name__ == "__main__":
    main()
