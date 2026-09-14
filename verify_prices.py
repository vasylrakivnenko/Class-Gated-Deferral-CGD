"""Audit every Azure-metered price in the registry against the live retail API.

    python verify_prices.py

`models.py` claimed a `verify_prices_live` helper for this. It never existed, so
no price in the registry had ever been re-checked against the catalogue -- and
`deepseek-v4-flash` had drifted to a figure that matches no meter at any tier or
scope, on a row that sits on the published Pareto frontier.

The check is deliberately weak but has almost no false positives: for each model
we ask whether our stored (input, output) pair appears ANYWHERE in the Foundry
catalogue. A price that exists as a real meter might still be the wrong tier for
us; a price that exists as no meter at all is simply wrong. The first needs
judgement, the second needs no argument, and it is the second this catches.

Fuzzy name matching is reported alongside but never used to decide, because
meter names are abbreviated beyond reliable matching ("56luna ShCo Opt Fl Gl",
"V4 Flash Inp glbl", "GPT 5 Mini cchd Inpt Glbl"). A human reads the candidates.

Tier and scope qualifiers are the whole game. The same model carries Flex,
Standard, Priority and Batch meters spanning up to 2x, and Global / DataZone /
Regional differ ~10% on top. Anything printed here is per 1M tokens.
"""
import json
import re
import subprocess
import sys

sys.path.insert(0, "src")

from downshift.models import HOSTED_MODELS

API = "https://prices.azure.com/api/retail/prices"
FILTER = ("serviceName eq 'Foundry Models' and priceType eq 'Consumption' "
          "and armRegionName eq 'eastus'")


def fetch() -> list[dict]:
    """Paginate the retail API via curl.

    curl rather than urllib: this machine sits behind a TLS-intercepting proxy
    whose CA is in the system store but not in certifi, so urllib fails the
    handshake while curl succeeds.
    """
    items, skip = [], 0
    while True:
        out = subprocess.run(
            ["curl", "-sS", "--max-time", "90", "--get", API,
             "--data-urlencode", f"$filter={FILTER}",
             "--data-urlencode", "$top=1000",
             "--data-urlencode", f"$skip={skip}"],
            capture_output=True, text=True, check=True).stdout
        page = json.loads(out)
        items += page["Items"]
        if not page.get("NextPageLink"):
            return items
        skip += 1000


def per_1m(item: dict) -> float | None:
    """Normalise to $/1M. Meters mix per-1K, per-1M and per-hour units."""
    unit = item["unitOfMeasure"]
    if "1K" in unit:
        return item["retailPrice"] * 1000
    if "1M" in unit:
        return item["retailPrice"]
    return None


def main() -> int:
    items = fetch()
    prices = {round(p, 6) for p in (per_1m(i) for i in items) if p is not None}
    names = [(i["meterName"], per_1m(i)) for i in items if per_1m(i) is not None]
    print(f"{len(items)} meters, {len(prices)} distinct rates\n")

    bad = []
    for spec in HOSTED_MODELS:
        # Only Azure-metered rows. Fireworks and Anthropic-direct rates are not
        # in this catalogue and cannot be checked here -- saying so is the point.
        if "prices.azure.com" not in spec.source:
            continue
        checks = [("in", spec.price_in), ("out", spec.price_out)]
        if spec.price_cached >= 0:
            checks.append(("cached", spec.price_cached))
        missing = [f"{lbl} ${val}" for lbl, val in checks
                   if round(val, 6) not in prices]
        if not missing:
            continue
        bad.append(spec.key)
        tokens = [t for t in re.split(r"[-_. ]", spec.key.lower()) if len(t) > 2]
        near = sorted({f"{m}  ${v:.4f}" for m, v in names
                       if sum(t in m.lower() for t in tokens) >= 2})[:8]
        print(f"[DRIFT] {spec.key}  ({spec.label})")
        print(f"        stored {spec.price_in}/{spec.price_out}"
              f"{'/' + str(spec.price_cached) if spec.price_cached >= 0 else ''} "
              f"-- no meter at: {', '.join(missing)}")
        print(f"        source: {spec.source[:110]}")
        for n in near:
            print(f"          candidate: {n}")
        print()

    checked = sum(1 for s in HOSTED_MODELS if "prices.azure.com" in s.source)
    skipped = [s.key for s in HOSTED_MODELS if "prices.azure.com" not in s.source]
    print(f"checked {checked} Azure-metered models, {len(bad)} with no matching meter")
    if skipped:
        print(f"NOT checkable here (not in this catalogue): {', '.join(skipped)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
