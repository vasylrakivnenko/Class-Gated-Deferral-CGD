"""Verify which hosted models are reachable before spending a long run on them.

Model identifiers go stale fast -- names get renamed, models get retired, a tier
disappears. Finding that out 20 minutes into an experiment, after the encoder has
already trained, is the expensive way. This makes one tiny call per configured
model and reports what actually resolves.

    python check_models.py
"""
import sys

sys.path.insert(0, "src")

import downshift  # noqa: F401  -- loads .env
from downshift.models import HOSTED_MODELS, LOCAL_MODELS, missing_keys


def probe(spec) -> tuple[bool, str]:
    import dspy
    from downshift.evaluate import build_lm
    try:
        lm = build_lm(spec, max_tokens=8, cache=False)
        out = lm(messages=[{"role": "user", "content": "Reply with the single word: ok"}])
        text = out[0] if isinstance(out, list) else out
        if isinstance(text, dict):
            text = text.get("text", "")
        usage = lm.history[-1].get("usage") or {}
        return True, f"{str(text)[:24]!r} (in={usage.get('prompt_tokens','?')} out={usage.get('completion_tokens','?')})"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:150]}"


def main() -> None:
    print("LOCAL (ollama):")
    for spec in LOCAL_MODELS:
        ok, detail = probe(spec)
        print(f"  [{'OK ' if ok else 'FAIL'}] {spec.call_id:<18} {detail}")

    print("\nHOSTED:")
    any_hosted = False
    for spec in HOSTED_MODELS:
        if not spec.available:
            continue
        any_hosted = True
        ok, detail = probe(spec)
        print(f"  [{'OK ' if ok else 'FAIL'}] {spec.call_id:<38} {detail}")
        if not ok:
            print(f"         ^ model id may be stale; fix `call_id` in src/downshift/models.py")
    if not any_hosted:
        print("  (none configured)")

    locked = missing_keys()
    if locked:
        print("\nSet these in .env to unlock more candidates:")
        for env_key, labels in locked.items():
            print(f"  {env_key:<22} -> {', '.join(labels)}")


if __name__ == "__main__":
    main()
