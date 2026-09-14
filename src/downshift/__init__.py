"""downshift — find the cheapest model and prompt that clears your accuracy bar."""

from pathlib import Path

# Load .env at import so `models.available()` sees the user's keys. Without this
# every hosted candidate silently drops out of the run and the chart quietly
# shrinks to local models only -- a failure that looks like a smaller experiment
# rather than a missing config.
try:
    from dotenv import load_dotenv

    for _candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        if _candidate.is_file():
            load_dotenv(_candidate, override=False)
            break
except ImportError:  # dotenv is optional; real env vars still work
    pass

__all__ = ["data", "models", "program", "metric", "evaluate", "optimize",
           "encoders", "stats", "chart", "experiment"]
