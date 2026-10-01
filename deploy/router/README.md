# router.zadum.ai

How the router playground (`legalbench_map/ask_ui.py`) is served on the Hetzner box.

```
browser ─https─▶ Caddy :443 ─▶ oauth2-proxy 127.0.0.1:4180 ─▶ ask_ui.py 127.0.0.1:8765 ─▶ Jev (api.typesafe.ai)
                 (TLS)         (Google sign-in)                (limits, /admin)          └─▶ Kev 127.0.0.1:8008
```

| File | Installed at |
|---|---|
| `Caddyfile` | `/etc/caddy/Caddyfile` |
| `oauth2-proxy.service` | `/etc/systemd/system/` (binary: `/usr/local/bin/oauth2-proxy`) |
| `zadum-router.service` | `/etc/systemd/system/` |
| `kev.service` | `/etc/systemd/system/` (Kev checkout: `/root/projects/kev`, `uv sync --extra serve`) |

Secrets are not in this repo:

- `/etc/oauth2-proxy/env` (mode 600): `OAUTH2_PROXY_CLIENT_ID`, `OAUTH2_PROXY_CLIENT_SECRET` (Google OAuth web client, redirect URI `https://router.zadum.ai/oauth2/callback`) and `OAUTH2_PROXY_COOKIE_SECRET` (`openssl rand -base64 32 | tr -- '+/' '-_'`).
- `.env` at the repo root (mode 600): `JEV_API=...`, `FIREWORKS_API_KEY=...` (Tier 2's gpt-oss-120b),
  `OPENROUTER_API_KEY=...` (Gemma on OpenRouter) and `ZADUM_GPU_TOKEN=...` (Gemma on our own GPU). A Tier 2
  reader whose key is missing can't be picked in /admin, and a switch to one that fails its test read is refused.

Tier 0 needs the `encoders` and `tier0` extras in the repo's `.venv`; on first start it downloads `cross-encoder/nli-deberta-v3-xsmall` (~280 MB) to the Hugging Face cache.

The classifier bank (`legalbench_map/router/bank/`) is gitignored; rebuild it with `legalbench_map/build_router_bank.py` or copy it from a machine that has it. Load it with the scikit-learn version that fit it.

Users, limits and the request log are in `/var/lib/zadum-router/usage.db`. Admins (`--admin` in `zadum-router.service`) see them at `/admin`.

Which tiers answer (Pre-Tier 0, Tier 0, Tier 1 = Jev, `router/stages.py`) and which reader is Tier 2
(`router/tier2.py`) are set in /admin's **Routing Pipeline** tab, not on the playground page, and are stored in the
same `usage.db`, so they survive a restart. The `--classifiers` and `--tier2` flags in `zadum-router.service` are only
the startup default, used until an admin saves something; after that, changing a flag has no effect until the saved
row is removed:

```
sqlite3 /var/lib/zadum-router/usage.db "DELETE FROM settings WHERE key IN ('stages', 'tier2');"   # back to the flags
sqlite3 /var/lib/zadum-router/usage.db "SELECT key, value, updated_at, updated_by FROM settings;"
```

Each request logs which tiers were on, as `requests.stages` (`p1 t1 j1 c0 r:fireworks-priority`), so a day's
accuracy, latency and free-classifier share can be read against the configuration that produced them.
