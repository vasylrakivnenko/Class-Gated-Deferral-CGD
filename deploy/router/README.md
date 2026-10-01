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
| `zadum-deploy.service` + `.timer` | `/etc/systemd/system/` (see Auto-deploy) |

## Auto-deploy

A push to `main` reaches the box within a minute. Turn it on once:

```
cd /root/projects/zadumai && git pull --ff-only && bash deploy/router/install-autodeploy.sh
```

`zadum-deploy.timer` runs `deploy.sh` every minute. It exits immediately when the checkout already
matches `origin/main`, so it is cheap. When something new is there it fast-forwards, syncs
dependencies if `pyproject.toml` changed, restarts `zadum-router`, and waits up to two minutes for
`http://127.0.0.1:8765/` to answer 200 (Tier 0 loads spaCy and the NLI model at startup, so a cold
restart is slow).

If the restarted service never answers, it **rolls the checkout back** to the commit it started from,
restarts again, and records the bad commit in `.deploy-failed` so the timer does not roll the service
forward and back every minute. Pushing anything new clears it.

```
systemctl list-timers zadum-deploy      # when it next runs
journalctl -u zadum-deploy -f           # what it did
systemctl start zadum-deploy            # deploy now, don't wait for the tick
systemctl disable --now zadum-deploy.timer
```

It refuses to run while `/root/projects/zadumai` has uncommitted changes to tracked files, so
debugging live on the box is never clobbered — commit or stash there before expecting a deploy.

`.github/workflows/deploy-router.yml` does the same thing instantly on push instead of polling; it
needs the repository secrets `ROUTER_SSH_KEY` and `ROUTER_HOST`. Both can run together.

Note: the `zadum-router.service` in this repo does not carry the `--tier2 priority` the live unit has.
Auto-deploy never touches unit files, only the checkout, so that difference is harmless — but do not
copy this one over the live one without re-adding the flag.

Secrets are not in this repo:

- `/etc/oauth2-proxy/env` (mode 600): `OAUTH2_PROXY_CLIENT_ID`, `OAUTH2_PROXY_CLIENT_SECRET` (Google OAuth web client, redirect URI `https://router.zadum.ai/oauth2/callback`) and `OAUTH2_PROXY_COOKIE_SECRET` (`openssl rand -base64 32 | tr -- '+/' '-_'`).
- `.env` at the repo root (mode 600): `JEV_API=...`, `FIREWORKS_API_KEY=...` (Tier 2's gpt-oss-120b) and,
  only if Tier 2 is set to Gemma, `OPENROUTER_API_KEY=...`. A Tier 2 option whose key is missing fails the
  save in /admin rather than every later question.

Tier 0 needs the `encoders` and `tier0` extras in the repo's `.venv`; on first start it downloads `cross-encoder/nli-deberta-v3-xsmall` (~280 MB) to the Hugging Face cache.

The classifier bank (`legalbench_map/router/bank/`) is gitignored; rebuild it with `legalbench_map/build_router_bank.py` or copy it from a machine that has it. Load it with the scikit-learn version that fit it.

Users, limits and the request log are in `/var/lib/zadum-router/usage.db`. Admins (`--admin` in `zadum-router.service`) see them at `/admin`.

Which tiers answer (Pre-Tier 0, Tier 0, Tier 1 = Jev, Tier 2) is set in /admin's **Routing Pipeline** tab, not on
the playground page, and is stored in the same `usage.db`, so it survives a restart. The `--classifiers` and
`--tier2` flags in `zadum-router.service` are only the startup default, used until an admin saves something; after
that, changing a flag has no effect until the saved row is removed:

```
sqlite3 /var/lib/zadum-router/usage.db "DELETE FROM settings WHERE key = 'stages';"   # back to the flags
sqlite3 /var/lib/zadum-router/usage.db "SELECT value, updated_at, updated_by FROM settings WHERE key = 'stages';"
```

Each request logs which tiers were on, as `requests.stages` (`p1 t1 j1 c0 r:gpt-oss-120b-priority`), so a day's
accuracy, latency and free-classifier share can be read against the configuration that produced them.
