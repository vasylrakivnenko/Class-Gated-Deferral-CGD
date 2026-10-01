#!/usr/bin/env bash
# Pulls main and restarts the router when GitHub has something new.
#
# Run by zadum-deploy.timer every minute, or by hand. Does nothing when the
# checkout already matches origin/main, so it is cheap to run often.
#
# If the restarted service does not answer, this rolls the checkout back to the
# commit it started from and restarts again: a bad commit costs one restart, not
# a dead router. Logs: journalctl -u zadum-deploy -f
set -euo pipefail

REPO=${REPO:-/root/projects/zadumai}
BRANCH=${BRANCH:-main}
SERVICE=${SERVICE:-zadum-router}
HEALTH=${HEALTH:-http://127.0.0.1:8765/}
HEALTH_TRIES=${HEALTH_TRIES:-60}   # x2s = 2 min; Tier 0 loads spaCy and the NLI model at startup
VENV="$REPO/.venv"
FAILED="$REPO/.deploy-failed"      # a commit that already failed its health check, so we stop retrying it

cd "$REPO"

# Never deploy over uncommitted work on the box: that is someone debugging live.
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "refusing to deploy: $REPO has uncommitted changes to tracked files" >&2
    git status --short >&2
    exit 1
fi

git fetch --quiet origin "$BRANCH"
was=$(git rev-parse HEAD)
want=$(git rev-parse "origin/$BRANCH")
[ "$was" = "$want" ] && exit 0

# A commit that already failed is not retried: without this the timer would roll
# the service forward and back every minute until someone pushed a fix. Pushing
# anything new clears it, because `want` changes.
if [ -f "$FAILED" ] && [ "$(cat "$FAILED")" = "$want" ]; then
    exit 0
fi

echo "deploying ${was:0:7} -> ${want:0:7}"
# --ff-only: if the checkout has drifted off main, stop rather than guess.
git merge --ff-only "origin/$BRANCH"

# Dependencies only when they changed; a failure here is not fatal on its own,
# the health check below decides.
if ! git diff --quiet "$was" "$want" -- pyproject.toml; then
    echo "pyproject.toml changed, syncing dependencies"
    "$VENV/bin/python" -m pip install -q -e ".[tier0,encoders]" || echo "dependency sync failed" >&2
fi

systemctl restart "$SERVICE"

for _ in $(seq "$HEALTH_TRIES"); do
    code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$HEALTH" || true)
    if [ "$code" = "200" ]; then
        rm -f "$FAILED"
        echo "deployed ${want:0:7}: $(git log -1 --format=%s)"
        exit 0
    fi
    sleep 2
done

echo "ROLLING BACK: $SERVICE did not answer $HEALTH after the restart (last code: ${code:-none})" >&2
echo "$want" > "$FAILED"
git reset --hard "$was"
systemctl restart "$SERVICE"
echo "rolled back to ${was:0:7}; ${want:0:7} will not be retried until something new is pushed" >&2
exit 1
