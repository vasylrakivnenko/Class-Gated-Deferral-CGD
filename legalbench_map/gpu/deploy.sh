#!/bin/bash
# From the router server: set up Gemma 4 26B-A4B on a GPU pod and point the router at it (gpu/SETUP.md).
#     gpu/deploy.sh HOST PORT [all|start]
# HOST and PORT: the pod's "SSH over exposed TCP" address (Runpod: Connect tab). `all` (default) builds,
# downloads, starts and benchmarks; `start` only restarts llama-server and the gateway on a pod that still
# has them (a pod with a /workspace volume, after a stop and start). Then: switch on /admin.
set -euo pipefail
HOST=${1:?usage: deploy.sh HOST PORT [all|start]}
PORT=${2:?usage: deploy.sh HOST PORT [all|start]}
MODE=${3:-all}
HERE=$(cd "$(dirname "$0")" && pwd)
KEY=/root/.ssh/runpod_ed25519
ENV=/root/.env
UNIT=/etc/systemd/system/zadum-gpu-tunnel.service
SSH=(ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=15
     -o ServerAliveInterval=15 -p "$PORT" "root@$HOST")
step() { echo; echo "== $(date -u +%T) $*"; }

step "SSH to root@$HOST:$PORT"
ssh-keygen -R "[$HOST]:$PORT" >/dev/null 2>&1 || true  # a new pod on a reused address has a new host key
"${SSH[@]}" 'echo "connected: $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader)"' < /dev/null

step "gateway token (ZADUM_GPU_TOKEN in $ENV, to /workspace/gateway.token)"
grep -q '^ZADUM_GPU_TOKEN=' $ENV ||
  printf 'ZADUM_GPU_TOKEN=%s\n' "$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" >> $ENV
grep '^ZADUM_GPU_TOKEN=' $ENV | cut -d= -f2- | tr -d '\n' |
  "${SSH[@]}" 'mkdir -p /workspace && umask 077 && cat > /workspace/gateway.token'
echo "written"

step "copy setup files"
scp -q -i $KEY -o IdentitiesOnly=yes -P "$PORT" "$HERE"/{setup_pod.sh,gpu_gateway.py,prefill_bench.py,bench_prompts.jsonl} \
  "root@$HOST:/workspace/"
echo "copied"

step "setup on the pod ($MODE); it runs detached there, so a dropped SSH doesn't stop it (log: /workspace/setup.log)"
"${SSH[@]}" "setsid nohup bash /workspace/setup_pod.sh $MODE > /workspace/setup.log 2>&1 < /dev/null &" < /dev/null
shown=0
while :; do
  sleep 10
  # only complete lines, from the first one not shown yet
  out=$("${SSH[@]}" "n=\$(wc -l < /workspace/setup.log); head -n \$n /workspace/setup.log | tail -n +$((shown + 1))" < /dev/null) || continue
  [ -z "$out" ] && continue
  printf '%s\n' "$out"
  shown=$((shown + $(printf '%s\n' "$out" | wc -l)))
  grep -q '^SETUP_FAILED' <<< "$out" && { echo "setup failed on the pod: see above"; exit 1; }
  grep -q '^SETUP_DONE' <<< "$out" && break
done

step "tunnel: 127.0.0.1:18000 here -> the pod's 127.0.0.1:8000 ($UNIT)"
sed "s/@HOST@/$HOST/g; s/@PORT@/$PORT/g" "$HERE/zadum-gpu-tunnel.service" > $UNIT
systemctl daemon-reload
systemctl enable -q zadum-gpu-tunnel
systemctl restart zadum-gpu-tunnel
sleep 3
systemctl is-active zadum-gpu-tunnel

step "check from the router: health, a test read, 96 real reads through the tunnel"
cd "$HERE/.." && ../.venv/bin/python gpu/check_gpu.py --bench 96
echo
echo "Done. To use it: /admin#tier2 (the Tier 2 reader tab) -> 'Gemma 4 26B-A4B (our GPU)' -> Switch to this."
