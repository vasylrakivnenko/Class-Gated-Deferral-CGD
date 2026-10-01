#!/bin/bash
# Sets up Gemma 4 26B-A4B behind our zadum-gpu/1 gateway on a GPU pod (gpu/SETUP.md).
# gpu/deploy.sh copies this and its files to /workspace and runs it; by hand, on the pod:
#     bash /workspace/setup_pod.sh [all|start|bench]
#   all    tools → llama.cpp build + model download (in parallel) → llama-server → gateway → benchmark
#   start  llama-server → gateway (after a pod restart when the build and the model are still there)
#   bench  the benchmark only
# Re-running is safe: a build at the pinned commit and a model with the right checksum are kept.
set -Eeuo pipefail  # -E: the ERR trap fires inside functions too
trap 'echo "SETUP_FAILED at line $LINENO"' ERR

ROOT=/workspace
MODE=${1:-all}
LLAMA_COMMIT=f7b384c1e5c5b2c5b321a4a7cefea04b15b54cb7  # 2026-09-30; the build every number in SETUP.md was measured on
MODEL_URL=https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/resolve/main/gemma-4-26B_q4_0-it.gguf
MODEL_SIZE=14439363584
MODEL_SHA256=3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d
MODEL=$ROOT/models/gemma-4-26B_q4_0-it.gguf
MODEL_NAME=gemma-4-26b-a4b-it-q4_0  # what the router logs
LLAMA=$ROOT/llama.cpp/build/bin
# Why each flag: SETUP.md, "The llama-server flags". --cache-ram 0 is the one that matters most: llama.cpp's
# host-RAM prompt cache copies 100-400 MB per request and stalls every other request (1.7k -> 10.7k tok/s).
SERVER_FLAGS=(-ngl 99 -fa on -c 32768 -np 16 -b 4096 -ub 2048 --kv-unified --cache-ram 0 --ctx-checkpoints 0
              --jinja --no-ui --host 127.0.0.1 --port 8090)
export PATH=/usr/local/cuda/bin:$PATH  # Runpod's images keep nvcc off PATH in non-interactive shells

step() { echo; echo "== $(date -u +%T) $*"; }

tools() {
  step "GPU and tools"
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
  local missing=""
  for t in cmake git g++ wget curl python3; do command -v $t >/dev/null || missing="$missing $t"; done
  if [ -n "$missing" ]; then
    echo "installing:$missing"
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake git build-essential wget curl python3 >/dev/null
  fi
  command -v nvcc >/dev/null || { echo "no CUDA compiler (nvcc): pick a CUDA *devel* image (SETUP.md, step 1)"; false; }
  nvcc --version | tail -1
  local have need free  # GB: ~2 for the build, plus what's left of the model's 14.4
  have=$(( $(stat -c %s $MODEL 2>/dev/null || echo 0) / 1000000000 ))
  need=$(( 2 + 15 - have ))
  free=$(df --output=avail -BG $ROOT | tail -1 | tr -dc 0-9)
  echo "disk: ${free} GB free in $ROOT, ~${need} GB needed"
  [ "$free" -ge "$need" ] || { echo "not enough disk in $ROOT"; false; }
}

download() {
  mkdir -p $ROOT/models
  if [ -f $MODEL.verified ] && [ "$(stat -c %s $MODEL 2>/dev/null)" = $MODEL_SIZE ]; then
    echo "model: kept ($MODEL, checksum verified before)"; return
  fi
  local t0=$SECONDS
  for i in $(seq 20); do  # wget stalled once mid-file: a read timeout and -c resume it
    [ "$(stat -c %s $MODEL 2>/dev/null)" = $MODEL_SIZE ] && break
    wget -q -c --read-timeout=30 --tries=3 -O $MODEL "$MODEL_URL" || sleep 5
  done
  [ "$(stat -c %s $MODEL 2>/dev/null)" = $MODEL_SIZE ] || { echo "model: download incomplete"; return 1; }
  echo "model: downloaded in $((SECONDS - t0)) s; checking sha256"
  echo "$MODEL_SHA256  $MODEL" | sha256sum -c --quiet || { echo "model: checksum mismatch, deleted"; rm -f $MODEL; return 1; }
  touch $MODEL.verified
  echo "model: ok, $(du -h $MODEL | cut -f1), sha256 matches, in $((SECONDS - t0)) s"
}

build() {
  if [ "$(git -C $ROOT/llama.cpp rev-parse HEAD 2>/dev/null)" = $LLAMA_COMMIT ] && [ -x $LLAMA/llama-server ]; then
    echo "llama.cpp: kept (${LLAMA_COMMIT:0:7} already built)"; return
  fi
  local t0=$SECONDS arch
  arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)  # 4090: 89
  rm -rf $ROOT/llama.cpp && git init -q $ROOT/llama.cpp && cd $ROOT/llama.cpp
  git remote add origin https://github.com/ggml-org/llama.cpp
  git fetch -q --depth 1 origin $LLAMA_COMMIT && git checkout -q FETCH_HEAD
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$arch -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
    > cmake.log 2>&1 || { tail -20 cmake.log; return 1; }
  cmake --build build -j "$(nproc)" --target llama-server llama-bench > make.log 2>&1 || { tail -30 make.log; return 1; }
  echo "llama.cpp: ${LLAMA_COMMIT:0:7} built for sm_$arch in $((SECONDS - t0)) s"
}

start() {
  step "llama-server"
  [ -x $LLAMA/llama-server ] && [ -f $MODEL.verified ] || { echo "no build or no verified model: run with 'all'"; false; }
  pkill -x llama-server || true  # by process name: -f would also match this script's own shell
  sleep 2
  cd $LLAMA && setsid nohup ./llama-server -m $MODEL "${SERVER_FLAGS[@]}" > $ROOT/server.log 2>&1 < /dev/null &
  for i in $(seq 180); do
    curl -sf 127.0.0.1:8090/health >/dev/null && break
    pgrep -x llama-server >/dev/null || { tail -20 $ROOT/server.log; echo "llama-server exited"; false; }
    sleep 1
  done
  curl -sf 127.0.0.1:8090/health >/dev/null || { tail -20 $ROOT/server.log; false; }
  echo "up after ${i}s; $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader) VRAM"
  grep -m1 -o "kv_unified = '[a-z]*'" $ROOT/server.log || true

  step "gateway (zadum-gpu/1 on 127.0.0.1:8000)"
  [ "$(wc -c < $ROOT/gateway.token)" -ge 16 ] || { echo "no $ROOT/gateway.token (deploy.sh writes it)"; false; }
  chmod 600 $ROOT/gateway.token
  pkill -f "^python3 $ROOT/gpu_gateway.py" || true
  sleep 1
  ZADUM_GPU_TOKEN=$(cat $ROOT/gateway.token) setsid nohup python3 $ROOT/gpu_gateway.py --engine llama.cpp \
    --upstream http://127.0.0.1:8090 --model $MODEL_NAME --port 8000 > $ROOT/gateway.log 2>&1 < /dev/null &
  sleep 2
  curl -sf -H "Authorization: Bearer $(cat $ROOT/gateway.token)" 127.0.0.1:8000/v1/health || { cat $ROOT/gateway.log; false; }
  echo
}

bench() {
  step "benchmark: prefill only (RTX 4090, 2026-10-01: 7.3k tok/s at 1, 10.7k at 16 parallel)"
  python3 $ROOT/prefill_bench.py $ROOT/bench_prompts.jsonl --conc 1,16
  step "benchmark: full reader answers (RTX 4090: 0.26 s p50, 0.32 s p90 one at a time)"
  python3 $ROOT/prefill_bench.py $ROOT/bench_prompts.jsonl --answer --conc 1 --n 96
}

case $MODE in
  all)
    tools
    step "llama.cpp build and model download, in parallel"
    download > $ROOT/download.log 2>&1 &
    dl=$!
    build
    wait $dl || { cat $ROOT/download.log; false; }
    cat $ROOT/download.log
    start
    bench ;;
  start) start ;;
  bench) bench ;;
  *) echo "usage: setup_pod.sh [all|start|bench]"; false ;;
esac
echo "SETUP_DONE"
