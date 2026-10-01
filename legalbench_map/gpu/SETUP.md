# Our own GPU: Gemma 4 26B-A4B from scratch

How to bring up the Tier 2 reader "Gemma 4 26B-A4B (our GPU)" on a new GPU pod and connect it to the router.
You do step 1 in the Runpod console (about 2 minutes). Step 2 is one command on the router server, and step 3
is one click on /admin. Last measured end to end on 2026-10-01: `deploy.sh` took **12 min 48 s** with the model already on disk. The llama.cpp build is 10.5 min of that on the pod's 12 CPUs. On a fresh pod, the 14.4 GB download runs during the build and takes 10–15 min, so plan on **about 15–18 minutes**.

```
router.zadum.ai (this server)                                   GPU pod (Runpod, RTX 4090)
ask_ui.py  Tier 2 option "gpu-gemma"                            /workspace/gpu_gateway.py   127.0.0.1:8000
  reader.OwnGPULLM -> gpu.OwnGPU (zadum-gpu/1 client)             speaks zadum-gpu/1, adapter "llama.cpp"
  -> http://127.0.0.1:18000                                       -> llama-server              127.0.0.1:8090
  -> zadum-gpu-tunnel.service (SSH -L 18000 -> pod :8000) ====>      -> Gemma 4 26B-A4B Q4_0 on the GPU
```

Nothing on the pod is reachable from the internet except SSH. The gateway also needs a bearer token,
`ZADUM_GPU_TOKEN`, stored in `/root/.env` here and `/workspace/gateway.token` on the pod.

## 1. Create the pod (Runpod console)

| Setting | Value | Why |
|---|---|---|
| GPU | 1× **RTX 4090, 24 GB** | Measured on this card. Any NVIDIA GPU with 24 GB+ works; the build detects the architecture. With less memory, lower `-c` or `-ub` (section 5). |
| Template | **Runpod PyTorch 2.8** (Ubuntu 24.04, CUDA 12.8.1) | Needs a CUDA *devel* image, which has `nvcc` in `/usr/local/cuda/bin`, to build llama.cpp. PyTorch itself isn't used. |
| Container disk | **30 GB** | The model is 14.4 GB, the build 0.5 GB, and the image takes the rest. |
| Volume (recommended) | **40 GB, mounted at `/workspace`** | Keeps the model and the build when the pod is stopped. A restart then takes under a minute (`deploy.sh ... start`, 50 s measured) instead of a full setup. Without a volume, Stop wipes everything (the pod of 2026-10-01 had none). |
| Ports | **TCP 22** exposed, public IP | SSH is the only way in; the router tunnels through it. |
| SSH key | Runpod **Settings → SSH Public Keys**: the line below | The template's start script installs it (`PUBLIC_KEY`). |

The router server's key (`/root/.ssh/runpod_ed25519.pub`):

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKo8PEgepofXyM5Su80WjvfvTXeXfEE01kq4d4JSriXe vasyl@alumni.gsb.stanford.edu
```

If the key wasn't in Settings when the pod started, open the pod's web terminal and run:
`mkdir -p ~/.ssh && echo '<the line above>' >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys`

Once it's running: **Connect → "SSH over exposed TCP"** shows `ssh root@<IP> -p <PORT>`. Those two values are
all step 2 needs. The port changes every time the pod restarts.

## 2. Deploy (one command, on the router server)

```
cd /root/projects/zadumai/legalbench_map && gpu/deploy.sh <IP> <PORT>
```

What it does, with what you should see:

| Step | Where | What | Expected |
|---|---|---|---|
| SSH | here | Forgets any old host key for that address (a new pod on a reused address has a new one), then connects. | `connected: <host>, NVIDIA GeForce RTX 4090` |
| Token | here → pod | Creates `ZADUM_GPU_TOKEN` in `/root/.env` if missing, then writes it to `/workspace/gateway.token` (mode 600) through SSH stdin, never on a command line. | `written` |
| Copy | here → pod | `setup_pod.sh`, `gpu_gateway.py`, `prefill_bench.py`, `bench_prompts.jsonl` to `/workspace/`. | `copied` |
| Tools | pod | GPU, `nvcc`, cmake, git, g++, disk space. Installs whatever is missing with apt. Fails if there's no `nvcc` (wrong image). | `Build cuda_12.8...`, `disk: N GB free` |
| Build ‖ download | pod | Both run at once. **llama.cpp** at the pinned commit `f7b384c` (2026-09-30), with CUDA, for the GPU's architecture: llama-server and llama-bench. **The model**, `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` / `gemma-4-26B_q4_0-it.gguf` (public, no token), 14,439,363,584 bytes. wget resumes after stalls, and the sha256 must be `3eca3b8f…eca51d`. | `llama.cpp: f7b384c built for sm_89 in 627 s`; `model: ok, 14G, sha256 matches` (download 10–15 min at the pod's 17–24 MB/s; 8 parallel connections were no faster; the sha256 check takes 23 s) |
| llama-server | pod | Starts with the flags in section 5; waits for `/health`. | `up after ~10s`, 19–22 GB VRAM, `kv_unified = 'true'` |
| Gateway | pod | `gpu_gateway.py --engine llama.cpp` on 127.0.0.1:8000. | `{"api": "zadum-gpu/1", "ok": true, ...}` |
| Benchmark | pod | `prefill_bench.py` with 286 real reader prompts (~495 tokens each). | prefill **~7.3k tok/s at 1, ~10.7k at 16 parallel**; answers **~0.26 s p50** one at a time |
| Tunnel | here | Writes `/etc/systemd/system/zadum-gpu-tunnel.service` from `gpu/zadum-gpu-tunnel.service` with the pod's address, then restarts it. | `active` |
| Check | here | `gpu/check_gpu.py --bench 96`: health, a test read, 96 real reads through the tunnel. | **~290 ms p50, ~360 p90, ~630 p99**, 0 not JSON. Right after a fresh start the tail is slower (p99 ~1.1 s on the first 96); re-run `check_gpu.py --bench 286` and it settles. |

The setup runs detached on the pod, so a dropped SSH connection doesn't stop it. `deploy.sh` follows
`/workspace/setup.log` and stops at `SETUP_DONE` or `SETUP_FAILED`. Re-running is safe: a build at the pinned
commit and a model with the right checksum are kept.

## 3. Switch the router to it

/admin → the **Tier 2 reader** tab (`/admin#tier2`) → "Gemma 4 26B-A4B (our GPU)" → **Switch to this**. It sends one test read first and
won't switch if that read fails. The card shows whether the GPU is up, and its last 7 days of reads.

## 4. Later: restarts, a new pod, problems

| Situation | Do |
|---|---|
| Pod stopped and started, **with** a volume | `gpu/deploy.sh <IP> <NEW PORT> start` (restarts llama-server and the gateway, then points the tunnel at the new port and checks; 50 s measured) |
| Pod stopped and started **without** a volume, or a brand-new pod | `gpu/deploy.sh <IP> <PORT>` (the full setup) |
| Is it working? | `cd legalbench_map && ../.venv/bin/python gpu/check_gpu.py [--bench 96]` |
| Tunnel down (the card says "can't reach the GPU gateway") | `systemctl status zadum-gpu-tunnel`; `journalctl -u zadum-gpu-tunnel -n 20`; usually the port changed, so re-run `deploy.sh ... start` |
| Pod gone or stopped while it's the live reader | Questions that need Tier 2 defer ("the reader was unavailable"); nothing fails. Switch to a Fireworks reader on /admin. |
| On the pod, by hand | `bash /workspace/setup_pod.sh start` (restart both), `... bench` (benchmark); logs in `/workspace/{setup,server,gateway,download}.log` |

**Cost:** the pod bills by the hour while it runs, idle or not. **Stop** pauses billing (a volume still costs a
little per GB). **Terminate** ends it, volume included.

## 5. The llama-server flags (why it's fast)

```
llama-server -m gemma-4-26B_q4_0-it.gguf -ngl 99 -fa on -c 32768 -np 16 -b 4096 -ub 2048 \
  --kv-unified --cache-ram 0 --ctx-checkpoints 0 --jinja --no-ui --host 127.0.0.1 --port 8090
```

| Flag | What it does | Measured / why |
|---|---|---|
| **`--cache-ram 0`** | Turns off llama.cpp's host-RAM prompt cache (on by default, 8 GiB). | **This is the one that matters.** With the cache on, every new request copied the slot's KV (100–400 MiB) into RAM and stalled the server loop for 130–190 ms, so it never batched: ~1.7k prompt tok/s at 1, 8 and 16 parallel alike. Off: 7.3k / 9.9k / 10.7k. The reader never reuses a prompt, so the cache only costs time. |
| `--ctx-checkpoints 0` | No context checkpoints. | Gemma uses sliding-window attention, so llama.cpp saves a checkpoint of the window per request for reuse that we never do. |
| `--kv-unified` | One KV pool shared by all slots. | Lets 16 prompts of different lengths share 32k tokens. Alone it didn't change speed (that was the cache). |
| `-c 32768 -np 16` | 16 parallel requests, 32k tokens of KV in all. | Reader prompts are ~500 tokens (at most ~1.5k) plus a ~25-token answer. |
| `-b 4096 -ub 2048` | Logical / physical batch size. | ub 2048 gave the best raw prefill in llama-bench: 10–12.9k tok/s on 512–8192-token prompts. |
| `-fa on`, `-ngl 99` | Flash attention; every layer on the GPU. | Standard for speed. |
| `--jinja` | The model's own chat template. | The gateway sends `enable_thinking: false`, and per request also `cache_prompt: false`, temperature 0 and the reader's JSON schema. |
| `--host 127.0.0.1` | Only the gateway on the same machine reaches it. | The gateway holds the token check. |

Measured on the RTX 4090, 2026-10-01:

| Test | Result |
|---|---|
| llama-bench, raw prefill (`-p 512,2048,8192 -ub 2048`) | 10–12.9k tok/s |
| Server prefill (286 reader prompts), 1 / 8 / 16 parallel | 7.3k / 9.9k / 10.7k tok/s |
| Full answer on the pod, 1 / 8 / 16 parallel | 0.26 / 0.89 / 1.46 s p50 |
| Full answer from the router (tunnel + gateway), one at a time | 0.29 s p50, 0.36 p90, 0.63 p99 |
| Answers vs Gemma on OpenRouter/NextBit (bake-off, 160 questions) | the same on 151; most of the other 9 differ by a "the" |

To run llama-bench by hand, first stop llama-server (`pkill -x llama-server`), since both need the VRAM:
`/workspace/llama.cpp/build/bin/llama-bench -m /workspace/models/gemma-4-26B_q4_0-it.gguf -ngl 99 -fa 1 -n 0 -p 512,2048,8192 -ub 2048 -r 3`
Then run `bash /workspace/setup_pod.sh start`.

## 6. Another engine or another model

The router only knows **zadum-gpu/1** (spec: the `router/gpu.py` docstring): `GET /v1/health` and
`POST /v1/generate` (messages, max_tokens, temperature, json_schema, thinking → text, usage, timing),
with a bearer token. So:

- **Another engine** (vLLM, SGLang, TGI…): if it serves OpenAI's `/v1/chat/completions` with JSON-schema
  `response_format`, start it on 127.0.0.1:8090 and run the gateway with `--engine openai --model <served name>`.
  Otherwise, add an `Engine` subclass to `gpu/gpu_gateway.py` with `health()` and `generate()`, about 20 lines.
  Either way, change `setup_pod.sh` (how it installs and starts the engine). Nothing changes on the router side.
- **Another model**: change the download (URL, size, sha256) and `MODEL_NAME` in `setup_pod.sh`. Then
  re-measure before using it live: `check_gpu.py --bench 286` for speed, and the bake-off for quality
  (`/root/zadumai_nli_proto/qtree/bakeoff.py`, `verify.py`). The Jev threshold is per model (`check_min` of
  the option in `router/tier2.py`: gpt-oss-120b 0.7, Gemma 0.8), and so is the option's name on /admin.
- Bump `LLAMA_COMMIT` only together with a benchmark: llama.cpp's defaults change often (the prompt cache that
  cost us 6× was a new default).

## 7. Problems seen so far

| Symptom | Cause | Fix |
|---|---|---|
| CMake: "No CMAKE_CUDA_COMPILER could be found" | `nvcc` isn't on PATH in non-interactive SSH on Runpod's images. | `setup_pod.sh` adds `/usr/local/cuda/bin`. If `nvcc` is missing altogether, the image isn't a CUDA devel image. |
| The download stops growing | wget stalled mid-file (happened once). | `setup_pod.sh` uses a 30 s read timeout and resumes with `-c`, up to 20 times, then checks the sha256. |
| Prefill stuck near 1.7k tok/s however many requests run at once | The host-RAM prompt cache. | `--cache-ram 0` (section 5). |
| "REMOTE HOST IDENTIFICATION HAS CHANGED" | A new pod on an address used before. | `deploy.sh` removes the old key; by hand: `ssh-keygen -R "[<IP>]:<PORT>"`. |
| The SSH session dies right after `pkill -f ...` | `pkill -f` also matches the shell whose command line contains the pattern. | Kill by name (`pkill -x llama-server`) or with an anchored pattern (`pkill -f "^python3 /workspace/gpu_gateway.py"`). |
| ~40–70 ms extra on some calls through the tunnel | Nagle / delayed ACK on the gateway's socket. | Fixed in the gateway: TCP_NODELAY, keep-alive, one write per reply. |
| `check_gpu.py`: HTTP 401 | The token on the pod and in `/root/.env` differ. | Re-run `deploy.sh` (it rewrites the pod's copy). |
| llama-server exits at start | Out of VRAM (another process?) or a wrong path. | `tail /workspace/server.log`; `nvidia-smi`. |

## Files

- Here, in `legalbench_map/gpu/`: `deploy.sh` (step 2), `setup_pod.sh` (runs on the pod), `gpu_gateway.py`
  (the gateway), `prefill_bench.py` + `bench_prompts.jsonl` (286 real reader prompts from the CUAD bake-off),
  `check_gpu.py` (router-side check), `zadum-gpu-tunnel.service` (the tunnel unit's template).
- On the router server: `/etc/systemd/system/zadum-gpu-tunnel.service`; `ZADUM_GPU_TOKEN` (and optionally
  `ZADUM_GPU_URL`, default `http://127.0.0.1:18000`) in `/root/.env`; SSH key `/root/.ssh/runpod_ed25519`.
- On the pod: `/workspace/{llama.cpp,models,gpu_gateway.py,gateway.token,setup_pod.sh,*.log}`.
