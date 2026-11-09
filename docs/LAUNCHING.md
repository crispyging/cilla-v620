# Launching Cilla

Run these commands from the root of a checkout of this fork, after building `build-v620/maya-ds4`, installing the Python dependencies and preparing a compatible model pack. See the [build/pack instructions](../README.md#build).

**These are portable examples, not private deployment scripts.** No `start-mix256.sh`, machine-specific SDK tree, model pack, credentials or prebuilt engine is included in this repository. Replace every `/path/to/...` value before starting. The published source also contains host-allocation safety fixes relative to the original deployed binary; these commands do not claim that the public source has received a fresh full-model HIP qualification.

## HTTP server: the normal public launch command

For the conservative starting config:

```bash
cp -n examples/cilla.local.example.json cilla.local.json
```

Edit `cilla.local.json`: set your actual absolute executable, repository working-directory, engine-log, model-pack and tokenizer paths. Choose the correct GPU list in **both** `args` and `gpu`. The sample's one-GPU index is not a promise that this model fits one V620.

Set your ROCm runtime paths (change `/opt/rocm` if your SDK is elsewhere), then launch:

```bash
export ROCM_PATH=/opt/rocm
export HIP_PATH="$ROCM_PATH"
export HIP_PLATFORM=amd
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.venv/bin/python serve/server.py --engine strata --config cilla.local.json --host 127.0.0.1 --port 8099
```

This starts the HTTP wrapper, which launches the engine with `--serve` itself. After the current engine boot reports ready, the web UI is at `http://127.0.0.1:8099/` and the OpenAI-compatible base URL is `http://127.0.0.1:8099/v1`. `/v1/models` alone is **not** an engine-readiness gate; do not send requests while the model is still loading. Run in the foreground initially; Ctrl+C stops the wrapper.

Keep loopback binding. The inherited UI includes management routes; API-key authentication alone does not protect the entire service. Use an authenticated tunnel/proxy for remote access, not a direct internet port. Local runtime configs and keys must stay out of Git.

## Four-V620 reference profile: opt-in, high host-RAM use

The original deployment used four cards, a compatible 40-layer DeepSeek-V4.1 mixed-quant pack with DSpark and vision, 524288 configured context, four conversation slots, a 4096-token prefill chunk, pinned Engram and copied CPU experts. Its runtime knobs and engine arguments are reproduced with generic paths in [`examples/cilla.four-v620.example.json`](../examples/cilla.four-v620.example.json).

**This is not the recommended first-boot default.** Pinned Engram plus `--mmap-experts off` can consume hundreds of GB of physical host RAM; admission must include NUMA placement, memlock limits, other processes and startup transients. DSpark and vision must actually be present and compatible in the selected pack. A configured 512K context does not establish every 512K workload is reliable. `DS4_YOCO=2` uses bounded rather than exact replay. See [the complete limitations](CILLA.md).

To select that profile, copy it to a separate ignored local file:

```bash
cp -n examples/cilla.four-v620.example.json cilla-four.local.json
```

Edit all `/path/to/...` entries in `cilla-four.local.json`. With the same ROCm exports above, the HTTP launch command is:

```bash
.venv/bin/python serve/server.py --engine strata --config cilla-four.local.json --host 127.0.0.1 --port 8099
```

The `env` mapping applies the engine tuning values; `STRATA_FAIR_SLICE_S=2` is also read by the HTTP server. Do not start both examples on the same port or against the same GPUs simultaneously. Do not alter an existing running deployment just to try this example.

### Underlying engine command — not an HTTP server

For protocol debugging only, the equivalent engine argument list is below. **This assumes the reference profile's `env` values have already been exported to the shell**; listing environment assignments in a document does not export them. Prefer the HTTP/config recipe above, which applies them for you.

```bash
./build-v620/maya-ds4 --serve \
  --pack /path/to/model-pack \
  --gpus 0,1,2,3 \
  --max-context 524288 \
  --reserve-mib 256 \
  --prefill-chunk 4096 \
  --vision on \
  --mmap-experts off \
  --dspark on
```

The engine's `--serve` speaks the internal stdin/stdout protocol, **not HTTP**. The HTTP wrapper additionally supplies `--layer-split auto`; the DS4 engine accepts that compatibility argument but uses its own layer-assignment logic.

## Eight V620s or a different model

There is **no tested eight-card preset** here. For an eight-card experiment, use a separate local config, change the GPU list to `0,1,2,3,4,5,6,7` in both `args` and `gpu`, and remove the four-card `DS4_SPLIT` entry. That is only a starting configuration, not a performance or correctness guarantee. Recalculate RAM/VRAM reservations, conversation/context settings and sidecar costs for the actual pack and topology.

GPU selection is not a parallelism mode. This source's `Options::tp` defaults to 2, and expert-format constraints may narrow it to 1. Inspect the startup messages and actual group/layer assignments; listing eight GPUs does not establish TP8 or even eight independent layer groups. Do not add `--tp 1` blindly: it also affects prompt-path selection.

The DS4 packer accepts `deepseek41`, not arbitrary GGUF architectures. Another V4.1 quant/fine-tune still needs tensor-format, tokenizer, memory and sidecar checks. Other architectures are not drop-in models for `maya-ds4`.
