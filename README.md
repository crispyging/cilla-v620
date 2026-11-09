# Cilla — DeepSeek on Radeon PRO V620

An experimental V620-focused source fork of [Project Maya](https://github.com/mw00/project-maya), descended from Strata. **Named after Cilla, the boss in New World's Lazarus Instrumentality expedition.** Independent community software; not affiliated with Amazon Games, AMD, DeepSeek, or upstream maintainers. No game artwork is redistributed.

The focus is DeepSeek-V4.1-Flash-1017 on Linux/HIP (`gfx1030`): multi-GPU layer splitting, HIP MMQ prompt experts, conversation slots and prefix-checkpoint reuse. The executable keeps its upstream name, `maya-ds4`.

**Experimental source release, not a turnkey appliance.** The exported engine revision was used on a four-V620 system. That does not qualify every motherboard, ROCm version, pack, context length or optional execution path. No performance guarantee is made. Read [Cilla notes and limitations](docs/CILLA.md).

## Included

Engine, packer, HTTP server, tests, and upstream documentation; mixed-quant pack overrides, Engram support, and the pinned seed-checkpoint fix. The Cilla branch retains public Maya ancestry but excludes private deployment commits, machine configuration, logs, weights, and credentials. Upstream licenses and credits remain intact.

The [upstream README](UPSTREAM_README.md) describes broader Maya functionality and upstream measurements, not qualification of this fork. [Provenance](docs/CILLA.md#provenance) identifies the snapshot and dependency pin.

## Requirements

- Linux x86-64, an AMD kernel driver, `/dev/kfd` and render-node access.
- A coherent ROCm/HIP SDK with **gfx1030 rocBLAS libraries**. The original build used **ROCm 7.2.4**; other versions need verification.
- CMake >=3.24, C++20 compiler, Git, Ninja or Make. **Python 3.11 is the tested release environment**; other Python versions are not qualified by this release.
- Separately obtained, compatible model weights and enough host RAM/VRAM for the exact pack. **One 32-GiB V620 does not by itself hold this large model.** Pinned Engram plus copied CPU experts can consume hundreds of GB of host RAM.

Do not copy another machine's kernel, P2P, NUMA or swap changes blindly.

## Build

Start with a clean checkout of the Cilla branch:

```bash
git clone --branch release/cilla-v620 https://github.com/crispyging/cilla-v620.git cilla
cd cilla
```

Create the tested Python environment, then fetch **llama.cpp revision `3cf03257f219afbe7334045ff7c6a06ac68c627d`**, the exact source pin required by this engine and its MMQ patch. These commands download dependency source, not model weights, and assume `third_party/llama.cpp` does not already exist:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install numpy jinja2 regex pyyaml tqdm psutil 'jsonschema>=4.18,<5'
git init third_party/llama.cpp
git -C third_party/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git
git -C third_party/llama.cpp fetch --depth=1 origin 3cf03257f219afbe7334045ff7c6a06ac68c627d
git -C third_party/llama.cpp checkout --detach FETCH_HEAD
git -C third_party/llama.cpp rev-parse HEAD
```

The final command must print that exact revision. Use the matching ROCm development headers, HIP compiler/runtime, device libraries, hipBLAS and rocBLAS packages, including gfx1030 rocBLAS assets. Ensure your account can access `/dev/kfd` and the required `/dev/dri/renderD*` nodes (usually through the host's `render`/`video` groups). These instructions do not install drivers or change permissions for you.

Set `ROCM_PATH` to your actual SDK. Configure a **new** build directory:

```bash
export ROCM_PATH=/opt/rocm
export HIP_PATH="$ROCM_PATH"
export HIP_PLATFORM=amd
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cmake -S . -B build-v620 \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_NATIVE_EXPERTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1030 \
  -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
  -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp" \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_WERROR=OFF \
  -DCMAKE_EXE_LINKER_FLAGS=-rdynamic
cmake --build build-v620 --target maya-ds4 -j 4
```

Use modest parallelism initially. CMake applies MMQ fused-SwiGLU and byte-stride changes to a **build-local copy**, not the downloaded dependency. Check for `Cilla: MMQ byte-stride edit applied`; patch failure or missing rewrite seams now stop configuration rather than silently falling back. No private patch script is needed.

## Pack and serve

The command below is an example for a complete, supported `deepseek41` GGUF set, **not a guaranteed recipe for arbitrary GGUFs**. A versioned family example is [Maya-S at revision `758b04c62d2572f4381d067b32d4686a1c4275b3`](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF/tree/758b04c62d2572f4381d067b32d4686a1c4275b3/Maya-S): six `DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-...gguf` shards. This pins artifact identity, not a fresh qualification of that checkpoint on your hardware. Keep all shards from the same revision; check their model license and memory requirements separately.

See `tools/ds4/pack_gguf.py --help` and upstream DeepSeek docs. A text-only pack without a drafter must explicitly opt out of sibling sidecar discovery:

```bash
.venv/bin/python tools/ds4/pack_gguf.py \
  --model /path/to/gguf-directory --out /path/to/new-pack \
  --drafter none --vision none
```

Not every third-party mixed GGUF works directly. Dense Q5_K/Q6_K may need conversion, and mixed experts need valid per-tensor overrides. `tools/ds4/requant_dense_q8.py` is an advanced conversion utility requiring a shard-index JSON, **not an automatic end-to-end preparation pipeline**. Borrowed drafters/vision towers need separate compatibility and output gates.

Copy `examples/cilla.local.example.json` to `cilla.local.json`. Replace executable, pack, tokenizer, working-directory and log paths with your own **absolute paths**, and set the GPU list. The example uses one GPU index, one conversation and 32K configured context, with pinned Engram disabled. It is a configuration starting point, **not a claim that a pack fits one V620**.

```bash
.venv/bin/python serve/server.py --engine strata --config cilla.local.json --host 127.0.0.1 --port 8099
```

Keep it on loopback. The inherited UI exposes management functions; API-key authentication alone does not protect the entire service. Use an authenticated tunnel/proxy for remote access, not direct internet exposure. Never commit your local config or keys.

Wait for the **engine-ready log message** before inference: `/v1/models` alone is not readiness. Test a deterministic answer, growing histories and prefix reuse before long-context or performance tests.

## CPU-only checks

With Python 3.11 and the pinned llama.cpp source installed, install the declared test dependencies and put CMake/Ninja from the venv on PATH:

```bash
.venv/bin/python -m pip install -r requirements-test-cilla.txt
export PATH="$PWD/.venv/bin:$PATH"
.venv/bin/python tools/check_hip_compat.py
.venv/bin/python -m unittest discover -s serve -p 'test_*.py' -t .
.venv/bin/python -m unittest discover -s tools -p 'test_*.py'
.venv/bin/python -m unittest discover -s tools/ds4 -p 'test_*.py'
```

Hardware/model-gated tests may skip. These checks do not qualify HIP numerics or full-model reliability. See [Cilla notes](docs/CILLA.md) for optional four-GPU tuning and known experimental paths.

## Credits and license

Project Maya by Peasant Smith and contributors; Strata by Niko1221 and contributors. Cilla builds on their work. See [LICENSE](LICENSE), [NOTICE](NOTICE), and third-party licenses. Source licensing does not grant rights to model weights, game artwork or trademarks. Remove secrets and private infrastructure details from issue reports.
