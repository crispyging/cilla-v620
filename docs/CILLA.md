# Cilla: implementation notes and limits

## Provenance

The source snapshot is based on a clean Cilla worktree at `f245caf7c4cae2543891f640310496b2c9ee68b1` (Project Maya derivative). The exported `serve/` files matched the server worktree used alongside that engine. This is a GitHub-network fork of Project Maya. Its Cilla branch is based on public Maya ancestry, with a sanitized snapshot commit rather than private local Cilla history. Private commit metadata and deleted deployment content are not imported.

- Upstream: <https://github.com/mw00/project-maya>; public base revision `3bf6867d344212166c09d2a5fe086b9ab90ac2fb`.
- Embedded assets: all 387 original snapshot assets with `.png`, `.whl`, `.bin`, `.gguf`, `.f32` or `.woff2` suffixes matched that public base byte-for-byte; none were private deployment captures.
- llama.cpp pin: `3cf03257f219afbe7334045ff7c6a06ac68c627d`, shared by `setup.py`, CMake and `third_party/ggml/VERSION.txt`.
- Original HIP build: ROCm 7.2.4, gfx1030, HIP on/CUDA off, native experts and MMQ enabled, `maya-ds4` target, `-rdynamic`.
- Publication changes: documentation, local-config ignore rules, generic path arguments for the advanced dense requantizer, home-redaction-aware report tests, fail-closed checks around the existing HIP MMQ build patch, and bounded/rollback-safe MMQ scratch allocation helpers with CPU fault injection. No engine optimization is claimed from these changes.

Upstream source, notices and credits are retained. Upstream installer/update features are inherited code, not a Cilla release/update service; do not use an upstream dashboard update to maintain this snapshot. Keep a backup and update explicitly from a reviewed source revision.

## Optional four-card recipe — not a universal default

For a copyable HTTP launch command and a ready-to-edit config containing these values, see [Launching Cilla](LAUNCHING.md#four-v620-reference-profile-opt-in-high-host-ram-use).

The following settings describe the original four-card layer-split recipe. They require a compatible 40-layer DeepSeek pack, ample physical host RAM and adequate locked-memory limits. Do **not** apply them to one/two-card machines or assume a 512K context was exhaustively qualified. Increase context only after memory admission and output tests.

```
GPUs: 0,1,2,3
prefill chunk: 4096
conversation slots: 4
mmap-experts: off
DS4_SPLIT=5,5,5,6,4,5,5,5
DS4_GEMM16=1
DS4_W16_MB=32
DS4_ATTN_A=1024
DS4_BULK_CHUNK=4096
DS4_YOCO=2
DS4_MMQ=1
DS4_CONVS=4
DS4_ENGRAM_RAM=1
DS4_COPY_NUMA=1
DS4_CKPIN=1
GPU_MAX_HW_QUEUES=4
```

Put engine settings in the config's `env` mapping. Queue-count behavior depends on runtime/firmware; the value above is not a generic AMD recommendation. `DS4_YOCO=2` uses bounded replay rather than the exact-replay setting; compare output on your own workload. Start with the conservative example config, not this list.

Pinned Engram (`DS4_ENGRAM_RAM=1`) and copied experts (`--mmap-experts off`) change the host-memory budget materially. Account for each NUMA node, scratch, startup transients, other services and headroom. Do not fix admission failures by blindly increasing memlock limits or disabling safeguards. The inherited engine also contains strict-residency features; inspect their contracts before enabling them.

## Eight cards and other models

Device selection and general layer assignment use dynamically sized vectors, so `--gpus 0,1,2,3,4,5,6,7` is representable. Eight V620s provide 256 GiB nominal total VRAM before reservations. **An eight-card deployment is not hardware-tested by this release.** The streaming-cost split optimizer is explicitly limited to at most six groups; eight one-card groups use the general free-memory-based layer assignment instead. Do not copy the four-card `DS4_SPLIT` above. Start without that override and inspect actual assignments, output correctness and memory.

Using eight GPUs is not equivalent to `--tp 8`; TP additionally depends on quantization block sizes, FFN widths, attention/output-group divisibility and transport correctness. Prefer ordinary layer splitting first. There is no promise of proportional throughput scaling.

The DS4 packer explicitly requires GGUF architecture `deepseek41`. A compatible V4.1 quant/fine-tune may work after format, memory, tokenizer and sidecar checks. GLM-5.3 has a separate inherited Maya engine; Cilla's DS4 optimizations do not automatically apply. Qwen, Llama and unrelated DeepSeek architectures are not drop-in models. Check the exact model repository, quantization and host-memory budget before recommending this fork.

## Paths not generally qualified

- The MMQ fast path is bounded to at most 8192 rows, embedding width 5120 and expert FFN width 2304; larger or invalid geometry uses the existing generic expert path. The CLI still reserves 320 MiB per device for MMQ. The fixed buffers at the maximum admitted geometry request about 189.1 MiB before context/pool overhead. Allocation helpers publish complete replacement groups only; a failure may retain earlier valid owned groups for retry, but does not publish a partial gate/up pair. These host-side changes have CPU fault-injection coverage, not a fresh HIP build or GPU-parity claim.
- Tensor-parallel and pipelined/decode-lane experiments remain experimental. CPU layout tests are not full-model TP correctness tests; prefer ordinary layer splitting initially.
- Threaded/chunk-parallel prompt execution with multiple conversation lanes has known failure history, including an Engram staging-buffer overflow. Do not enable `--threads on` for this source release.
- Chunks above 4096 have fault/instability history on the original topology. Do not infer safety from one successful request.
- DSpark changes greedy trajectories; use drafting disabled for deterministic numerical comparisons. Re-enable only with a compatible drafter and separate quality/latency gates.
- The C++ `Model` public API is mutable and **single-owner**, not safe for concurrent external callers. Serialize admission, slot selection/switching, feed/step/checkpoint and teardown. Internal per-device workers and conversation slots do not change that contract; use the server's serialized engine protocol rather than calling those methods concurrently.
- Slot selection, identical-prompt reuse, history continuation, eviction, cancellation and concurrent histories are different acceptance cases. A cache-resend win is not a decode-throughput gain.
- The inherited server includes UI/config/update/MCP management capabilities. Bind to loopback and use external authentication/access control if remote use is needed. Do not assume protecting `/v1/*` protects management routes.
- Optional wheel binaries and synthetic fixtures are inherited from upstream and need their own provenance/license consideration. No built Cilla engine or weights are distributed.

## Verification contract

A CPU-only test pass establishes only the checked parsing, mock HTTP, packer and CPU behavior. It does not establish a clean HIP build, GPU numerical parity, working model artifacts, sustained service, all 512K contexts or performance on another machine.

Before relying on a new installation: compile the exact source, verify startup and memory admission, wait for the real engine-ready line, generate a deterministic answer, gate real growing conversations/tool calls and cancellations, then measure with frozen inputs. Preserve rollback artifacts and raw failures. Do not benchmark through a GPU fault or report upstream numbers as Cilla measurements.

No fresh model load/restart or GPU stress run is required to prepare this source snapshot; operational qualification is separate from source publication. See `RELEASE_CHECKS.md` for the checks actually executed during preparation.
