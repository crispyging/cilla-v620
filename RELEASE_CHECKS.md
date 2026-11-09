# Source-release checks

This is an **experimental source release**, not a new hardware qualification or performance claim. No model/GPU load, serving restart, or hardware stress test was performed to prepare it.

## Executed checks

| Check | Result |
| --- | --- |
| HTTP server tests, Python 3.11.16, OpenAI SDK 3.46.0 | 336 run; OK, 6 skipped |
| Installer/tools and release-boundary tests | 350 run; OK, 28 skipped |
| DS4 packer tests | 20 run; OK, 1 skipped |
| CPU-only CMake/Ninja build and CTest | Build succeeded; 11 passed, 1 explicitly skipped for unavailable AVX-512 features |
| HIP compatibility-name static check | Passed; no unmapped compiled CUDA names |
| HIP MMQ build-patch boundary tests | Pinned source applies; missing Git/patch seam rejected; upstream source untouched |
| MMQ allocation fault injection | Production helper and extracted caller compiled with CPU fake allocation; all five allocation-failure points, retry/ownership, iota byte size, steady 8192-row capacity and oversized/invalid geometry checked |
| Generated MMQ source vs original deployed build | Byte-identical; SHA-256 `41d0e828a80cae8895b25bb8cc36bdcd0f76fd198b96b315d72c3d1facaca1ec` |
| External llama.cpp build-source comparison | 1,708 files checked against pinned public source; zero differences |
| Inherited binary/fixture/font/screenshot/wheel comparison | All 387 inspected asset files matched public Maya base byte-for-byte |
| Gitleaks 8.30.1 source-tree scan | No leaks found; no allowlist/suppression added |
| Owner-specific identifier/private endpoint scan | No matches in the staged release source |
| Whitespace/diff checks | Passed |

CI uses Python 3.11 and `requirements-test-cilla.txt`, including the SDK conformance oracle. These pins describe the tested environment, not a complete transitive dependency lock or an engine runtime dependency requirement.

## Failures retained and resolved or bounded

- Initial test collection lacked the pinned llama.cpp Python package and then `pyyaml`/`regex`. Dependencies were installed and the complete suites rerun.
- Tools initially could not find CMake on PATH. The release check environment and CI put the venv executables on PATH.
- Three report tests initially expected unredacted home paths. The implementation correctly redacts the home directory; test expectations were corrected, not privacy behavior weakened.
- The first new MMQ boundary test checked the wrong textual marker. It was corrected to test the actual fused function plus negative-stride addressing; the generated file also independently matched the deployed build hash.
- The initial CTest run reported a failure where the review CPU lacked the required AVX-512 ISA. The test now returns an explicit CTest hardware skip; it does not claim numerical parity on unsupported hardware, and benchmark mode still refuses unsupported CPUs.
- An independent review environment using OpenAI SDK **2.24.0** rejected 10 Responses event conformance cases. The declared test oracle **3.46.0** passes the full server suite. Legacy-SDK schema-oracle compatibility is not established; do not interpret this release as qualification of every SDK version.
- Independent code review found a real gate/up scratch-allocation failure leak. Replacement allocation is now transactional per group, and MMQ refuses requests above its bounded geometry. The review's proposed 2048-vs-8192 reservation mismatch was not confirmed: the original CLI already adds 320 MiB per device for 8192-row scratch, separately from the set grouping limit. Caller-level fault injection also caught a candidate iota byte-count regression before publication; the final caller requests `rows * sizeof(int)` and the integration test checks it. Earlier successful allocation groups remain valid/owned on a later failure and may be reused on retry. No math kernel or main reservation amount was changed, and no fresh HIP build/GPU numerical qualification is claimed for this host-side fix.
- HIP MMQ patch failure previously fell back silently. The public candidate now rejects patch failure or missing required rewrite seams. This is CPU-tested build-configuration hardening; a clean HIP rebuild of the public candidate has not been performed here.

## Not established

No clean-machine HIP build, fresh full-model parity/endurance test, eight-card run, new model-pack qualification, maximum-context qualification, throughput guarantee, or security audit of the entire inherited management UI is claimed. Preserve loopback binding and the limitations in `docs/CILLA.md`. An eight-card user must qualify their own topology, exact model/quant and memory budget.

Upstream licenses and public author credits are deliberately retained. Local private commit history and deployment artifacts are not imported into the Cilla branch.
