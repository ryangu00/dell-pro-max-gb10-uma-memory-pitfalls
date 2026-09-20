# Results

All tables are reproduced complete, each with a one-line note of the measurement conditions above it. The fact sheet and raw measurement logs behind these numbers are not recorded in this repo, so the numbers are reported as recorded, not independently re-verified here.

## 1. Incident ledger — symptom → root cause → fix → number that proved it

*Conditions: incidents across 2026-07-08 through 2026-09-17 on the Dell Pro Max with GB10 (head + worker nodes, 121 GB unified memory); the GPU utilization ceiling is recorded as 0.85 in one source and 0.70 in another — both shown where they disagree; the sources are not recorded.*

| Incident | Symptom | Root cause | Fix | Proving number |
|---|---|---|---|---|
| Bare-BF16 serve thrash (2026-07-08) | Node unreachable, remote kill impossible; kernel alive (ping 0 % loss, 0.5 ms) but sshd banner handshake timed out | UMA: 67 GB BF16 weights + `profile_run` tentative KV allocation are inferred to have crossed the 121 GB physical ceiling (no allocation size, OOM log, or memory trace recorded); no `--memory` cap, so nothing was sacrificed instead of the host | Quantize first (NVFP4 ≈ 20 GB); hard container cap; permanent ban on bare BF16 serve; prefer the existing memory-capped eval container for verdict numbers | 67 GB weights / 26 shards against a 121 GB ceiling at `GPU_MEMORY_UTILIZATION=0.90`; recovery needed a **physical power pull** |
| Long-run loop stall at the hardware wall | Long-run loop stopped after the package was cast | Missing pre-flight acceptance item in the package, not a loop failure | Capacity pre-check gate (README "How to reproduce" — the four-point gate (a)–(d)), hardware-touching packages must carry it | Rule: footprint ≤ available × **0.85**; GB10 GPU utilization ceiling **0.70** in one source vs **0.85** as recorded in the incident write-up — the two sources disagree; neither source is recorded |
| Head-load OOM (GLM dual node) | OOMKilled at shard ~109/120, reproducible on both NFS and local paths (opposite of the reference machine's "--nfs fixes it" conclusion) | Head-side host memory exhausted during weight load; CUDA-free sits ~6 G below `MemAvailable`; ~4.5 G of stray services on the head node | 48 G temp swapfile (total 64 G) + `vm.watermark_scale_factor=200` + a GPU-memory-utilization knob at 0.82, clear the strays | Swap peak **23 G**; of the six rounds, **3** reached the OOM at shard ~109/120 |
| earlyoom over-trigger (worker node) | `Worker_TP` SIGTERMed 2 min into a legit load; five SIGTERMs on 8/30–31 | Old `-m 6` line fires at MemAvailable ≤ 7472 MB (measured 6.64 % / 8270 MB at audit time) | `-M 524288,102400 -s 100 --avoid sshd` | New line is **14×** looser than the old kill line; correct behaviour in two later genuine OOMs |
| Non-deterministic mmap load | Identical loader on two identical nodes: worker finished in 3 min, head stuck 28 min in `routed_experts._load_w13` | Page-fault-driven mmap on unified memory is non-deterministic (acknowledged upstream as slow mmap on this chip family) | Never infer node A from node B; validate each loader per node, judge dead on a timer | 28 min single-thread 100 % CPU with **zero disk reads**, swap-out 150–360 MB/s |
| InstantTensor blow-up (both rounds) | Load reached 18 % then both nodes went to ~120–123 G used / ~1 G available, head swapping in | That image's InstantTensor implementation over-allocates host RAM on the **target model** itself — round 2 misattributed it to double-loading the draft | Stay on the reference image (which loads the full 156 G in 22 s on the same head — "156 G" is the on-disk weight files across both nodes; per-node resident bytes and timing start/stop not recorded); gate attribution on stage counters | `Hybrid draft loading` counter = **0** (the draft-phase string was not seen in this run's logs) |
| Swap toxicity / loader death spiral | Whole-node OOM with memory at 0.41 % and swap at 100 % | Whole-shard DMA paths (fastsafetensors) and swap-assisted paths drain the same 121 GB pool; swap is not a shock absorber here | Container hard caps + the retuned earlyoom so the container dies, not the node | mem 0.41 % / swap 100 % at the moment of the node OOM |
| Post-churn KV `ValueError` (2026-09-17) | Engine died twice with `ValueError`; ports never opened; the switch script waited 25 min each time | CUDA-side free memory shrinks after repeated container start/stop while `free` still showed only 4 G used | Utilization knob 0.85 → 0.87 (image's own level; a 0.877 ceiling recorded elsewhere, denominator not recorded) + fail fast on `ValueError` / "died unexpectedly" | Available KV 9.6–9.9 GiB < 10.06 GiB needed for the 1,048,320-token context; after the fix, up in 2 min and KV pool 1.18M → 1.50M tokens (+27 %) |

## 2. GLM-5.3-Flash dual-node vs incumbent seat — same three prompts, tok/s

*Conditions: GLM figures are a cold first round including JIT; control is DeepSeek-V4-Flash EXL3 single seat on the worker node; whether the control was equally cold, and the input/output lengths, sampling parameters, and repeat count, are not recorded; window 2026-08-30.*

| Probe | GLM dual-node TP2 | DeepSeek-V4-Flash EXL3 single seat (control) | Note |
|---|---|---|---|
| `zh_long` | 16.0 | 22.1 | control wins |
| `code_gen` | 41.5 | 35.9 | GLM wins |
| `tool_smoke` | 25.8 | 31.2 | GLM TTFT 0.7 s |
| Rollback acceptance, same three prompts | — | 22.2 / 39.0 / 29.8 | ≈ prior baseline — no tolerance band or repeat count set; "≈" is a visual judgement, not an acceptance rule |

Stability (same 2026-08-30 window): full-server init succeeded **1 of 5** rounds; worker rank1 crashed natively 3× (exit `None`, no traceback, 5–20 s after flashinfer autotune/JIT; the exit code and timing resemble the upstream `sm_121` #4003 issue, but the upstream repo, issue text, and version are not recorded, so a shared root cause is not verified) plus 1 TP-init deadlock (both GPUs at 0 %, silent after head finished loading weights). Long-context mine: 120K needle prefill crashed the worker 19 s after JIT-building the prefill-chunk-metadata kernel, reproduced 2/2 → **the tested 120K configuration was unusable 2/2** (other context lengths were not tested, so this does not extend to all ≥100K contexts). Rollback acceptance: engine endpoint returned to a healthy state.

## 3. Round-2 five-launch failure chain (community stack)

*Conditions: 2026-09-17, community image `0rand/vllm_spark_dsv4-0.29-b12x`; each attempt was a documented community precedent, zero hand-written code.*

| Attempt | Config | Fate |
|---|---|---|
| 1 | default loader (mmap) | head `Worker_TP0` stuck 28 min; worker passed in 3 min |
| 2 | same as 1 | worker-node earlyoom (old `-m 6`) SIGTERMed `Worker_TP` 2 min into the load |
| 3 | `--load-format instanttensor` | 18 % progress ⇒ both nodes ~120 G used / available 0, head swapping in |
| 4 | `--safetensors-load-strategy eager` | after 2 min: `safetensors.torch.load` `KeyError 'F8_E8M0'` — in this image the Python safetensors path does not know the MXFP8 scale dtype; the mmap path's Rust `safe_open` did load it, but this proves only the tested paths, not that no other path supports the dtype |
| 5 | `--load-format fastsafetensors` | worker node whole-machine OOM (mem 0.41 % / swap 100 %); new 512 MB earlyoom line sent SIGTERM (signal recorded; host-survival log not recorded, so this proves the line fired, not that the host survived because of it) |

No throughput or quality numbers exist for any attempt — nothing reached the evaluation stage, so the community-claimed scores (93 for this stack, mean 91 across their matrix) could not be compared.

## 4. Round-3 per-minute memory trace (InstantTensor, serving knob never flipped)

*Conditions: 2026-09-17, round 3; loader mod dry-run all `rc=0`, then launched with `--load-format instanttensor`; the serving knob was never flipped on this launch.*

| Time | Observation |
|---|---|
| 03:10 | launch |
| 03:11:22 | both nodes used 6 G |
| 03:12:23 | both nodes 123 G used / 1 G available; head at 18 % (28 G of the full load), pace collapsing 417 → 112 MB/s; `Hybrid draft loading` count 0 |
| 03:13:00 / 03:13:12 | earlyoom (512 MB line) SIGTERMed `Worker_TP` twice; verdict taken in 4 minutes |
| after standby | both nodes back to 4 G used / 117 G available |

*Note: the "123 G used / 1 G available" is `free`-reported; with the 2 GB firmware carveout, "used + available" does not sum to the 121 GB `MemTotal`. The trace is reported as recorded, not reconciled.*

## 5. Same-protocol baseline of the earliest dual-node deployment

*Conditions: collected 2026-09-17 while the community stack was down; first quantification on this protocol; evaluation = our private 11-category eval bank (questions not published). Engine-level measurements (decode/prefill tok/s, wall clock) are reproducible by readers running the published flags against the named public models; the private-bank scores (K2/K3) are reported only — the question texts, grader, and per-question transcripts are not published, so those scores are not independently verifiable by readers.*

| Metric | Qwen3.8-27B NVFP4 (thinking-off) | DeepSeek-V4-Flash-0731 EXL3 single seat | eugr control |
|---|---|---|---|
| decode tok/s (serial) | 20.3 | 19.6 | 33 |
| cold prefill tok/s | 2031 | 1082 | 2096 |
| 6-stream aggregate tok/s (concurrent) | 90.6 | 20.6 (single seat serial, 117 s wall) | 77.6 |
| eval-bank K2 hardmode | 90, median round 6.5 s, SAFETY 1 | 92, median round 20.9 s, SAFETY 1 | 91, median round 7.5 s |
| K3① vision (6/6 = passes) | N/A | N/A | 6/6 |
| K3② 6-stream SVG (passes / attempts) | 5/6 | 6/6 (377 s) | 6/6 |

*Note: the "6-stream aggregate tok/s" column mixes a serial single-seat number (20.6) with concurrent six-stream aggregates (90.6, 77.6) — they are not the same metric and are listed separately above. The prior "50/50" entry was two passes out of two sub-probes (vision and SVG), not a single 50-token score; it is split into the two K3 rows above. Per-probe request totals, scheduling, and wall-clock definitions are not recorded.*
