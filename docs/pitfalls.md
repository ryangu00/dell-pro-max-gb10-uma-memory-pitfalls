# Pitfalls

Each pitfall is expanded as **symptom / root cause / fix / how we found it**. Numbers are reported as recorded, with their conditions; the fact sheet and raw logs behind them are not recorded in this repo.

---

## P1. `earlyoom` banner text mistaken for the real kill

- **Symptom.** `earlyoom`'s restart banner prints "sending SIGTERM when mem ≤ X%". A loose grep on that string looks like a kill, so a worker process appears to have been `earlyoom`-killed when it was only a banner.
- **Root cause.** The banner line ("sending SIGTERM when mem ≤ X%") and the real kill line ("sending SIGTERM to process") are different strings; a single broad grep matches the banner and reports a kill that did not happen.
- **Fix.** Match the two strings separately. Confirm a kill only on "sending SIGTERM to process", never on the banner.
- **How we found it.** While triaging the round-2 worker SIGTERM (2 min into a legit load) the banner grep reported a kill that the real kill line did not corroborate.

## P2. `/etc/default/earlyoom` refuses to start

- **Symptom.** After editing `/etc/default/earlyoom`, the service fails to start.
- **Root cause.** A trailing `#` comment on an `EnvironmentFile` line is parsed as an argument, not a comment; the line is malformed.
- **Fix.** Put comments on their own line; never append `#` to an `EARLYOOM_ARGS=` value.
- **How we found it.** Retuning the worker-node `earlyoom` to `-M 524288,102400 -s 100 --avoid sshd`; the edited file did not start until the trailing comment was moved.

## P3. A "successful" load on one node says nothing about its twin

- **Symptom.** Identical loader on two identical nodes: the worker finished in 3 min, the head stuck 28 min in `routed_experts._load_w13`.
- **Root cause.** Page-fault-driven mmap on unified memory is suspected to be non-deterministic; one 3-min vs 28-min divergence on two identical nodes is not enough to prove a mmap root cause (cache state, memory pressure, and version-controlled loaders were not held constant, and the cited upstream "slow mmap on this chip family" note is not recorded in this repo). One node's behaviour did not predict the other's in this run.
- **Fix.** Never infer node A from node B. Validate each loader per node and judge dead on a timer.
- **How we found it.** Round 2 attempt 1 (default mmap loader): 28 min single-thread 100 % CPU with **zero disk reads**, swap-out 150–360 MB/s on the head, while the worker cleared in 3 min.

## P4. Attribution of a memory blow-up to the wrong phase

- **Symptom.** A launch reaches 18 % then both nodes go to ~120–123 G used / ~1 G available, head swapping in; round 2 blamed double-loading the draft.
- **Root cause.** The InstantTensor implementation in that image over-allocates host RAM on the **target model** itself. Round 2 misattributed it to the draft phase.
- **Fix.** Read stage counters (`Hybrid draft loading`) before blaming a component; stay on the reference image (which loads the full 156 G in 22 s on the same head — "156 G" is the on-disk weight files across both nodes; the per-node resident bytes and the timing start/stop points are not recorded).
- **How we found it.** A single grep count of `Hybrid draft loading` = **0** (the draft-phase string was not seen in this run's logs). A zero count alone cannot rule out log loss, a level change, or a different control flow that never logs the string; it is bounded here to "the draft phase was not reached in this run's logs".

## P5. Mod dry-runs give false confidence

- **Symptom.** Round 3's dry-run patches all applied cleanly (`rc=0`), yet the real launch still blew host memory at 18 %.
- **Root cause.** A dry-run proves the patch applies; it does not prove it solves the problem.
- **Fix.** Treat a dry-run as a patch-applicability check only. Judge the real launch on its own memory trace.
- **How we found it.** Round 3: `--check` / patch / MARKER / import all passed and all five mods stacked in Ollie order with `rc=0`; the subsequent launch still hit 123 G used / 1 G available at 18 %.

## P6. `/health` 200 is not "engine alive"

- **Symptom.** The `/health` endpoint keeps returning 200 while the engine is not actually serving.
- **Root cause.** Collective-timeout hangs keep the endpoint answering; a port answering does not mean a request can be served.
- **Fix.** Probe with a real request, not a port check.
- **How we found it.** Post-launch checks that relied on ports only produced two 25-minute waits (in the switch script) before anyone judged the launch dead. The record shows the port never opened; it does not show a `/health` 200 coexisting with a failed inference, so this pitfall is bounded to "a port probe is not a serving probe", not to a proved collective-timeout root cause.

## P7. Probing memory with `torch.cuda.mem_get_info` perturbs the box

- **Symptom.** A memory probe itself changes the state it is trying to measure, spinning up a fresh CUDA context (NVRM `0x51`).
- **Root cause.** `torch.cuda.mem_get_info` can initialize a CUDA context as a side effect of the call.
- **Fix.** Use OS-level counters (`free`, `/proc/meminfo`) instead of CUDA API probes. OS-level free is not a drop-in for CUDA-allocatable memory: on this platform CUDA-free sat ~6 G below `MemAvailable` in the observed window, but that gap is an observed value, not a constant — reusing an existing CUDA context to measure is preferred where possible, and otherwise treat the gap as a conservative budget to subtract.
- **How we found it.** Memory probes during the head-load and InstantTensor incidents showed NVRM `0x51` on a probe that should have been read-only.

## P8. Kernel-level OOM killer may simply not fire in a thrash

- **Symptom.** Whole-machine thrash with the kernel alive (ping 0 % loss, 0.5 ms) but `sshd` banner handshake timed out; no kernel OOM kill happened.
- **Root cause.** In a unified-memory thrash there is no discrete VRAM to reclaim; the in-kernel OOM killer may not fire before userspace is starved.
- **Fix.** Cap the container (`--memory`) so a runaway container is bounded by its own limit; keep `earlyoom` as the node-level last resort with `--avoid sshd`. The driver/cgroup memory accounting and limits that make the container die first are not recorded, so this is the intended design, not a proved ordering under this kernel.
- **How we found it.** Incident 1 (2026-07-08): bare BF16 serve at 67 GB / 26 shards against a 121 GB ceiling, `GPU_MEMORY_UTILIZATION=0.90`, no container cap; recovery needed a physical power pull in this run.

## P9. Head-node memory looks plentiful in `free` while CUDA is short

- **Symptom.** `free` reports ample head memory, but CUDA-side allocation fails.
- **Root cause.** CUDA-free sits ~6 G below `MemAvailable`; `free` overstates what CUDA can actually use.
- **Fix.** Account for both OS-level free memory and the CUDA-free gap; do not size a launch on `free` alone.
- **How we found it.** The head-load OOM workaround: CUDA-free ~6 G below `MemAvailable`, plus ~4.5 G of stray services on the head node.

## P10. Weight-load OOM on a head node that has the RAM

- **Symptom.** OOMKilled at shard ~109/120 during weight load, reproducible on both NFS and local paths (opposite of the reference machine's "--nfs fixes it" conclusion).
- **Root cause.** Head-side host memory exhausted during weight load, not a path/NFS problem.
- **Fix.** Add the temporary swapfile + watermark tuning combination (48 G temp swapfile → total 64 G, `vm.watermark_scale_factor=200`, a GPU-memory-utilization knob at 0.82) and clear stray services, rather than shrinking the model. Both NFS and local paths reproduced the failure, but that does not prove the path is irrelevant; the swapfile, watermark, utilization knob, and service cleanup changed together, so this is a combined mitigation, not an isolated fix. The swapfile's creation, recovery, and removal steps are not recorded.
- **How we found it.** GLM dual-node trial (2026-08-30): swap peak **23 G**; all **3** rounds crossed shard 109 on both NFS and local paths.

## P11. Community "good numbers" are stack- and image-specific

- **Symptom.** Community-claimed scores (93 for this stack, mean 91 across their matrix) could not be reproduced or even compared.
- **Root cause.** The community numbers depend on exact knobs in the exact image; a knob that looks equivalent may be absent in your image.
- **Fix.** Before adopting a recipe, check that the exact knob exists in the exact image. (An earlier note called the in-house draft-loader mod "a requirement, not an optimization"; that necessity claim is dropped — the mod applied cleanly in a dry-run but the real launch still failed, so the mod is neither proved necessary nor proved sufficient.)
- **How we found it.** Round-2 chain: all five attempts (four distinct loader configurations) died before evaluation; no throughput or quality numbers exist for any attempt.

## P12. Long experiments are more expensive than fast verdicts

- **Symptom.** A launch hangs for 28 min (round 2 attempt 1) before being judged dead.
- **Root cause.** No "N minutes without progress = dead" threshold was set before launching.
- **Fix.** Set a progress deadline before launching. Round 3's 4-minute call versus a 28-minute first failure.
- **How we found it.** Direct comparison of the round-2 first failure (28 min) against the round-3 verdict taken in 4 minutes.

## P13. Worker node without egress

- **Symptom.** A worker node cannot pull images or weights; everything must arrive via the head node.
- **Root cause.** The worker has no external network (intranet only); `docker save|load` / `rsync` over the 200GbE link is the only path.
- **Fix.** Plan transfer time (~273 MB/s observed) into every window; tag references and assert image IDs on both nodes (a digest-pinned `docker load` drops the repo digest and the worker tries to reach the internet it does not have).
- **How we found it.** `docker save|load` with digest-pinned images: the load step dropped the repo digest, so the worker's pinned reference missed.
