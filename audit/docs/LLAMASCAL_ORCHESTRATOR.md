# Llamascal orchestrator rules
Version 1.2 · 2026-11-26 · HP Z4 G4 · Pascal-first

**Operating rule:** Use all three GPUs for independent discovery when the host passes the concurrency check. Use exclusive, matched A/B runs to certify gains. GPU residency removes most steady-state host transfers; it does not isolate CPU scheduling, disk/page cache, RAM, PCIe, cooling, or power.

## 1. Start here in a fresh context

1. Read this file and the repo's `AGENTS.md`, then the latest ledger entry, model manifest and private host inventory. Reconcile outstanding experiments, owners, exact commits, and device leases. Treat missing information as unverified.
2. Confirm the active repo and that each GPU is available for the assigned work. This is a **Pascal-first** project; no Maxwell port or preset is assumed. Start with the current Llamascal baseline. Support discovery on a new card or a different model is a new result, not inherited evidence.
3. Assign one owner per GPU lane. The orchestrator owns the global schedule, common fixtures, baseline definitions, admission and promotion. Workers own only their experiment worktree, build, logs and GPU lease. Start a card only when its hypothesis, scope, stop condition and expected evidence are written down.
4. Separate coding from bench ownership. Review a small patch with one primary hypothesis before merging it. Never retune several switches inside an A/B pair. Do not change host services, the desktop, power or fan policy, unrelated jobs or repo visibility to smooth over a failed experiment. Stop at the failing gate and write a handoff.

### Antigravity: lean orchestration by default

**Gemini Neon plans and reviews. Gemini Hydrogen executes.** Use Antigravity's orchestration mode when available. Neon maintains the queue and grants leases, writes bounded cards, and decides whether the result warrants escalation. Hydrogen runs only the assigned work and hands back a receipt. Keep the planner out of routine log reading and keep executors out of unrelated branches. Do not start workers just to watch other workers.

| Effort | Use for | Stop when |
| --- | --- | --- |
| Low / quick | File/metadata check, one dispatch probe, one fit check or short smoke | The requested evidence or first concrete blocker is recorded |
| Medium / targeted | One bounded implementation or matched short A/B on one lane | Patch and checks are reproducible, or the card's stated budget/pass criterion fails |
| High / escalated | Unexplained regression, shared-path interaction, or evidence needed for a default or portable claim | The named uncertainty is resolved or the remaining blocker is documented |

Start at the lowest useful effort, adjustable per card. Neon records a concrete reason before raising it. Use fresh short-context workers by default. Do not inherit entire chats, resurrect old workers, or send every artifact to every lane. Reuse context only when indispensable transient state would otherwise be lost (for example an owned live experiment), or when a specific handoff proves it saves substantial repeat work without carrying stale assumptions. Resume from receipts instead of memory: exact commits, fixture, lease, command, result and next action. Batch only closely related tiny probes; parallelize across unoccupied GPU lanes when already qualified, not across dependent gates.

One screen per card is enough: `ID / owner / tier / hypothesis / files and lane / GPU and time budget / exact control and fixture / command or allowed actions / pass + abort criteria / expected receipt and next gate`. On timebox expiry, return the partial result and release only resources the card owns; no silent extension or rerunning an entire matrix.

## 2. Six candidate files and three comparison types

All six are **Q4_K_M text-only weight files**. Sizes below are listings, not measured VRAM. Do not add a vision projector. Q4_K_M is a mixed tensor recipe; verify the actual quant type and shape of the tensor your patch targets.

| Z4 lane | Dense | Routed MoE |
| --- | --- | --- |
| P620, 2 GB, also driving display | [Qwen3.5-0.8B, Unsloth](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/tree/main), `Qwen3.5-0.8B-Q4_K_M.gguf`, 533 MB | [Granite 3.1 1B-A400M Instruct, QuantFactory](https://huggingface.co/QuantFactory/granite-3.1-1b-a400m-instruct-GGUF/tree/main), `granite-3.1-1b-a400m-instruct.Q4_K_M.gguf`, 850 MB |
| P2000, 5 GB | [Qwen3.5-4B, Unsloth](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/main), `Qwen3.5-4B-Q4_K_M.gguf`, 2.74 GB | [Granite 3.1 3B-A800M Instruct, TensorBlock](https://huggingface.co/tensorblock/granite-3.1-3b-a800m-instruct-GGUF), `granite-3.1-3b-a800m-instruct-Q4_K_M.gguf`, 2.017 GB |
| P104, tested 8 GB | [Qwen3.5-9B, Unsloth](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/main), `Qwen3.5-9B-Q4_K_M.gguf`, 5.68 GB | [OLMoE 1B-active/7B-total 0924 Instruct, bartowski](https://huggingface.co/bartowski/OLMoE-1B-7B-0924-Instruct-GGUF), `OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf`, 4.21 GB |

- **LADDER:** Each card runs its two right-sized candidates. Start at short context. Fit includes all resident weights (including inactive expert weights), KV/state, graph/scratch and the P620 desktop. Active experts describe per-token work, not what must fit. Insufficient VRAM means record a fit miss or define a separately named fallback with its own baseline; no silent CPU spill. Keep the measured 9B Q8 KV setup as its own practical profile unless the matched arms explicitly use that type. Preserve the reported P104 140 W cap if still in effect and record the actual cap.
- **COMMON:** When checking a promising gain across hardware, use the exact small Qwen GGUF on all three for dense, or the exact small Granite GGUF on all three for MoE. Freeze GPU residency, template/token sequence, quant, batch/ubatch, K/V types, context and output budget to the P620's available headroom. Compare change versus its local control per GPU. Never compare the three different ladder sizes as if their only difference were GPU.
- **TARGETED:** Choose one tensor shape and actual path for the patch. If the common fixture never dispatches it, report “not exercised”; pick a suitable fixture or make a narrower claim. OLMoE is a second routed architecture, useful to check transfer after a Granite win, not evidence that all MoEs behave alike.

Manifest each GGUF SHA-256, exact file and upstream model revision, tokenizer/template, architecture and target tensor type/shape, plus realized context and allocation. The listed repos are converters; reference logits must come from the original matching checkpoint. Six files define the ladder, not an acceptance matrix to rerun after every edit. Loop/recurrent architectures can be added later under a separate profile.

## 2a. Branch and baseline policy

- **One repo and default binary.** Ordinary dense matmul and expert-indexed matmul can dispatch to different kernels: see [llama.cpp CUDA backend](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu) and [shared quantized matvec](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu). They can still share quantized compute, scratch/graphs, routing or launch decisions. Put each optimization behind the precise operation, architecture, layout, tensor quant and shape guards it needs. Preserve the old path and a reversible control toggle. Don't assume dense-only changes spare MoE if they touch shared code.
- **Two pinned references.** Keep a runnable immutable stock upstream commit for occasional external comparison, and tag the current dense-optimized Llamascal commit as the daily control. Start a **short-lived** `exp/moe-<idea>` branch/worktree from that Llamascal control; do not fork the product. Run stock versus the current dense control once on one representative loaded MoE profile to measure any inherited gain or regression. Then test MoE candidate versus that control on matched settings. Report “versus stock” and “incremental versus Llamascal” separately; rebaseline deliberately, not per edit. The reported 30–40% dense improvement is a lead until receipt-backed with its exact workload and comparator.
- Carry a cross-family smoke test when shared dispatch or primitives change. When two patches can interact, compare **both off / dense only / MoE only / both on** using toggles in one otherwise identical build on representative fixtures. Keep a second permanent build only if a demonstrated compile-time conflict, footprint or regression has no practical isolated dispatch solution.

## 3. Admission and shared-host scheduling

- **GPU lease:** One benchmark/inference experiment per physical GPU. Pin process visibility to its private UUID; changing visible order changes ordinal numbers. Confirm actual selected backend and physical lane after initialization. Never emit UUID, device serial, credentials or private paths in shared receipts; use P620/P2000/P104 aliases. Leave P620 display ownership alone; if it is busy, postpone its timed run rather than killing the desktop.
- **Host gate:** Coordinate all launches and use a shared lock for model loads, builds, downloads, cache-heavy work and other heavy I/O. Stagger models, letting each completed load settle and warm before loading the next. A host-exclusive timing window must block **new admissions and heavy work** and wait for every other owned experimental lane to drain, using the same orchestration gate so timing cannot race launch. Per-GPU leases stay held until the owned process exits. Never clear an unknown lease or kill an unrelated process.
- **Memory:** Budget from measured free memory and peak demand, not nominal VRAM. Include weights, requested worst-case KV/state, scratch/graph buffers and desktop headroom. Check actual layer placement, backend and KV/state residency from loader/runtime evidence. Any intentional CPU components must be documented, proven unchanged, and excluded from a “fully resident” claim if they are part of the model compute or cache. Host tokenization and file mmap are not CPU layer offload. Budget host RAM for concurrent peak staging, OS/page cache and workers. Stop admitting on swap activity, memory pressure, OOM, instability, throttling, or loss of desktop usability. Drain owned work in priority order; do not disable swap, alter clocks, or fill RAM with a build while timing.
- **Other shared constraints:** Record the PCIe link width and generation at an appropriate idle scope if safely available. CPU affinity, overlapping power/thermal limits and background load can matter despite residency. Use only low-rate, known-safe observations; no GPU polling or monitoring loop that can hang the host. Never change a power limit or latch a fan at a new setting just to finish a run.
- **Build and storage isolation:** Pin known-good baseline source, executable/library hashes and effective build flags. Separate worktree, CMake cache, build dir, output dir, ports, telemetry and logs per experiment/lane/arm as needed. Never overwrite a baseline or direct a clean script at a shared output. Admit fresh builds only after verifying toolchain/driver compatibility and actual target architecture. Read-only model cache may be shared after hash verification; no simultaneous writes.

### Qualify parallel discovery before stacking three experiments

1. Obtain two or more repeatable solo **A/A** runs per lane on its intended concurrent workload, after the same warm-up. Get per-phase medians and spread. Start with a **3% drift guard** for prefill and decode medians, relative to each lane's solo baseline. If solo variance itself crosses the guard, fix or flag the setup first; do not silently loosen the guard after seeing results.
2. Stagger loads. Run the **same** pinned binaries/workloads solo versus 2-up then 3-up, with timing overlap recorded. Match P620 desktop state, clocks/power policy, warm-up/cache state, context and output length. Report each lane's percent drift and spread separately, plus RAM/swap/pressure and observable throttling. Check workload and output equivalence, not just total tokens/s.
3. If all participating lanes pass, schedule one exploratory job per GPU and record overlap. Otherwise use the passing pair, serialize the affected lane or reduce shared pressure, then requalify. Requalify following material workload, topology, power, thermal, driver or desktop changes. Parallel data remains **exploratory**, even when drift is small; it can be reported separately as host aggregate throughput.

## 4. Iterate fast, validate only the changes that earn it

**Fast path per patch:** One hypothesis, one target GPU, one appropriately sized model and one exercised shape. Start with a short warm-up and two balanced matched pairs (AB/BA) on the pinned daily control and candidate. Log a brief dispatch witness separately from timing, pp and tg separately, and a short deterministic output sanity check for valid finite output and stops. Work toward completion in minutes, not six full runs. If the patch touches a shared path, also do one short opposite-family smoke on the same card if it fits; if it doesn't, use a fitting common model and label the scope. Unchanged paths need no routine opposite-family test. Qualified concurrent discovery is allowed, but don't present overlapping timings as a validated speedup.

**Escalate only for a repeatable promising hit, a shared/default change, a meaningful quality warning, or a portability/public claim.** Use exclusive, interleaved runs and at least five matched pairs when practical, full artifact provenance, relevant target and opposite-family checks, and the common fixtures on the additional cards only as needed by the intended claim. Compare the same GPU, exact model and tokens, seed/policy, context, batch, slots, K/V, warm-up/cache, offload, clocks/power policy and build flags. Separate cold loading from measured warm pp/tg and end-to-end. Preserve all attempts and predeclare invalidation causes and the minimum worthwhile gain. If small/noisy or the target path never ran, keep the finding exploratory or stop. Test both switches together when shared-code effects could interact. No packaging or default promotion based solely on a quick hit.

### Cheap, one-sided KLD guard

1. For the models where numerical changes need checking, cache a small fixed teacher-forced token sequence and **unquantized FP16 or FP32 reference logits** once per original checkpoint, off the Z4 if needed and authorized. Pin reference checkpoint, tokenizer/template and token IDs, positions, vocab alignment, logits and runner. 32–128 scored positions across a few short prompts is a starter screen, not a quality benchmark. No Q8 substitution or comparison across different base models.
2. On the same positions and distribution vocabulary, compare the quantized control and candidate using `delta = mean KL(ref || candidate) - mean KL(ref || control)` in nats/token, computed stably from full logits, not top-k. Keep precision/normalization the same in the two arms. If control and candidate logits on the probe are byte-identical, no new KL calculation is needed. NaN, Inf, broken output or invalid stops fail the sanity check independently.
3. **Only a reproducible positive delta above the predeclared one-sided tolerance triggers a quality regression investigation.** Calibrate using a short control/control repeat; as a starting engineering guard, use the greater of the observed repeat-noise bound and 0.001 nats/token, then keep it fixed for that profile during comparison. Repeat once on a warning before escalating; negative deltas do not trigger. Report mean delta and denominator, not a global quality promise. Use the tiny probe for suspected numerical/shared changes or promotion; no mandatory KL pass on every dispatch-only edit. If full-precision logits aren't available, label the speed result exploratory and quality unchecked against reference.

## 5. Experiment receipt and restart

Write one ledger entry per experiment, with a stable ID and relative artifact paths. Minimum fields:

    id | owner | status | started/ended UTC | card alias | profile ID | solo/overlap
    baseline/candidate commit and binary SHA | GGUF SHA | compiler/CUDA/driver/build flags
    effective args + prompt/token fixture hash | device/offload/KV proof | dispatch witness
    VRAM peak/free floor | host RAM/swap/pressure | power/thermal/desktop state
    raw pp/tg/e2e metrics | repeats/pairs/invalid counts + reasons | parity/quality
    conclusion + scope | artifact links/hashes | cleanup/lease status | next bounded card

Use statuses `queued`, `exploratory`, `validated`, `inconclusive`, `failed`, `blocked`. Keep immutable raw receipts, failed hypotheses, and exact baseline hashes; record model and resource availability as observed. At handoff list the last clean baseline per lane, exact unmerged patches, known confounds, current owner/leases and the next experiment. If a worker disappears, verify ownership before resuming or clearing anything. When a new session starts: reconcile receipts first, skip completed experiments, and resume the next bounded one.

## AGENTS.md pointer

Save this file as `LLAMASCAL_ORCHESTRATOR.md` beside the root `AGENTS.md`. Add:

> Before assigning or running a GPU experiment, read [LLAMASCAL_ORCHESTRATOR.md](./LLAMASCAL_ORCHESTRATOR.md). Its workload, lane ownership, shared-host admission, matched timing and evidence rules govern Llamascal benchmark claims.

This document defines the workflow. It does not certify the Z4's installed builds, actual model fits, concurrency, or any speed result.
