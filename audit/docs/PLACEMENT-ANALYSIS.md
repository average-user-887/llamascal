# P104 placement analysis (2026-11-24)

Sources (read-only copies, sha256 in SHA256SUMS): ~/p104-bench/lab-docs/lab-facts-20261124/ = lab-host:~/LAB_FACTS.md (updated 2026-11-09, plus later registry), m6000-next-plan-20261110.md, serving-registry.json (updated 2026-11-22), fleet-{models,services,topology}.json from <lan-ip>:8090 (live 2026-11-24). lab-host:~/CURRENT_HANDOFF.md is an unrelated Uroboros software handoff (no placement facts). RESEARCH-MODELS.md is the local model-candidate doc. Items marked [gap] are not in any fetched doc.

## (a) Inventory
| Card / host | VRAM | Resident now | Footprint / ctx / slots | Caller / role | Duty |
|---|---|---|---|---|---|
| M6000 (the lab orchestrator host), 250 W cap | 24 GB | ONE of: Qwen3.8-27B C4 `qwen38-27b-dense` (1 slot x 131k, ~26 tg short, 14 @100k) or Qwen3.6-35B-A3B `qwen36-35b-moe`/`occamy` (2 x 98k, ~76-84 tg, pp16k 515). Live 2026-11-24: occamy loaded | swap 35-50 s via llama-swap :11445 | Hermes (provider lab-host-m6000-swap), OpenWebUI, deepseek-harness lead, labcoord lane (cap 2) | on call, one at a time; the "big orchestrator" |
| GTX 1660 Ti (the lab orchestrator host) | 6 GB | Kev-4B Q4_K_M, 2 x 64k, q4_0 KV, ~5.5/6.1 GiB; pp ~257 / tg ~47 | door :18089 (alias kev, decision) | fast judge / decision | boot-resident; also drives the desktop |
| P2000 (hpserver) | 5 GB | Qwen3-Embedding-4B Q4_K_M (~2.9 GiB) + decision-d1-2B (~2.0 GiB, -np 1, max prompt 6144); 4.9/5.1 GiB idle | llama-swap :18000 (raw embed :18081, MRL gateway on the lab orchestrator host :18084) | embeddings mainly Honcho via relay; d1 Hermes profile labd1judge | both resident; card FULL. On-call specialists (lfm25-1.2b, qwen35-2b) cannot load while d1 is resident. Vision is also on this door (per LAB_FACTS) |
| Steam Deck APU (Vulkan) | shared RAM, ~4.6 GiB avail | Qwen3.5-9B Q4_K_M `deck-main`, 2 x 64k; pp ~110 / tg ~12; plus Qwen3-Reranker-0.6B Q8_0 :18084 | <lan-ip>:18082 (Bearer) | worker lane (extractdeck cap 2), Hermes/harness direct; pending: Honcho deriver -> Deck | Registry/fleet show both DOWN on 2026-11-24 (Deck expected offline) |
| CPU (the lab orchestrator host) | - | Whisper large-v3-turbo int8, RTF ~0.5, 6 CPUs/4 GiB, :11447 | HP STT via forward | voice STT | on-demand; GPU Whisper retired 2026-10-30 |
| P104-100 (z4-p104) | 8 GiB (~7.7 usable) | registry alias qwen3-8b-q4km (stale: topology probe shows qwen35-9b-mtp-q4km, 2 slots, ~49k ctx) | :18100 | optional worker, "card z4-onboarding" | 140 W cap live (persistent cap unit not installed, needs sudo) |
| P620 (z4-p104) | small, display | not used | - | - | [gap] VRAM not in docs |

Not found in docs: any Qwen TTS/voice-model host ("TTS shares the P2000 slot via dedalus lock", voice forwards :8000/:8001 to the lab orchestrator host, no model/VRAM given), Qwen-Image or any image generator (only "vision on the P2000 door"), any reranker on a GPU other than Deck CPU/Vulkan. [gap] These need the human.

Orchestration facts: one door per GPU (llama-swap), registry = single source (`labslot set/check`). Hermes kanban lanes = one per endpoint, cap = real slots (m6000 2, extractdeck 2, p2000 2, extract1660 0). Honcho deriver / dream-router currently broken-ish (router points at retired :7862); deriver planned to move to Deck. Design goal (task brief, not in LAB_FACTS): big model orchestrates, small ones supply embeddings/RAG and pruned tool output. Constraints in docs: M6000 refuses loads while gpu-bench.lock held; large prompt batches thrash M6000; models must live on local SSD; Honcho host has 8 GB RAM; P2000 decision-head needs ubatch = ctx.

## (b) Underused capacity and P104 roles (8 GiB Pascal, dp4a, fp16 crippled, PCIe 1.0 x4 ~1 GB/s: model load slow, no multi-GPU split)
Gaps: nobody does pruning/summarising of tool output for the 27B/35B (their pp is the costly part: 27B 14 tg@100k; the plan's 35B pp16k 515 t/s means a 16k tool dump costs ~30 s); the Deck (12 tg, often offline) is the deriver/worker bottleneck; P2000 is full, so no on-call specialists; Kev is a single 4B at 47 tg.

Ranked P104 roles:
1. **Pruner / tool-output compressor for the M6000 models** (LAN call, text in, short text out). Quantised GEMV/dp4a LLM work is exactly what the card does well; ~8 GiB fits 4B at 128k. Wins by cutting M6000 pp. Best fit.
2. **Honcho deriver / worker lane replacing the Deck** (always-on, 24/7, wired, not battery). Fits: current 9B/4B-class model is already proven here (39-43 tg). Beats Deck's 12 tg by 3x+.
3. **Embeddings server (Qwen3-Embedding-4B Q4_K_M ~2.9 GiB) + small reranker (Qwen3-Reranker-0.6B ~0.7 GiB)**: frees the P2000 of embeddings so it can host d1 + specialists, but embeddings/rerankers are pp-bound batch jobs, Q4 dp4a is OK. Co-resides with a 1.7-2B pruner (see c). Moderate value; only worthwhile if P2000 pressure is the problem. Risk: moving embedding model changes nothing in vectors only if same model/quant (it would be; cos 0.99 across cards seen before).
4. Second chat/judge lane (extra slots for labcoord) - low unique value.
5. Whisper large / Qwen voice (ASR/TTS) - **bad fit**: fp16-heavy (CTranslate2/PyTorch), Pascal GP104 fp16 is 1/64 rate; int8 on CPU already RTF 0.5. Better: the M6000 (Maxwell, fp16 also non-native; fp32 fine) or 1660 Ti (Turing, real fp16/tensor, but only 6 GB shared with desktop+Kev) or keep CPU. TTS likewise: 1660 Ti or CPU.
6. Qwen-Image / diffusion - **bad fit**: fp16 + >8 GiB at useful sizes. Needs the M6000 (24 GB, fp32 workable but slow) or off-lab.
7. Reranker/embeddings on the 1660 Ti: better compute (tensor/fp16) than P104 but its 6 GB is full with Kev.

## (c) Replacement-model proposals vs roles; co-residency in 8109 MiB budget (values from RESEARCH-MODELS.md, computed)
| Model | Best role | Alone | Co-resident sidecar options |
|---|---|---|---|
| Qwen3.5-4B Q4_K_M (2614 MiB wts; 64k 5312, 128k 7360) | deriver/worker, orchestrator-lite; general pruner with tool use | yes to 128k | at 64k leaves ~2.8 GiB: + Qwen3-Embedding-4B (~2.9 GiB incl. buffers: tight, probably not) or Reranker-0.6B (~0.8 GiB) yes. At 32k (~3.9 GiB) + Embedding-4B yes |
| LFM3-1.7B Q8_0 (1761 MiB; 64k 3079, 128k 3847, 256k 5383) | pruner/router at high volume, long-context compressor (256k) | yes | at 64k (3.1 GiB) + Embedding-4B (2.9) + Reranker-0.6B (0.8) = ~6.8 GiB: fits. Best co-residency candidate; 128k+embedding also fits (~6.7) |
| LFM2.5-8B-A1B Q4_K_M (4917; 64k 6285, 128k 7053) | quality pruner with 8B capacity at ~100 tg, if kernels behave | yes | only reranker at 64k (6285+~800 = ~7.1 GiB ok); no embedding |
| Nemotron-3.5-Bolt-E4B UD-Q4_K_XL (4826; 64k 6590, 128k 7614) | tool-calling worker | 64k yes, 128k tight | at 64k + Reranker only if ctx lowered; no embedding |
| Qwen3.5-2B Q8_0 (1919; 64k 3307, 256k 5611) | fallback pruner, 256k, same family as Deck/Kev | yes | at 64k + Embedding-4B + Reranker = ~7.0 GiB fits; same footprint class as LFM3 |
Note the research doc says the same-card pairing of two chat models at long ctx fails (LFM3 32k + Qwen3.5-4B 64k = 7.8 GiB over); that holds. Embedding/reranker footprints above are estimates [unverified on this card; must be measured with our gguard]. Suggested single-P104 combos: (A) LFM3-1.7B 64-128k pruner + Embedding-4B + Reranker-0.6B ("RAG appliance + pruner"); (B) Qwen3.5-4B 64-128k alone as deriver/worker (second card would host A).

## (d) Second P104 scenario (PCIe 1.0 x4 each, no SLI/NVLink, no cross-card split)
Layer split across two PCIe 1.0 x4 cards is poor (activations over ~1 GB/s; fine for pipeline of tiny hidden states but extra complexity); 8B-class models already fit one card, so splitting buys only ctx. Prefer role-per-card, or replica for throughput:
- **Role-per-card (recommended)**: card 1 = Qwen3.5-4B (or Bolt) deriver/worker 128k; card 2 = LFM3-1.7B pruner 128k + Embedding-4B + Reranker-0.6B, which also lets the P2000 drop embeddings and host d1 + on-call specialists (lfm25-1.2b, qwen35-2b) again. Deck stays optional.
- **Replica**: two identical 4B workers behind one door (llama-swap or LAN round-robin) only if labcoord lanes need >2 slots; cheap to do, no model changes.
- **Split** (tensor/layer): only to run something that does not fit 8 GiB (e.g. LFM2.5-24B-A2B or 12B MoE ~13 GB weights across 16 GiB, ~14 GiB usable); PCIe 1.0 x4 makes this slow and unvalidated.
Checks (not guessed): PSU rating and spare PCIe power connectors in the Z4 G4 (state: unknown, human to read label/spec: each P104 is 180 W stock, 140 W live cap; plus P620, CPU); slot lane wiring/chipset vs CPU (x4 electrical, two x4 slots + P620 slot); airflow/noise: run1 stopped ~80 C without fan, fan needed; card spacing, whether the P104 blower gets intake; persistent power cap unit still uninstalled (sudo) so a reboot returns to 180 W; second card's temp traces with the first under load.

## (e) Open questions for the human (max 8)
1. Is the P104 meant to be the Honcho deriver/Deck replacement, the M6000 pruner, or the RAG (embedding+rerank) appliance first? Scoring above assumes pruner > deriver > RAG.
2. Where do Qwen voice (TTS/ASR) and Qwen-Image actually run today (docs only show Whisper on CPU and "TTS on the P2000 slot" via dedalus lock; no image host)? Are they wanted on a GPU at all?
3. Should the P2000 keep embeddings + d1 resident (no on-call specialists), or may embeddings move to the P104 so specialists return?
4. Is the Deck's role as deriver/reranker host still planned given it is down/expected offline?
5. Which callers (Hermes lane? harness? labcoord?) may call the P104 over LAN, and is a new lane cap/registry slot name wanted (current registry alias is stale: qwen3-8b-q4km vs live qwen35-9b-mtp-q4km)?
6. Pruning protocol: who calls the pruner (orchestrator tool wrapper, MCP labteam, or harness insert) and what max prompt do we need (drives 64k vs 128k vs 256k)?
7. Z4 G4 PSU wattage, available PCIe 6/8-pin connectors, and whether the case can airflow two blower cards (second P104 plan)?
8. Is noise/thermal ceiling acceptable at 140 W per card 24/7, and may we install the powercap unit (sudo)?
