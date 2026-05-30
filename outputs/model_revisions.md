# Model revisions — round 1 (from multi-agent review)

Changes made to address the review. Each notes the fix and its numeric impact.

## Correctness fixes applied

1. **MLA KV for the MoE preset (was GQA).** `config.py`: `ModelConfig` gains
   `kv_mode` ("gqa"/"mla") + `mla_kv_dim`; `d_kv` returns the single compressed
   latent (576) for MLA instead of `2·n_kv_heads·head_dim` (2048). The
   DeepSeek-V3-class preset now uses MLA (the `moe_large_gqa` key is kept as an
   alias).
   **Impact:** MoE KV ~3.5× smaller → less HBM pressure → baselines fit larger
   batches. MoE offload benefit fell from ~+35% to **~+24%** (EP-heavy 1×64:
   33.5k→41.4k tok/s), and absolute throughput rose. Dense-70B unaffected.

2. **MoE weight-read saturates toward P_total with batch.** `analytical.py`:
   new `weight_read_bytes(model, b_d)` = non-routed (always read) + routed
   experts × coverage(b_d), where coverage = 1−(1−ρ)^b_d, ρ = active/total
   expert mass. Equals P_act at b_d=1 (so dense and small-batch unchanged) and
   → P_total as the batch saturates experts. Replaces the old P_act-only term
   that under-counted MoE weight traffic at large batch.

3. **C2C charged once, not twice.** Decode-C2C is on the TPOT critical path
   (and ≤ bw_c2c by construction), so `tps` no longer divides by total C2C util
   (which double-charged it). Throttling now uses only **append-induced** C2C
   contention (`c2c_util["append_util"]`), which is the genuinely additive part
   not already in TPOT. `c2c_util` now reports `decode_util`/`append_util`
   separately.

4. **C2C bidirectional vs one-way.** `HardwareConfig.bw_c2c_oneway` = bw_c2c/2.
   One-directional bulk transfers (old-KV load, new-KV flush) now use the
   one-way figure (was full aggregate → had halved transfer time). Bidirectional
   decode Q/O exchange still uses the aggregate.

5. **Decode GPU-attention compute floor.** `gpu_attn_time` now takes
   `max(KV-bytes/BW_HBM, attn-FLOPs/F_G)` for symmetry with the CPU/append
   terms. Harmless for long context (memory-bound), correct at large batch /
   short context.

6. **Append-prefill HBM floor.** `append_time` HBM term now includes the
   weight-read (`weight_read_bytes`) so small-A, weight-bandwidth-bound appends
   aren't under-counted.

7. **Honest docstring.** `tps` no longer claims a "min over resource bounds"
   that it never took; the bound is encoded inside TPOT's rooflines, and the
   `components` dict is documented as diagnostic-only.

## Tests
27/27 pass (was 24). Added: MLA-vs-GQA KV size, MoE weight-read batch
saturation (+ dense batch-independence), C2C-not-double-charged. Updated the
old-KV-reflush test to the one-way bandwidth.

## Round-1 verification (round-2 review)
A second multi-agent review confirmed all 6 round-1 fixes are mathematically
correct and self-consistent within config.py/analytical.py, with no regression.
MLA d_kv=576 matches DeepSeek's published ~70 KB/token. Two soft spots: the MoE
weight-read asymptote is 0.975·P_total (not P_total) — a small conservative
decomposition sliver; and rho is defined on expert *mass*, not *count* — exact
for equal-size experts (current preset) but should be on top_k/E eventually.

## Round 2 — DONE so far (code/correctness P0+P1a)
- **parallelism.py reconciled with analytical:** per-rank weight read now uses
  the same batch-saturation model (`_rank_weight_read_bytes`); old-KV load uses
  `bw_c2c_oneway`; per-layer `t_sync` added to the decode C2C term.
- **Optimistic AND conservative TPOT both reported** (parallelism CSV columns
  `system_tps_optimistic`/`_conservative`; recommended_configs shows an opt..cons
  band). This exposed a key finding: **sparse MoE offload is +24% optimistic but
  −13% conservative** for 1×72 — the offload benefit exists only if CPU+C2C
  overlap GPU work. All headline gains are now labeled best-case.
- **Expert TP-sharding added** (`ParallelConfig.shard_experts_with_tp`,
  `expert_pieces`): experts shard across min(ep·tp, tp·cp) in-stage GPUs. With it,
  non-EP replicas fit the 671B MoE → **"EP mandatory" RETRACTED** (it was an
  artifact of EP-only expert sharding). EP all-to-all / TP all-reduce already
  approximate the respective comm, so no double-counted comm term was added.
- Tests still 27/27.

## Round 2 — COMPLETE

All P0/P1/P2/P3 items from the round-2 review are now addressed:

**P0 (done in the code-fix iteration):**
- ✅ Optimistic + conservative both reported (parallelism opt..cons band;
  CSV columns). Headlines labeled best-case. Exposed: sparse MoE offload is
  +24% optimistic / −13% conservative.
- ✅ parallelism.rank_step reconciled with analytical (batch-saturating weight
  read, `bw_c2c_oneway`, per-layer `t_sync`).

**P1 (done):**
- ✅ Expert TP-sharding added (`expert_pieces`); "EP mandatory" RETRACTED — a
  non-EP TP8,PP4 replica fits with expert-TP, fails under EP-only. TP all-reduce
  / EP all-to-all already approximate the respective comm (no double-count).
- ✅ Admission-bound sensitivity sweep -> `outputs/admission_sensitivity.md`
  (varies HBM, r, offload f): the bound dissolves as HBM grows (192→768 GB:
  admit-stall 46891→947 s) and eases with lower r — so it's a property of
  (HBM size, r, single-GPU), not intrinsic to the workload.

**P2 (done — documentation honesty):**
- ✅ "never/always" claims scoped to the single-node interactive-SLO regime
  (SUMMARY #3, phase2_findings, validation_report).
- ✅ "reproduces FastDecode/Adrenaline" reframed: FastDecode is ⚠️ out-of-regime
  (multi-node dense CPU BW aggregation), NOT reproduced; others 🟰 "shares shape".
- ✅ Sparse-attention ACCURACY caveat (~2.1–2.5%) added wherever sparse offload
  is recommended.
- ✅ Trace chars/token relabeled as an ANCHOR (not validation) + sensitivity
  note (cpt=4.0 shifts input ~1.6%) + cache-hit optimistic-prefix caveat.

**P3 (done — test quality):**
- ✅ Replaced DP-linearity (was ==4× by construction) with
  `test_expert_tp_sharding_lets_non_ep_fit`; replaced same-seed determinism with
  `test_sim_seed_changes_arrivals_but_same_seed_reproduces`; replaced the `>=`
  no-op offload test with `test_sparse_offload_strictly_helps_optimistic_but_
  can_hurt_conservative`; added a hand-computed tiny-trace parser fixture.
- Tests: 18 + 4 + 6 = **28/28 pass**.

**P4 (acknowledged, not "fixed" — these are documented limitations):** MoE
absolute tok/s are optimistic ceilings ~2–4× high (inter-node fabric cost
under-modeled; PP bubble ~1.01 because microbatches=B); expert ρ is on mass not
count (exact for equal experts); single scalar η across resources. These are now
stated as caveats in `recommended_configs.md` / `SUMMARY.md` rather than silently
baked in.

## Round 3 — serving (prefill-inclusive) throughput + theoretical ceiling

Driven by the "does it reflect the real serving throughput ceiling?" review.

- **`model/roofline.py` — `serving_roofline()`**: the theoretical output-token
  ceiling = 1/[max(decode-KV/BW, decode-compute) + prefill/O], per GPU. Makes the
  unified prefill+decode roofline explicit; `utilization()` places any operating
  point as % of ceiling. dense-70B = 150 tok/s/GPU (KV-bandwidth bound), MoE-MLA
  = 354 (prefill-compute bound); admission-bound ops ≈ 16% of ceiling. Verified:
  decode TPS converges to BW_HBM/KV at large batch.
- **`analytical.serving_tps()` + `optimize_f(objective="serving")`**: headline
  throughput now folds in the per-call prefill burst (GPU-s/call = T_prefill +
  O·TPOT/B). `scenario.py` and `parallelism.py` report SERVING throughput.
  Decode-only `tps()` kept for back-compat (it over-states on prefill-heavy work).
- **Impact:** parallelism MoE numbers drop from the old decode-only optimistic
  ~33–41k to a realistic **~11–12k serving tok/s** (consistent with DeepSeek's
  published ~14.8k/node) — a major realism correction. Single-GPU dense headline
  ≈ unchanged (admission-bound B=1 is decode-dominated, prefill is small there).
- Tests: 47/47 (added roofline convergence, offload-raises-ceiling,
  serving≤decode, serving→ceiling).

## Net effect of two improvement rounds
The model's *quantitative honesty* is now in line with its *qualitative* claims:
every headline is scoped (single-node interactive regime), labeled (optimistic
bound), and caveated (sparse accuracy cost, fabric cost, EP-not-mandatory). The
direction-level conclusions are unchanged and well-supported; the absolute
numbers are explicitly bounds, not predictions.

# Model revisions — round 3 (accuracy pass from a 3-agent review)

Independent accuracy / usability / understandability reviews. Accuracy fixes:

1. **C2C decode bandwidth: one-way, not bidirectional (A1).** Per-layer Q-out
   (GPU→CPU) and O-back (CPU→GPU) are SEQUENTIAL (O waits on CPU attention), so
   each leg uses one C2C direction → `bw_c2c_oneway`, not the bidirectional
   aggregate. The bidirectional peak only applies under a layer-ahead
   double-buffered pipeline. Fixed in `analytical.c2c_decode_time`,
   `concurrent._decode_tpot` + rate accounting, `contention`, `parallelism`.
   *Impact:* the C2C decode term was ~2× too optimistic; **headline gains
   unchanged** (1.05× dense / ~1.99× sparse) because C2C is not the binding
   resource at long context. A correctness fix, not a conclusion change.

2. **Conservative band is now a TRUE lower bound on the offload path (A4).**
   `concurrent.serving_mix` previously kept CPU/C2C as separate, fully-overlapping
   rate servers even at ov=0 — so the "conservative" end silently assumed the
   offloaded CPU attention was free, and was NOT a lower bound on the offloaded
   work. The call-rate now uses the SAME overlap structure as `_decode_tpot`:
   GPU compute∥HBM overlap by ov, and the offload path (CPU attn → C2C, serial)
   overlaps the GPU step by ov. ov=0 ⇒ sum of all four resources (genuine
   no-overlap); ov=1 ⇒ max() throughout. *Impact:* headlines unchanged (f=0 and
   optimistic best-f untouched); the conservative numbers are now honest.

3. **Corrected the "loose SLO revives gain@con" claim (A3, follow-on of A4).**
   With the offload now serialized on the rate path at ov=0, a loose latency
   budget revives gain@con **only for SPARSE offload** (≈1.04× at 1 Grace, ≈2.6×
   at 4 Grace); **DENSE stays ≈1.00× even with an unbounded SLO**, because the
   serialized ~178ms/token CPU attention dominates the call rate regardless of B.
   The old "~1.17× for dense" was an artifact of the A4 rate-overlap bug.
   `test_offload_revives_under_loose_slo` was rewritten accordingly (now also
   pins the dense-no-revival boundary). Real precondition: `high overlap OR
   (loose SLO AND sparse)`. Also documented that `gain@con≈1` in the tight-SLO
   dense regime is partly STRUCTURALLY FORCED (baseline pinned at SLO + offload
   serialized), not an independent discovery — the loose-SLO sparse case is the
   non-tautological check.

4. **C2C util on a consistent one-way basis + link-sharing throttle (A2).**
   `analytical.c2c_util` normalized decode by `bw_c2c` and append by the one-way
   bw (two bases). Both now use `bw_c2c_oneway`. The throughput throttle now
   models append competing for the RESIDUAL link after decode
   (`append_util > 1 − decode_util`), instead of throttling on append alone —
   without re-throttling decode (already in TPOT). Numerically tiny in current
   presets; a latent-correctness fix.

Tests: 72/72 after the round (one test rewritten to the corrected physics, no
new regressions). Verified-correct and explicitly NOT changed: `d_attn` FLOP
factor, MoE `1−(1−ρ)^B` weight coverage, `1/(1−f)` capacity coupling, `_combine`
overlap interpolation, band ordering, MLA-KV preset.
