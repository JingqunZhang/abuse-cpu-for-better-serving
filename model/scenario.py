"""Tunable scenario runner: set ALL hardware knobs on the command line and get
the analytical model's answers (single-GPU f*/TPOT/throughput, and the best
multi-GPU parallel deployment).

Every hardware feature is a flag:
  --gpus N                 number of GPUs in the system
  --gpu-flops F            per-GPU BF16 FLOP/s            (e.g. 2.5e15)
  --hbm-cap GB             per-GPU HBM capacity (GB)
  --hbm-bw TB/s            per-GPU HBM bandwidth (TB/s)
  --nvlink-bw TB/s         per-GPU NVLink bandwidth (TB/s)
  --c2c-bw GB/s            CPU<->GPU (NVLink-C2C) bandwidth, bidirectional (GB/s)
  --cpu-dram-bw GB/s       per-CPU DRAM bandwidth (GB/s)
  --cpu-flops F            per-CPU BF16 FLOP/s
  --cpu-mem GB             per-CPU memory (GB)
  --cpus-per-gpu R         CPUs serving each GPU (scales CPU bw/mem/flops)
  --eta E                  achievable efficiency fraction (0..1)
  --model NAME             dense_70b | moe_large_mla
  --sparse S               CPU attention sparsity (1.0 dense, 0.1 ScoutAttention)
  --slo-tpot S  --slo-ttft S
  --mix SPEC               concurrent workload mix for section [3], e.g.
                           'long:0.7,short:0.3' (classes: long short mid prefill
                           [+ 'custom' = the --workload config])
  --workload SPEC          model YOUR workload instead of the Codex mean, e.g.
                           's_cached=16000,a_append=2000,o_output=400' (aliases
                           sc,a,o). Drives section [1] and, unless --mix is set, [3].
  --measured-tps X         one measured co-execution throughput (tok/s) -> fits
                           ov via fit_overlap and collapses the [3] band to a
                           single calibrated gain (None if outside the band).

Section [3] runs the concurrent multi-scenario fluid model (model/concurrent.py):
prefill+decode of the mix contend for GPU-compute ∥ HBM-bw + CPU/C2C, and it
prints a one-line VERDICT, the binding-resource meaning, and the offload gain for
both overlap ends (gain@con≈1 means the win needs CPU/GPU overlap).

Example (default GB200 NVL72):           python -m model.scenario
Example (H100-ish, 4 CPUs/GPU, MoE):
  python -m model.scenario --gpus 8 --gpu-flops 2e15 --hbm-cap 80 --hbm-bw 3.35 \
      --c2c-bw 64 --cpu-dram-bw 400 --cpus-per-gpu 4 --model moe_large_mla --sparse 0.1
"""

from __future__ import annotations

import argparse

from . import analytical as an
from . import parallelism as par
from . import concurrent as cc
from .config import (Coeffs, HardwareConfig, ModelConfig, PolicyConfig,
                     SLOConfig, WorkloadConfig)

# Named workload classes for the --mix flag (share one serving cluster).
MIX_CLASSES = {
    "long": WorkloadConfig(name="long_agentic", s_cached=64_338,
                           a_append=3_991, o_output=520),   # Codex mean
    "short": WorkloadConfig(name="short_chat", s_cached=1_500,
                            a_append=500, o_output=300),
    "mid": WorkloadConfig(name="mid_rag", s_cached=16_000,
                          a_append=1_000, o_output=400),
    "prefill": WorkloadConfig(name="prefill_heavy", s_cached=68_000,
                              a_append=12_000, o_output=20),
}


def parse_workload(spec):
    """'s_cached=64000,a_append=2000,o_output=400' -> WorkloadConfig.

    Lets a user model THEIR OWN workload instead of the built-in Codex mean.
    Unspecified fields keep the WorkloadConfig default. Accepts a few aliases
    (sc/a/o) for brevity."""
    alias = {"sc": "s_cached", "a": "a_append", "o": "o_output",
             "idle": "t_idle"}
    fields = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        k = alias.get(k.strip(), k.strip())
        if k not in {"s_cached", "a_append", "o_output", "t_idle", "name"}:
            raise SystemExit(f"unknown --workload field {k!r}; use "
                             "s_cached,a_append,o_output[,t_idle] (or sc,a,o)")
        fields[k] = v.strip() if k == "name" else float(v)
    fields.setdefault("name", "custom_workload")
    return WorkloadConfig(**fields)


def parse_mix(spec, custom=None):
    """'long:0.7,short:0.3' -> [(WorkloadConfig, weight), ...].

    A class named 'custom' resolves to the --workload config (so a custom
    workload can be mixed with the presets, e.g. --mix 'custom:0.8,short:0.2')."""
    classes = dict(MIX_CLASSES)
    if custom is not None:
        classes["custom"] = custom
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, w = part.partition(":")
        name = name.strip()
        if name not in classes:
            raise SystemExit(f"unknown mix class {name!r}; "
                             f"choose from {list(classes)} or name:weight")
        out.append((classes[name], float(w) if w else 1.0))
    if not out:
        raise SystemExit("empty --mix")
    return out


def build(args):
    hw_peak = HardwareConfig.system(
        name="custom", n_gpus=args.gpus, cpus_per_gpu=args.cpus_per_gpu,
        gpu_flops=args.gpu_flops, hbm_capacity_gb=args.hbm_cap,
        hbm_bw_tb_s=args.hbm_bw, nvlink_bw_tb_s=args.nvlink_bw,
        c2c_bw_gb_s=args.c2c_bw, cpu_flops_per_cpu=args.cpu_flops,
        cpu_dram_bw_gb_s=args.cpu_dram_bw, cpu_mem_gb_per_cpu=args.cpu_mem)
    hw = hw_peak.effective(args.eta)
    model = ModelConfig.preset(args.model)
    work = parse_workload(args.workload) if args.workload else WorkloadConfig()
    slo = SLOConfig(slo_tpot=args.slo_tpot, slo_ttft=args.slo_ttft)
    return hw_peak, hw, model, work, slo


def fmt(x, unit=""):
    if x == float("inf"):
        return "inf"
    return f"{x:,.1f}{unit}"


def run(args):
    hw_peak, hw, model, work, slo = build(args)

    print("=" * 72)
    print(f"SCENARIO  model={model.name}  eta={args.eta}  sparse={args.sparse}")
    print(f"  {args.gpus} GPU(s): {fmt(args.gpu_flops/1e12,'TF')} BF16, "
          f"HBM {args.hbm_cap:.0f}GB @ {args.hbm_bw}TB/s, NVLink {args.nvlink_bw}TB/s")
    print(f"  CPU/GPU: {args.cpus_per_gpu}x [{args.cpu_dram_bw:.0f}GB/s DRAM, "
          f"{fmt(args.cpu_flops/1e12,'TF')}, {args.cpu_mem:.0f}GB]  "
          f"C2C {args.c2c_bw:.0f}GB/s")
    print(f"  effective CPU per GPU: {hw.bw_cpu/1e9:.0f}GB/s, {hw.m_cpu/1e9:.0f}GB")
    print(f"  SLO: TPOT<={slo.slo_tpot*1e3:.0f}ms  TTFT<={slo.slo_ttft:.0f}s")
    print("=" * 72)

    # ---- single-GPU analysis: optimal offload f* under the SLO ----
    wl_src = ("--workload" if args.workload else "Codex/SWEBenchPro mean call")
    print(f"\n[1] SINGLE-GPU  (workload: {wl_src} -- "
          f"Sc={work.s_cached:,.0f} A={work.a_append:,.0f} O={work.o_output:,.0f}; "
          "NOT affected by --mix)")
    print("    headline = SERVING output tok/s (prefill burst INCLUDED -- the "
          "real-world rate, not decode-only)")
    for sp_label, sp in [("no-offload baseline", 1.0), (f"sparse={args.sparse}", args.sparse)]:
        # objective="serving": optimize the prefill-inclusive throughput
        fs = an.optimize_f(work, hw, model, slo, sparse=sp, objective="serving")
        if not fs.feasible:
            print(f"  {sp_label:22s}: INFEASIBLE under SLO ({fs.reason})")
            continue
        print(f"  {sp_label:22s}: serving_output_tok/s={fs.tps:.0f}  "
              f"gain={fs.gain:.2f}x  | f*={fs.f:.2f} B_d={fs.b_d} "
              f"| per-req TPOT={fs.tpot*1e3:.1f}ms TTFT={fs.ttft:.2f}s")
    print("  (gain = serving output tokens/sec ratio; TPOT/TTFT = per-request "
          "latency, kept <= SLO)")

    # ---- multi-GPU analysis (only if >1 GPU) ----
    if args.gpus > 1:
        print(f"\n[2] MULTI-GPU  ({args.gpus} GPUs): best parallel deployments")
        results = par.run_sweep(model=model, hw=hw, work=work, slo=slo)
        feas = sorted([r for _, r in results if r.feasible],
                      key=lambda r: -r.system_tps)
        if not feas:
            print("  no feasible deployment under the SLO.")
        for r in feas[:5]:
            p = r.pcfg
            off = f"f={r.f}" if r.f > 0 else "none"
            print(f"  TP{p.tp} PP{p.pp} EP{p.ep} DP{p.dp} ({p.total_gpus}gpu) "
                  f"offload={off}: system_output_tok/s={r.system_tps:.0f}.."
                  f"{r.system_tps_cons:.0f} (opt..cons), B_d={r.b_d}, "
                  f"per-req TPOT={r.tpot*1e3:.1f}ms")
        print("  -> full table + verdicts: outputs/recommended_configs.md")

    # ---- concurrent multi-scenario (fluid two-resource contention) ----
    custom = parse_workload(args.workload) if args.workload else None
    # If the user gave --workload but left --mix at its default, model THEIR
    # workload as the (single-class) concurrent mix instead of the Codex preset.
    mix_spec = ("custom:1.0" if (custom is not None and args.mix == "long:1.0")
                else args.mix)
    classes = parse_mix(mix_spec, custom=custom)
    names = ", ".join(f"{w.name.split('_')[0]}:{p:g}" for w, p in classes)
    print(f"\n[3] CONCURRENT MIX  ({names})")
    print("    prefill+decode of the mix contend concurrently for GPU-compute ∥ "
          "HBM-bw + CPU/C2C; offload gain shown for both overlap ends.")
    # scenario hw already aggregates CPU by cpus_per_gpu (system() builder), so
    # pass cpus_per_gpu=1.0 here to avoid double-scaling.
    base = cc.serving_mix(classes, model, hw, f=0.0, sparse=1.0,
                          cpus_per_gpu=1.0, slo=slo)
    con, opt = cc.serving_band(classes, model, hw, f=0.0, sparse=1.0,
                               cpus_per_gpu=1.0, slo=slo)
    if not base["fits"]:
        print(f"  f=0: INFEASIBLE -- {base['binding']} (KV doesn't fit free HBM "
              "on one GPU; needs tensor-parallel sharding)")
        print()
        return

    rows = {}
    for sp_label, sp in [("dense", 1.0), (f"sparse={args.sparse}", args.sparse)]:
        bf_o, g_o, *_ , row_o = cc.best_f(classes, model, hw, sparse=sp,
                                          cpus_per_gpu=1.0, slo=slo,
                                          overlap="optimistic")
        _, g_c, *_ = cc.best_f(classes, model, hw, sparse=sp,
                               cpus_per_gpu=1.0, slo=slo,
                               overlap="conservative")
        rows[sp_label] = (sp, bf_o, g_o, g_c)
        print(f"  {sp_label:14s}: f=0 [{con:.0f}..{opt:.0f}] tok/s "
              f"({base['binding']}, B={base['B']:.1f}, "
              f"TPOT={base['tpot']*1e3:.0f}ms) -> best f@opt={bf_o:.2f} "
              f"gain {g_o:.2f}x(opt) / {g_c:.2f}x(con)")

    # ---- binding-resource legend (what each value means for offload) ----
    legend = {
        "gpu_hbm": "HBM-bandwidth-bound -> offload CAN help (frees HBM bw + capacity)",
        "gpu_compute": "GPU-compute-bound -> offload WON'T help (compute isn't freed)",
        "cpu_dram": "CPU-DRAM-bound -> CPU is the new bottleneck (add CPUs / more sparsity)",
        "c2c": "C2C-link-bound -> the GPU<->CPU interconnect is the new bottleneck",
    }
    print(f"  binding @ f=0: {base['binding']} = "
          f"{legend.get(base['binding'], 'see model/concurrent.py')}")

    # ---- VERDICT: state the decision in words (uses values already computed) ----
    sp, bf, g_opt, g_con = rows.get(f"sparse={args.sparse}", rows["dense"])
    if base["binding"] == "gpu_compute" or g_opt <= 1.02:
        verdict = (f"DON'T OFFLOAD -- workload is {base['binding']}; "
                   f"even perfect overlap gives only {g_opt:.2f}x.")
    else:
        if g_con < 1.03:
            precond = ("the win REQUIRES CPU/GPU overlap (no bankable gain "
                       "without a layer-ahead pipeline OR a looser SLO)")
        elif g_opt > g_con * 1.2:
            precond = (f"a {g_con:.2f}x floor is bankable without overlap; the "
                       f"rest of the upside ({g_opt:.2f}x) needs overlap")
        else:
            precond = "robust to the overlap assumption"
        verdict = (f"OFFLOAD f={bf:.2f} (sparse={sp:g}) -- {g_con:.2f}x (no "
                   f"overlap) .. {g_opt:.2f}x (perfect overlap); {precond}.")
    print(f"  VERDICT: {verdict}")

    # ---- optional calibration: collapse the band with one measured point ----
    if args.measured_tps is not None:
        ov = cc.fit_overlap(classes, model, hw, args.measured_tps,
                            f=0.0, cpus_per_gpu=1.0, slo=slo)
        if ov is None:
            print(f"  CALIBRATION: measured {args.measured_tps:.0f} tok/s is OUTSIDE "
                  f"the f=0 band [{con:.0f}..{opt:.0f}] -- overlap alone can't "
                  "explain it (check bandwidth model / admission stalls / kernel "
                  "interference), so the band is the honest diagnosis.")
        else:
            cal = cc.best_f(classes, model, hw, sparse=sp, cpus_per_gpu=1.0,
                            slo=slo, overlap=ov)[1]
            print(f"  CALIBRATION: measured {args.measured_tps:.0f} tok/s -> "
                  f"fitted ov={ov:.2f} -> calibrated offload gain {cal:.2f}x "
                  f"(single prediction, not a band).")
    print()


def main():
    d = HardwareConfig()          # defaults
    ap = argparse.ArgumentParser(description="Tunable CPU-GPU offload scenario")
    ap.add_argument("--gpus", type=int, default=72)
    ap.add_argument("--gpu-flops", type=float, default=d.f_gpu)
    ap.add_argument("--hbm-cap", type=float, default=192.0, help="GB")
    ap.add_argument("--hbm-bw", type=float, default=8.0, help="TB/s")
    ap.add_argument("--nvlink-bw", type=float, default=1.8, help="TB/s")
    ap.add_argument("--c2c-bw", type=float, default=900.0, help="GB/s (bidir)")
    ap.add_argument("--cpu-dram-bw", type=float, default=500.0, help="GB/s per CPU")
    ap.add_argument("--cpu-flops", type=float, default=2.0e12, help="FLOP/s per CPU")
    ap.add_argument("--cpu-mem", type=float, default=240.0, help="GB per CPU")
    ap.add_argument("--cpus-per-gpu", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--model", default="dense_70b",
                    choices=["dense_70b", "moe_large_mla", "moe_large_gqa"])
    ap.add_argument("--sparse", type=float, default=0.1)
    ap.add_argument("--slo-tpot", type=float, default=0.05, help="seconds")
    ap.add_argument("--slo-ttft", type=float, default=10.0, help="seconds")
    ap.add_argument("--mix", default="long:1.0",
                    help="concurrent workload mix, e.g. 'long:0.7,short:0.3' "
                         "(classes: long short mid prefill [+ custom])")
    ap.add_argument("--workload", default=None,
                    help="model YOUR workload instead of the Codex mean, e.g. "
                         "'s_cached=16000,a_append=2000,o_output=400' (aliases "
                         "sc,a,o). Drives section [1] and, unless --mix is set, [3].")
    ap.add_argument("--measured-tps", type=float, default=None,
                    help="one measured co-execution throughput (tok/s); "
                         "collapses the [3] overlap band to a single calibrated "
                         "gain via fit_overlap (None if outside the band).")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
