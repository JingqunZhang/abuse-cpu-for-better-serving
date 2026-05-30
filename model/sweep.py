"""Phase 2 sweep driver: produce outputs/sweep_results.csv and the four plots.

Sweeps (section 2 of the plan, "Sweep"):
    f   in {0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0}
    B_d in {8, 16, 32, 64, 128, 256}
    r   in {0.25, 0.5, 1.0}
    eta in {0.3, 0.5, 0.7}

Run:  python -m model.sweep
"""

from __future__ import annotations

import csv
import os

from . import analytical as an
from .config import (Coeffs, HardwareConfig, ModelConfig, PolicyConfig,
                     WorkloadConfig)

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
OUT = os.path.abspath(OUT)

F_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
BD_GRID = [8, 16, 32, 64, 128, 256]
R_GRID = [0.25, 0.5, 1.0]
ETA_GRID = [0.3, 0.5, 0.7]
SPARSE_GRID = [1.0, 0.1]   # dense CPU attn vs ScoutAttention-style 10% blocks


def run_sweep():
    from dataclasses import replace
    work = WorkloadConfig()
    model = ModelConfig()
    hw0 = HardwareConfig()
    coeffs = Coeffs()
    rows = []

    for eta in ETA_GRID:
        hw = hw0.effective(eta)
        for sparse in SPARSE_GRID:
            for r in R_GRID:
                for b_d in BD_GRID:
                    base = PolicyConfig.policy("gpu_hot", r=r, b_d=b_d)
                    base_tps = an.tps(base, work, hw, model, coeffs).tps
                    for f in F_GRID:
                        kind = ("gpu_hot" if f == 0 else
                                "full_cpu_attn" if f == 1.0 else "partial_cpu_attn")
                        pol = PolicyConfig.policy(kind, f=f, r=r, b_d=b_d,
                                                  sparse=sparse)
                        th = an.tps(pol, work, hw, model, coeffs)
                        # Report all per-step metrics at the batch actually run.
                        pol_eff = replace(pol, b_d=th.batch)
                        tp = an.tpot(pol_eff, work, hw, model, coeffs)
                        hbm = an.hbm_capacity(pol_eff, work, hw, model, coeffs)
                        cu = an.c2c_util(pol_eff, work, hw, model, coeffs)
                        appt = an.append_time(pol_eff, work, hw, model, coeffs)
                        rows.append(dict(
                            eta=eta, sparse=sparse, r=r, b_d=b_d, f=f, policy=kind,
                            tps=th.tps, gain=(th.tps / base_tps if base_tps else 0),
                            bound=th.bound, b_eff=th.batch,
                            tpot_opt=tp.optimistic, tpot_cons=tp.conservative,
                            gpu_nonattn=tp.gpu_nonattn, gpu_attn=tp.gpu_attn,
                            cpu_attn=tp.cpu_attn, c2c_decode=tp.c2c,
                            append_total=appt.total, append_oldkv=appt.old_kv_load,
                            append_flush=appt.new_kv_flush,
                            hbm_used_gb=hbm.used / 1e9, hbm_fits=hbm.fits,
                            c2c_util=cu["util"], decode_batch_cap=th.batch_cap,
                        ))

    os.makedirs(OUT, exist_ok=True)
    csv_path = os.path.join(OUT, "sweep_results.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}  ({len(rows)} rows)")
    return rows, (work, model, hw0, coeffs)


def make_plots(rows, ctx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reference slice: eta=0.5, r=1.0, B_d=256 (target large enough that HBM
    # capacity binds, so offload's batch headroom is exercised).
    def slice_for(sparse):
        s = [r for r in rows if r["eta"] == 0.5 and r["r"] == 1.0
             and r["b_d"] == 256 and r["sparse"] == sparse]
        s.sort(key=lambda x: x["f"])
        return s

    dense = slice_for(1.0)
    sp = slice_for(0.1)
    fs = [r["f"] for r in dense]

    def save_overlay(name, key, ylabel, title, scale=1.0, baseline=None, log=False):
        plt.figure(figsize=(6.2, 4))
        plt.plot(fs, [r[key] * scale for r in dense], "o-", label="dense CPU attn (sparse=1.0)")
        plt.plot(fs, [r[key] * scale for r in sp], "s-", label="sparse CPU attn (sparse=0.1)")
        if baseline is not None:
            plt.axhline(baseline, ls="--", c="gray", label="baseline f=0")
        if log:
            plt.yscale("log")
        plt.xlabel("CPU attention fraction f")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(OUT, name)
        plt.savefig(p, dpi=120)
        plt.close()
        print(f"wrote {p}")

    save_overlay("gain_vs_f.png", "gain", "Gain = TPS(f)/TPS(0)",
                 "Throughput gain vs f  (eta=0.5, r=1, B_d_target=256)",
                 baseline=1.0)
    save_overlay("tpot_vs_f.png", "tpot_opt", "TPOT (ms)",
                 "Time-per-output-token vs f (optimistic overlap)", scale=1e3, log=True)
    save_overlay("cpu_bw_util_vs_f.png", "c2c_util", "C2C utilization",
                 "CPU<->GPU (C2C) bandwidth utilization vs f", baseline=1.0)
    save_overlay("hbm_capacity_vs_f.png", "decode_batch_cap",
                 "Max decode batch that fits HBM",
                 "HBM-limited decode batch capacity vs f")


if __name__ == "__main__":
    rows, ctx = run_sweep()
    make_plots(rows, ctx)
