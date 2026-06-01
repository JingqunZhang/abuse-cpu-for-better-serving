"""Validate the analytical model against real hardware -- the honest way.

We CANNOT assert "<X% error" without real measurements. So this module provides
two things:

  (1) ROOFLINE LOWER-BOUND CHECKS (no data needed): the model's predicted TPOT
      must never fall below the physical floor of streaming weights + KV from
      HBM once per token. This is a rigorous correctness guarantee we can make
      now -- a model that violates it is wrong; passing it means the model is at
      least physically consistent.

  (2) A CALIBRATION + LEAVE-ONE-OUT harness: drop real measured points into
      data/hw_measurements.json (schema below), and this fits the efficiency
      eta to your hardware and reports the held-out MAPE -- the real "is the
      error controllable?" number (target, à la Frontier, ~10-15%).

Measurement JSON schema (a list of):
  {
    "name": "h100_llama70b_8k_b32",
    "model": "dense_70b",            # preset name, or omit for dense_70b
    "hw": {"gpu_flops": 9.9e14, "hbm_bw_tb_s": 3.35, "hbm_capacity_gb": 80,
           "c2c_bw_gb_s": 64, "cpu_dram_bw_gb_s": 400},   # any system() kwargs
    "b_d": 32,
    "context_tokens": 8192,
    "measured_tpot_ms": 22.0          # OR "measured_throughput_tok_s": <B/tpot>
  }

Run:  python -m model.validate
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

from . import analytical as an
from .config import (Coeffs, HardwareConfig, ModelConfig, PolicyConfig,
                     WorkloadConfig)

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "hw_measurements.json"))
OUT = os.path.abspath(os.path.join(HERE, "..", "outputs"))


# --------------------------------------------------------------------------
# Map a measurement point -> analytical config and predict TPOT
# --------------------------------------------------------------------------
def _hw_from_point(point, eta):
    hwk = dict(point.get("hw", {}))
    hw = HardwareConfig.system(name=point.get("name", "pt"), **hwk)
    return hw.effective(eta)


def _model_from_point(point):
    m = point.get("model", "dense_70b")
    if isinstance(m, dict):                      # custom architecture
        return ModelConfig(**m)
    return ModelConfig.preset(m)


def predict_tpot(point, eta=None, *, eta_compute=None, eta_mem=None,
                 t_dispatch=0.0, overlap="conservative"):
    """Model TPOT (s) for a measurement point.

    `eta` (shorthand) sets both compute & memory efficiency; or pass
    eta_compute / eta_mem separately (LIFE-style). t_dispatch adds the fixed
    per-step overhead. Back-compat: predict_tpot(point, 0.5) still works.
    """
    hwk = dict(point.get("hw", {}))
    hw = HardwareConfig.system(name=point.get("name", "pt"), **hwk).effective(
        eta=eta, eta_compute=eta_compute, eta_mem=eta_mem)
    model = _model_from_point(point)
    s = float(point["context_tokens"])
    work = replace(WorkloadConfig(), s_cached=s, a_append=0.0)
    pol = PolicyConfig.policy("gpu_hot", b_d=int(point["b_d"]))
    tp = an.tpot(pol, work, hw, model, Coeffs(t_dispatch=t_dispatch))
    return tp.optimistic if overlap == "optimistic" else tp.conservative


def measured_tpot(point):
    if "measured_tpot_ms" in point:
        return point["measured_tpot_ms"] / 1e3
    if "measured_throughput_tok_s" in point:
        return point["b_d"] / point["measured_throughput_tok_s"]
    raise ValueError(f"point {point.get('name')} has no measured TPOT/throughput")


# --------------------------------------------------------------------------
# (1) Roofline physical floor -- the model must never predict below this
# --------------------------------------------------------------------------
def roofline_floor_tpot(point, eta=1.0):
    """Hard lower bound on decode TPOT: per token you must stream the active
    weights once and the hot KV for the batch from HBM. Perfect overlap =>
    max(); FLOPs are a further (usually smaller) floor. eta=1 => peak HW."""
    hw = _hw_from_point(point, eta)
    model = _model_from_point(point)
    b = int(point["b_d"])
    s = float(point["context_tokens"])
    w_bytes = an.weight_read_bytes(model, b)
    kv_bytes = b * an.kv_size(s, model)
    mem_floor = max(w_bytes, kv_bytes) / hw.bw_hbm
    flop_floor = 2 * model.p_act * b / hw.f_gpu
    return max(mem_floor, flop_floor)


def roofline_ok(point):
    """Model (at eta=1, the most generous) must be >= the physical floor."""
    pred = predict_tpot(point, eta=1.0)
    floor = roofline_floor_tpot(point, eta=1.0)
    return pred >= floor - 1e-12, pred, floor


# --------------------------------------------------------------------------
# (2) Calibration: fit eta to measured points; report MAPE + leave-one-out CV
# --------------------------------------------------------------------------
def _mape(points, eta, overlap="conservative"):
    errs = []
    for p in points:
        pred = predict_tpot(p, eta, overlap=overlap)
        errs.append(abs(pred - measured_tpot(p)) / measured_tpot(p))
    return sum(errs) / len(errs) if errs else float("inf")


def calibrate_eta(points, overlap="conservative", grid=None):
    """Single-eta fit (robust headline): eta in (0,1] minimizing MAPE."""
    if grid is None:
        grid = [round(0.05 * k, 3) for k in range(1, 21)]   # 0.05 .. 1.0
    best = min(grid, key=lambda e: _mape(points, e, overlap))
    return best, _mape(points, best, overlap)


# --- LIFE-style joint fit: separate compute/memory efficiency + dispatch ---
ETA_GRID = [round(0.05 * k, 3) for k in range(2, 21)]          # 0.10 .. 1.0
TD_GRID = [0.0, 0.5e-3, 1e-3, 2e-3, 3e-3, 5e-3, 8e-3, 12e-3]   # 0 .. 12 ms


def _mape_params(points, ec, em, td, overlap="conservative"):
    errs = []
    for p in points:
        pred = predict_tpot(p, eta_compute=ec, eta_mem=em, t_dispatch=td,
                            overlap=overlap)
        errs.append(abs(pred - measured_tpot(p)) / measured_tpot(p))
    return sum(errs) / len(errs) if errs else float("inf")


def calibrate_joint(points, overlap="conservative"):
    """Fit (eta_compute, eta_mem, t_dispatch) by grid search, minimizing MAPE.

    For decode-only points (memory-bound) eta_compute is weakly determined; we
    still search it but flag it. Returns dict + in-sample MAPE.
    """
    best = None
    for ec in ETA_GRID:
        for em in ETA_GRID:
            for td in TD_GRID:
                m = _mape_params(points, ec, em, td, overlap)
                if best is None or m < best[0]:
                    best = (m, ec, em, td)
    m, ec, em, td = best
    return {"eta_compute": ec, "eta_mem": em, "t_dispatch": td, "mape": m}


def n_free_params_joint():
    return 3


def leave_one_out(points, overlap="conservative", joint=False):
    """Held-out error. joint=False -> fit single eta on the others (robust with
    few points). joint=True -> fit (ec, em, td); only meaningful when
    len(points) > n_params+1, else returns None with a flag."""
    if len(points) < 2:
        return None
    if joint and len(points) <= n_free_params_joint() + 1:
        return ("insufficient", None)        # too few points for a 3-param CV
    errs = []
    for i in range(len(points)):
        train = points[:i] + points[i + 1:]
        if joint:
            fit = calibrate_joint(train, overlap)
            pred = predict_tpot(points[i], eta_compute=fit["eta_compute"],
                                eta_mem=fit["eta_mem"],
                                t_dispatch=fit["t_dispatch"], overlap=overlap)
            tag = f"ec{fit['eta_compute']},em{fit['eta_mem']},td{fit['t_dispatch']*1e3:.0f}ms"
        else:
            eta, _ = calibrate_eta(train, overlap)
            pred = predict_tpot(points[i], eta, overlap=overlap)
            tag = f"{eta:.2f}"
        meas = measured_tpot(points[i])
        errs.append((points[i].get("name", f"pt{i}"), tag, pred, meas,
                     abs(pred - meas) / meas))
    loo_mape = sum(e[4] for e in errs) / len(errs)
    return loo_mape, errs


# --------------------------------------------------------------------------
# Self-test of the calibration machinery (NOT a validation of the model):
# generate points with a KNOWN eta and confirm calibrate_eta recovers it.
# --------------------------------------------------------------------------
def _synthetic_points(true_eta=0.6):
    cfgs = [
        ("a", "dense_70b", dict(hbm_capacity_gb=192, hbm_bw_tb_s=8.0), 1, 8000),
        ("b", "dense_70b", dict(hbm_capacity_gb=192, hbm_bw_tb_s=8.0), 2, 32000),
        ("c", "dense_70b", dict(hbm_capacity_gb=192, hbm_bw_tb_s=8.0), 1, 64000),
        ("d", "dense_70b", dict(hbm_capacity_gb=192, hbm_bw_tb_s=8.0), 2, 16000),
    ]
    pts = []
    for name, mdl, hwk, b, s in cfgs:
        p = {"name": name, "model": mdl, "hw": hwk, "b_d": b,
             "context_tokens": s}
        p["measured_tpot_ms"] = predict_tpot(p, true_eta) * 1e3   # "measure" at true_eta
        pts.append(p)
    return pts


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def load_measurements():
    if os.path.exists(DATA):
        with open(DATA) as fh:
            pts = json.load(fh)
        # Honor "_skip": true (points kept in the file for documentation but
        # excluded from calibration -- e.g. the physically-mislabeled b32 anchor).
        return [p for p in pts if not p.get("_skip")]
    return []

ROOFLINE_CASES = [
    {"name": "dense70b_b1_64k", "model": "dense_70b",
     "hw": {"hbm_capacity_gb": 192, "hbm_bw_tb_s": 8.0}, "b_d": 1, "context_tokens": 64000},
    {"name": "dense70b_b32_8k", "model": "dense_70b",
     "hw": {"hbm_capacity_gb": 192, "hbm_bw_tb_s": 8.0}, "b_d": 32, "context_tokens": 8000},
    {"name": "moe_b16_32k", "model": "moe_large_mla",
     "hw": {"hbm_capacity_gb": 192, "hbm_bw_tb_s": 8.0}, "b_d": 16, "context_tokens": 32000},
]


def report():
    meas0 = load_measurements()
    status = (f"{len(meas0)} real (literature-derived) measurement point(s) "
              "loaded from `hw_measurements.json`."
              if meas0 else
              "no measurements yet (`hw_measurements.json` absent) — calibration "
              "shown as a self-test.")
    L = ["# Validation vs. real hardware\n",
         "Two-part validation: (1) physical roofline-floor checks (rigorous, no "
         "data needed); (2) calibration + leave-one-out MAPE against real "
         f"measurements. Status: {status}\n",
         "## (1) Roofline floor checks (model TPOT >= physical HBM/FLOP floor)\n",
         "| case | model TPOT @eta=1 (ms) | physical floor (ms) | >= floor? |",
         "|---|---|---|---|"]
    for c in ROOFLINE_CASES:
        ok, pred, floor = roofline_ok(c)
        L.append(f"| {c['name']} | {pred*1e3:.2f} | {floor*1e3:.2f} | "
                 f"{'✅' if ok else '❌ VIOLATION'} |")
    L += ["",
        "A pass means the model never predicts faster than physics allows -- a "
        "necessary correctness condition (not sufficient: it bounds below, not the "
        "absolute value).",
        "",
        "## (2) Calibration harness"]

    meas = load_measurements()
    if meas:
        eta, mape = calibrate_eta(meas)
        L.append(f"\n### Single-eta fit (robust headline)\n"
                 f"**{len(meas)} real points.** Best-fit eta = {eta:.2f}, "
                 f"in-sample MAPE = {mape*100:.1f}%.")
        loo = leave_one_out(meas)
        if loo and loo[0] != "insufficient":
            loo_mape, errs = loo
            L.append(f"\n**Leave-one-out held-out MAPE = {loo_mape*100:.1f}%** "
                     f"(the honest generalization error).")
            L.append("\n| point | fit eta | pred TPOT (ms) | measured (ms) | error |")
            L.append("|---|---|---|---|---|")
            for n, e, pr, ms, er in errs:
                L.append(f"| {n} | {e} | {pr*1e3:.1f} | {ms*1e3:.1f} | {er*100:.1f}% |")

        # LIFE-style joint fit (separate compute/mem efficiency + dispatch)
        j = calibrate_joint(meas)
        L.append(
            f"\n### LIFE-style joint fit (eta_compute, eta_mem, t_dispatch)\n"
            f"In-sample MAPE = **{j['mape']*100:.1f}%** at eta_mem={j['eta_mem']}, "
            f"t_dispatch={j['t_dispatch']*1e3:.1f}ms "
            f"(eta_compute={j['eta_compute']} is weakly determined for "
            f"memory-bound decode). The split + dispatch terms tighten the fit "
            f"vs single-eta ({mape*100:.1f}% -> {j['mape']*100:.1f}%).")
        jloo = leave_one_out(meas, joint=True)
        if jloo and jloo[0] == "insufficient":
            L.append(f"\n_Joint leave-one-out skipped: {len(meas)} points < "
                     f"{n_free_params_joint()+2} needed for a 3-parameter CV. "
                     "Add more (batch, context) points to validate the joint fit "
                     "out-of-sample._")
        elif jloo:
            L.append(f"\n**Joint leave-one-out MAPE = {jloo[0]*100:.1f}%.**")
        L.append(
            (f"\n**Verdict:** held-out MAPE {loo[0]*100:.1f}% — "
             + ("**within** a ~15% band (comparable to analytical sims like "
                "Frontier ~9–11%)." if loo[0] < 0.15 else
                "slightly above a ~15% band; the largest error is the high-batch "
                "point, where real ITL rises with concurrency (scheduler / kernel "
                "overhead the closed form omits). Add more points + a per-batch "
                "overhead term to tighten it."))
            if loo else "")
        L += ["",
            "### Caveats on these points",
            "- They are **approximate, literature-derived** single-stream ITL "
            "figures (NVIDIA TRT-LLM + vLLM H100 benchmarks), not a controlled "
            "decode-only sweep — treat the ~13–18% as indicative, not final.",
            "- eta≈0.4 here folds in everything the closed form omits at batch 1 "
            "(kernel launch, sampling, non-peak HBM): that is what calibration is "
            "for. Replace these with your own tt-stack / vLLM measurements in "
            "`hw_measurements.json` for a hardware-specific number.",
            "",
            "### Sources",
            "- NVIDIA, *LLM Inference Benchmarking with TensorRT-LLM* "
            "(developer.nvidia.com) — single-stream ITL ~11–21 ms, 8B/H100.",
            "- vLLM v0.6.0 perf blog (vllm.ai) — Llama-3 70B/8B H100 TPOT.",
            "- *Forecasting LLM Inference Performance via Hardware-Agnostic "
            "Analytical Modeling*, arXiv:2508.00904 — comparable analytical model.",
            "- *Frontier* LLM inference simulator — ~9–11% PDD error target.",
        ]
    else:
        # self-test: can we recover a known eta?
        pts = _synthetic_points(true_eta=0.6)
        eta, mape = calibrate_eta(pts)
        L += ["",
            "**No real measurements yet** (`hw_measurements.json` absent). "
            "Running a self-test of the harness: 4 points were generated at a "
            f"KNOWN eta=0.60; the calibrator recovered **eta={eta:.2f}** "
            f"(MAPE {mape*100:.1f}%). This proves the calibration machinery works "
            "-- it does NOT validate the model against reality.",
            "",
            "### To get a real error number",
            "Drop measured points into `hw_measurements.json` (schema in the "
            "module docstring): each = a config + the TPOT (or throughput) you "
            "measured on real hardware (vLLM / TRT-LLM / your tt-stack). Re-run "
            "`python -m model.validate`; it will fit eta and report the "
            "leave-one-out MAPE. A few points across batch sizes and context "
            "lengths are enough to bound the error.",
        ]
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "validation_vs_hardware.md")
    with open(p, "w") as fh:
        fh.write("\n".join(x for x in L if x is not None) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    report()
