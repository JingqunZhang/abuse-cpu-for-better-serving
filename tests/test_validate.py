"""Tests for the hardware-validation harness (Phase: validation).

Run:  python tests/test_validate.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import validate as v


def test_model_never_beats_roofline_floor():
    """The model's TPOT at eta=1 must be >= the physical HBM/FLOP floor for a
    range of configs -- a hard correctness guarantee."""
    cases = v.ROOFLINE_CASES + [
        {"name": "x", "model": "dense_70b",
         "hw": {"hbm_capacity_gb": 192, "hbm_bw_tb_s": 8.0}, "b_d": 8,
         "context_tokens": 4000},
        {"name": "y", "model": {"name": "m", "layers": 32, "hidden": 4096,
         "n_heads": 32, "n_kv_heads": 8, "head_dim": 128, "p_act": 8e9,
         "p_total": 8e9}, "hw": {"hbm_bw_tb_s": 3.35, "hbm_capacity_gb": 80},
         "b_d": 1, "context_tokens": 2048},
    ]
    for c in cases:
        ok, pred, floor = v.roofline_ok(c)
        assert ok, f"{c['name']}: pred {pred*1e3:.2f}ms < floor {floor*1e3:.2f}ms"


def test_calibration_recovers_known_eta():
    """Generate points at a known eta; the calibrator should recover it closely
    and drive MAPE ~0 -- proving the fitting machinery is correct."""
    for true_eta in (0.4, 0.6, 0.8):
        pts = v._synthetic_points(true_eta=true_eta)
        eta, mape = v.calibrate_eta(pts)
        assert abs(eta - true_eta) <= 0.05, (true_eta, eta)
        assert mape < 0.05, mape


def test_leave_one_out_runs_and_is_bounded():
    """LOO on the (self-consistent) synthetic points should be near-zero error."""
    pts = v._synthetic_points(true_eta=0.6)
    loo = v.leave_one_out(pts)
    assert loo is not None
    loo_mape, errs = loo
    assert loo_mape < 0.1 and len(errs) == len(pts)


def test_joint_calibration_recovers_memory_efficiency():
    """Synthetic points generated at eta=0.6 (decode = memory-bound): the joint
    fit should recover eta_mem~0.6 and drive MAPE ~0; t_dispatch ~0."""
    pts = v._synthetic_points(true_eta=0.6)
    fit = v.calibrate_joint(pts)
    assert abs(fit["eta_mem"] - 0.6) <= 0.1, fit
    assert fit["mape"] < 0.05, fit
    assert fit["t_dispatch"] <= 1e-3, fit


def test_joint_loo_guards_small_samples():
    """Joint LOO must refuse a 3-parameter CV on too-few points."""
    pts = v._synthetic_points(true_eta=0.6)[:3]
    res = v.leave_one_out(pts, joint=True)
    assert res[0] == "insufficient"


def test_higher_bandwidth_lowers_predicted_tpot():
    """Sanity: doubling HBM bandwidth should reduce predicted decode TPOT."""
    base = {"name": "b", "model": "dense_70b",
            "hw": {"hbm_bw_tb_s": 4.0, "hbm_capacity_gb": 192}, "b_d": 1,
            "context_tokens": 8000}
    fast = dict(base); fast["hw"] = {"hbm_bw_tb_s": 8.0, "hbm_capacity_gb": 192}
    assert v.predict_tpot(fast, 0.5) < v.predict_tpot(base, 0.5)


def _run_all():
    fns = [x for k, x in globals().items() if k.startswith("test_") and callable(x)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
