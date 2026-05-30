"""Phase 5 tests for the parallelism extension.

Run:  python tests/test_parallelism.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import parallelism as par
from model.config import (HardwareConfig, ModelConfig, SLOConfig, WorkloadConfig)

MOE = ModelConfig.preset("moe_large_gqa")
HW = HardwareConfig().effective(0.5)
WORK = WorkloadConfig()


def test_sharding_reduces_per_rank_params():
    none = par.shard(MOE, par.ParallelConfig(tp=1, pp=1, ep=1))
    shed = par.shard(MOE, par.ParallelConfig(tp=8, pp=8, ep=8))
    assert shed.p_tot_rank < none.p_tot_rank / 10
    assert shed.layers_rank < none.layers_rank
    assert shed.kv_shard < none.kv_shard


def test_moe_does_not_fit_unsharded_but_fits_with_ep():
    slo = SLOConfig()
    unshard = par.best_for_config(MOE, HW, par.ParallelConfig(dp=1, tp=1, pp=1),
                                  WORK, slo, f=0.0, sparse=1.0)
    assert not unshard.fits
    ep = par.best_for_config(MOE, HW, par.ParallelConfig(dp=1, tp=8, pp=8, ep=8),
                             WORK, slo, f=0.0, sparse=1.0)
    assert ep.fits and ep.feasible


def test_pp_bubble_increases_with_pp():
    assert par.pp_bubble(par.ParallelConfig(pp=1), 8) == 1.0
    b4 = par.pp_bubble(par.ParallelConfig(pp=4, vpp=1), 8)
    b8 = par.pp_bubble(par.ParallelConfig(pp=8, vpp=1), 8)
    assert 1.0 < b4 < b8
    # VPP reduces the bubble
    bv = par.pp_bubble(par.ParallelConfig(pp=8, vpp=4), 8)
    assert bv < b8


def test_expert_tp_sharding_lets_non_ep_fit():
    """The round-2 correction: with TP sharding experts, a low-PP non-EP replica
    fits the 671B MoE; with EP-only sharding (the old assumption) it does not.
    This is the real test behind retracting 'EP mandatory' (not a tautology)."""
    slo = SLOConfig()
    p_on = par.ParallelConfig(dp=1, tp=8, pp=4, ep=1, shard_experts_with_tp=True)
    p_off = par.ParallelConfig(dp=1, tp=8, pp=4, ep=1, shard_experts_with_tp=False)
    on = par.best_for_config(MOE, HW, p_on, WORK, slo, f=0.0, sparse=1.0)
    off = par.best_for_config(MOE, HW, p_off, WORK, slo, f=0.0, sparse=1.0)
    assert on.fits and not off.fits, (on.fits, off.fits)


def test_sparse_offload_strictly_helps_optimistic_but_can_hurt_conservative():
    """Non-tautological: under optimistic overlap sparse offload STRICTLY beats
    no-offload (grows the SLO-limited batch), but the conservative (serialized)
    bound is lower -- the gain is overlap-dependent, not a free win."""
    slo = SLOConfig()
    p = par.ParallelConfig(dp=1, tp=8, pp=9)         # fits without EP
    base = par.best_for_config(MOE, HW, p, WORK, slo, f=0.0, sparse=1.0)
    off = par.best_for_config(MOE, HW, p, WORK, slo, f=0.3, sparse=0.1)
    assert off.feasible and base.feasible
    assert off.system_tps > base.system_tps * 1.02        # strict optimistic win
    assert off.system_tps_cons < off.system_tps           # conservative is lower
    # and here the conservative offload point is actually below the no-offload base
    assert off.system_tps_cons < base.system_tps


def test_total_gpu_budget_respected():
    for _, p in par.deployment_grid():
        assert p.total_gpus <= par.RACK_GPUS


def test_hardware_system_builder_and_counts():
    """HardwareConfig.system() aggregates CPU resources by cpus_per_gpu and
    sets the tunable counts; deployment_grid honors a smaller n_gpus."""
    from model.config import HardwareConfig
    hw1 = HardwareConfig.system(cpus_per_gpu=1.0, cpu_dram_bw_gb_s=500.0)
    hw4 = HardwareConfig.system(cpus_per_gpu=4.0, cpu_dram_bw_gb_s=500.0)
    assert abs(hw4.bw_cpu - 4 * hw1.bw_cpu) < 1e-3      # CPU BW scales with count
    assert abs(hw4.m_cpu - 4 * hw1.m_cpu) < 1e-3
    assert abs(hw1.bw_hbm - 8.0e12) < 1e-3              # TB/s -> bytes/s
    assert abs(hw1.m_hbm - 192e9) < 1e6                 # GB -> bytes
    # smaller rack -> no layout exceeds it; a 16-GPU rack admits none of the
    # (>=64-GPU) layouts, a 64-GPU rack admits the 64-GPU ones.
    assert par.deployment_grid(rack=16) == []
    r64 = par.deployment_grid(rack=64)
    assert len(r64) >= 1 and all(p.total_gpus <= 64 for _, p in r64)
    # effective() preserves the tunable counts
    eff = hw4.effective(0.5)
    assert eff.n_gpus == hw4.n_gpus and eff.cpus_per_gpu == 4.0


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
