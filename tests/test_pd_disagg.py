"""PD-disaggregation vs co-location (Mooncake follow-up). Locks the central,
counterintuitive finding: in a pure-THROUGHPUT fluid model, disaggregation does
NOT beat co-location/interleaving (co-location overlaps prefill-compute with
decode-HBM for free), so disagg's real value lives in latency isolation +
prefix reuse, not steady-state throughput."""
import pytest

from model.config import HardwareConfig, ModelConfig, SLOConfig, WorkloadConfig
from model.pd_disagg import (_decode_rate_per_gpu, _prefill_rate_per_gpu,
                             system_colocated, system_pd)

MODEL = ModelConfig()
HW = HardwareConfig().effective(0.5)
SLO = SLOConfig(slo_tpot=0.05)
MOONCAKE = WorkloadConfig(s_cached=3795, a_append=3795, o_output=182)
CODEX = WorkloadConfig(s_cached=64_338, a_append=3_991, o_output=520)


@pytest.mark.parametrize("work", [MOONCAKE, CODEX])
def test_disagg_does_not_beat_coloc_under_perfect_overlap(work):
    # optimistic co-location overlaps prefill(compute) with decode(HBM) for
    # free; dedicating GPUs to one phase can only lose -> disagg <= coloc.
    co = system_colocated(work, MODEL, HW, 72, slo=SLO, overlap="optimistic")
    pd = system_pd(work, MODEL, HW, 72, slo=SLO, overlap="optimistic")
    assert pd["tps"] <= co["tps"] + 1e-6


@pytest.mark.parametrize("work", [MOONCAKE, CODEX])
def test_pd_split_is_a_valid_balanced_partition(work):
    pd = system_pd(work, MODEL, HW, 72, slo=SLO, overlap="optimistic")
    assert pd["n_p"] > 0 and pd["n_d"] > 0
    assert pd["n_p"] + pd["n_d"] == pytest.approx(72.0, rel=1e-9)
    # balance: the two pools' aggregate rates are equalized (or RDMA-capped)
    if pd["binding"] == "balanced":
        assert pd["n_p"] * pd["lam_p"] == pytest.approx(pd["n_d"] * pd["lam_d"], rel=1e-6)


def test_prefill_heavy_regime_wants_more_prefill_gpus_than_decode_heavy():
    # Mooncake 720:1 should allocate a far higher prefill:decode GPU ratio than
    # the decode-heavy Codex regime (sanity on the balance direction).
    pd_m = system_pd(MOONCAKE, MODEL, HW, 72, slo=SLO, overlap="optimistic")
    pd_c = system_pd(CODEX, MODEL, HW, 72, slo=SLO, overlap="optimistic")
    assert pd_m["ratio"] > pd_c["ratio"]
