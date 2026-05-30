"""Smoke + parsing tests for the scenario CLI's concurrent-mix wiring."""

import pytest

from model import scenario as sc
from model.config import HardwareConfig, ModelConfig, SLOConfig
from model import concurrent as cc


def test_parse_mix_named_and_weights():
    m = sc.parse_mix("long:0.7,short:0.3")
    assert [round(w, 3) for _, w in m] == [0.7, 0.3]
    assert m[0][0].name == "long_agentic" and m[1][0].name == "short_chat"


def test_parse_mix_default_weight_and_single():
    m = sc.parse_mix("mid")
    assert len(m) == 1 and m[0][1] == 1.0 and m[0][0].name == "mid_rag"


def test_parse_mix_rejects_unknown():
    with pytest.raises(SystemExit):
        sc.parse_mix("bogus:1.0")


def test_scenario_classes_feed_serving_mix():
    """The CLI's named classes drive serving_mix without error and obey the
    same OOM/feasibility contract."""
    hw = HardwareConfig().effective(0.5)
    classes = sc.parse_mix("long:0.7,short:0.3")
    r = cc.serving_mix(classes, ModelConfig(), hw, f=0.0, cpus_per_gpu=1.0,
                       slo=SLOConfig())
    assert r["fits"] and r["tps"] > 0


def test_parse_workload_fields_and_aliases():
    """--workload sets the named fields; aliases sc/a/o work; S is derived."""
    w = sc.parse_workload("s_cached=16000,a_append=2000,o_output=400")
    assert (w.s_cached, w.a_append, w.o_output) == (16000, 2000, 400)
    assert w.s_context == 18000          # S = S_c + A, derived
    wa = sc.parse_workload("sc=16000,a=2000,o=400")
    assert (wa.s_cached, wa.a_append, wa.o_output) == (16000, 2000, 400)


def test_parse_workload_rejects_unknown_field():
    with pytest.raises(SystemExit):
        sc.parse_workload("bogus=1")


def test_parse_mix_custom_class_resolves_to_workload():
    """A 'custom' class in --mix resolves to the --workload config so a user's
    own workload can be mixed with the presets."""
    custom = sc.parse_workload("s_cached=9000,a_append=100,o_output=50")
    m = sc.parse_mix("custom:0.8,short:0.2", custom=custom)
    assert m[0][0].s_cached == 9000 and round(m[0][1], 3) == 0.8
    # without a custom config, 'custom' is not a known class
    with pytest.raises(SystemExit):
        sc.parse_mix("custom:1.0")
