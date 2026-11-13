"""Tests for farmsim: invariants, edge cases and checks against exact analytic values.

Run with:  python3 -m pytest -q
"""
from dataclasses import replace
from itertools import product

import numpy as np
import pytest

import farmsim as fs
from farmsim import Config, POLICIES

BUDGETED = ["availability", "fleet_weighted"]
ALL_ALLOC = ["fifo", "availability", "fleet_weighted"]


# ------------------------------------------------------------------ setup functions

@pytest.mark.parametrize("skew", [0.0, 0.5, 1.0, 1.5, 3.0])
@pytest.mark.parametrize("k", [1, 2, 8, 20])
def test_fleet_shares_valid(skew, k):
    s = fs.fleet_shares(Config(n_classes=k, fleet_skew=skew))
    assert s.shape == (k,)
    assert np.all(s > 0)
    assert s.sum() == pytest.approx(1.0)
    assert np.all(np.diff(s) <= 1e-12)  # oldest class holds the largest share
    if skew == 0:
        assert np.allclose(s, 1 / k)


@pytest.mark.parametrize("mode", ["recency", "fleet"])
@pytest.mark.parametrize("devices", [8, 9, 13, 40, 41, 100])
@pytest.mark.parametrize("k", [1, 3, 8])
def test_farm_devices_sum_and_minimum(mode, devices, k):
    cfg = Config(n_classes=k, farm_devices=devices, farm_mode=mode)
    dev = fs.farm_devices_per_class(cfg)
    assert dev.sum() == devices
    assert dev.min() >= 1
    assert dev.dtype.kind == "i"


def test_farm_recency_grows_with_recency():
    dev = fs.farm_devices_per_class(Config())
    assert np.all(np.diff(dev) >= 0)
    assert list(dev) == [1, 2, 3, 4, 6, 7, 8, 9]  # values quoted in the paper


def test_farm_fleet_mode_follows_shares():
    cfg = Config(farm_mode="fleet")
    dev = fs.farm_devices_per_class(cfg)
    assert np.all(np.diff(dev) <= 0)
    assert dev[0] == 15


def test_run_capacity_formula():
    cfg = Config()
    cap = fs.run_capacity(cfg)
    expected = np.floor(fs.farm_devices_per_class(cfg) * 60 / 4)
    assert np.array_equal(cap, expected)
    assert cap[0] == 15  # quoted in the paper


def test_manifest_probs_range():
    q = fs.manifest_probs(Config())
    assert q[0] == pytest.approx(0.50) and q[-1] == pytest.approx(0.15)
    assert np.all(np.diff(q) < 0)


# ------------------------------------------------------------------ allocation policies

def _cap(*vals):
    return np.array(vals, dtype=int)


@pytest.mark.parametrize("policy", ALL_ALLOC)
@pytest.mark.parametrize("n", [0, 1, 5, 37, 200])
@pytest.mark.parametrize("cap", [_cap(0, 0, 0), _cap(1, 0, 5), _cap(3, 3, 3), _cap(15, 30, 45), _cap(100, 100, 100)])
def test_policy_respects_capacity(policy, n, cap):
    share = np.array([0.6, 0.3, 0.1])
    fn, _ = POLICIES[policy]
    tested = fn(n, cap, share, np.random.default_rng(0))
    assert tested.shape == (n, 3)
    assert tested.dtype == bool
    used = tested.sum(axis=0)
    assert np.all(used <= cap), (policy, used, cap)


@pytest.mark.parametrize("policy", ALL_ALLOC)
def test_policy_does_not_mutate_capacity(policy):
    cap = _cap(5, 5, 5)
    before = cap.copy()
    fn, _ = POLICIES[policy]
    fn(10, cap, np.array([0.5, 0.3, 0.2]), np.random.default_rng(0))
    assert np.array_equal(cap, before)


@pytest.mark.parametrize("policy", ALL_ALLOC)
def test_policy_full_testing_when_capacity_ample(policy):
    """With plenty of capacity, every change should be tested on every class."""
    n = 20
    cap = _cap(1000, 1000, 1000, 1000)
    fn, _ = POLICIES[policy]
    tested = fn(n, cap, np.array([0.4, 0.3, 0.2, 0.1]), np.random.default_rng(0))
    assert tested.all()


@pytest.mark.parametrize("policy", ALL_ALLOC)
@pytest.mark.parametrize("n", [7, 23, 50, 130, 133, 599])
def test_policy_uses_all_capacity_when_demand_exceeds_it(policy, n):
    """Work-conserving: if changes still want runs, no device-hour should sit idle.

    Demand on class c is n runs (one per change). Usable runs on class c are
    min(n, cap[c]). A work-conserving policy uses exactly that much."""
    cap = _cap(15, 30, 45, 60, 90, 105, 120, 135)
    share = fs.fleet_shares(Config())
    fn, _ = POLICIES[policy]
    tested = fn(n, cap, share, np.random.default_rng(0))
    assert np.array_equal(tested.sum(axis=0), np.minimum(n, cap)), (
        policy, n, tested.sum(axis=0), np.minimum(n, cap))


@pytest.mark.parametrize("policy", BUDGETED)
def test_budgeted_policies_spread_evidence(policy):
    """Round-robin: per-change run counts differ by at most one when classes are interchangeable."""
    cap = _cap(10, 10, 10)
    fn, _ = POLICIES[policy]
    tested = fn(9, cap, np.array([1 / 3] * 3), np.random.default_rng(0))
    per = tested.sum(axis=1)
    assert per.max() - per.min() <= 1


def test_fleet_weighted_prefers_large_share():
    cap = _cap(2, 2, 2)
    share = np.array([0.1, 0.7, 0.2])
    fn, _ = POLICIES["fleet_weighted"]
    tested = fn(2, cap, share, np.random.default_rng(0))
    # with 6 runs and 2 changes, each change gets every class; check first pick order instead
    tested1 = fn(6, _cap(1, 1, 1), share, np.random.default_rng(0))
    assert tested1[0, 1]  # first change takes the 0.7-share class


def test_availability_prefers_free_capacity():
    fn, _ = POLICIES["availability"]
    tested = fn(1, _cap(1, 9, 3), np.array([0.8, 0.1, 0.1]), np.random.default_rng(0))
    # budget = min(13 // 1, 3) = 3, so this change gets everything; use a tighter case
    tested = fn(13, _cap(1, 9, 3), np.array([0.8, 0.1, 0.1]), np.random.default_rng(0))
    assert tested[0, 1]  # first change goes to the class with the most free runs


def test_fifo_front_loads():
    fn, _ = POLICIES["fifo"]
    tested = fn(5, _cap(2, 2, 2), np.array([1 / 3] * 3), np.random.default_rng(0))
    assert tested[0].all() and tested[1].all()
    assert not tested[2:].any()


# ------------------------------------------------------------------ simulation: exact analytic checks

def _expected_exposure_per_defect(cfg):
    """E[sum of shares over affected classes], including the 'at least one class' rule."""
    q = fs.manifest_probs(cfg)
    share = fs.fleet_shares(cfg)
    p_none = np.prod(1 - q)
    p_aff = q + p_none / cfg.n_classes
    return float((share * p_aff).sum())


def _expected_escape_full_testing(cfg):
    """P(defect escapes) when every class is tested once: E[(1-d)^|A|], exact by enumeration."""
    q = fs.manifest_probs(cfg)
    k = cfg.n_classes
    total = 0.0
    for bits in product([0, 1], repeat=k):
        a = np.array(bits, bool)
        p = np.prod(np.where(a, q, 1 - q))
        size = a.sum() if a.any() else 1  # forced single class
        total += p * (1 - cfg.p_detect) ** size
    return total


@pytest.mark.parametrize("policy", list(POLICIES))
def test_no_capacity_means_every_defect_escapes(policy):
    cfg = Config(hours_per_device=0.0, weeks=400, multiplier=2)
    r = fs.simulate(cfg, policy, seed=1)
    assert r["mean_fwec"] == 0.0
    if not POLICIES[policy][1]:
        assert r["escape_rate"] == 1.0
        per_defect = r["exposure_per_week"] * cfg.weeks / r["defective"]
        assert per_defect == pytest.approx(_expected_exposure_per_defect(cfg), rel=0.03)


def test_full_testing_escape_rate_matches_exact_value():
    cfg = Config(hours_per_device=10_000.0, weeks=600, multiplier=1)
    r = fs.simulate(cfg, "availability", seed=3)
    assert r["mean_fwec"] == pytest.approx(1.0)
    exact = _expected_escape_full_testing(cfg)
    se = np.sqrt(exact * (1 - exact) / r["defective"])
    assert abs(r["escape_rate"] - exact) < 4 * se, (r["escape_rate"], exact)


@pytest.mark.parametrize("policy", list(POLICIES))
def test_perfect_detection_and_full_testing_means_no_escapes(policy):
    cfg = Config(hours_per_device=10_000.0, p_detect=1.0, weeks=60)
    r = fs.simulate(cfg, policy, seed=0)
    assert r["escaped"] == 0
    assert r["exposure_per_week"] == 0.0


def test_zero_detection_means_every_defect_escapes():
    cfg = Config(hours_per_device=10_000.0, p_detect=0.0, weeks=60)
    r = fs.simulate(cfg, "availability", seed=0)
    assert r["escape_rate"] == 1.0


def test_no_defects_means_no_exposure():
    r = fs.simulate(Config(p_defect=0.0, multiplier=5), "fifo", seed=0)
    assert r["defective"] == 0 and r["exposure_per_week"] == 0.0


def test_canary_perfect_and_tiny_removes_untested_exposure():
    """With no lab capacity, a perfect canary of zero homes leaves zero exposure."""
    cfg = Config(hours_per_device=0.0, canary_frac=0.0, p_canary_detect=1.0, canary_waves=10**6, weeks=100)
    r = fs.simulate(cfg, "fleet_weighted_canary", seed=0)
    assert r["exposure_per_week"] == 0.0


def test_canary_exact_value_without_lab():
    """No lab: every affected class goes through canary. Exposure per defect =
    E[sum share_k * (pc*f + (1-pc)*1)] over affected k."""
    cfg = Config(hours_per_device=0.0, weeks=600, multiplier=2, canary_waves=10**6)
    r = fs.simulate(cfg, "fleet_weighted_canary", seed=5)
    factor = cfg.p_canary_detect * cfg.canary_frac + (1 - cfg.p_canary_detect)
    expected = _expected_exposure_per_defect(cfg) * factor
    per_defect = r["exposure_per_week"] * cfg.weeks / r["defective"]
    assert per_defect == pytest.approx(expected, rel=0.06)


def test_canary_never_worse_than_same_policy_without_it():
    for m in [1, 4, 8]:
        a = fs.simulate(Config(multiplier=m), "fleet_weighted", seed=11)
        b = fs.simulate(Config(multiplier=m), "fleet_weighted_canary", seed=11)
        # same seed and same allocation: identical lab outcome, canary can only reduce exposure
        assert b["exposure_per_week"] <= a["exposure_per_week"] + 1e-12


def test_change_arrivals_are_poisson_mean():
    cfg = Config(multiplier=3, weeks=500)
    r = fs.simulate(cfg, "fifo", seed=2)
    mean = r["changes"] / cfg.weeks
    assert mean == pytest.approx(60, rel=0.03)


def test_defect_rate():
    cfg = Config(multiplier=3, weeks=500)
    r = fs.simulate(cfg, "fifo", seed=2)
    assert r["defective"] / r["changes"] == pytest.approx(0.20, abs=0.01)


def test_vcr_formula():
    r = fs.simulate(Config(multiplier=6, weeks=2), "fifo", seed=0)
    assert r["vcr_oldest"] == pytest.approx(120 / 15)
    assert r["vcr_newest"] == pytest.approx(120 / 135)


def test_reproducible_with_same_seed():
    a = fs.simulate(Config(multiplier=4), "availability", seed=7)
    b = fs.simulate(Config(multiplier=4), "availability", seed=7)
    assert a == b


def test_different_seeds_differ():
    a = fs.simulate(Config(multiplier=4), "availability", seed=7)
    b = fs.simulate(Config(multiplier=4), "availability", seed=8)
    assert a["escaped"] != b["escaped"] or a["changes"] != b["changes"]


def test_exposure_monotone_in_volume_for_status_quo():
    prev = -1
    for m in [1, 2, 4, 6, 8, 10]:
        vals = [fs.simulate(Config(multiplier=m), "availability", s)["escape_rate"] for s in range(5)]
        cur = float(np.mean(vals))
        assert cur > prev
        prev = cur


def test_more_devices_never_hurts_on_average():
    small = np.mean([fs.simulate(Config(multiplier=6), "availability", s)["exposure_per_week"] for s in range(8)])
    big = np.mean([fs.simulate(Config(multiplier=6, farm_devices=80), "availability", s)["exposure_per_week"] for s in range(8)])
    assert big < small


# ------------------------------------------------------------------ randomized stress test

def test_random_configs_stay_in_range():
    rng = np.random.default_rng(2026)
    for trial in range(200):
        k = int(rng.integers(1, 12))
        cfg = Config(
            n_classes=k,
            fleet_skew=float(rng.uniform(0, 2)),
            farm_devices=int(rng.integers(k, 80)),
            hours_per_device=float(rng.uniform(0, 120)),
            hours_per_run=float(rng.uniform(0.5, 10)),
            base_rate=float(rng.uniform(0, 40)),
            multiplier=float(rng.uniform(0.5, 12)),
            p_defect=float(rng.uniform(0, 1)),
            q_oldest=float(rng.uniform(0, 1)),
            q_newest=float(rng.uniform(0, 1)),
            p_detect=float(rng.uniform(0, 1)),
            canary_frac=float(rng.uniform(0, 0.2)),
            p_canary_detect=float(rng.uniform(0, 1)),
            weeks=int(rng.integers(1, 15)),
            farm_mode=str(rng.choice(["recency", "fleet"])),
        )
        for p in POLICIES:
            r = fs.simulate(cfg, p, seed=trial)
            assert 0 <= r["escaped"] <= r["defective"] <= r["changes"]
            assert 0.0 <= r["escape_rate"] <= 1.0
            assert 0.0 <= r["mean_fwec"] <= 1.0 + 1e-9
            # each escaped defect can expose at most the whole fleet once
            assert 0.0 <= r["exposure_per_week"] * cfg.weeks <= r["escaped"] + 1e-9


# ------------------------------------------------------------------ 1.1.0: paired draws and limited canary capacity

@pytest.mark.parametrize("m", [1, 4, 8])
def test_common_random_numbers_across_policies(m):
    """Every policy sees the same changes and defects for the same seed."""
    runs = [fs.simulate(Config(multiplier=m, weeks=20), p, seed=4) for p in POLICIES]
    assert len({r["changes"] for r in runs}) == 1
    assert len({r["defective"] for r in runs}) == 1


@pytest.mark.parametrize("m", [1, 6])
def test_zero_canary_waves_equals_no_gating(m):
    for base, gated in [("fleet_weighted", "fleet_weighted_canary"), ("availability", "availability_canary")]:
        a = fs.simulate(Config(multiplier=m, canary_waves=0), base, seed=9)
        b = fs.simulate(Config(multiplier=m, canary_waves=0), gated, seed=9)
        assert a["exposure_per_week"] == pytest.approx(b["exposure_per_week"])
        assert a["escape_rate"] == b["escape_rate"]


@pytest.mark.parametrize("waves", [0, 1, 5, 20])
def test_canary_waves_limit_coverage_gain(waves):
    """Canary can add at most one change's untested share per wave per week."""
    cfg = Config(multiplier=6, canary_waves=waves, weeks=30)
    r = fs.simulate(cfg, "fleet_weighted_canary", seed=1)
    gain = (r["mean_fwec_after_canary"] - r["mean_fwec"]) * r["changes"]
    assert gain <= waves * cfg.weeks + 1e-9
    assert r["mean_fwec_after_canary"] >= r["mean_fwec"] - 1e-12


def test_more_canary_waves_never_hurt_same_seed():
    prev = None
    for waves in [0, 1, 5, 20, 10**6]:
        e = fs.simulate(Config(multiplier=6, canary_waves=waves), "fleet_weighted_canary", seed=2)["exposure_per_week"]
        if prev is not None:
            assert e <= prev + 1e-12
        prev = e


def test_reached_fleet_never_exceeds_missed_by_lab():
    for p in POLICIES:
        r = fs.simulate(Config(multiplier=6), p, seed=3)
        assert r["reached_fleet_rate"] <= r["escape_rate"] + 1e-12
        if not POLICIES[p][1]:
            assert r["reached_fleet_rate"] == pytest.approx(r["escape_rate"])


def test_canary_sees_only_untested_classes():
    """If the lab tests every class, canary gating changes nothing."""
    cfg = Config(hours_per_device=10_000.0, weeks=60)
    a = fs.simulate(cfg, "fleet_weighted", seed=0)
    b = fs.simulate(cfg, "fleet_weighted_canary", seed=0)
    assert a["exposure_per_week"] == pytest.approx(b["exposure_per_week"])


@pytest.mark.parametrize("cap", [1, 2, 4, 8, 100])
def test_max_oldest_caps_and_conserves_devices(cap):
    cfg = Config(farm_mode="fleet", max_oldest=cap)
    dev = fs.farm_devices_per_class(cfg)
    assert dev.sum() == cfg.farm_devices
    assert dev[0] == min(cap, fs.farm_devices_per_class(Config(farm_mode="fleet"))[0])
    assert dev.min() >= 1


def test_max_oldest_ignored_in_recency_mode():
    assert np.array_equal(fs.farm_devices_per_class(Config(max_oldest=2)), fs.farm_devices_per_class(Config()))
