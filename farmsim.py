"""
farmsim: a small discrete-time simulation of physical verification capacity
for AI-native device software.

Each simulated week:
  * N changes arrive that need physical evidence (N = base_rate * multiplier).
  * A fraction of changes carry a defect. A defect manifests on a random
    subset of hardware classes; older classes are more likely to be affected.
  * The device farm has a fixed number of test runs per class per week.
  * An allocation policy decides which classes each change is tested on.
  * A test run on an affected class detects the defect with probability d.
    A detected defect is fixed before release (the change does not escape).
  * Undetected defective changes ship. Their "fleet exposure" is the fleet
    share of the classes they affect (optionally reduced by canary gating).

All parameters are in Config and are documented in README.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class Config:
    n_classes: int = 8              # hardware classes, index 0 = oldest
    fleet_skew: float = 1.0         # Zipf-like skew of fleet toward older classes
    farm_devices: int = 40          # physical devices in the farm
    hours_per_device: float = 60.0  # usable device-hours per device per week
    hours_per_run: float = 4.0      # device-hours per test run (flash, boot, suite, soak)
    base_rate: float = 20.0         # changes/week needing physical evidence, human-only baseline
    multiplier: float = 1.0         # agent-driven increase in change volume
    p_defect: float = 0.20          # probability a change carries a hardware-visible defect
    q_oldest: float = 0.50          # prob. a defect manifests on the oldest class
    q_newest: float = 0.15          # prob. a defect manifests on the newest class
    p_detect: float = 0.90          # prob. one run on an affected class detects the defect
    canary_frac: float = 0.02       # share of a class's homes in the canary group
    p_canary_detect: float = 0.95   # prob. canary group catches a defect before full rollout
    weeks: int = 52
    farm_mode: str = "recency"      # "recency": farm stocks newer boards; "fleet": stocked by fleet share


def fleet_shares(cfg: Config) -> np.ndarray:
    """Older classes hold a larger share of the installed base (rank-based Zipf)."""
    ranks = np.arange(1, cfg.n_classes + 1)  # class 0 (oldest) has rank 1
    w = 1.0 / ranks ** cfg.fleet_skew
    return w / w.sum()


def farm_devices_per_class(cfg: Config) -> np.ndarray:
    """Farms tend to stock newer hardware: device count grows with recency.
    Every class gets at least one device."""
    if cfg.farm_mode == "fleet":
        weight = fleet_shares(cfg)
    else:
        weight = np.arange(1, cfg.n_classes + 1, dtype=float)  # newest class has the largest weight
    raw = weight / weight.sum() * cfg.farm_devices
    dev = np.maximum(1, np.floor(raw)).astype(int)
    # hand out remaining devices by largest fractional remainder
    rem = raw - np.floor(raw)
    for i in np.argsort(-rem, kind="stable"):
        if dev.sum() >= cfg.farm_devices:
            break
        dev[i] += 1
    # if the minimum-of-one rule overshot, take back from the largest pools
    while dev.sum() > cfg.farm_devices:
        dev[np.argmax(dev)] -= 1
    return dev


def run_capacity(cfg: Config) -> np.ndarray:
    """Test runs per class per week."""
    return np.floor(farm_devices_per_class(cfg) * cfg.hours_per_device / cfg.hours_per_run).astype(int)


def manifest_probs(cfg: Config) -> np.ndarray:
    return np.linspace(cfg.q_oldest, cfg.q_newest, cfg.n_classes)


# ---------------------------------------------------------------- policies
# Each policy returns a boolean matrix tested[n_changes, n_classes].

def policy_fifo(n, cap, share, rng):
    """First come, first served: each change asks for every class in turn."""
    k = len(cap)
    cap = cap.copy()
    tested = np.zeros((n, k), bool)
    order = np.arange(k)[::-1]  # teams start with the newest board
    for i in range(n):
        for c in order:
            if cap[c] > 0:
                tested[i, c] = True
                cap[c] -= 1
    return tested


def _budgeted(n, cap, rank_key):
    """Shared core for the fair-share policies.

    Runs are handed out round-robin: in each round every change gets at most one
    more run, on the first class in rank_key(capacity_left) that still has
    capacity and that the change has not used yet. Rounds continue until a full
    round assigns nothing, so no device time is left idle while some change
    could still use it (work-conserving), and per-change run counts stay even.
    """
    k = len(cap)
    cap = cap.copy()
    tested = np.zeros((n, k), bool)
    if n == 0:
        return tested
    assigned = True
    while assigned and cap.sum() > 0:
        assigned = False
        for i in range(n):
            order = rank_key(cap)
            for c in order:
                if cap[c] > 0 and not tested[i, c]:
                    tested[i, c] = True
                    cap[c] -= 1
                    assigned = True
                    break
    return tested


def policy_availability(n, cap, share, rng):
    """Status quo: test where devices are free (most remaining capacity first)."""
    return _budgeted(n, cap, lambda left: np.argsort(-left, kind="stable"))


def policy_fleet(n, cap, share, rng):
    """Fleet-weighted: spend runs on the classes with the largest fleet share."""
    order = np.argsort(-share, kind="stable")
    return _budgeted(n, cap, lambda left: order)


POLICIES = {
    "fifo": (policy_fifo, False),
    "availability": (policy_availability, False),
    "fleet_weighted": (policy_fleet, False),
    "fleet_weighted_canary": (policy_fleet, True),
}


# ---------------------------------------------------------------- simulation

def simulate(cfg: Config, policy: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    share = fleet_shares(cfg)
    cap = run_capacity(cfg)
    q = manifest_probs(cfg)
    fn, canary = POLICIES[policy]
    k = cfg.n_classes

    lam = cfg.base_rate * cfg.multiplier
    tot_changes = tot_defective = tot_escaped = 0
    exposure = 0.0
    fwec_sum = 0.0
    for _ in range(cfg.weeks):
        n = rng.poisson(lam)
        tot_changes += n
        if n == 0:
            continue
        tested = fn(n, cap, share, rng)
        fwec_sum += (tested * share).sum()
        defective = rng.random(n) < cfg.p_defect
        for i in np.flatnonzero(defective):
            tot_defective += 1
            affected = rng.random(k) < q
            if not affected.any():
                affected[rng.integers(k)] = True
            runs_on_affected = tested[i] & affected
            caught = (rng.random(runs_on_affected.sum()) < cfg.p_detect).any()
            if caught:
                continue
            tot_escaped += 1
            for c in np.flatnonzero(affected):
                if canary and not tested[i, c]:
                    # untested classes are held to the canary group first
                    if rng.random() < cfg.p_canary_detect:
                        exposure += cfg.canary_frac * share[c]
                        continue
                exposure += share[c]
    demand_runs = lam * k  # every change would ideally run on every class
    with np.errstate(divide="ignore"):
        vcr = demand_runs / k / cap  # per-class: changes/week vs runs/week (inf if no capacity)
    return {
        "policy": policy,
        "seed": seed,
        **{f"cfg_{k_}": v for k_, v in asdict(cfg).items()},
        "changes": tot_changes,
        "defective": tot_defective,
        "escaped": tot_escaped,
        "escape_rate": tot_escaped / max(tot_defective, 1),
        "exposure_per_week": exposure / cfg.weeks,
        "mean_fwec": fwec_sum / max(tot_changes, 1),
        "vcr_max": float(vcr.max()),
        "vcr_oldest": float(vcr[0]),
        "vcr_newest": float(vcr[-1]),
    }
