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
    share of the classes they affect (optionally reduced by canary gating,
    which has a limited number of canary waves per week).
  * All randomness comes from one policy-independent stream per seed, so
    every policy sees the same changes, defects and detection luck.

All parameters are in Config and are documented in README.md.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
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
    p_canary_detect: float = 0.75   # prob. a canary wave catches a defect it can see (below lab detection)
    canary_waves: int = 5           # canary waves per week (one change per wave); limits canary capacity
    weeks: int = 52
    farm_mode: str = "recency"      # "recency": farm stocks newer boards; "fleet": stocked by fleet share
    max_oldest: int = 0             # fleet mode only: cap on units of the oldest class (0 = no cap),
                                    # e.g. because the model is discontinued; the rest is spread by share


def fleet_shares(cfg: Config) -> np.ndarray:
    """Older classes hold a larger share of the installed base (rank-based Zipf)."""
    ranks = np.arange(1, cfg.n_classes + 1)  # class 0 (oldest) has rank 1
    w = 1.0 / ranks ** cfg.fleet_skew
    return w / w.sum()


def _allocate(weight: np.ndarray, total: int) -> np.ndarray:
    """Split `total` devices in proportion to `weight`, at least one per class,
    remaining units by largest fractional remainder."""
    raw = weight / weight.sum() * total
    dev = np.maximum(1, np.floor(raw)).astype(int)
    rem = raw - np.floor(raw)
    for i in np.argsort(-rem, kind="stable"):
        if dev.sum() >= total:
            break
        dev[i] += 1
    # if the minimum-of-one rule overshot, take back from the largest pools
    while dev.sum() > total:
        dev[np.argmax(dev)] -= 1
    return dev


def farm_devices_per_class(cfg: Config) -> np.ndarray:
    """Devices per hardware class.

    recency: farms tend to stock newer hardware, so device count grows with recency.
    fleet: stocked in proportion to fleet share. With max_oldest > 0 the oldest class
    is capped (e.g. a discontinued model) and the remaining devices are split over the
    other classes by fleet share. Every class gets at least one device."""
    if cfg.farm_mode == "fleet":
        share = fleet_shares(cfg)
        if cfg.max_oldest > 0:
            full = _allocate(share, cfg.farm_devices)
            oldest = max(1, min(cfg.max_oldest, int(full[0])))
            rest = _allocate(share[1:], cfg.farm_devices - oldest)
            return np.concatenate([[oldest], rest]).astype(int)
        return _allocate(share, cfg.farm_devices)
    return _allocate(np.arange(1, cfg.n_classes + 1, dtype=float), cfg.farm_devices)


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
    # name: (lab allocation, canary gating on?)
    "fifo": (policy_fifo, False),
    "availability": (policy_availability, False),
    "fleet_weighted": (policy_fleet, False),
    "availability_canary": (policy_availability, True),
    "fleet_weighted_canary": (policy_fleet, True),
}


# ---------------------------------------------------------------- simulation

def _draw_week(rng_env, cfg, q):
    """Everything random about one week, drawn from a stream that does not depend on
    the policy. Every policy therefore faces the same changes, the same defects and
    the same detection luck (common random numbers), so comparisons are truly paired."""
    k = cfg.n_classes
    n = rng_env.poisson(cfg.base_rate * cfg.multiplier)
    defective = rng_env.random(n) < cfg.p_defect
    affected = rng_env.random((n, k)) < q
    forced = rng_env.integers(k, size=n)
    none = ~affected.any(axis=1)
    affected[none, forced[none]] = True          # every defect affects at least one class
    affected &= defective[:, None]                # non-defective changes affect nothing
    u_lab = rng_env.random((n, k))                # detection luck for a run on (change, class)
    u_canary = rng_env.random(n)                  # detection luck for the change's canary wave
    return n, defective, affected, u_lab, u_canary


def simulate(cfg: Config, policy: str, seed: int) -> dict:
    """Simulate cfg.weeks weeks.

    Lab: a run on an affected class catches the defect with probability p_detect;
    a lab catch fixes the change everywhere before release.

    Canary gating (policies ending in _canary): after the lab, up to canary_waves
    changes per week are sent first to a canary group (canary_frac of the homes of
    each class the change was NOT tested on); changes with the most untested fleet
    share go first. Classes the change WAS tested on get the full rollout at once.
    A canary wave sees the defect only if it affects at least one untested class;
    it then catches it with probability p_canary_detect, and the change is pulled
    from all untested classes, so only canary homes were exposed there. Changes
    that get no canary wave go to every untested class in full, as without gating.

    Exposure of a shipped defect = sum of the fleet shares of the affected classes
    it reached in full (plus canary_frac x share for canary-only exposure). Shares
    sum to 1, so exposure counts defect-bearing updates per average device.
    """
    rng_env = np.random.default_rng([seed, 0])
    share = fleet_shares(cfg)
    cap = run_capacity(cfg)
    q = manifest_probs(cfg)
    fn, canary = POLICIES[policy]

    tot_changes = tot_defective = missed_by_lab = reached_fleet = 0
    exposure = fwec_lab = fwec_after = 0.0
    for _ in range(cfg.weeks):
        n, defective, affected, u_lab, u_canary = _draw_week(rng_env, cfg, q)
        tot_changes += n
        tot_defective += int(defective.sum())
        if n == 0:
            continue
        tested = fn(n, cap, share, None)
        untested_share = ((~tested) * share).sum(axis=1)
        fwec_lab += (tested * share).sum()

        in_canary = np.zeros(n, bool)
        if canary and cfg.canary_waves > 0:
            order = np.argsort(-untested_share, kind="stable")
            chosen = [i for i in order if untested_share[i] > 0][: cfg.canary_waves]
            in_canary[chosen] = True
        fwec_after += (tested * share).sum() + untested_share[in_canary].sum()

        lab_catch = (tested & affected & (u_lab < cfg.p_detect)).any(axis=1)
        for i in np.flatnonzero(defective & ~lab_catch):
            missed_by_lab += 1
            hit_tested = affected[i] & tested[i]
            hit_untested = affected[i] & ~tested[i]
            exp_i = (share * hit_tested).sum()
            if in_canary[i] and hit_untested.any() and u_canary[i] < cfg.p_canary_detect:
                exp_i += cfg.canary_frac * (share * hit_untested).sum()
                reached = hit_tested.any()
            else:
                exp_i += (share * hit_untested).sum()
                reached = True
            exposure += exp_i
            reached_fleet += int(reached)

    lam = cfg.base_rate * cfg.multiplier
    with np.errstate(divide="ignore"):
        vcr = lam / cap  # per class: changes/week needing evidence vs runs/week (inf if no capacity)
    return {
        "policy": policy,
        "seed": seed,
        **{f"cfg_{k_}": v for k_, v in asdict(cfg).items()},
        "changes": tot_changes,
        "defective": tot_defective,
        "escaped": missed_by_lab,                       # kept for compatibility: missed by the lab
        "escape_rate": missed_by_lab / max(tot_defective, 1),
        "reached_fleet_rate": reached_fleet / max(tot_defective, 1),
        "exposure_per_week": exposure / cfg.weeks,
        "mean_fwec": fwec_lab / max(tot_changes, 1),
        "mean_fwec_after_canary": fwec_after / max(tot_changes, 1),
        "vcr_max": float(vcr.max()),
        "vcr_oldest": float(vcr[0]),
        "vcr_newest": float(vcr[-1]),
    }
