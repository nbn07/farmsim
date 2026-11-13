"""Run all experiments and write results/*.csv. Usage: python3 experiments.py"""
from dataclasses import replace
from pathlib import Path
import pandas as pd

from farmsim import Config, POLICIES, simulate, fleet_shares, farm_devices_per_class, run_capacity

SEEDS = range(30)
MULTS = [1, 2, 3, 4, 5, 6, 8, 10]
OUT = Path("results")
OUT.mkdir(exist_ok=True)


def sweep(name, configs, policies=POLICIES):
    """Run every config x policy x seed. Seeds share random draws across policies
    (common random numbers), so rows with the same config and seed are paired."""
    rows = []
    for cfg in configs:
        for p in policies:
            for s in SEEDS:
                rows.append(simulate(cfg, p, s))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{name}.csv", index=False)
    return df


def both_farms(cfgs):
    """Each config with the status-quo farm and with the farm restocked to the fleet."""
    return [c for cfg in cfgs for c in (cfg, replace(cfg, farm_mode="fleet"))]


def main():
    base = Config()
    setup = pd.DataFrame({
        "class": range(base.n_classes),
        "fleet_share": fleet_shares(base),
        "farm_devices_recency": farm_devices_per_class(base),
        "runs_per_week_recency": run_capacity(base),
        "farm_devices_fleet": farm_devices_per_class(replace(base, farm_mode="fleet")),
        "runs_per_week_fleet": run_capacity(replace(base, farm_mode="fleet")),
    })
    setup.to_csv(OUT / "setup.csv", index=False)

    # RQ1/RQ2: agent volume sweep, both farm stockings, all policies
    sweep("volume", both_farms([replace(base, multiplier=m) for m in MULTS]))

    # RQ3: one-factor sensitivity at 6x, both farm stockings, all policies
    six = replace(base, multiplier=6)
    sweep("sens_skew", both_farms([replace(six, fleet_skew=v) for v in [0.0, 0.5, 1.0, 1.5]]))
    sweep("sens_flat_q", both_farms([replace(six, q_oldest=0.325, q_newest=0.325)]))
    sweep("sens_lab_detect", both_farms([replace(six, p_detect=v) for v in [0.6, 0.75, 0.9]]))
    sweep("sens_canary_detect", both_farms([replace(six, p_canary_detect=v) for v in [0.5, 0.75, 0.9, 0.95]]))
    sweep("sens_canary_waves", both_farms([replace(six, canary_waves=v) for v in [1, 2, 5, 10, 20, 1_000_000]]))
    sweep("sens_canary_frac", both_farms([replace(six, canary_frac=v) for v in [0.01, 0.02, 0.05]]))
    # restocking when the oldest model can no longer be bought: cap its units (ignored by the recency farm)
    sweep("sens_restock_cap", both_farms([replace(six, max_oldest=v) for v in [1, 2, 4, 8]]))
    print("done")


if __name__ == "__main__":
    main()
