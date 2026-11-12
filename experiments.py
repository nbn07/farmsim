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
    rows = []
    for cfg in configs:
        for p in policies:
            for s in SEEDS:
                rows.append(simulate(cfg, p, s))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{name}.csv", index=False)
    return df


def main():
    base = Config()
    # setup table
    setup = pd.DataFrame({
        "class": range(base.n_classes),
        "fleet_share": fleet_shares(base),
        "farm_devices_recency": farm_devices_per_class(base),
        "runs_per_week_recency": run_capacity(base),
        "farm_devices_fleet": farm_devices_per_class(replace(base, farm_mode="fleet")),
        "runs_per_week_fleet": run_capacity(replace(base, farm_mode="fleet")),
    })
    setup.to_csv(OUT / "setup.csv", index=False)

    # RQ1/RQ2: agent volume sweep, status-quo farm
    sweep("rq1_rq2_volume", [replace(base, multiplier=m) for m in MULTS])
    # RQ2b: farm restocked by fleet share
    sweep("rq2_restocked", [replace(base, multiplier=m, farm_mode="fleet") for m in MULTS])
    # RQ3: sensitivity at 6x
    sens = []
    for skew in [0.0, 0.5, 1.0, 1.5]:
        sens.append(replace(base, multiplier=6, fleet_skew=skew))
    sweep("rq3_skew", sens)
    sweep("rq3_flat_q", [replace(base, multiplier=6, q_oldest=0.325, q_newest=0.325)])
    sweep("rq3_detect", [replace(base, multiplier=6, p_detect=d) for d in [0.6, 0.75, 0.9]])
    print("done")


if __name__ == "__main__":
    main()
