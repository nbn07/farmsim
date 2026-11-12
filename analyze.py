"""Summaries and figures from results/*.csv. Usage: python3 analyze.py"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
F = Path("figures")
F.mkdir(exist_ok=True)


def per100(df):
    """Exposure per 100 changes: expected defect-bearing updates an average device receives
    for every 100 changes released (fleet shares sum to 1, so a sum of shares is a per-device count)."""
    df = df.copy()
    df["exp100"] = df["exposure_per_week"] / (df["changes"] / df["cfg_weeks"]) * 100
    return df


def ci(x):
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def main():
    vol = per100(pd.read_csv(R / "rq1_rq2_volume.csv"))
    rest = per100(pd.read_csv(R / "rq2_restocked.csv"))
    rest["policy"] = "restocked_" + rest["policy"]
    both = pd.concat([vol, rest])

    s = both.groupby(["cfg_multiplier", "policy"]).agg(
        escape_rate=("escape_rate", "mean"), escape_ci=("escape_rate", ci),
        exp100=("exp100", "mean"), exp100_ci=("exp100", ci),
        fwec=("mean_fwec", "mean"), vcr_oldest=("vcr_oldest", "mean"),
        vcr_newest=("vcr_newest", "mean")).reset_index()
    s.to_csv(R / "summary_volume.csv", index=False)

    # Table: interventions at 6x
    t = s[s.cfg_multiplier == 6].set_index("policy")
    ref = t.loc["availability", "exp100"]
    t["exposure_vs_status_quo"] = t["exp100"] / ref - 1
    keep = ["availability", "fifo", "fleet_weighted", "fleet_weighted_canary",
            "restocked_availability", "restocked_fleet_weighted_canary"]
    t = t.loc[keep, ["escape_rate", "escape_ci", "fwec", "exp100", "exp100_ci", "exposure_vs_status_quo"]]
    t.to_csv(R / "table_6x.csv")
    print(t.round(3).to_string())

    # sensitivity table at 6x
    rows = []
    for name, f, col in [("skew", "rq3_skew.csv", "cfg_fleet_skew"),
                         ("flat_q", "rq3_flat_q.csv", "cfg_q_oldest"),
                         ("detect", "rq3_detect.csv", "cfg_p_detect")]:
        d = per100(pd.read_csv(R / f))
        g = d.groupby([col, "policy"])["exp100"].mean().unstack()
        for v, r in g.iterrows():
            rows.append({"factor": name, "value": v,
                         "fleet_vs_avail": r["fleet_weighted"] / r["availability"] - 1,
                         "canary_vs_avail": r["fleet_weighted_canary"] / r["availability"] - 1})
    sens = pd.DataFrame(rows)
    sens.to_csv(R / "sensitivity_6x.csv", index=False)
    print(sens.round(3).to_string())

    # Figure 2: exposure per 100 changes vs agent multiplier
    labels = {
        "fifo": "First come, first served",
        "availability": "Test where devices are free (status quo)",
        "fleet_weighted": "Fleet-weighted allocation",
        "restocked_availability": "Farm restocked to match fleet",
        "fleet_weighted_canary": "Fleet-weighted + canary gating",
    }
    styles = {
        "fifo": dict(color="#888780", ls=":"),
        "availability": dict(color="#444441", ls="-"),
        "fleet_weighted": dict(color="#378ADD", ls="--"),
        "restocked_availability": dict(color="#BA7517", ls="-."),
        "fleet_weighted_canary": dict(color="#185FA5", ls="-", lw=2.2),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    for p, lab in labels.items():
        d = s[s.policy == p]
        ax.errorbar(d.cfg_multiplier, d.exp100, yerr=d.exp100_ci, label=lab,
                    marker="o", ms=3.5, capsize=2, **styles[p])
    ax.set_xlabel("Change volume relative to human-only baseline (×)")
    ax.set_ylabel("Defect-bearing updates per device\nper 100 changes released")
    ax.set_xticks([1, 2, 3, 4, 5, 6, 8, 10])
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(F / "fig2_exposure.png")
    fig.savefig(F / "fig2_exposure.pdf")
    print("figures written")


if __name__ == "__main__":
    main()
