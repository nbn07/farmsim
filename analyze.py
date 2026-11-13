"""Summaries, paired comparisons and figures from results/*.csv. Usage: python3 analyze.py"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
F = Path("figures")
F.mkdir(exist_ok=True)

STATUS_QUO = "recency/availability"
CONFIGS = {  # label used in tables -> (farm_mode, policy)
    "recency/availability": "Status quo: test where devices are free",
    "recency/fifo": "First come, first served",
    "recency/fleet_weighted": "Fleet-weighted allocation",
    "fleet/availability": "Farm restocked to match the fleet",
    "recency/availability_canary": "Canary gating (5 waves/week)",
    "fleet/availability_canary": "Restocked farm + canary gating",
}


def exp100(df):
    """Exposure per 100 changes: defect-bearing updates an average device receives for
    every 100 changes released (fleet shares sum to 1, so a sum of shares counts
    updates per device)."""
    return df["exposure_per_week"] / (df["changes"] / df["cfg_weeks"]) * 100


def prep(df):
    df = df.copy()
    df["exp100"] = exp100(df)
    df["config"] = df["cfg_farm_mode"] + "/" + df["policy"]
    return df


def ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def paired_vs(df, keys, ref=STATUS_QUO, metric="exp100"):
    """Paired relative change vs the reference config, per seed (common random numbers).
    Returns mean relative change and its 95% CI for every config."""
    rows = []
    for k, g in df.groupby(keys):
        piv = g.pivot_table(index="seed", columns="config", values=metric)
        base = piv[ref]
        for c in piv.columns:
            rel = (piv[c] - base) / base.mean()
            rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))),
                         "config": c, "rel_change": rel.mean(),
                         "ci_low": rel.mean() - ci(rel), "ci_high": rel.mean() + ci(rel)})
    return pd.DataFrame(rows)


def main():
    vol = prep(pd.read_csv(R / "volume.csv"))
    s = vol.groupby(["cfg_multiplier", "config"]).agg(
        missed_by_lab=("escape_rate", "mean"), reached_fleet=("reached_fleet_rate", "mean"),
        exp100=("exp100", "mean"), exp100_ci=("exp100", ci),
        fwec=("mean_fwec", "mean"), fwec_after_canary=("mean_fwec_after_canary", "mean"),
        vcr_oldest=("vcr_oldest", "mean"), vcr_newest=("vcr_newest", "mean")).reset_index()
    s.to_csv(R / "summary_volume.csv", index=False)

    paired = paired_vs(vol, ["cfg_multiplier"])
    paired.to_csv(R / "paired_vs_status_quo.csv", index=False)

    t = s[s.cfg_multiplier == 6].set_index("config").loc[list(CONFIGS)]
    p6 = paired[paired.cfg_multiplier == 6].set_index("config").loc[list(CONFIGS)]
    t = t.join(p6[["rel_change", "ci_low", "ci_high"]])
    t.insert(0, "label", [CONFIGS[c] for c in t.index])
    t.to_csv(R / "table_6x.csv")
    print(t[["label", "missed_by_lab", "reached_fleet", "fwec", "fwec_after_canary", "exp100", "exp100_ci",
             "rel_change", "ci_low", "ci_high"]].round(3).to_string())

    # sensitivity: paired relative change vs status quo for each factor value
    sens = []
    for name, col in [("skew", "cfg_fleet_skew"), ("flat_q", "cfg_q_oldest"), ("lab_detect", "cfg_p_detect"),
                      ("canary_detect", "cfg_p_canary_detect"), ("canary_waves", "cfg_canary_waves"),
                      ("canary_frac", "cfg_canary_frac"), ("restock_cap", "cfg_max_oldest")]:
        d = prep(pd.read_csv(R / f"sens_{name}.csv"))
        pv = paired_vs(d, [col])
        pv.insert(0, "factor", name)
        pv = pv.rename(columns={col: "value"})
        sens.append(pv)
    sens = pd.concat(sens)
    sens.to_csv(R / "sensitivity_6x.csv", index=False)
    wide = sens.pivot_table(index=["factor", "value"], columns="config", values="rel_change")
    print(wide[[c for c in CONFIGS if c != STATUS_QUO]].round(3).to_string())

    # Figure 2: exposure per 100 changes vs agent multiplier
    styles = {
        "recency/fifo": dict(color="#888780", ls=":"),
        "recency/availability": dict(color="#444441", ls="-"),
        "recency/fleet_weighted": dict(color="#378ADD", ls="--"),
        "fleet/availability": dict(color="#BA7517", ls="-."),
        "recency/availability_canary": dict(color="#185FA5", ls="-", lw=2.0),
        "fleet/availability_canary": dict(color="#0F6E56", ls="-", lw=2.0, marker="s"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    for c, lab in CONFIGS.items():
        d = s[s.config == c]
        st = dict(marker="o", ms=3.5)
        st.update(styles[c])
        ax.errorbar(d.cfg_multiplier, d.exp100, yerr=d.exp100_ci, label=lab, capsize=2, **st)
    ax.set_xlabel("Change volume relative to human-only baseline (×)")
    ax.set_ylabel("Defect-bearing updates per home\nper 100 changes released")
    ax.set_xticks([1, 2, 3, 4, 5, 6, 8, 10])
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(F / "fig2_exposure.png")
    fig.savefig(F / "fig2_exposure.pdf", metadata={"CreationDate": None, "ModDate": None})
    print("figures written")


if __name__ == "__main__":
    main()
