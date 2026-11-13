# farmsim

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24181448.svg)](https://doi.org/10.5281/zenodo.24181448)

A small, reproducible simulation of physical verification capacity for AI-native device software. It accompanies the article "Cheap Code, Scarce Hardware: Verifying AI-Native Device Software" (submitted to *IEEE Computer*).

## What it models

Each simulated week:

1. `N ~ Poisson(base_rate × multiplier)` changes arrive that need evidence from physical hardware. The multiplier stands for the extra change volume coding agents produce.
2. Each change carries a hardware-visible defect with probability `p_defect`. A defect shows up on each hardware class independently, with a probability that falls linearly from `q_oldest` (oldest class) to `q_newest` (newest class). Every defect affects at least one class.
3. The device farm supplies `floor(devices_k × hours_per_device / hours_per_run)` test runs per class per week.
4. An allocation policy decides which classes each change is tested on (at most one run per class per change).
5. One run on an affected class catches the defect with probability `p_detect`. A caught defect is fixed before release.
6. A defect that is not caught ships. Its *exposure* is the sum of the fleet shares of the classes it affects. Fleet shares sum to 1, so exposure equals the expected number of defect-bearing updates received by an average device.
7. With canary gating, up to `canary_waves` changes per week go through a canary wave before full rollout, starting with the changes that have the most untested fleet share. A wave sends the build to `canary_frac` of the homes in each class the change was not tested on. If the defect affects one of those classes, the wave catches it with probability `p_canary_detect` and the change is pulled from all untested classes, so only canary homes were exposed there. Classes the change was tested on get the full rollout at once. Changes that get no wave go to every class in full.
8. All randomness for a seed comes from one stream that does not depend on the policy (common random numbers). Every policy sees the same changes, defects and detection luck, so runs with the same seed are paired.

Reported metrics:

- **Missed by the lab** (`escape_rate`): share of defective changes the lab did not catch.
- **Reached the fleet** (`reached_fleet_rate`): share of defective changes that reached at least one class in full (after canary gating, if any).
- **Mean FWEC**: fleet-weighted evidence coverage per change after the lab; `mean_fwec_after_canary` also counts classes covered by a canary wave.
- **Exposure per 100 changes**: defect-bearing updates per device for every 100 changes released.
- **VCR per class**: changes per week ÷ test runs per week for that class.

## Policies

| Name | Rule |
|---|---|
| `fifo` | Changes in arrival order; each takes every class with capacity left, newest first. |
| `availability` (status quo) | Round-robin over changes, one run per change per round, on the class with the most free capacity, until no change can use any free device. |
| `fleet_weighted` | Same round-robin, but each run goes to the class with the largest fleet share that still has capacity. |
| `availability_canary` | Status quo lab plus canary gating. |
| `fleet_weighted_canary` | `fleet_weighted` plus canary gating. |

`farm_mode="recency"` stocks the farm in proportion to recency (newer boards have more devices). `farm_mode="fleet"` stocks it in proportion to fleet share; `max_oldest` caps the units of the oldest class (for a discontinued model) and spreads the rest by fleet share. Every class has at least one device.

## Default parameters (all assumptions, varied in the sensitivity runs)

| Parameter | Value | Basis |
|---|---|---|
| Hardware classes | 8 | Typical number of concurrently supported generations; assumption |
| Fleet skew | Zipf, s = 1 (oldest class ≈ 37%) | Long-lived installed base; varied 0–1.5 |
| Farm devices | 40 | Assumption; only the ratio to demand matters |
| Usable hours per device per week | 60 | Allows for reflashing, maintenance and failures |
| Device-hours per run | 4 | Flash, boot, suite and short soak; same order as published HIL evaluations |
| Baseline changes per week | 20 | Human-only rate; assumption |
| Defect probability | 0.20 | Any hardware-visible defect; assumption |
| Manifestation probability | 0.50 (oldest) to 0.15 (newest) | Older boards assumed more fragile; flat 0.325 also tested |
| Detection per run | 0.90 | Varied 0.60–0.90 |
| Canary waves per week | 5 | One per working day; varied 1 to unlimited |
| Canary fraction | 2% of a class's homes | Varied 1–5% |
| Canary detection | 0.75 | Below lab detection: field telemetry misses subtle defects; varied 0.5–0.95 |
| Weeks per run / seeds | 52 / 200 for the volume sweep, 30 for sensitivity sweeps | Paired comparisons are precise with 30 seeds; absolute levels at low volume need more |

## Reproduce

```
pip install -r requirements.txt
python3 experiments.py   # about 25 minutes; writes results/*.csv
python3 analyze.py       # writes summary_volume.csv, paired_vs_status_quo.csv, table_6x.csv, sensitivity_6x.csv, figures/
```

## Tests

```
pip install pytest
python3 -m pytest -q
```

The suite has 215 tests. They cover:

- **Setup invariants:** shares sum to 1, the farm holds exactly the configured number of devices, capacity follows its formula.
- **Policy invariants:** no class is over-booked, no capacity sits idle while a change could use it, and fair-share policies spread runs evenly.
- **Exact analytic checks:** escape rate with full testing, exposure per defect with no lab, and exposure with canary gating only.
- **Boundary cases:** no defects, perfect or zero detection.
- **Determinism and pairing:** the same seed gives the same result, and every policy sees the same changes and defects.
- **Canary limits:** zero waves equals no gating, coverage gained never exceeds one change per wave, more waves never hurt, wave selection order, the exposure rule for each defect, and exact expected exposure with limited waves.
- **Mutation testing:** 15 of 16 deliberately planted bugs are caught; the remaining one changes data on non-defective changes that is never read, so no test can observe it.
- **Randomized stress test:** 200 random configurations with range checks on every output.

## Changelog

- 1.1.0: the volume sweep uses 200 seeds (30 left the 1× baseline noisy); canary capacity is limited (`canary_waves`) and canary detection defaults to 0.75; a canary catch pulls the change from all untested classes; common random numbers make runs with the same seed paired; new metrics `reached_fleet_rate` and `mean_fwec_after_canary`; `analyze.py` writes paired comparisons with 95% CIs; the restocked farm is in every sensitivity sweep; new sweeps for canary detection, waves and size, and for a capped oldest class (`max_oldest`). These changes followed an independent review. With limited canary capacity, canary gating's benefit shrinks from the large reduction reported under 1.0.0 to a few percent at 5 waves per week.
- 1.0.0: first public release. Fair-share policies are work-conserving, so no device time is left idle while a change could still use it. An earlier internal draft left up to 20% of capacity unused at some volumes; the test suite caught this before release.

## Citation and license

Cite the archived release: version 1.1.0 is [doi:10.5281/zenodo.24204059](https://doi.org/10.5281/zenodo.24204059), and [doi:10.5281/zenodo.24181448](https://doi.org/10.5281/zenodo.24181448) always points to the latest version; `CITATION.cff` has the details, and GitHub's "Cite this repository" button uses it. The code is released under the MIT License.

## Limitations

The model is deliberately simple. Defects are independent across changes. Evidence never carries over between similar boards. Release happens weekly with no queueing across weeks. Agent-written changes and human-written changes have the same defect probability, and changes carry no risk signal, so risk-aware scheduling, change-impact analysis and batching with bisection are not modeled. Canary waves take no time in the model, so rollout delay is not measured. The results show relative effects under stated assumptions; they do not predict any particular operator's fleet.
