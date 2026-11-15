"""Fill the results section of README.md from results/*.json."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
NAME = {
    "linear": "Linear",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "doubleensemble": "DoubleEnsemble",
    "mlp": "MLP",
    "gru": "GRU",
    "lstm": "LSTM",
    "alstm": "ALSTM",
}
ORDER = list(NAME)


def main() -> None:
    ic = json.loads((RES / "ic_audit.json").read_text())
    bt = json.loads((RES / "backtest_audit.json").read_text())
    rep, dec, st, tests = ic["reproduction"], ic["decay"], ic["staleness"], ic["tests"]
    ab, pr = ic["ablation"], ic["predictions"]
    ens, null = bt["ensembles"], bt["null"]
    fe = json.loads((RES / "feature_audit.json").read_text())["strong_mean_aligned"]
    drop1 = 100 * (1 - fe["published"] / fe["train"])
    drop2 = 100 * (1 - fe["post"] / fe["published"])
    models = [m for m in ORDER if m in dec["models"]]

    rows = [
        "| Model | Label | Published table | Ours, 2017–07/2020 | Ours, 08/2020–11/2026 | Kept | Refit, 2023–2026 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    rank_label = set(ic["label_split"]["rank_models"])
    for m in sorted(models, key=lambda m: -dec["models"][m]["post"]):
        d = dec["models"][m]
        r = rep["models"][m]
        refit = st.get(m, {}).get("refit")
        rows.append(
            f"| {NAME[m]} | {'ranked' if m in rank_label else 'z-score'} | {r['pub_rank_ic']:.4f} | {d['published']:.4f} | "
            f"{d['post']:.4f} | {100 * d['post'] / d['published']:.0f}% | {'–' if refit is None else f'{refit:.4f}'} |"
        )
    table = "\n".join(rows)

    lgb = ab["lightgbm_rank"]
    kept = ic["label_split"]["kept"]
    kr = [kept[m] for m in ic["label_split"]["rank_models"]]
    kz = [kept[m] for m in ic["label_split"]["z_models"]]
    tau = ic["rank_stability"]["ours_published_vs_ours_post"]["tau"]
    mcs_pub = ", ".join(NAME[m] for m in ORDER if m in tests["pub/published"]["mcs_included"])
    mcs_post = ", ".join(NAME[m] for m in ORDER if m in tests["pub/post"]["mcs_included"])
    mcs_rec = ", ".join(NAME[m] for m in ORDER if m in tests["refit/recent"]["mcs_included"])
    pubs = [ens["pub"][m]["published"]["annual_excess"] for m in models if m in ens["pub"]]
    posts = [ens["pub"][m]["post"]["annual_excess"] for m in models if m in ens["pub"]]
    above_pub = sum(pubs) / len(pubs) - null["published"]["annual_excess"]
    above_post = sum(posts) / len(posts) - null["post"]["annual_excess"]

    text = f"""Rank IC is the mean over days of the cross-sectional Spearman correlation between the score and the return from the next close to the one after (Qlib's label); "Kept" is the post-2020 value as a share of the published-window value. Sorted by the post-2020 column.

{table}

1. **The table reproduces.** On its own window, our mean Rank IC is within {max(abs(r["ours_rank_ic"] - r["pub_rank_ic"]) for r in rep["models"].values() if "pub_rank_ic" in r):.4f} of the published value for every model, and the two rankings agree (Kendall τ = {rep["kendall_published_vs_ours"]["rank_ic"]["tau"]:.2f}).
2. **Most of the signal is gone after July 2020.** The average Rank IC falls from {dec["mean_published"]:.4f} to {dec["mean_post"]:.4f} ({dec["mean_change_pct"]:.0f}%). Against a TopK strategy run on random scores, the average excess return falls from {100 * above_pub:.1f}% to {100 * above_post:.1f}% a year. Refitting on 2014–2020 recovers only part of the loss.
3. **The ranking does not carry over.** Differences between models exceed seed noise in every window, yet the rank correlation between the two windows is τ = {tau:.2f}. The 90% model confidence set is {{{mcs_pub}}} in the published window, {{{mcs_post}}} after it, and {{{mcs_rec}}} on 2023–2026 with every model refitted.
4. **The label explains much of it.** Configs that train on a cross-sectionally ranked label kept {100 * min(kr):.0f}–{100 * max(kr):.0f}% of their Rank IC; those that train on a z-scored label kept {100 * min(kz):.0f}–{100 * max(kz):.0f}%. Changing only LightGBM's label from z-scores to ranks raises its post-2020 Rank IC from {lgb["base_model_post"]:.4f} to {lgb["variant_post"]:.4f}. This hypothesis was [written down](notes/prediction-2026-11-15.md) before the remaining runs finished; {pr["n_held"]} of its 4 predictions held.
5. **The decay started before the benchmark window.** The Alpha158 features with a clear sign in 2008–2016 had already lost {drop1:.0f}% of their Rank IC by 2017–2020, and lost another {drop2:.0f}% after.

![Rolling Rank IC](paper/figures/rolling_rank_ic.png)

![Rank IC by window](paper/figures/dumbbell.png)
"""
    readme = ROOT / "README.md"
    s = readme.read_text()
    s = re.sub(r"<!-- RESULTS -->.*?<!-- /RESULTS -->", "<!-- RESULTS -->", s, flags=re.S)
    s = s.replace("<!-- RESULTS -->", f"<!-- RESULTS -->\n{text}\n<!-- /RESULTS -->")
    readme.write_text(s)


if __name__ == "__main__":
    main()
