"""Figures for the paper and README (PDF and PNG in ``paper/figures``)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oosaudit import audit  # noqa: E402
from published import published_table  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"

# Reference palette (light mode): categorical slots 1-3, ink, chrome.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, CONTEXT = "#e1e0d9", "#c3c2b7", "#c3c2b7"
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

LABEL = {
    "linear": "Linear",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "doubleensemble": "DoubleEnsemble",
    "mlp": "MLP",
    "gru": "GRU",
    "lstm": "LSTM",
    "alstm": "ALSTM",
    "transformer": "Transformer",
}
PUB_END = pd.Timestamp("2020-08-01")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def shade_published(ax) -> None:
    ax.axvspan(pd.Timestamp("2017-01-01"), PUB_END, color="#f0efec", zorder=0, lw=0)


def fig_rolling(ic: pd.DataFrame, models: list[str]) -> None:
    """Centred 250-day mean of daily Rank IC, every model, published protocol."""
    sub = ic[(ic.protocol == "pub") & (ic.seed == audit.ENSEMBLE)]
    w = sub.pivot(index="datetime", columns="model", values="rank_ic").sort_index()
    w.index = pd.to_datetime(w.index)
    roll = w[models].rolling(250, min_periods=125, center=True).mean()
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    shade_published(ax)
    for m in models:
        if m != "linear":
            ax.plot(roll.index, roll[m], color=CONTEXT, lw=0.9, zorder=2)
    mean = roll.mean(axis=1)
    ax.plot(roll.index, mean, color=BLUE, lw=2, zorder=4, label="Mean of all models")
    ax.plot(roll.index, roll["linear"], color=ORANGE, lw=2, zorder=5, label="Linear (OLS)")
    ax.plot([], [], color=CONTEXT, lw=0.9, label="Each other model")
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax.set_ylabel("Rank IC, centred 250-day mean")
    tr = ax.get_xaxis_transform()
    ax.text(
        pd.Timestamp("2018-10-15"), 0.05, "published test window", ha="center", color=INK2, fontsize=7.5, transform=tr
    )
    ax.text(pd.Timestamp("2020-09-15"), 0.05, "after the window", ha="left", color=INK2, fontsize=7.5, transform=tr)
    ax.legend(loc="upper right", fontsize=7.5, ncol=3, bbox_to_anchor=(1.0, 1.18))
    ax.set_xlim(roll.dropna(how="all").index[0], roll.index[-1])
    save(fig, "rolling_rank_ic")


def fig_dumbbell(audit_json: dict, models: list[str]) -> None:
    dec = audit_json["decay"]["models"]
    st = audit_json["staleness"]
    pub = published_table()
    order = sorted([m for m in models if m in dec], key=lambda m: dec[m]["published"])
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(4.8, 0.32 * len(order) + 0.9))
    for i, m in enumerate(order):
        a, b = dec[m]["published"], dec[m]["post"]
        ax.plot([b, a], [i, i], color=AXIS, lw=1.6, zorder=1)
        ax.errorbar(a, i, xerr=dec[m]["published_sd"], fmt="none", ecolor=BLUE, lw=1, capsize=0, zorder=2)
        ax.errorbar(b, i, xerr=dec[m]["post_sd"], fmt="none", ecolor=ORANGE, lw=1, capsize=0, zorder=2)
    ax.scatter(
        [dec[m]["published"] for m in order], y, s=34, color=BLUE, zorder=3, label="2017–07/2020 (published window)"
    )
    ax.scatter(
        [dec[m]["post"] for m in order], y, s=34, color=ORANGE, zorder=3, label="08/2020–11/2026 (after publication)"
    )
    rf = [(i, st[m]["refit"]) for i, m in enumerate(order) if m in st and "refit" in st[m]]
    if rf:
        ax.scatter(
            [v for _, v in rf],
            [i for i, _ in rf],
            s=30,
            marker="D",
            color=AQUA,
            zorder=3,
            label="2023–2026, refit on 2014–2020",
        )
    pv = [(i, pub.loc[m, "rank_ic"]) for i, m in enumerate(order) if m in pub.index]
    ax.scatter(
        [v for _, v in pv],
        [i for i, _ in pv],
        s=40,
        facecolors="none",
        edgecolors=INK,
        lw=0.9,
        zorder=4,
        label="Published table",
    )
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(y, [LABEL.get(m, m) for m in order], color=INK)
    ax.set_xlabel("Mean daily Rank IC (CSI 300)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), fontsize=7, ncol=2)
    save(fig, "dumbbell")


def fig_seeds(ic: pd.DataFrame, audit_json: dict, models: list[str]) -> None:
    """Per-seed mean Rank IC against the sampling error of one test window."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=False)
    for ax, (period, color, title) in zip(
        axes,
        [("published", BLUE, "Published window, 2017–07/2020"), ("post", ORANGE, "After publication, 08/2020–11/2026")],
        strict=True,
    ):
        sub = audit.in_period(ic[(ic.protocol == "pub") & (ic.seed != audit.ENSEMBLE)], period)
        runs = sub.groupby(["model", "seed"])["rank_ic"].mean().reset_index()
        ms = [m for m in models if m in set(runs.model)]
        ms = sorted(ms, key=lambda m: runs[runs.model == m].rank_ic.mean())
        for i, m in enumerate(ms):
            v = runs[runs.model == m].rank_ic.to_numpy()
            ax.scatter(np.full(len(v), i), v, s=14, color=color, alpha=0.85, zorder=3, lw=0)
        se = audit_json["tests"].get(f"pub/{period}", {}).get("hac_se", {})
        if se:
            half = 1.96 * float(np.median(list(se.values())))
            mid = runs.rank_ic.mean()
            xb = len(ms) + 0.2
            ax.errorbar(xb, mid, yerr=half, fmt="none", ecolor=INK2, lw=1.2, capsize=4, zorder=3)
            ax.text(xb, mid - half, "±1.96 s.e.\nof one\nwindow\nmean", ha="center", va="top", fontsize=6.5, color=INK2)
            ax.set_ylim(mid - half * 1.9, mid + half * 1.15)
            ax.set_xlim(-0.6, len(ms) + 0.8)
        ax.set_xticks(range(len(ms)), [LABEL.get(m, m) for m in ms], rotation=45, ha="right", color=INK)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Mean daily Rank IC, one dot per seed")
    fig.tight_layout()
    save(fig, "seed_noise")


def fig_features() -> None:
    pm = pd.read_csv(RES / "feature_period_means.csv", index_col=0)
    strong = pm[pm["train_t"].abs() > 3]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1.1, 1]})
    x, y = strong["published_aligned"], strong["post_aligned"]
    lim = max(x.abs().max(), y.abs().max()) * 1.08
    ax1.plot([-lim, lim], [-lim, lim], color=AXIS, lw=0.9, ls="--", zorder=1)
    ax1.axhline(0, color=AXIS, lw=0.8)
    ax1.axvline(0, color=AXIS, lw=0.8)
    ax1.scatter(x, y, s=12, color=BLUE, alpha=0.8, lw=0, zorder=3)
    k, c = np.polyfit(x, y, 1)
    xs = np.array([-lim, lim])
    ax1.plot(xs, k * xs + c, color=ORANGE, lw=1.6, zorder=4, label=f"fit, slope {k:.2f}")
    ax1.set_xlim(-lim * 0.4, lim)
    ax1.set_ylim(-lim * 0.6, lim)
    ax1.set_xlabel("Rank IC, published window")
    ax1.set_ylabel("Rank IC, after publication")
    ax1.set_title("Each Alpha158 feature", loc="left")
    ax1.legend(loc="upper left", fontsize=7)
    win = strong["window"].fillna(1).astype(int)
    g = strong.groupby(win)[["train_aligned", "published_aligned", "post_aligned"]].mean()
    labels = ["1 day" if w == 1 else f"{w}d" for w in g.index]
    xi = np.arange(len(g))
    bw = 0.26
    for j, (col, color, name) in enumerate(
        [
            ("train_aligned", AQUA, "2008–2016"),
            ("published_aligned", BLUE, "2017–07/2020"),
            ("post_aligned", ORANGE, "08/2020–2026"),
        ]
    ):
        ax2.bar(xi + (j - 1) * bw, g[col], width=bw - 0.03, color=color, label=name, zorder=3)
    ax2.set_xticks(xi, labels, color=INK)
    ax2.set_xlabel("Look-back window of the feature")
    ax2.set_ylabel("Mean Rank IC (sign as in 2008–2016)")
    ax2.set_title("By look-back window", loc="left")
    ax2.grid(axis="x", visible=False)
    ax2.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, "feature_decay")


def fig_cumulative(models: list[str]) -> None:
    ex = pd.read_parquet(RES / "excess_returns.parquet")
    cols = {m: f"pub/{m}/ens" for m in models if f"pub/{m}/ens" in ex.columns}
    if not cols:
        return
    cum = ex[list(cols.values())].fillna(0).cumsum()
    cum.columns = list(cols)
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    shade_published(ax)
    for m in cum.columns:
        if m != "linear":
            ax.plot(cum.index, cum[m], color=CONTEXT, lw=0.9, zorder=2)
    ax.plot(cum.index, cum.mean(axis=1), color=BLUE, lw=2, zorder=4, label="Mean of all models")
    if "linear" in cum:
        ax.plot(cum.index, cum["linear"], color=ORANGE, lw=2, zorder=5, label="Linear (OLS)")
    ax.plot([], [], color=CONTEXT, lw=0.9, label="Each other model")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel("Cumulative excess return\nover CSI 300, after costs")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_xlim(cum.index[0], cum.index[-1])
    save(fig, "cumulative_excess")


def main() -> None:
    ic = pd.read_parquet(RES / "daily_ic.parquet")
    aj = json.loads((RES / "ic_audit.json").read_text())
    models = [m for m in LABEL if m in set(ic[ic.protocol == "pub"].model)]
    fig_rolling(ic, models)
    fig_dumbbell(aj, models)
    fig_seeds(ic, aj, models)
    fig_features()
    if (RES / "excess_returns.parquet").exists():
        fig_cumulative(models)


if __name__ == "__main__":
    main()
