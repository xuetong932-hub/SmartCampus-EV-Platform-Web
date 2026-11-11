"""
a3_matplotlib_vs_seaborn.py
===========================
Section A(c) - the same two executive charts built twice, once with raw
Matplotlib and once with Seaborn, so the effort and the result can be
compared side by side.

Chart 1: multi-bar plot of hourly charging revenue, AC vs DC.
Chart 2: correlation heatmap - ambient temperature, charging duration,
         grid load.

Run from the project root:  python section_a/a3_matplotlib_vs_seaborn.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
import seaborn as sns            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "cleaned_energy_sales_data.csv"
FIG_DIR = ROOT / "outputs" / "figures"
COLORS = {"AC": "#1f77b4", "DC": "#ff7f0e"}
HEAT_COLS = {"Ambient_Temp_C": "Ambient temp (°C)",
             "Duration_Mins": "Charging duration (min)",
             "Grid_Load_kW": "Grid load (kW)"}


def hourly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue per start hour (06:00-22:00) split by charger type."""
    table = df.pivot_table(index="Start_Hour", columns="Charger_Type",
                           values="Net_Billable_MYR", aggfunc="sum",
                           fill_value=0)
    return table.loc[6:22]


def bars_matplotlib(table: pd.DataFrame, ax: plt.Axes) -> None:
    """Chart 1 in Matplotlib: bar offsets, colours, legend set by hand."""
    x = np.arange(len(table))
    width = 0.4
    for i, charger in enumerate(["AC", "DC"]):
        ax.bar(x + (i - 0.5) * width, table[charger], width,
               color=COLORS[charger], label=charger)
    ax.set_xticks(x, table.index)
    ax.set(title="Matplotlib: hourly revenue by charger",
           xlabel="Start hour", ylabel="Revenue (RM)")
    ax.legend(title="Charger")


def bars_seaborn(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Chart 1 in Seaborn: hue= does the grouping and the legend."""
    data = df[df["Start_Hour"].between(6, 22)]
    sns.barplot(data=data, x="Start_Hour", y="Net_Billable_MYR",
                hue="Charger_Type", hue_order=["AC", "DC"],
                estimator="sum", errorbar=None,
                palette=COLORS, ax=ax)
    ax.set(title="Seaborn: hourly revenue by charger",
           xlabel="Start hour", ylabel="Revenue (RM)")
    ax.legend(title="Charger")


def heat_matplotlib(corr: pd.DataFrame, ax: plt.Axes) -> None:
    """Chart 2 in Matplotlib: imshow + a manual loop for every label."""
    img = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = list(HEAT_COLS.values())
    ax.set_xticks(range(3), labels, rotation=20, ha="right")
    ax.set_yticks(range(3), labels)
    for r in range(3):
        for c in range(3):
            ax.text(c, r, f"{corr.iat[r, c]:.2f}", ha="center",
                    va="center", color="white" if abs(corr.iat[r, c]) > 0.6
                    else "black")
    ax.grid(False)
    plt.colorbar(img, ax=ax, fraction=0.046)
    ax.set_title("Matplotlib: correlation heatmap")


def heat_seaborn(corr: pd.DataFrame, ax: plt.Axes) -> None:
    """Chart 2 in Seaborn: one call, annotations and colour bar built in."""
    sns.heatmap(corr.rename(index=HEAT_COLS, columns=HEAT_COLS),
                annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                square=True, linewidths=1, ax=ax)
    ax.set_title("Seaborn: correlation heatmap")
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)


def main() -> None:
    """Build and save both comparison figures."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    df = pd.read_csv(DATA_FILE)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    bars_matplotlib(hourly_revenue(df), axes[0])
    bars_seaborn(df, axes[1])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "A_c_chart1_multibar.png", dpi=300)
    plt.close(fig)

    corr = df[list(HEAT_COLS)].corr()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    heat_matplotlib(corr, axes[0])
    heat_seaborn(corr, axes[1])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "A_c_chart2_heatmap.png", dpi=300)
    plt.close(fig)
    print(corr.round(3).to_string())
    print("[FIGURE] saved A_c_chart1_multibar.png, A_c_chart2_heatmap.png")


if __name__ == "__main__":
    main()
