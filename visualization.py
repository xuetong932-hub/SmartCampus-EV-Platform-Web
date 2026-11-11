"""
visualization.py
================
Section B(b) - publication-quality charts with Matplotlib + Seaborn.

Panel 1: hourly grid-demand peaks vs solar generation (time series)
Panel 2: charging cost by User Type (box plot)
Panel 3: battery temperature vs energy delivered with trendlines (scatter)

The panels are saved together (dashboard figure) and one by one, at
300 dpi PNG, into ``outputs/figures``.

Run:  python visualization.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")                    # render without a display window
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import seaborn as sns                    # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "cleaned_energy_sales_data.csv"
FIG_DIR = BASE_DIR / "outputs" / "figures"

# One palette for the whole project so colours mean the same thing
# everywhere (AC = blue, DC = orange, solar = gold, grid = navy).
CHARGER_COLORS = {"AC": "#1f77b4", "DC": "#ff7f0e"}
USER_ORDER = ["Student", "Staff", "Visitor"]
GRID_COLOR, SOLAR_COLOR, PEAK_SHADE = "#1b2a49", "#f2b705", "#e63946"


def set_style() -> None:
    """Apply a clean, print-friendly theme."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update({"font.family": "DejaVu Sans",
                         "axes.titleweight": "bold",
                         "axes.spines.top": False,
                         "axes.spines.right": False,
                         "savefig.dpi": 300,
                         "savefig.bbox": "tight"})


def load_data(path: Path = DATA_FILE) -> pd.DataFrame:
    """Load the cleaned dataset produced by ``data_pipeline.py``."""
    try:
        return pd.read_csv(path, parse_dates=["Session_Date"])
    except FileNotFoundError as exc:
        raise SystemExit(f"[ERROR] {path} missing - run data_pipeline.py"
                         " first.") from exc


def panel_grid_vs_solar(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Panel 1 - peak (max) grid demand and mean solar output by hour."""
    hourly = (df.groupby("Start_Hour")
              .agg(Grid_Peak=("Grid_Load_kW", "max"),
                   Grid_Mean=("Grid_Load_kW", "mean"),
                   Solar=("Solar_Generation_kW", "mean"))
              .reindex(range(24)).interpolate())
    ax.axvspan(11.5, 17.5, color=PEAK_SHADE, alpha=0.08,
               label="Peak tariff window")
    ax.plot(hourly.index, hourly["Grid_Peak"], color=GRID_COLOR, lw=2.4,
            marker="o", ms=4, label="Grid demand - hourly peak")
    ax.plot(hourly.index, hourly["Grid_Mean"], color=GRID_COLOR, lw=1.2,
            ls="--", alpha=0.7, label="Grid demand - hourly mean")
    ax.fill_between(hourly.index, hourly["Solar"], color=SOLAR_COLOR,
                    alpha=0.35)
    ax.plot(hourly.index, hourly["Solar"], color=SOLAR_COLOR, lw=2.4,
            label="Solar generation (mean)")
    top = hourly["Grid_Peak"].idxmax()
    ax.annotate(f"Peak {hourly.loc[top, 'Grid_Peak']:.0f} kW",
                xy=(top, hourly.loc[top, "Grid_Peak"]),
                xytext=(top + 2.5, hourly["Grid_Peak"].max() * 0.98),
                arrowprops={"arrowstyle": "->", "color": "grey"})
    ax.set(title="Panel 1: Hourly Grid Demand vs Solar Generation",
           xlabel="Hour of day", ylabel="Power (kW)", xticks=range(0, 24, 2),
           xlim=(0, 23))
    ax.legend(loc="upper left", fontsize=8, frameon=True)


def panel_cost_by_user(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Panel 2 - distribution of Net_Billable_MYR per User Type."""
    sns.boxplot(data=df, x="User_Type", y="Net_Billable_MYR",
                order=USER_ORDER, hue="User_Type", palette="Set2",
                legend=False, width=0.55, fliersize=3, ax=ax)
    stats = df.groupby("User_Type")["Net_Billable_MYR"]
    medians, q3 = stats.median(), stats.quantile(0.75)
    for i, user in enumerate(USER_ORDER):
        ax.text(i + 0.30, q3[user] + 0.6, f"median\nRM{medians[user]:.2f}",
                ha="left", fontsize=8, weight="bold")
    ax.set_ylim(0, df["Net_Billable_MYR"].quantile(0.99) * 1.15)
    ax.set(title="Panel 2: Charging Session Cost by User Type",
           xlabel="User type", ylabel="Net billable amount (RM)")
    ax.text(0.99, 0.97, "y-axis capped at 99th percentile",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color="grey")


def panel_temp_vs_energy(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Panel 3 - scatter with one linear trendline per charger type."""
    for charger, colour in CHARGER_COLORS.items():
        sub = df[df["Charger_Type"] == charger]
        sns.regplot(data=sub, x="Battery_Temp_C", y="Energy_kWh", ax=ax,
                    color=colour, scatter_kws={"s": 14, "alpha": 0.45},
                    line_kws={"lw": 2.2},
                    label=f"{charger} (r = "
                          f"{sub['Battery_Temp_C'].corr(sub['Energy_kWh']):.2f})")
    ax.axvline(45, color=PEAK_SHADE, ls=":", lw=1.5)
    ax.text(45.3, ax.get_ylim()[1] * 0.93, "45 C safety limit",
            color=PEAK_SHADE, fontsize=8)
    ax.set(title="Panel 3: Battery Temperature vs Energy Delivered",
           xlabel="Battery temperature (°C)",
           ylabel="Energy delivered (kWh)")
    ax.legend(title="Charger type", fontsize=8)


def build_dashboard(df: pd.DataFrame) -> Path:
    """Combine the three panels into one figure and save it."""
    fig = plt.figure(figsize=(14, 9.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35,
                            wspace=0.22)
    panel_grid_vs_solar(df, fig.add_subplot(grid[0, :]))
    panel_cost_by_user(df, fig.add_subplot(grid[1, 0]))
    panel_temp_vs_energy(df, fig.add_subplot(grid[1, 1]))
    fig.suptitle("Smart Campus EV Fleet - Energy & Charging Analytics "
                 "(1-20 April 2026, n = 1,200 sessions)",
                 fontsize=14, weight="bold", y=0.995)
    out = FIG_DIR / "B_b_dashboard_three_panels.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def save_single_panels(df: pd.DataFrame) -> list[Path]:
    """Save each panel on its own for the report and README."""
    jobs = [("B_b_panel1_grid_vs_solar.png", panel_grid_vs_solar, (11, 4.5)),
            ("B_b_panel2_cost_by_user.png", panel_cost_by_user, (7, 4.5)),
            ("B_b_panel3_temp_vs_energy.png", panel_temp_vs_energy, (7, 4.5))]
    paths = []
    for name, func, size in jobs:
        fig, ax = plt.subplots(figsize=size)
        func(df, ax)
        fig.savefig(FIG_DIR / name)
        plt.close(fig)
        paths.append(FIG_DIR / name)
    return paths


def main() -> None:
    """Generate every Section B(b) figure."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    df = load_data()
    outputs = [build_dashboard(df), *save_single_panels(df)]
    for path in outputs:
        print(f"[FIGURE] saved {path.relative_to(BASE_DIR)}")
    slope = np.polyfit(df["Battery_Temp_C"], df["Energy_kWh"], 1)[0]
    print(f"[INSIGHT] Energy rises ~{slope:.2f} kWh per +1 C battery temp.")


if __name__ == "__main__":
    main()
