"""
a1_numpy_pandas.py
==================
Section A(a) - NumPy vs Pandas on campus energy telemetry.

* NumPy op 1 : 2D slicing of a (days x 24 hours) grid-load matrix to pull
               out the 12:00-17:00 peak window.
* NumPy op 2 : boolean masking to find battery temperatures above 45 C.
* Pandas op 1: median imputation of missing power data.
* Pandas op 2: group-wise charging revenue with .groupby().
* Benchmark  : Python loop vs NumPy vectorisation on 1 million readings.

Run from the project root:  python section_a/a1_numpy_pandas.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def numpy_peak_slice(clean: pd.DataFrame) -> None:
    """NumPy op 1 - reshape hourly grid load into a 2D matrix and slice it."""
    hourly = (clean.groupby(["Session_Date", "Start_Hour"])["Grid_Load_kW"]
              .max().unstack(fill_value=0))
    load = hourly.reindex(columns=range(24), fill_value=0).to_numpy()
    print(f"Grid-load matrix shape (days, hours): {load.shape}")
    peak = load[:, 12:18]                 # every row, columns 12..17
    print(f"Peak-window slice shape            : {peak.shape}")
    print(f"Mean peak load  : {peak.mean():8.1f} kW")
    print(f"Highest reading : {peak.max():8.1f} kW at day "
          f"{np.unravel_index(peak.argmax(), peak.shape)[0] + 1}")


def numpy_temperature_mask(raw: pd.DataFrame) -> None:
    """NumPy op 2 - boolean mask for batteries hotter than 45 C."""
    temps = raw["Battery_Temp_C"].to_numpy()
    hot = temps > 45.0                    # array of True / False
    print(f"Sessions above 45 C: {hot.sum()} of {temps.size} "
          f"({hot.mean():.1%}); hottest = {temps[hot].max():.1f} C")


def pandas_median_impute(raw: pd.DataFrame) -> pd.DataFrame:
    """Pandas op 1 - fill missing Energy_kWh / Duration_Mins by median."""
    df = raw.copy()
    print("Missing before:", df[["Energy_kWh", "Duration_Mins"]]
          .isna().sum().to_dict())
    for col in ["Energy_kWh", "Duration_Mins"]:
        df[col] = df[col].fillna(df[col].median())
    print("Missing after :", df[["Energy_kWh", "Duration_Mins"]]
          .isna().sum().to_dict())
    return df


def pandas_groupby_revenue(clean: pd.DataFrame) -> None:
    """Pandas op 2 - revenue per charger type and user type."""
    table = (clean.groupby(["Charger_Type", "User_Type"])["Net_Billable_MYR"]
             .agg(["count", "sum", "mean"]).round(2))
    print(table.to_string())


def benchmark(n: int = 1_000_000) -> None:
    """Compare a pure-Python loop with a vectorised NumPy expression."""
    rng = np.random.default_rng(1)
    energy = rng.uniform(3, 50, n)
    start = time.perf_counter()
    carbon_loop = [e * 0.475 for e in energy]
    t_loop = time.perf_counter() - start
    start = time.perf_counter()
    carbon_vec = energy * 0.475
    t_vec = time.perf_counter() - start
    assert np.allclose(carbon_loop, carbon_vec)
    print(f"Python loop : {t_loop * 1000:8.2f} ms")
    print(f"NumPy vector: {t_vec * 1000:8.2f} ms  "
          f"(~{t_loop / t_vec:.0f}x faster)")


if __name__ == "__main__":
    raw_df = pd.read_csv(DATA_DIR / "ev_campus_dataset.csv")
    clean_df = pd.read_csv(DATA_DIR / "cleaned_energy_sales_data.csv")
    for title, func, arg in [
        ("NumPy 1 - 2D slicing of peak hours", numpy_peak_slice, clean_df),
        ("NumPy 2 - boolean mask > 45 C", numpy_temperature_mask, raw_df),
        ("Pandas 1 - median imputation", pandas_median_impute, raw_df),
        ("Pandas 2 - groupby revenue", pandas_groupby_revenue, clean_df),
    ]:
        print(f"\n--- {title} ---")
        func(arg)
    print("\n--- Benchmark: 1,000,000 carbon calculations ---")
    benchmark()
