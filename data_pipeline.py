"""
data_pipeline.py
================
Section B(a) - Data pipeline for the Smart Campus Energy & EV Fleet Platform.

Steps
-----
1. Load the raw charging-session file (``ev_campus_dataset.csv``) and the
   hub sales file (``sales_data.csv``).
2. Cleanse: validate categorical columns and impute missing numerical
   values with the column median.
3. Enrich: add simulated "model" columns that the raw file does not hold
   (session date, ambient temperature, solar generation, campus grid load
   and battery state-of-charge). A fixed random seed keeps them
   reproducible.
4. Feature engineering: Power Rate (kW), Carbon Savings (kg CO2) and the
   Net Billable Amount including the peak-hour surcharge.
5. Export ``cleaned_energy_sales_data.csv`` (+ ``cleaned_sales_data.csv``)
   and print summary statistics to the console.

Run:  python data_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration (business rules recovered from the reference cleaned file)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_EV_FILE = DATA_DIR / "ev_campus_dataset.csv"
RAW_SALES_FILE = DATA_DIR / "sales_data.csv"
CLEAN_EV_FILE = DATA_DIR / "cleaned_energy_sales_data.csv"
CLEAN_SALES_FILE = DATA_DIR / "cleaned_sales_data.csv"

RANDOM_SEED = 72604                 # module code, so results are repeatable
CARBON_FACTOR = 0.475               # kg CO2 avoided per kWh (from brief)
BASE_RATE_MYR = {"AC": 6.0, "DC": 15.0}   # RM per charging hour
PEAK_HOURS = range(12, 18)          # 12:00-17:59 counts as grid peak
PEAK_SURCHARGE_RATE = 0.25          # +25 % of the base fee in peak hours
OVERSTAY_GRACE_MINS = 15            # first 15 idle minutes are free
OVERSTAY_FEE_PER_MIN = 0.50         # RM per minute after the grace period
BATTERY_CAPACITY_KWH = 60.0         # typical fleet EV pack size
SOLAR_CAPACITY_KW = 350.0           # rooftop PV capacity of the campus
START_DATE, END_DATE = "2026-04-01", "2026-04-20"  # aligned with sales_data

VALID_CATEGORIES = {
    "Charger_Type": {"AC", "DC"},
    "User_Type": {"Student", "Staff", "Visitor"},
    "Grid_Status_Label": {"Normal", "Grid_Surge", "Overheated"},
}
EV_NUMERIC = ["Start_Hour", "Duration_Mins", "Energy_kWh",
              "Overstay_Mins", "Battery_Temp_C"]
SALES_VALID = {
    "Hub_Location": {"Lakeside_Hub", "Subang_Hub", "KL_Campus_Hub",
                     "Taylor_Tower_Hub"},
    "Category": {"Charging_Service", "EV_Accessory", "Energy_Credit",
                 "Fleet_Pass"},
    "Customer_Type": {"Student", "Staff", "Public_Fleet"},
    "Payment_Method": {"TouchNGo", "Credit_Card", "Campus_Wallet"},
}


# --------------------------------------------------------------------------
# 1. Loading
# --------------------------------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file and fail with a clear message if it is unusable.

    Args:
        path: Location of the CSV file.

    Returns:
        The raw DataFrame.

    Raises:
        SystemExit: If the file is missing, empty or not valid CSV.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"[ERROR] Input file not found: {path}")
    except pd.errors.EmptyDataError:
        sys.exit(f"[ERROR] Input file is empty: {path}")
    except pd.errors.ParserError as exc:
        sys.exit(f"[ERROR] Could not parse {path.name}: {exc}")
    print(f"[LOAD] {path.name}: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


# --------------------------------------------------------------------------
# 2. Cleansing
# --------------------------------------------------------------------------
def validate_categoricals(df: pd.DataFrame,
                          rules: dict[str, set[str]]) -> pd.DataFrame:
    """Trim text, then replace values outside the allowed set by the mode.

    Args:
        df: DataFrame to validate (a copy is returned).
        rules: Mapping of column name -> set of allowed labels.

    Returns:
        DataFrame whose categorical columns only hold valid labels.
    """
    df = df.copy()
    for col, allowed in rules.items():
        if col not in df.columns:
            raise KeyError(f"Expected categorical column missing: {col}")
        df[col] = df[col].astype("string").str.strip()
        invalid = ~df[col].isin(allowed)
        if invalid.any():
            mode_value = df.loc[~invalid, col].mode().iat[0]
            df.loc[invalid, col] = mode_value
            print(f"[CLEAN] {col}: {invalid.sum()} invalid label(s) "
                  f"-> '{mode_value}'")
        else:
            print(f"[CLEAN] {col}: all labels valid {sorted(allowed)}")
        df[col] = df[col].astype("category")
    return df


def impute_median(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce columns to numbers and fill NaNs with each column's median.

    The median is used instead of the mean because energy and duration
    values are right-skewed (a few very long DC sessions), so the median
    is a more robust "typical" value.

    Args:
        df: DataFrame to impute (a copy is returned).
        columns: Numerical columns to treat.

    Returns:
        DataFrame with no missing values in ``columns``.
    """
    df = df.copy()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n_missing = int(df[col].isna().sum())
        if n_missing:
            median = df[col].median()
            df[col] = df[col].fillna(median)
            print(f"[CLEAN] {col}: imputed {n_missing} missing value(s) "
                  f"with median {median:.2f}")
    return df


def clean_ev_sessions(raw: pd.DataFrame) -> pd.DataFrame:
    """Full cleansing routine for the EV charging-session table."""
    df = validate_categoricals(raw, VALID_CATEGORIES)
    df = impute_median(df, EV_NUMERIC)
    # Physical sanity rules: no negative durations / energy, valid hour.
    df["Start_Hour"] = df["Start_Hour"].clip(0, 23).astype(int)
    df["Overstay_Mins"] = df["Overstay_Mins"].clip(lower=0).astype(int)
    df = df[(df["Duration_Mins"] > 0) & (df["Energy_kWh"] > 0)]
    df = df.drop_duplicates(subset="Session_ID")
    return df.reset_index(drop=True)


def clean_sales(raw: pd.DataFrame) -> pd.DataFrame:
    """Cleanse the hub sales table.

    Missing revenue is first rebuilt from ``Units_Sold * Unit_Price_MYR``
    (an exact rule), and only the rows that still have gaps are filled
    with the median.
    """
    df = validate_categoricals(raw, SALES_VALID)
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"],
                                            errors="coerce")
    rebuilt = df["Total_Revenue_MYR"].isna() & df["Units_Sold"].notna()
    df.loc[rebuilt, "Total_Revenue_MYR"] = (
        df.loc[rebuilt, "Units_Sold"] * df.loc[rebuilt, "Unit_Price_MYR"])
    print(f"[CLEAN] Total_Revenue_MYR: rebuilt {rebuilt.sum()} value(s) "
          "from units x price")
    missing_units = df["Units_Sold"].isna()
    df = impute_median(df, ["Units_Sold", "Total_Revenue_MYR"])
    # Keep revenue consistent with the imputed units.
    df.loc[missing_units, "Total_Revenue_MYR"] = (
        df.loc[missing_units, "Units_Sold"]
        * df.loc[missing_units, "Unit_Price_MYR"])
    df["Units_Sold"] = df["Units_Sold"].round().astype(int)
    return df.dropna(subset=["Transaction_Date"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Simulated model columns
# --------------------------------------------------------------------------
def ambient_profile(hours: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Tropical campus air temperature: ~26 C at dawn, ~33 C at 14:00."""
    base = 26.0 + 7.0 * np.exp(-((hours - 14) / 4.5) ** 2)
    return np.round(base + rng.normal(0, 0.8, hours.size), 1)


def solar_profile(hours: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    """PV output (kW): half-sine between 07:00 and 19:00 scaled by cloud."""
    shape = np.clip(np.sin(np.pi * (hours - 7) / 12), 0, None)
    shape[(hours < 7) | (hours > 19)] = 0.0
    return np.round(SOLAR_CAPACITY_KW * shape * cloud, 1)


def campus_base_load(hours: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Non-EV campus demand (kW): night base + office-hour hump."""
    hump = 520.0 * np.exp(-((hours - 13.5) / 3.8) ** 2)
    return 380.0 + hump + rng.normal(0, 25, hours.size)


def add_simulated_columns(df: pd.DataFrame,
                          seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Attach reproducible, physically plausible model columns.

    New columns: Session_Date, Ambient_Temp_C, Solar_Generation_kW,
    Grid_Load_kW, Start_SoC_pct, End_SoC_pct.

    Grid_Load_kW = campus base load for that date-hour + the summed
    charging power of every EV session that started in the same hour,
    so busy charging hours genuinely push the grid towards its peak.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    df["Session_Date"] = rng.choice(dates, size=len(df))
    hours = df["Start_Hour"].to_numpy(dtype=float)

    df["Ambient_Temp_C"] = ambient_profile(hours, rng)
    cloud_by_day = dict(zip(dates, rng.uniform(0.55, 1.0, dates.size)))
    cloud = df["Session_Date"].map(cloud_by_day).to_numpy()
    df["Solar_Generation_kW"] = solar_profile(hours, cloud)

    ev_load = df.groupby(["Session_Date", "Start_Hour"])["Power_Rate_kW"] \
        .transform("sum")
    df["Grid_Load_kW"] = np.round(campus_base_load(hours, rng) + ev_load, 1)

    df["Start_SoC_pct"] = rng.integers(10, 61, len(df))
    gained = df["Energy_kWh"] / BATTERY_CAPACITY_KWH * 100
    df["End_SoC_pct"] = np.minimum(100, df["Start_SoC_pct"] + gained).round(1)
    return df


# --------------------------------------------------------------------------
# 4. Feature engineering
# --------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute power, carbon and billing metrics (all vectorised).

    * Power_Rate_kW        = Energy_kWh / Duration_Hours
    * Carbon_Savings_kgCO2 = Energy_kWh * 0.475
    * Base_Fee_MYR         = hourly rate (AC 6 / DC 15) * Duration_Hours
    * Peak_Surge_MYR       = 25 % of base fee when Start_Hour in 12-17
    * Overstay_Fee_MYR     = RM0.50 per idle minute after a 15 min grace
    * Net_Billable_MYR     = base + peak surge + overstay fee
    """
    df = df.copy()
    df["Duration_Hours"] = df["Duration_Mins"] / 60
    df["Power_Rate_kW"] = df["Energy_kWh"] / df["Duration_Hours"]
    df["Carbon_Savings_kgCO2"] = (df["Energy_kWh"] * CARBON_FACTOR).round(2)
    df["Base_Rate_MYR"] = df["Charger_Type"].map(BASE_RATE_MYR).astype(float)
    df["Base_Fee_MYR"] = df["Base_Rate_MYR"] * df["Duration_Hours"]
    df["Is_Peak_Hour"] = df["Start_Hour"].isin(PEAK_HOURS)
    df["Peak_Surge_MYR"] = np.where(df["Is_Peak_Hour"],
                                    df["Base_Fee_MYR"] * PEAK_SURCHARGE_RATE,
                                    0.0)
    billable_idle = (df["Overstay_Mins"] - OVERSTAY_GRACE_MINS).clip(lower=0)
    df["Overstay_Fee_MYR"] = billable_idle * OVERSTAY_FEE_PER_MIN
    df["Net_Billable_MYR"] = (df["Base_Fee_MYR"] + df["Peak_Surge_MYR"]
                              + df["Overstay_Fee_MYR"]).round(2)
    return df


# --------------------------------------------------------------------------
# 5. Export & reporting
# --------------------------------------------------------------------------
def export_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV, reporting permission problems clearly."""
    try:
        out = df.copy()
        for col in out.select_dtypes(include="datetime").columns:
            out[col] = out[col].dt.strftime("%Y-%m-%d")
        out.to_csv(path, index=False)
        print(f"[EXPORT] {path.name}: {len(out)} rows saved")
    except PermissionError:
        print(f"[ERROR] {path.name} is open in another program - close it "
              "and run again.")
        raise


def print_summary(df: pd.DataFrame) -> None:
    """Print the console summary required by the brief."""
    line = "=" * 70
    print(f"\n{line}\nSUMMARY STATISTICS - cleaned_energy_sales_data.csv\n{line}")
    print(f"Sessions: {len(df)} | Missing values left: "
          f"{int(df.isna().sum().sum())}")
    cols = ["Duration_Mins", "Energy_kWh", "Power_Rate_kW",
            "Carbon_Savings_kgCO2", "Battery_Temp_C", "Net_Billable_MYR"]
    print(df[cols].describe().round(2).to_string())
    print("\nRevenue and energy by charger type:")
    print(df.groupby("Charger_Type", observed=True)
          .agg(Sessions=("Session_ID", "count"),
               Energy_kWh=("Energy_kWh", "sum"),
               Revenue_MYR=("Net_Billable_MYR", "sum"))
          .round(2).to_string())
    print(f"\nTotal energy delivered : {df['Energy_kWh'].sum():,.2f} kWh")
    print(f"Total carbon offset    : {df['Carbon_Savings_kgCO2'].sum():,.2f} kg")
    print(f"Total net revenue      : RM {df['Net_Billable_MYR'].sum():,.2f}")
    print(line)


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute every stage and return (energy_df, sales_df)."""
    ev = clean_ev_sessions(load_csv(RAW_EV_FILE))
    ev = engineer_features(ev)          # Power_Rate needed for grid load
    ev = add_simulated_columns(ev)
    sales = clean_sales(load_csv(RAW_SALES_FILE))
    export_csv(ev, CLEAN_EV_FILE)
    export_csv(sales, CLEAN_SALES_FILE)
    print_summary(ev)
    return ev, sales


if __name__ == "__main__":
    run_pipeline()
