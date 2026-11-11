"""
a2_functional_surcharge.py
==========================
Section A(b) - functional processing of charging-session records.

Pipeline:  load -> filter(high risk) -> map(15 % surcharge) -> reduce(total)

High-risk rule: Overstay_Mins > 20  AND  Battery_Temp_C > 40.

Run from the project root:  python section_a/a2_functional_surcharge.py
"""

import csv
from functools import reduce
from pathlib import Path
from typing import Iterable, Iterator

DATA_FILE = (Path(__file__).resolve().parents[1]
             / "data" / "cleaned_energy_sales_data.csv")

OVERSTAY_LIMIT_MINS = 20
TEMP_LIMIT_C = 40.0
SURCHARGE_RATE = 0.15          # 15 % on the base energy fee


def load_records(path: Path) -> Iterator[dict]:
    """Yield one session at a time as a dict.

    Memory note: this is a generator, so the 1,200-row file is streamed
    line by line instead of being copied into a list. Only the fields we
    need are converted to float, keeping each dict small.
    """
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                yield {
                    "Session_ID": row["Session_ID"],
                    "Overstay_Mins": float(row["Overstay_Mins"]),
                    "Battery_Temp_C": float(row["Battery_Temp_C"]),
                    "Base_Fee_MYR": float(row["Base_Fee_MYR"]),
                }
    except FileNotFoundError:
        print(f"[ERROR] {path} not found - run data_pipeline.py first.")
    except (KeyError, ValueError) as exc:
        print(f"[ERROR] Bad record in {path.name}: {exc}")


# Lambda design: each lambda is a single, side-effect-free expression, so
# it can be passed straight into filter()/map()/reduce() and is easy to
# test on its own. Named constants keep the thresholds out of the logic.
is_high_risk = lambda s: (s["Overstay_Mins"] > OVERSTAY_LIMIT_MINS   # noqa: E731
                          and s["Battery_Temp_C"] > TEMP_LIMIT_C)

add_surcharge = lambda s: {                                           # noqa: E731
    **s,                                   # copy original fields (no mutation)
    "Surcharge_MYR": round(s["Base_Fee_MYR"] * SURCHARGE_RATE, 2),
    "Adjusted_Fee_MYR": round(s["Base_Fee_MYR"] * (1 + SURCHARGE_RATE), 2),
}


def process_sessions(records: Iterable[dict]) -> tuple[list[dict], float]:
    """Run the filter -> map -> reduce chain.

    filter() and map() are lazy in Python 3: they return iterators that
    pull one record at a time from the generator above, so no
    intermediate list of all 1,200 sessions is ever created. We only
    materialise the small list of alerts because it is reported twice.
    """
    alerts_iter = filter(is_high_risk, records)        # step 1: filter
    surcharged = list(map(add_surcharge, alerts_iter))  # step 2: map
    total = reduce(lambda acc, s: acc + s["Surcharge_MYR"],  # step 3: reduce
                   surcharged, 0.0)                    # 0.0 = start value
    return surcharged, round(total, 2)


def main() -> None:
    """Print the alert sessions and the total surcharge revenue."""
    alerts, total = process_sessions(load_records(DATA_FILE))
    print(f"High-risk sessions (overstay > {OVERSTAY_LIMIT_MINS} min and "
          f"temp > {TEMP_LIMIT_C:.0f} C): {len(alerts)}")
    print(f"{'Session':<11}{'Overstay':>9}{'Temp C':>8}{'Base RM':>9}"
          f"{'+15% RM':>9}{'New RM':>8}")
    # List comprehension used only for formatting the report lines.
    lines = [f"{a['Session_ID']:<11}{a['Overstay_Mins']:>9.0f}"
             f"{a['Battery_Temp_C']:>8.1f}{a['Base_Fee_MYR']:>9.2f}"
             f"{a['Surcharge_MYR']:>9.2f}{a['Adjusted_Fee_MYR']:>8.2f}"
             for a in alerts]
    print("\n".join(lines))
    print(f"\nTOTAL SURCHARGE REVENUE: RM {total:,.2f}")


if __name__ == "__main__":
    main()
