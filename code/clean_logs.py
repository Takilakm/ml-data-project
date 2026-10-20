"""
CSI-4170/5170 Final Project — daily log cleaning

Takes the raw Google Sheets export and produces the anonymised, validated
dataset that goes into data/daily_log_anonymised.csv.

Run it again every time more rows come in. It is deliberately repeatable:
nothing is cleaned by hand, so the whole process is reproducible and can be
described in the report.

    python clean_logs.py raw_export.csv

Writes:
    daily_log_anonymised.csv    the clean data
    cleaning_report.txt         what was changed and why
"""

import sys
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# configuration
# ------------------------------------------------------------------

# first name (lowercased) -> participant code. The mapping itself is NOT
# published; only the codes appear in the output file.
PEOPLE = {"jing": "P1", "christina": "P2", "pablo": "P3"}

FEATURES = ["Sleep (hours)", "Exercise (min)", "Screen time (hours)",
            "Study (hours)", "Caffeine (cups)"]
TARGET = "Stress (1-10)"
EXTRA = ["Mood (1-10)", "Focus (1-10)"]

# plausible ranges — a value outside these is a data-entry error, not an outlier
VALID = {
    "Sleep (hours)":       (0, 16),
    "Exercise (min)":      (0, 480),
    "Screen time (hours)": (0, 24),
    "Study (hours)":       (0, 18),
    "Caffeine (cups)":     (0, 15),
    "Mood (1-10)":         (1, 10),
    "Stress (1-10)":       (1, 10),
    "Focus (1-10)":        (1, 10),
}

DEFAULT_YEAR = 2026


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def parse_date(value):
    """Handle the three formats that appear in the sheet:
    10/4/2026, 2026-10-5, and 10/20 (year missing)."""
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()
    if s == "":
        return pd.NaT
    # bare M/D with no year
    if s.count("/") == 1:
        s = f"{s}/{DEFAULT_YEAR}"
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(s, errors="coerce")


def normalise_person(value):
    """'Jing ', 'Pablo Avila', 'christina' all map to a participant code."""
    if pd.isna(value):
        return None
    first = str(value).strip().split()[0].lower() if str(value).strip() else ""
    return PEOPLE.get(first)


# ------------------------------------------------------------------
# the cleaning
# ------------------------------------------------------------------

def clean_log(csv_path, verbose=True):
    raw = pd.read_csv(csv_path)
    log = []

    def note(msg):
        log.append(msg)
        if verbose:
            print(msg)

    note(f"Loaded {csv_path}: {raw.shape[0]} rows, {raw.shape[1]} columns")

    df = raw.copy()
    n0 = len(df)

    # 1 — drop rows that are entirely empty (spacer rows in the sheet)
    df = df.dropna(how="all")
    note(f"  [1] dropped {n0 - len(df)} completely empty rows")

    # 2 — normalise the person column, drop rows with no identifiable person
    n = len(df)
    unmatched = sorted({str(v).strip() for v in df["Person"].dropna()
                        if normalise_person(v) is None})
    df["Person"] = df["Person"].map(normalise_person)
    df = df[df["Person"].notna()]
    note(f"  [2] dropped {n - len(df)} rows with no identifiable person")
    if unmatched:
        note(f"      WARNING unrecognised names, check PEOPLE mapping: {unmatched}")

    # 3 — parse dates
    df["Date"] = df["Date"].map(parse_date)
    n_bad = int(df["Date"].isna().sum())
    if n_bad:
        note(f"  [3] WARNING {n_bad} rows have an unparseable date")
    else:
        note("  [3] all dates parsed")

    # 4 — force the measurement columns to numeric
    for col in FEATURES + [TARGET] + EXTRA:
        before = df[col].notna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        lost = before - df[col].notna().sum()
        if lost:
            note(f"  [4] WARNING {col}: {lost} value(s) were not numeric and became blank")

    # 5 — range check: impossible values become blank rather than silently training
    for col, (lo, hi) in VALID.items():
        if col not in df.columns:
            continue
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        if bad.any():
            note(f"  [5] WARNING {col}: {int(bad.sum())} value(s) outside {lo}-{hi}, blanked")
            df.loc[bad, col] = np.nan

    # 6 — duplicates: one entry per person per day
    dup = df.duplicated(subset=["Date", "Person"], keep="last")
    if dup.any():
        note(f"  [6] dropped {int(dup.sum())} duplicate person+date rows (kept the last)")
        df = df[~dup]
    else:
        note("  [6] no duplicate person+date rows")

    # 7 — incomplete rows: a model needs every feature and the target
    need = FEATURES + [TARGET]
    incomplete = df[need].isna().any(axis=1)
    if incomplete.any():
        note(f"  [7] dropped {int(incomplete.sum())} incomplete row(s):")
        for _, r in df[incomplete].iterrows():
            missing = [c for c in need if pd.isna(r[c])]
            d = r["Date"].date() if pd.notna(r["Date"]) else "?"
            note(f"        {d}  {r['Person']}  missing: {', '.join(missing)}")
        df = df[~incomplete]
    else:
        note("  [7] no incomplete rows")

    # 8 — Notes column is free text and sometimes used as a chat thread.
    #     Keep it out of the published file entirely.
    if "Notes" in df.columns:
        df = df.drop(columns=["Notes"])
        note("  [8] dropped the Notes column (free text, not a feature)")

    df = df.sort_values(["Date", "Person"]).reset_index(drop=True)
    note(f"\nCLEAN: {len(df)} usable rows out of {n0} raw rows")

    return df, log


# ------------------------------------------------------------------
# reporting
# ------------------------------------------------------------------

def coverage_report(df, verbose=True):
    """Who logged on which day — the thing that decides whether the study works."""
    out = []

    def note(m):
        out.append(m)
        if verbose:
            print(m)

    note("\n" + "=" * 62)
    note("COVERAGE")
    note("=" * 62)

    per = df.groupby("Person")["Date"].agg(["min", "max", "count"])
    per.columns = ["first entry", "last entry", "rows"]
    note(per.to_string())

    latest = df["Date"].max()
    note("")
    for person, row in per.iterrows():
        gap = (latest - row["last entry"]).days
        flag = "  <-- STOPPED LOGGING" if gap >= 3 else ""
        note(f"  {person}: last logged {gap} day(s) before the newest entry{flag}")

    note("\nDaily grid (1 = logged):")
    grid = (df.assign(v=1)
              .pivot_table(index="Date", columns="Person", values="v", aggfunc="size")
              .fillna(0).astype(int))
    grid.index = grid.index.strftime("%Y-%m-%d")
    note(grid.to_string())
    return out


def label_report(df, verbose=True):
    """Two ways to turn a 1-10 stress rating into a class, and what each gives."""
    out = []

    def note(m):
        out.append(m)
        if verbose:
            print(m)

    note("\n" + "=" * 62)
    note("LABEL BALANCE")
    note("=" * 62)

    note("\nAbsolute cutoff:")
    for c in (4, 5, 6, 7):
        y = (df[TARGET] >= c).astype(int)
        note(f"  stress >= {c}:  low={int((y == 0).sum()):3d}  "
             f"high={int((y == 1).sum()):3d}   ({y.mean() * 100:.0f}% high)")

    note("\nPer-person mean stress:")
    note(df.groupby("Person")[TARGET].agg(["mean", "std", "count"]).round(2).to_string())

    z = df.groupby("Person")[TARGET].transform(
        lambda s: (s - s.mean()) / (s.std() if s.std() else 1))
    y = (z > 0).astype(int)
    note(f"\nPer-person calibrated (above that person's own average):")
    note(f"  low={int((y == 0).sum())}  high={int((y == 1).sum())}  "
         f"({y.mean() * 100:.0f}% high)")
    return out


# ------------------------------------------------------------------

def main(path, out_csv="daily_log_anonymised.csv", out_txt="cleaning_report.txt"):
    df, log = clean_log(path)
    log += coverage_report(df)
    log += label_report(df)

    df.to_csv(out_csv, index=False)
    with open(out_txt, "w") as fh:
        fh.write("\n".join(log) + "\n")

    print(f"\nwrote {out_csv}  ({len(df)} rows)")
    print(f"wrote {out_txt}")
    return df


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "raw.csv")
