"""Profile v6 data and create chronological, purged numeric modeling partitions."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["numerai_ticker", "date"]
TARGETS = ["target_jupiter_60", "target_jupiter_20"]


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    if frame[KEYS].isna().any().any():
        raise ValueError("Missing stock/date key")
    frame["numerai_ticker"] = frame["numerai_ticker"].astype("str").str.strip()
    if frame["numerai_ticker"].eq("").any():
        raise ValueError("Empty ticker")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame["date"].isna().any():
        raise ValueError("Unparseable or NaT stock/date key")
    return frame


def key_profile(path: Path) -> tuple[pd.DataFrame, dict]:
    keys = normalize_keys(pd.read_parquet(path, columns=KEYS))
    duplicated = int(keys.duplicated(KEYS).sum())
    if duplicated:
        # Repeated stock-date rows can have conflicting labels: never silently keep first.
        raise ValueError(f"{path.name}: {duplicated} duplicate stock-date keys")
    report = {"rows": len(keys), "duplicate_keys": duplicated,
              "unique_stocks": int(keys.numerai_ticker.nunique()),
              "dates": int(keys.date.nunique()), "date_min": str(keys.date.min().date()),
              "date_max": str(keys.date.max().date()),
              "schema": {field.name: str(field.type) for field in pq.ParquetFile(path).schema_arrow}}
    return keys, report


def prepare(version="v6.0", holdout_start="2023-01-01", embargo_bdays=65):
    if embargo_bdays < 60:
        raise ValueError("Jupiter60 requires at least 60 weekday embargo")
    raw = ROOT / "data/raw" / version
    out = ROOT / "data/processed" / version
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((raw / "features.json").read_text(encoding="utf-8"))
    features = list(dict.fromkeys(metadata["feature_sets"]["small"] + metadata["feature_sets"]["medium"]))
    excluded = [f for f in features if f == "feature_country"]
    features = [f for f in features if f not in excluded]
    original_numeric = set(metadata["feature_sets"].get("signals", [])) - {"feature_country"}
    train_keys, train_report = key_profile(raw / "train.parquet")
    valid_keys, valid_report = key_profile(raw / "validation.parquet")
    train_max = train_keys.date.max()
    if not train_keys.merge(valid_keys, on=KEYS, how="inner").empty:
        raise ValueError("Training and validation stock-date keys overlap")
    holdout_requested = pd.Timestamp(holdout_start)
    holdout_min = valid_keys.loc[valid_keys.date >= holdout_requested, "date"].min()
    if pd.isna(holdout_min):
        raise ValueError("No holdout dates")
    tuning_after = train_max + pd.offsets.BDay(embargo_bdays)
    tuning_before = holdout_min - pd.offsets.BDay(embargo_bdays)
    if tuning_before <= tuning_after:
        raise ValueError("Tuning window is empty after embargo")
    del train_keys, valid_keys
    plan = {"version": version, "primary_target": TARGETS[0], "auxiliary_target": TARGETS[1],
            "feature_sets": ["small", "medium"], "numeric_features": len(features),
            "excluded_text_features": excluded, "holdout_start_requested": holdout_start,
            "full_train_max_date_before_sampling": str(train_max.date()),
            "tuning_date_strictly_after": str(tuning_after.date()),
            "tuning_date_strictly_before": str(tuning_before.date()),
            "holdout_date_at_or_after": str(holdout_min.date()),
            "embargo_weekdays": embargo_bdays,
            "embargo_calendar": "pandas BDay weekdays; official example uses 60, this run adds 5 weekdays",
            "selection_policy": "Select using tuning NCORR only; freeze before scoring holdout",
            "sampling_policy": "Cleaning retains all eligible rows; model training samples fixed 200000 rows seed 7",
            "missing_feature_policy": "Preserve native NaN for LightGBM; never fit preprocessing on validation",
            "missing_label_policy": "Drop rows only when primary target is null; auxiliary model filters its own label",
            "duplicate_policy": "Fail on duplicate stock-date keys rather than silently choosing a row"}
    reports = {"train": train_report, "validation": valid_report}
    writers = {}
    split_counts = Counter()
    date_counts = Counter()
    missing_by_feature = Counter()
    null_by_target = Counter()
    data_types = Counter()
    removed = Counter()
    try:
        for filename, report in [("train.parquet", train_report), ("validation.parquet", valid_report)]:
            pf = pq.ParquetFile(raw / filename)
            if "data_type" not in pf.schema_arrow.names:
                raise ValueError(f"{filename} is missing required data_type column")
            required = KEYS + features + TARGETS
            missing_columns = set(required) - set(pf.schema_arrow.names)
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}")
            columns = required + (["data_type"] if "data_type" in pf.schema_arrow.names else [])
            seen = 0
            for batch in pf.iter_batches(batch_size=100000, columns=columns):
                frame = normalize_keys(batch.to_pandas())
                seen += len(frame)
                for col in features:
                    values = frame[col]
                    missing_by_feature[(filename, col)] += int(values.isna().sum())
                    if not pa.types.is_integer(pf.schema_arrow.field(col).type) and not pa.types.is_floating(pf.schema_arrow.field(col).type):
                        raise ValueError(f"Non-numeric feature unexpectedly included: {col}")
                    nonnull = values.dropna().to_numpy()
                    if not np.isfinite(nonnull).all():
                        raise ValueError(f"Nonfinite values: {filename}/{col}")
                    if col not in original_numeric and not np.isin(nonnull, [0, 1, 2, 3, 4]).all():
                        raise ValueError(f"Invalid binned feature values: {col}")
                for col in TARGETS:
                    null_by_target[(filename, col)] += int(frame[col].isna().sum())
                    if not np.isfinite(frame[col].dropna().to_numpy()).all():
                        raise ValueError(f"Infinite label: {col}")
                    if not frame[col].dropna().between(0, 1).all():
                        raise ValueError(f"Jupiter label outside 0..1: {col}")
                if "data_type" in frame:
                    data_types.update((filename, str(x)) for x in frame.data_type)
                    eligible_type = frame.data_type.eq("train" if filename == "train.parquet" else "validation")
                else:
                    eligible_type = pd.Series(True, index=frame.index)
                removed[(filename, "other_data_type")] += int((~eligible_type).sum())
                mature = frame[TARGETS[0]].notna()
                removed[(filename, "null_primary_target")] += int((eligible_type & ~mature).sum())
                eligible = eligible_type & mature
                if filename == "train.parquet":
                    masks = {"train": eligible}
                else:
                    masks = {"tuning": eligible & (frame.date > tuning_after) & (frame.date < tuning_before),
                             "holdout": eligible & (frame.date >= holdout_min)}
                    used = masks["tuning"] | masks["holdout"]
                    removed[(filename, "embargo")] += int((eligible & ~used).sum())
                for split, mask in masks.items():
                    part = frame.loc[mask, required].copy()
                    if part.empty:
                        continue
                    part[TARGETS] = part[TARGETS].astype("float32")
                    # Do not standardize, rank, clip or impute anonymized features.
                    table = pa.Table.from_pandas(part, preserve_index=False)
                    if split not in writers:
                        writers[split] = pq.ParquetWriter(out / (split + ".parquet.part"), table.schema, compression="zstd")
                    writers[split].write_table(table)
                    split_counts[split] += len(part)
                    for date, n in part.groupby("date").size().items():
                        date_counts[(split, str(date.date()))] += int(n)
                print(f"{filename}: profiled {seen:,}/{report['rows']:,} rows", flush=True)
            report["default_target_alias_verified"] = False
            if "target" in pf.schema_arrow.names:
                for b in pf.iter_batches(batch_size=200000, columns=["target", TARGETS[0]]):
                    a = b.to_pandas()
                    if not np.allclose(a["target"], a[TARGETS[0]], rtol=0, atol=0, equal_nan=True):
                        raise ValueError("Generic target is not an exact Jupiter60 alias")
                report["default_target_alias_verified"] = True
    finally:
        for writer in writers.values():
            writer.close()
    if set(split_counts) != {"train", "tuning", "holdout"}:
        raise ValueError(f"Missing modeling partitions: {split_counts}")
    for split in split_counts:
        (out / (split + ".parquet.part")).replace(out / (split + ".parquet"))
    dates = pd.DataFrame([{"split": s, "date": d, "rows": n} for (s, d), n in sorted(date_counts.items())])
    dates.to_csv(out / "date_counts.csv", index=False)
    for split, n in split_counts.items():
        ss = dates[dates.split == split]
        plan.setdefault("partitions", {})[split] = {"rows": n, "dates": len(ss), "date_min": ss.date.min(), "date_max": ss.date.max()}
    quality = {"raw": reports, "split_counts": dict(split_counts),
               "removed_rows": [{"file": f, "reason": r, "rows": n} for (f, r), n in removed.items()],
               "missing_targets": [{"file": f, "target": t, "rows": n} for (f, t), n in null_by_target.items()],
               "missing_features": [{"file": f, "feature": c, "rows": n} for (f, c), n in missing_by_feature.items() if n],
               "data_type_counts": [{"file": f, "data_type": t, "rows": n} for (f, t), n in data_types.items()],
               "key_overlap_train_validation": 0,
               "checks": ["required columns", "nonnull unique keys", "no train/validation key overlap", "strict dates", "numeric finite features", "0..4 binned features", "finite mature labels", "target alias identity"]}
    (out / "split_manifest.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (out / "cleaning_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(json.dumps(plan["partitions"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v6.0")
    parser.add_argument("--holdout-start", default="2023-01-01")
    parser.add_argument("--embargo-bdays", type=int, default=65)
    args = parser.parse_args()
    if args.embargo_bdays < 60:
        parser.error("Jupiter60 requires at least 60 weekday embargo")
    prepare(args.version, args.holdout_start, args.embargo_bdays)
