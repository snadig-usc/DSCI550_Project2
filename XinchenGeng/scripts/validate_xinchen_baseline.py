"""Independently verify Xinchen Geng's completed baseline and historical scores.

Recomputes CORR on every saved date with an independent rank/power formula,
rebuilds the training sample, and verifies one complete holdout date against
the saved model and a separate least-squares NCORR calculation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import norm, rankdata
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["numerai_ticker", "date"]
TARGET = "target_jupiter_60"
MODEL = "B0_small_main"
TOL = 1e-10


def require(condition, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_close(actual: float, expected: float, label: str) -> float:
    require(np.isclose(actual, expected, atol=TOL, rtol=0, equal_nan=True),
            f"{label}: recorded={actual}, independently computed={expected}")
    return float(abs(actual - expected)) if np.isfinite(actual) else 0.0


def independent_corr(predictions: np.ndarray, targets: np.ndarray) -> float:
    gaussian = norm.ppf((rankdata(predictions, method="average") - .5) / len(targets))
    centered_native = targets - targets.mean()
    pred_power = np.sign(gaussian) * np.abs(gaussian) ** 1.5
    # The official scorer transforms the native float32 target before Pearson
    # promotes it to float64. Promoting earlier changes quantization.
    target_power = np.sign(centered_native) * np.abs(centered_native) ** 1.5
    return float(np.corrcoef(pred_power, target_power.astype(float))[0, 1])


def independent_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    require(len(values) > 0 and np.isfinite(values).all(), "Invalid score array")
    mean = float(np.sum(values) / len(values))
    std = float(np.sqrt(np.mean((values - mean) ** 2)))
    cumulative = np.r_[0.0, np.cumsum(values)]
    return {
        "n_scored_dates": len(values), "mean": mean, "std_ddof0": std,
        "sharpe": mean / std if std else np.nan,
        "worst_date_score": float(np.min(values)),
        "max_drawdown": float(np.max(np.maximum.accumulate(cumulative) - cumulative)),
        "positive_fraction": float(np.count_nonzero(values > 0) / len(values)),
    }


def frame_stats(frame: pd.DataFrame) -> dict:
    return {"rows": len(frame), "dates": frame.date.nunique(),
            "min_date": frame.date.min(), "max_date": frame.date.max()}


def filtered_date(path: Path, date: pd.Timestamp, columns: list[str]) -> pd.DataFrame:
    dtype = pq.read_schema(path).field("date").type
    value = (date.strftime("%Y-%m-%d") if pa.types.is_string(dtype) or pa.types.is_large_string(dtype)
             else date.date() if pa.types.is_date(dtype) else date.to_pydatetime())
    frame = pd.read_parquet(path, columns=columns, filters=[("date", "==", value)])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def audit(root: Path, output: Path) -> dict:
    run = read_json(output / "run.json")
    require(run.get("status") == "complete", "Refusing to audit an incomplete run")
    require(run.get("data_version") == "v6.0", "Unexpected data version")
    require(run.get("primary_metric") == "CORR", "Baseline must report proposal CORR as primary")
    require(run.get("model") == MODEL, "Unexpected model")
    processed = root / "data/processed/v6.0"
    split_plan = read_json(processed / "split_manifest.json")
    protocol = read_json(output / "protocol.json")
    features = run["features"]
    metadata = read_json(root / "data/raw/v6.0/features.json")
    expected_features = [f for f in metadata["feature_sets"]["small"] if f != "feature_country"]
    require(features == expected_features and len(features) == len(set(features)), "Small feature schema differs")
    require(all(f.startswith("feature_") for f in features), "Target/key entered feature schema")
    for key in ("features", "params", "seed", "max_train_rows"):
        require(protocol[key] == run[key], f"Frozen protocol differs from run: {key}")
    require(pd.Timestamp(protocol["frozen_at_utc"]) <= pd.Timestamp(run["started_utc"])
            <= pd.Timestamp(run["completed_utc"]), "Protocol/run timestamps are inconsistent")
    require(sha256_file(output / "protocol.json") == run["protocol_sha256"], "Frozen protocol changed")
    require(sha256_file(output / "models" / f"{MODEL}.txt") == run["model_sha256"], "Serialized model changed")
    for key, expected in {"n_estimators": 300, "learning_rate": .05, "max_depth": 5,
                          "num_leaves": 31, "colsample_bytree": .1}.items():
        require(run["params"][key] == expected, f"Official baseline parameter changed: {key}")

    require(bool(run.get("input_sha256")), "No input hashes")
    checked_inputs = []
    for relative, expected_hash in run["input_sha256"].items():
        require(sha256_file(root / relative.replace("\\", "/")) == expected_hash, f"Input changed: {relative}")
        checked_inputs.append(relative)
    require(bool(run.get("code_sha256")), "No code hashes")
    for relative, expected_hash in run["code_sha256"].items():
        require(sha256_file(root / relative.replace("\\", "/")) == expected_hash, f"Source changed: {relative}")

    predictions = pd.read_parquet(output / "predictions.parquet")
    predictions["date"] = pd.to_datetime(predictions["date"])
    require(not predictions[KEYS + [TARGET, MODEL, "split"]].isna().any().any(), "Missing prediction values")
    require(not predictions.duplicated(KEYS).any(), "Duplicate saved stock/date key")
    require(set(predictions.split) == {"tuning", "holdout"}, "Unexpected prediction partitions")
    require(predictions[TARGET].between(0, 1).all(), "Target outside 0..1")
    require(np.isfinite(predictions[[TARGET, MODEL]].to_numpy()).all(), "Nonfinite predictions or labels")

    train = pd.read_parquet(processed / "train.parquet", columns=KEYS + [TARGET])
    train["date"] = pd.to_datetime(train["date"])
    require(not train[KEYS + [TARGET]].isna().any().any(), "Missing training keys/labels")
    require(not train.duplicated(KEYS).any(), "Duplicate source training keys")
    require(train[TARGET].between(0, 1).all(), "Invalid training labels")
    require(len(train) == run["train_input_rows"], "Full training row count differs")
    require(train.date.max() == pd.Timestamp(run["train_max_date_before_sample"]), "Before-sampling train max differs")
    n_sample = min(int(run["max_train_rows"]), len(train))
    sample = train.sample(n=n_sample, random_state=int(run["seed"])).reset_index(drop=True)
    sample_hash = hashlib.sha256(sample[KEYS].to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")).hexdigest()
    require(sample_hash == run["train_sample_keys_sha256"], "Training sample keys differ")
    require(n_sample == run["train_sample_rows"] == run["training"]["training_rows"], "Training sample count differs")
    require(run["training"]["target"] == TARGET and run["training"]["n_features"] == len(features), "Training schema metadata differs")

    source_stats = {"train": frame_stats(train)}
    del sample
    train_max = train.date.max()
    del train
    for split in ("tuning", "holdout"):
        source = pd.read_parquet(processed / f"{split}.parquet", columns=KEYS + [TARGET])
        source["date"] = pd.to_datetime(source["date"])
        require(not source.duplicated(KEYS).any(), f"Duplicate source keys: {split}")
        saved = predictions[predictions.split == split]
        joined = source.merge(saved[KEYS + [TARGET]], on=KEYS, how="outer", validate="one_to_one",
                              indicator=True, suffixes=("_source", "_saved"))
        require(joined["_merge"].eq("both").all(), f"Prediction coverage differs: {split}")
        require(np.array_equal(joined[TARGET + "_source"], joined[TARGET + "_saved"]), f"Saved target differs: {split}")
        source_stats[split] = frame_stats(source)
        del source, saved, joined
    for split, stats in source_stats.items():
        recorded = run["splits"][split]
        plan_part = split_plan["partitions"][split]
        for key in ("rows", "dates"):
            require(stats[key] == recorded[key] == plan_part[key], f"Partition count differs: {split}/{key}")
        for key, plan_key in (("min_date", "date_min"), ("max_date", "date_max")):
            require(stats[key] == pd.Timestamp(recorded[key]) == pd.Timestamp(plan_part[plan_key]),
                    f"Partition boundary differs: {split}/{key}")
    embargo = int(split_plan["embargo_weekdays"])
    require(embargo >= 65, "Embargo weaker than frozen cleaning protocol")
    tuning_min, tuning_max = source_stats["tuning"]["min_date"], source_stats["tuning"]["max_date"]
    holdout_min = source_stats["holdout"]["min_date"]
    require(tuning_min > train_max + pd.offsets.BDay(embargo), "Train/tuning embargo violated")
    require(holdout_min > tuning_max + pd.offsets.BDay(embargo), "Tuning/holdout embargo violated")

    per_date = pd.read_csv(output / "per_date_metrics.csv", parse_dates=["date"])
    require(set(per_date.model) == {MODEL}, "Unexpected model in per-date scores")
    require(not per_date.duplicated(["split", "date"]).any(), "Duplicate per-date score")
    groups = predictions.groupby(["split", "date"], sort=True)
    require(set(groups.groups) == set(zip(per_date.split, per_date.date)), "Per-date score coverage differs")
    lookup = per_date.set_index(["split", "date"])
    max_corr_error = 0.0
    for index, ((split, date), frame) in enumerate(groups, start=1):
        # Sort identically for native float32 mean operation order, but evaluate
        # the ranking/power/correlation formula without calling numerai_tools.
        frame = frame.sort_values("numerai_ticker")
        row = lookup.loc[(split, date)]
        require(int(row.n_rows) == len(frame), f"Scored row count differs: {split}/{date}")
        corr = independent_corr(frame[MODEL].to_numpy(), frame[TARGET].to_numpy())
        max_corr_error = max(max_corr_error, assert_close(float(row.CORR), corr, f"CORR {split}/{date}"))
        if index == 1 or index % 100 == 0 or index == len(groups):
            print(f"Independently verified CORR: {index}/{len(groups)} dates", flush=True)

    summary = pd.read_csv(output / "metrics.csv")
    require(set(summary.model) == {MODEL}, "Unexpected summary model")
    require(not summary.duplicated(["split", "metric"]).any(), "Duplicate summary rows")
    expected_summary = {(split, metric) for split in ("tuning", "holdout")
                        for metric in ("CORR", "NCORR")}
    require(set(zip(summary.split, summary.metric)) == expected_summary, "Summary metric coverage differs")
    max_summary_error = 0.0
    for _, row in summary.iterrows():
        group = per_date[per_date.split == row["split"]].sort_values("date")
        values = group[row.metric].dropna().to_numpy()
        require(len(group) == row.n_dates, "Summary n_dates differs")
        expected = (independent_summary(values) if len(values) else
                    {"n_scored_dates": 0, **{k: np.nan for k in ("mean", "std_ddof0", "sharpe",
                     "worst_date_score", "max_drawdown", "positive_fraction")}})
        for key, value in expected.items():
            max_summary_error = max(max_summary_error, assert_close(float(row[key]), float(value), f"Summary {row['split']}/{row.metric}/{key}"))
        require(int(row.training_rows) == n_sample and row.target == TARGET and int(row.n_features) == len(features),
                "Summary training metadata differs")

    date = holdout_min
    saved_date = predictions[(predictions.split == "holdout") & predictions.date.eq(date)].sort_values("numerai_ticker")
    source_date = filtered_date(processed / "holdout.parquet", date, KEYS + features + [TARGET])
    require(not source_date.duplicated(KEYS).any(), "Duplicate source date keys")
    aligned = saved_date[KEYS + [MODEL]].merge(source_date, on=KEYS, validate="one_to_one", how="left")
    require(len(aligned) == len(source_date) == len(saved_date), "Saved-model sample coverage differs")
    booster = lgb.Booster(model_file=str(output / "models" / f"{MODEL}.txt"))
    require(booster.feature_name() == features, "Serialized model feature schema differs")
    require(booster.num_trees() == run["params"]["n_estimators"], "Serialized model tree count differs")
    regenerated = booster.predict(aligned[features], num_threads=1)
    model_error = float(np.max(np.abs(regenerated - aligned[MODEL].to_numpy())))
    require(model_error <= TOL, f"Saved model predictions differ: {model_error}")

    ncorr_check = {"status": "unavailable", "reason": "No recorded NCORR for sampled date"}
    if np.isfinite(float(lookup.loc[("holdout", date), "NCORR"])):
        neutralizer_path = root / "data/raw/v6.0/validation_neutralizer.parquet"
        neutralizer_cols = [c for c in pq.read_schema(neutralizer_path).names if c.startswith("neutralizer_column_")]
        require(bool(neutralizer_cols), "Missing neutralizer columns")
        neutralizers = filtered_date(neutralizer_path, date, KEYS + neutralizer_cols)
        require(not neutralizers.duplicated(KEYS).any(), "Duplicate neutralizer keys")
        combined = saved_date[KEYS + [MODEL, TARGET]].merge(neutralizers, on=KEYS, validate="one_to_one", how="left")
        require(len(combined) == len(saved_date) and np.isfinite(combined[neutralizer_cols].to_numpy()).all(),
                "Incomplete sampled neutralizer coverage")
        gaussian = norm.ppf((rankdata(combined[MODEL].to_numpy(), method="average") - .5) / len(combined))
        design = np.column_stack([combined[neutralizer_cols].to_numpy(dtype=float), np.ones(len(combined))])
        coefficients = np.linalg.lstsq(design, gaussian, rcond=1e-6)[0]
        residual = gaussian - design @ coefficients
        y = combined[TARGET].to_numpy()
        centered = y - y.mean()
        ncorr = float(np.corrcoef(residual, centered.astype(float))[0, 1])
        ncorr_error = assert_close(float(lookup.loc[("holdout", date), "NCORR"]), ncorr, "Sampled independent NCORR")
        ncorr_check = {"status": "passed", "date": date.strftime("%Y-%m-%d"),
                       "rows": len(combined), "neutralizer_columns": len(neutralizer_cols),
                       "absolute_error": ncorr_error}

    return {
        "status": "passed", "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "absolute_tolerance": TOL, "all_date_corr": {"dates": len(groups), "rows": len(predictions),
        "maximum_absolute_error": max_corr_error, "formula": "average ranks -> Gaussian -> signed power 1.5; native target centering/power -> float64 Pearson"},
        "all_summary_metrics": {"rows": len(summary), "maximum_absolute_error": max_summary_error},
        "training_sample": {"rows": n_sample, "keys_sha256": sample_hash, "status": "matched"},
        "embargo": {"configured_weekdays": embargo,
                    "train_to_tuning_weekdays": int(np.busday_count(train_max.date(), tuning_min.date())),
                    "tuning_to_holdout_weekdays": int(np.busday_count(tuning_max.date(), holdout_min.date()))},
        "saved_model_prediction_check": {"date": date.strftime("%Y-%m-%d"), "rows": len(saved_date),
                                         "maximum_absolute_error": model_error},
        "independent_ncorr_spot_check": ncorr_check,
        "checked_input_hashes": checked_inputs, "checked_code_hashes": list(run["code_sha256"]),
        "scope": "Every CORR date and summary row; exact source/prediction key and target coverage; training sample and split bounds; input/source hashes; one full holdout date model and NCORR check. Does not retrain model or establish unseen-test status.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output if args.output is not None else
              root / "results/xinchen_baseline_20261030").resolve()
    with threadpool_limits(limits=1):
        result = audit(root, output)
    (output / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
