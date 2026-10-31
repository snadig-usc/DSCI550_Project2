"""Run Xinchen Geng's fixed LightGBM baseline and historical time validation.

Uses the locally mirrored official v6 tutorial's 200,000-row teaching baseline.
The existing historical holdout has already been observed in earlier project runs.
This command trains one baseline, with no model selection or live submission.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from numerai_tools.scoring import neutral_correlation, numerai_corr
from threadpoolctl import threadpool_limits

from run_experiments import (
    KEYS, MAIN_TARGET, PeakMemoryMonitor, read_neutralizers, sha256_file,
    summarize_scores, validate_frame, write_json,
)

MODEL = "B0_small_main"
SPLITS = ("train", "tuning", "holdout")


def partition_info(frame: pd.DataFrame) -> dict:
    return {"rows": len(frame), "dates": frame.date.nunique(),
            "min_date": frame.date.min(), "max_date": frame.date.max()}


def check_partition(frame: pd.DataFrame, name: str, plan: dict) -> dict:
    actual = partition_info(frame)
    expected = plan["partitions"][name]
    if actual["rows"] != expected["rows"] or actual["dates"] != expected["dates"]:
        raise ValueError(f"{name} row/date counts differ from split_manifest.json")
    for actual_key, expected_key in [("min_date", "date_min"), ("max_date", "date_max")]:
        if actual[actual_key] != pd.Timestamp(expected[expected_key]):
            raise ValueError(f"{name} date boundary differs from split_manifest.json")
    return actual


def score_baseline(frame: pd.DataFrame, neutralizers: pd.DataFrame | None,
                   split: str) -> pd.DataFrame:
    records = []
    total_dates = frame.date.nunique()
    for i, (date, group) in enumerate(frame.groupby("date", sort=True), 1):
        group = group.sort_values("numerai_ticker")
        predictions = group[[MODEL]]
        if not np.isfinite(predictions.to_numpy()).all() or group[MODEL].nunique() < 2:
            raise ValueError(f"Nonfinite or constant prediction on {date}")
        target = group[MAIN_TARGET]
        corr = float(numerai_corr(predictions, target, max_filtered_index_ratio=0)[MODEL])
        ncorr = (float(neutral_correlation(predictions, neutralizers.loc[group.index], target,
                                          max_filtered_index_ratio=0)[MODEL])
                 if neutralizers is not None else np.nan)
        if not np.isfinite(corr) or (neutralizers is not None and not np.isfinite(ncorr)):
            raise ValueError(f"Undefined score on {date}")
        records.append({"split": split, "date": date, "model": MODEL,
                        "n_rows": len(group), "CORR": corr, "NCORR": ncorr})
        if i == 1 or i % 50 == 0 or i == total_dates:
            print(f"Scored {split}: {i}/{total_dates} dates", flush=True)
    return pd.DataFrame(records)


def run(args: argparse.Namespace, monitor: PeakMemoryMonitor) -> None:
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserving existing results: {output}; choose a new --output")
    processed = root / "data/processed/v6.0"
    raw = root / "data/raw/v6.0"
    metadata_path = raw / "features.json"
    plan_path = processed / "split_manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["version"] != "v6.0" or plan["primary_target"] != MAIN_TARGET:
        raise ValueError("Prepared data version/target differs from the baseline protocol")
    embargo = int(plan["embargo_weekdays"])
    if embargo < 60:
        raise ValueError("Jupiter60 requires at least 60 weekdays of separation")
    features = [f for f in metadata["feature_sets"]["small"] if f != "feature_country"]
    if not features or len(features) != len(set(features)) or any(
            not f.startswith("feature_") for f in features):
        raise ValueError("Invalid small numeric feature list")
    source_path = root / "references/official_sources.json"
    sources = json.loads(source_path.read_text(encoding="utf-8"))
    tutorial = next(s for s in sources["sources"] if s["id"] == "official_v6_tutorial")
    mirror_path = root / tutorial["local_mirror"]
    if sha256_file(mirror_path) != tutorial["local_sha256"]:
        raise ValueError("Official tutorial mirror hash differs from source record")
    params = dict(tutorial["baseline"]["parameters"])
    params.update(n_jobs=args.n_jobs, random_state=args.seed, feature_fraction_seed=args.seed,
                  bagging_seed=args.seed, data_random_seed=args.seed,
                  deterministic=True, force_col_wise=True)
    protocol = {
        "frozen_at_utc": datetime.now(timezone.utc), "owner": "Xinchen Geng",
        "scope": "Baseline & Validation", "model": MODEL, "primary_metric": "CORR",
        "additional_metric": "NCORR when official neutralizers are present",
        "data_version": "v6.0", "target": MAIN_TARGET, "features": features,
        "params": params, "seed": args.seed, "max_train_rows": args.max_train_rows,
        "baseline_source": tutorial["url"], "baseline_source_commit": tutorial["commit_sha"],
        "embargo_weekdays": embargo, "embargo_calendar": "pandas BDay (Monday-Friday)",
        "sampling": "Fixed pandas row sample from eligible training rows only",
        "selection": "None; fixed baseline; no early stopping, tuning, or refitting on validation",
        "historical_holdout_previously_observed": True,
        "interpretation": "Historical reproducibility/development evidence; no new unseen test or live result",
        "split_manifest": plan,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    write_json(output / "protocol.json", protocol)
    manifest = {
        "status": "started", "started_utc": datetime.now(timezone.utc),
        "owner": "Xinchen Geng", "kind": "offline_fixed_baseline_reproduction",
        "data_version": "v6.0", "model": MODEL, "primary_metric": "CORR",
        "target": MAIN_TARGET, "features": features, "params": params,
        "seed": args.seed, "max_train_rows": args.max_train_rows,
        "baseline_source": tutorial["url"], "embargo_weekdays": embargo,
        "historical_holdout_previously_observed": True,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["lightgbm", "numpy", "pandas", "pyarrow", "numerai-tools", "scipy",
                      "psutil", "threadpoolctl"]},
        "definitions": {"CORR": "official numerai_tools.scoring.numerai_corr, separately per date",
                        "NCORR": "official numerai_tools.scoring.neutral_correlation, separately per date",
                        "mean": "Equal weight per scored date, not per stock row",
                        "std": "Population standard deviation across dates, ddof=0",
                        "sharpe": "mean/std, not annualized; target windows overlap",
                        "max_drawdown": "Additive cumulative score drop including initial zero; not return"},
        "code_sha256": {str(p.relative_to(root)): sha256_file(p) for p in
                        [Path(__file__).resolve(), root / "scripts/run_experiments.py"]},
    }
    write_json(output / "run.json", manifest)
    monitor.run_started = True
    started = time.perf_counter()
    input_paths = [metadata_path, plan_path, processed / "cleaning_report.json", source_path,
                   mirror_path] + [processed / f"{name}.parquet" for name in SPLITS]
    neutralizer_path = raw / "validation_neutralizer.parquet"
    if neutralizer_path.exists():
        input_paths.append(neutralizer_path)
    print("Hashing fixed inputs and recording the baseline protocol...", flush=True)
    manifest["input_sha256"] = {str(p.relative_to(root)): sha256_file(p) for p in input_paths}
    write_json(output / "run.json", manifest)
    train = validate_frame(pd.read_parquet(processed / "train.parquet",
                           columns=KEYS + features + [MAIN_TARGET]), features, "train")
    split_info = {"train": check_partition(train, "train", plan)}
    train_max = train.date.max()  # Compute before sampling, as required by the tutorial.
    train_input_rows = len(train)
    train = train.sample(n=min(args.max_train_rows, len(train)),
                         random_state=args.seed).reset_index(drop=True)
    sample_hash = hashlib.sha256(train[KEYS].to_csv(
        index=False, date_format="%Y-%m-%d").encode("utf-8")).hexdigest()
    print(f"Training {MODEL}: {len(train):,}/{train_input_rows:,} eligible rows, "
          f"{len(features)} features, seed {args.seed}", flush=True)
    tick = time.perf_counter()
    model = lgb.LGBMRegressor(**params)
    model.fit(train[features], train[MAIN_TARGET])
    training = {"training_rows": len(train), "target": MAIN_TARGET,
                "n_features": len(features), "training_seconds": time.perf_counter() - tick}
    model_path = output / "models" / f"{MODEL}.txt"
    model.booster_.save_model(str(model_path))
    print(f"Training finished in {training['training_seconds']:.1f}s", flush=True)
    del train
    manifest.update(training=training, train_input_rows=train_input_rows,
                    train_sample_rows=training["training_rows"], train_sample_keys_sha256=sample_hash,
                    train_max_date_before_sample=train_max)
    write_json(output / "run.json", manifest)
    all_predictions, all_scores, timings, ncorr_status = [], [], {}, {}
    previous_max = train_max
    for split in ("tuning", "holdout"):
        tick = time.perf_counter()
        frame = validate_frame(pd.read_parquet(processed / f"{split}.parquet",
                               columns=KEYS + features + [MAIN_TARGET]), features, split)
        split_info[split] = check_partition(frame, split, plan)
        if frame.date.min() <= previous_max + pd.offsets.BDay(embargo):
            raise ValueError(f"{split} violates the strict {embargo}-weekday embargo")
        previous_max = frame.date.max()
        predictions = frame[KEYS + [MAIN_TARGET]].copy()
        predictions[MODEL] = model.predict(frame[features])
        del frame
        neutralizers, ncorr_status[split] = read_neutralizers(neutralizer_path, predictions)
        scores = score_baseline(predictions, neutralizers, split)
        del neutralizers
        all_scores.append(scores)
        all_predictions.append(predictions.assign(split=split))
        timings[split] = time.perf_counter() - tick
    per_date = pd.concat(all_scores, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    per_date.to_csv(output / "per_date_metrics.csv", index=False, date_format="%Y-%m-%d")
    predictions.to_parquet(output / "predictions.parquet", index=False)
    summaries = []
    for split, group in per_date.groupby("split", sort=True):
        for metric in ("CORR", "NCORR"):
            summaries.append({"split": split, "model": MODEL, "metric": metric,
                              "n_dates": len(group),
                              **summarize_scores(group.sort_values("date")[metric]),
                              **training, "seed": args.seed})
    metrics = pd.DataFrame(summaries)
    metrics.to_csv(output / "metrics.csv", index=False)
    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc),
                    elapsed_seconds=time.perf_counter() - started, splits=split_info,
                    evaluation_seconds=timings, ncorr=ncorr_status,
                    model_sha256=sha256_file(model_path),
                    protocol_sha256=sha256_file(output / "protocol.json"))
    write_json(output / "run.json", manifest)
    print(metrics[["split", "metric", "n_scored_dates", "mean", "std_ddof0"]].to_string(index=False),
          flush=True)
    print(f"Complete: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()
    if args.max_train_rows < 1 or args.n_jobs < 1:
        parser.error("max-train-rows and n-jobs must be positive")
    args.output = (args.output if args.output is not None else
                   args.root / "results/xinchen_baseline_20261030").resolve()
    monitor = PeakMemoryMonitor()
    failure = None
    try:
        with monitor, threadpool_limits(limits=args.n_jobs):
            run(args, monitor)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if monitor.run_started:
            path = args.output / "run.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["memory"] = monitor.result()
            if failure is not None:
                manifest.update(status="failed", error=failure, failed_at_utc=datetime.now(timezone.utc))
            write_json(path, manifest)


if __name__ == "__main__":
    main()
