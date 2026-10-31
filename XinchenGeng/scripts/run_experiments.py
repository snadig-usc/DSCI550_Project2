"""Run the predeclared Numerai v6 pilot and freeze selection before holdout.

Baseline source (retrieved 2026-10-26):
https://github.com/numerai/example-scripts/blob/master/numerai/v6_example_model.ipynb
This is an offline pilot, not a live submission or a final competition result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from numerai_tools.scoring import neutral_correlation, numerai_corr
from threadpoolctl import threadpool_limits

MAIN_TARGET = "target_jupiter_60"
AUX_TARGET = "target_jupiter_20"
KEYS = ["numerai_ticker", "date"]
MODEL_NAMES = ["B0_small_main", "E1_medium_main", "E2_small_aux20", "E3_rank_ensemble"]
BASELINE_URL = "https://github.com/numerai/example-scripts/blob/master/numerai/v6_example_model.ipynb"


class PeakMemoryMonitor:
    """Sample process RSS each second and use the OS high-water mark if exposed."""
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.peak_rss_bytes = 0
        self.run_started = False
        self.thread = threading.Thread(target=self._watch, daemon=True, name="experiment-memory")

    def _sample(self) -> None:
        info = self.process.memory_info()
        self.peak_rss_bytes = max(self.peak_rss_bytes, info.rss, getattr(info, "peak_wset", 0))

    def _watch(self) -> None:
        while not self.stop_event.wait(self.interval):
            self._sample()

    def __enter__(self):
        self._sample()
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self._sample()

    def result(self) -> dict:
        return {"peak_rss_bytes": self.peak_rss_bytes,
                "peak_rss_gib": self.peak_rss_bytes / 1024 ** 3,
                "sample_interval_seconds": self.interval,
                "scope": "experiment Python process; OS peak working set used when available"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summarize_scores(values: pd.Series) -> dict:
    """Equal weight per date; drawdown is positive additive CORR units from zero."""
    values = values.dropna().astype(float)
    if values.empty:
        return {"n_scored_dates": 0, **{key: np.nan for key in
            ["mean", "std_ddof0", "sharpe", "worst_date_score", "max_drawdown", "positive_fraction"]}}
    if not np.isfinite(values).all():
        raise ValueError("Non-finite per-date scores")
    std = float(values.std(ddof=0))
    cumulative = np.concatenate([[0.0], values.cumsum().to_numpy()])
    drawdown = np.maximum.accumulate(cumulative) - cumulative
    return {
        "n_scored_dates": len(values), "mean": float(values.mean()), "std_ddof0": std,
        "sharpe": float(values.mean() / std) if std > 0 else np.nan,
        "worst_date_score": float(values.min()), "max_drawdown": float(drawdown.max()),
        "positive_fraction": float((values > 0).mean()),
    }


def rank_ensemble(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Average tie-preserving percentile ranks separately within each date."""
    ranks = frame.groupby("date", sort=False)[columns].rank(method="average", pct=True)
    return ranks.mean(axis=1)


def validate_frame(frame: pd.DataFrame, features: list[str], name: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"{name} has no rows")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if frame[KEYS + [MAIN_TARGET]].isna().any().any():
        raise ValueError(f"{name} has missing keys or main targets")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{name} has duplicate ticker/date keys")
    if not frame[MAIN_TARGET].between(0, 1).all():
        raise ValueError(f"{name} has invalid target values")
    if any(not pd.api.types.is_numeric_dtype(frame[c]) for c in features):
        raise ValueError(f"{name} has nonnumeric features")
    # Preserve native feature NaNs: LightGBM handles them without learning a
    # validation-derived imputation value. Infinity is never a valid feature.
    if np.isinf(frame[features].to_numpy()).any():
        raise ValueError(f"{name} has infinite feature values")
    return frame.reset_index(drop=True)


def read_neutralizers(path: Path, scoring_frame: pd.DataFrame) -> tuple[pd.DataFrame | None, dict]:
    if not path.exists():
        return None, {"status": "unavailable", "reason": "validation_neutralizer.parquet is absent"}
    schema = pq.read_schema(path)
    columns = [name for name in schema.names if name.startswith("neutralizer_column_")]
    if not columns:
        raise ValueError("Neutralizer file exists but has no neutralizer_column_ columns")
    dates = pd.DatetimeIndex(scoring_frame["date"].unique())
    date_type = schema.field("date").type
    if pa.types.is_string(date_type) or pa.types.is_large_string(date_type):
        filter_dates = dates.strftime("%Y-%m-%d").tolist()
    elif pa.types.is_date(date_type):
        filter_dates = list(dates.date)
    else:
        filter_dates = list(dates.to_pydatetime())
    neutralizers = pd.read_parquet(path, columns=KEYS + columns, filters=[("date", "in", filter_dates)])
    neutralizers["date"] = pd.to_datetime(neutralizers["date"])
    if neutralizers.duplicated(KEYS).any():
        raise ValueError("Duplicate neutralizer ticker/date keys")
    aligned = scoring_frame[KEYS].merge(neutralizers, how="left", on=KEYS, validate="one_to_one", sort=False)
    if len(aligned) != len(scoring_frame) or aligned[columns].isna().any().any():
        raise ValueError("Incomplete neutralizer key coverage; refusing to silently filter scoring rows")
    if not np.isfinite(aligned[columns].to_numpy()).all():
        raise ValueError("Neutralizer matrix contains non-finite values")
    return aligned[columns], {"status": "available", "n_columns": len(columns), "coverage": 1.0}


def score_predictions(frame: pd.DataFrame, neutralizers: pd.DataFrame | None, split: str) -> pd.DataFrame:
    """Use the official scoring implementation without any permitted row dropping."""
    records = []
    n_dates = frame["date"].nunique()
    started = time.perf_counter()
    for i, (date, rows) in enumerate(frame.groupby("date", sort=True), start=1):
        rows = rows.sort_values("numerai_ticker")
        preds = rows[MODEL_NAMES]
        target = rows[MAIN_TARGET]
        if any(preds[c].nunique() < 2 for c in MODEL_NAMES):
            raise ValueError(f"Constant model prediction on {date}; correlation is undefined")
        corr = numerai_corr(preds, target, max_filtered_index_ratio=0)
        ncorr = (neutral_correlation(preds, neutralizers.loc[rows.index], target,
                                     max_filtered_index_ratio=0)
                 if neutralizers is not None else pd.Series(np.nan, index=MODEL_NAMES))
        if not np.isfinite(corr.to_numpy()).all():
            raise ValueError(f"Undefined CORR on {date}")
        if neutralizers is not None and not np.isfinite(ncorr.to_numpy()).all():
            raise ValueError(f"Undefined NCORR on {date}")
        for name in MODEL_NAMES:
            records.append({"split": split, "date": date, "model": name, "n_rows": len(rows),
                            "CORR": float(corr[name]), "NCORR": float(ncorr[name])})
        if i == 1 or i % 25 == 0 or i == n_dates:
            print(f"Scored {split}: {i}/{n_dates} dates, latest {pd.Timestamp(date).date()}, "
                  f"{time.perf_counter() - started:.1f}s", flush=True)
    return pd.DataFrame.from_records(records)


def predict_all(frame: pd.DataFrame, models: dict, model_features: dict) -> pd.DataFrame:
    predictions = frame[KEYS + [MAIN_TARGET]].copy()
    for name, model in models.items():
        predictions[name] = model.predict(frame[model_features[name]])
    predictions["E3_rank_ensemble"] = rank_ensemble(predictions, ["B0_small_main", "E2_small_aux20"])
    return predictions


def run(args: argparse.Namespace, memory_monitor: PeakMemoryMonitor | None = None) -> None:
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output} is not empty; use a new output path or explicit --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    processed, raw = root / "data/processed/v6.0", root / "data/raw/v6.0"
    metadata_path = raw / "features.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_sets = {name: [f for f in metadata["feature_sets"][name] if f != "feature_country"]
                    for name in ["small", "medium"]}
    if not all(feature_sets.values()):
        raise ValueError("Empty feature set")
    all_features = list(dict.fromkeys(feature_sets["small"] + feature_sets["medium"]))
    if any(not f.startswith("feature_") for f in all_features):
        raise ValueError("Only feature_ columns can enter training")
    params = dict(n_estimators=300, learning_rate=0.05, max_depth=5, num_leaves=31,
                  colsample_bytree=0.1, verbosity=-1, n_jobs=args.n_jobs,
                  random_state=args.seed, feature_fraction_seed=args.seed,
                  bagging_seed=args.seed, data_random_seed=args.seed,
                  deterministic=True, force_col_wise=True)
    run_info = {
        "status": "started", "started_utc": datetime.now(timezone.utc), "kind": "offline_pilot",
        "baseline_source": BASELINE_URL, "data_version": "v6.0", "main_target": MAIN_TARGET,
        "aux_target": AUX_TARGET, "seed": args.seed, "max_train_rows": args.max_train_rows,
        "params": params, "feature_counts": {k: len(v) for k, v in feature_sets.items()},
        "models": MODEL_NAMES, "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "pyarrow", "lightgbm", "numerai-tools", "scipy", "psutil", "threadpoolctl"]},
        "blas_thread_limit": args.n_jobs,
        "definitions": {"CORR": "official numerai_tools.scoring.numerai_corr",
                        "NCORR": "official numerai_tools.scoring.neutral_correlation",
                        "std": "population standard deviation ddof=0 across dates",
                        "sharpe": "mean/std, not annualized; overlapping 60-day targets",
                        "max_drawdown": "positive peak-to-trough drop in additive cumulative score including initial zero"},
    }
    write_json(output / "run.json", run_info)
    if memory_monitor is not None:
        memory_monitor.run_started = True
    started = time.perf_counter()
    neutralizer_file = raw / "validation_neutralizer.parquet"
    if not neutralizer_file.exists() and not args.allow_corr_fallback:
        raise FileNotFoundError("NCORR is the primary selection metric; download validation_neutralizer.parquet "
                                "or explicitly pass --allow-corr-fallback for a differently defined pilot")
    train = validate_frame(pd.read_parquet(processed / "train.parquet",
        columns=KEYS + all_features + [MAIN_TARGET, AUX_TARGET]), all_features, "train")
    tuning = validate_frame(pd.read_parquet(processed / "tuning.parquet",
        columns=KEYS + all_features + [MAIN_TARGET]), all_features, "tuning")
    last_train_date = train["date"].max()  # Before random row sampling.
    if tuning["date"].min() <= last_train_date + pd.offsets.BDay(60):
        raise ValueError("Train/tuning boundary violates the 60-business-day embargo")
    train_input_rows = len(train)
    train = train.sample(n=min(args.max_train_rows, len(train)), random_state=args.seed).reset_index(drop=True)
    training_sample_hash = hashlib.sha256(
        train[KEYS].to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")).hexdigest()
    models, training_info = {}, {}
    model_features = {"B0_small_main": feature_sets["small"], "E1_medium_main": feature_sets["medium"],
                      "E2_small_aux20": feature_sets["small"]}
    for name, target in [("B0_small_main", MAIN_TARGET), ("E1_medium_main", MAIN_TARGET),
                         ("E2_small_aux20", AUX_TARGET)]:
        mask = train[target].notna()
        if not mask.any():
            raise ValueError(f"No mature training rows for {target}")
        tick = time.perf_counter()
        model = lgb.LGBMRegressor(**params)
        model.fit(train.loc[mask, model_features[name]], train.loc[mask, target])
        seconds = time.perf_counter() - tick
        model.booster_.save_model(str(model_dir / f"{name}.txt"))
        models[name] = model
        training_info[name] = {"training_rows": int(mask.sum()), "target": target,
                               "n_features": len(model_features[name]), "training_seconds": seconds}
        print(f"Trained {name}: {mask.sum():,} rows, {seconds:.1f}s", flush=True)
    training_info["E3_rank_ensemble"] = {
        "training_rows": len(train), "target": "equal_weight_rank(B0,E2)", "n_features": len(feature_sets["small"]),
        "training_seconds": 0, "members": ["B0_small_main", "E2_small_aux20"]}
    del train
    tuning_preds = predict_all(tuning, models, model_features)
    tuning_neutralizers, tuning_status = read_neutralizers(raw / "validation_neutralizer.parquet", tuning_preds)
    tuning_metrics = score_predictions(tuning_preds, tuning_neutralizers, "tuning")
    selection_metric = "NCORR" if tuning_neutralizers is not None else "CORR"
    tuning_means = tuning_metrics.groupby("model")[selection_metric].mean()
    # Stable name tie-break is predeclared; holdout scores cannot affect selection.
    selected = sorted(MODEL_NAMES, key=lambda name: (-tuning_means[name], name))[0]
    selection = {"selected_model": selected, "selection_split": "tuning", "metric": selection_metric,
                 "tuning_mean_scores": tuning_means.to_dict(), "ncorr": tuning_status,
                 "frozen_at_utc": datetime.now(timezone.utc), "holdout_read_before_selection": False,
                 "tie_break": "alphabetical model name", "holdout_use": "frozen variants compared once; no reselection"}
    write_json(output / "selection.json", selection)
    print(f"Selection frozen using tuning {selection_metric}: {selected}; NCORR {tuning_status}", flush=True)
    del tuning, tuning_neutralizers
    # Holdout is first read after the experiment matrix and model selection are frozen.
    holdout = validate_frame(pd.read_parquet(processed / "holdout.parquet",
        columns=KEYS + all_features + [MAIN_TARGET]), all_features, "holdout")
    if holdout["date"].min() <= tuning_preds["date"].max() + pd.offsets.BDay(60):
        raise ValueError("Tuning/holdout boundary violates the 60-business-day embargo")
    holdout_preds = predict_all(holdout, models, model_features)
    holdout_neutralizers, holdout_status = read_neutralizers(raw / "validation_neutralizer.parquet", holdout_preds)
    holdout_metrics = score_predictions(holdout_preds, holdout_neutralizers, "holdout")
    per_date = pd.concat([tuning_metrics, holdout_metrics], ignore_index=True)
    summary = []
    for (split, model), group in per_date.groupby(["split", "model"], sort=True):
        for metric in ["CORR", "NCORR"]:
            summary.append({"split": split, "model": model, "metric": metric,
                            "n_dates": len(group), **summarize_scores(group.sort_values("date")[metric]),
                            **training_info[model], "selected_on_tuning": model == selected,
                            "seed": args.seed})
    pd.DataFrame(summary).drop(columns="members", errors="ignore").to_csv(output / "metrics.csv", index=False)
    per_date.to_csv(output / "per_date_metrics.csv", index=False, date_format="%Y-%m-%d")
    predictions = pd.concat([tuning_preds.assign(split="tuning"), holdout_preds.assign(split="holdout")], ignore_index=True)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    input_paths = [metadata_path] + [processed / f"{name}.parquet" for name in ["train", "tuning", "holdout"]]
    neutralizer_path = raw / "validation_neutralizer.parquet"
    if neutralizer_path.exists():
        input_paths.append(neutralizer_path)
    run_info.update({"status": "complete", "completed_utc": datetime.now(timezone.utc),
        "elapsed_seconds": time.perf_counter() - started, "training": training_info,
        "train_input_rows": train_input_rows, "train_sample_rows": min(args.max_train_rows, train_input_rows),
        "train_sample_keys_sha256": training_sample_hash, "train_max_date_before_sample": last_train_date,
        "splits": {name: {"rows": len(frame), "dates": frame["date"].nunique(),
                           "min_date": frame["date"].min(), "max_date": frame["date"].max()}
                   for name, frame in [("tuning", tuning_preds), ("holdout", holdout_preds)]},
        "ncorr": {"tuning": tuning_status, "holdout": holdout_status}, "selection": selection,
        "input_sha256": {str(path.relative_to(root)): sha256_file(path) for path in input_paths}})
    run_info["cleaning_provenance"] = {}
    for name in ["cleaning_report.json", "split_manifest.json"]:
        path = processed / name
        if path.exists():
            run_info["cleaning_provenance"][name] = {
                "path": str(path.relative_to(root)), "sha256": sha256_file(path),
                "content": json.loads(path.read_text(encoding="utf-8"))}
    write_json(output / "run.json", run_info)
    print(f"Complete: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/pilot"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--allow-corr-fallback", action="store_true",
                        help="Explicitly allow CORR selection if the primary NCORR neutralizer file is absent")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly rerun and replace output files")
    args = parser.parse_args()
    if args.max_train_rows < 1 or args.n_jobs < 1:
        parser.error("max-train-rows and n-jobs must be positive")
    memory_monitor = PeakMemoryMonitor(interval=1.0)
    failure = None
    try:
        with memory_monitor, threadpool_limits(limits=args.n_jobs):
            run(args, memory_monitor)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Do not alter an earlier completed run when the overwrite guard rejects
        # a launch. Stop/join the monitor before persisting either final status.
        if memory_monitor.run_started:
            manifest_path = args.output.resolve() / "run.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["memory"] = memory_monitor.result()
            if failure is not None:
                manifest.update(status="failed", error=failure, failed_at_utc=datetime.now(timezone.utc))
            write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
