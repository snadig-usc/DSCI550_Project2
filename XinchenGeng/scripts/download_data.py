"""Download public Numerai files atomically and record content hashes. No account needed."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from numerapi import NumerAPI
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
FILES = ["features.json", "train.parquet", "validation.parquet",
         "validation_neutralizer.parquet", "validation_example_preds.parquet"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(version: str, name: str) -> dict:
    dest = ROOT / "data" / "raw" / version / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    api = NumerAPI(show_progress_bars=False)
    query = "query($filename: String!) { dataset(filename: $filename) }"
    remote = f"{version}/{name}"
    url = api.raw_query(query, {"filename": remote})["data"]["dataset"]
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1)))
    status = "cached"
    if not dest.exists():
        partial = dest.with_suffix(dest.suffix + ".part")
        with session.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            expected = int(response.headers.get("Content-Length", 0))
            last_log = time.monotonic()
            with partial.open("wb") as f:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                    if time.monotonic() - last_log > 20:
                        print(f"{name}: {f.tell() / 1e6:.1f} MB downloaded", flush=True)
                        last_log = time.monotonic()
            if expected and partial.stat().st_size != expected:
                raise IOError(f"Incomplete download: {remote}")
        partial.replace(dest)
        status = "downloaded"
    if dest.suffix == ".parquet":
        import pyarrow.parquet as pq
        pq.ParquetFile(dest)  # Catch truncated/invalid Parquet files before use.
    else:
        json.loads(dest.read_text(encoding="utf-8"))
    record = {"dataset": remote, "path": str(dest.relative_to(ROOT)),
              "bytes": dest.stat().st_size, "sha256": digest(dest),
              "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": status, "seconds": round(time.perf_counter() - started, 2)}
    print(f"{name}: {status}, {record['bytes'] / 1e6:.1f} MB", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v6.0")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    api = NumerAPI(show_progress_bars=False)
    available = api.list_datasets()
    missing = [f"{args.version}/{f}" for f in FILES if f"{args.version}/{f}" not in available]
    if missing:
        raise RuntimeError(f"Required public files unavailable: {missing}")
    records = []
    manifest_path = ROOT / "data" / "raw" / args.version / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(lambda f: download(args.version, f), FILES):
            records.append(result)
            manifest_path.write_text(json.dumps({"version": args.version,
                "available_files": [x for x in available if x.startswith(args.version + "/")],
                "files": records}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
