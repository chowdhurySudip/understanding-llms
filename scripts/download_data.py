#!/usr/bin/env python3
"""
Cross-platform data downloader for the project.

Usage:
    python scripts/download_data.py            # uses ./data by default
    python scripts/download_data.py --data-dir path/to/data

This script checks for the following files in the data directory and downloads them if missing:
- TinyStoriesV2-GPT4-train.txt
- TinyStoriesV2-GPT4-valid.txt
- owt_train.txt  (downloaded as owt_train.txt.gz then gunzipped)
- owt_valid.txt  (downloaded as owt_valid.txt.gz then gunzipped)

The script uses the standard library only (urllib, gzip) so it works on Windows, Linux and macOS without extra dependencies.
"""
from __future__ import annotations
import argparse
import gzip
import os
import shutil
import sys
import urllib.request
from pathlib import Path

FILES = [
    {
        "name": "TinyStoriesV2-GPT4-train.txt",
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
        "gz": False,
    },
    {
        "name": "TinyStoriesV2-GPT4-valid.txt",
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
        "gz": False,
    },
    {
        "name": "owt_train.txt",
        "url": "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz",
        "gz": True,
        "gz_name": "owt_train.txt.gz",
    },
    {
        "name": "owt_valid.txt",
        "url": "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz",
        "gz": True,
        "gz_name": "owt_valid.txt.gz",
    },
]

CHUNK = 1024 * 64


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # fallback for some Windows consoles
        print(*[str(a).encode('utf-8', errors='ignore').decode('ascii', errors='ignore') for a in args], **kwargs)


def download_url(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".downloading")
    safe_print(f"Downloading: {url} -> {dest}")
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            total = resp.getheader("Content-Length")
            if total is not None:
                total = int(total)
            downloaded = 0
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    safe_print(f"\r  {downloaded}/{total} bytes ({pct}%)", end="")
            if total:
                safe_print()
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        raise


def gunzip_file(gz_path: Path, out_path: Path) -> None:
    safe_print(f"Decompressing: {gz_path} -> {out_path}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def ensure_file(entry: dict, data_dir: Path) -> None:
    target = data_dir / entry["name"]

    if entry.get("gz"):
        gz_name = entry.get("gz_name")
        gz_path = data_dir / gz_name
        # If target already exists, nothing to do
        if target.exists():
            safe_print(f"Found existing file: {target}")
            return
        # If gz exists but not target, decompress
        if gz_path.exists() and not target.exists():
            gunzip_file(gz_path, target)
            return
        # Otherwise download gz then decompress
        if not gz_path.exists():
            download_url(entry["url"], gz_path)
            gunzip_file(gz_path, target)
            return
    else:
        if target.exists():
            safe_print(f"Found existing file: {target}")
            return
        # direct download
        download_url(entry["url"], target)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download dataset files into a data directory")
    parser.add_argument("--data-dir", default="data", help="Directory to store dataset files (default: ./data)")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    safe_print(f"Using data directory: {data_dir.resolve()}")

    for entry in FILES:
        try:
            ensure_file(entry, data_dir)
        except Exception as e:
            safe_print(f"Error while handling {entry}: {e}")
            safe_print("Aborting.")
            sys.exit(1)

    safe_print("All files present.")


if __name__ == "__main__":
    main()
