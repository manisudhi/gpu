#!/usr/bin/env python3
"""
scripts/download_data.py
Idempotent data acquisition script for NCBI Pathogen Detection AMR data.
Targets: Klebsiella pneumoniae (PDG000000012.2550.amr.metadata.tsv)
"""

import os
import sys
import argparse
import logging
import urllib.request
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("download_data")

NCBI_KLEBSIELLA_URL = "https://ftp.ncbi.nlm.nih.gov/pathogen/Results/Klebsiella/latest_kmer/AMR/PDG000000012.2550.amr.metadata.tsv"
DEFAULT_FILENAME = "PDG000000012.2550.amr.metadata.tsv"


def download_file(url: str, dest_path: Path, max_bytes: int = None, chunk_size: int = 1024 * 1024) -> None:
    """Download a file with resume support or bounded max_bytes."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info(f"Target file already exists: {dest_path} ({dest_path.stat().st_size / (1024*1024):.2f} MB). Skipping download.")
        return

    logger.info(f"Connecting to NCBI FTP: {url}")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)

    start_time = time.time()
    downloaded = 0

    with urllib.request.urlopen(req, timeout=30) as response, open(temp_path, "wb") as out_file:
        total_size = response.headers.get("Content-Length")
        if total_size:
            total_size = int(total_size)
            logger.info(f"Remote file size: {total_size / (1024*1024):.2f} MB")
        else:
            logger.info("Remote file size: Unknown (streaming)")

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            elapsed = time.time() - start_time
            speed_mb = (downloaded / (1024 * 1024)) / (elapsed + 1e-6)
            logger.info(f"Downloaded {downloaded / (1024*1024):.2f} MB [{speed_mb:.2f} MB/s]")
            if max_bytes and downloaded >= max_bytes:
                logger.info(f"Reached requested limit of {max_bytes / (1024*1024):.2f} MB.")
                break

    temp_path.rename(dest_path)
    logger.info(f"Download complete: {dest_path} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")


def inspect_downloaded_data(file_path: Path) -> dict:
    """Read header and inspect row counts and AST phenotype coverage."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    total_rows = 0
    ast_rows = 0
    header = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        header_line = f.readline().strip()
        header = header_line.split("\t")
        if "AST_phenotypes" not in header or "AMR_genotypes" not in header:
            logger.warning("Header missing standard AST_phenotypes or AMR_genotypes columns!")
            return {"header": header, "rows": 0, "ast_rows": 0}

        ast_idx = header.index("AST_phenotypes")
        for line in f:
            total_rows += 1
            parts = line.strip().split("\t")
            if len(parts) > ast_idx and parts[ast_idx] not in ("NULL", ""):
                ast_rows += 1

    summary = {
        "file": str(file_path),
        "columns_count": len(header),
        "total_records": total_rows,
        "ast_records": ast_rows,
        "ast_coverage_pct": (ast_rows / max(total_rows, 1)) * 100,
    }
    logger.info(f"Data Summary: {summary['total_records']:,} records, {summary['ast_records']:,} with AST phenotypes ({summary['ast_coverage_pct']:.2f}% coverage)")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Download NCBI AMR Metadata for Klebsiella pneumoniae")
    parser.add_argument("--url", default=NCBI_KLEBSIELLA_URL, help="NCBI TSV URL")
    parser.add_argument("--dest-dir", default="data/raw", help="Target directory")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="Target filename")
    parser.add_argument("--max-mb", type=int, default=None, help="Max MB to download (optional)")
    args = parser.parse_args()

    dest_path = Path(args.dest_dir) / args.filename
    max_bytes = args.max_mb * 1024 * 1024 if args.max_mb else None

    download_file(args.url, dest_path, max_bytes=max_bytes)
    summary = inspect_downloaded_data(dest_path)
    print("\n--- Download & Inspection Complete ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
