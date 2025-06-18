import argparse
import datetime as dt
import gzip
import shutil
import sys
from pathlib import Path
from typing import List

import requests
from tqdm import tqdm

# -------------------------------------------------------------
# Binance depth snapshot URL pattern (spot market)
# Example:
# https://data.binance.vision/data/spot/daily/depth/BTCUSDT/20/2023-12-01.zip
# -------------------------------------------------------------
BASE_URL = "https://data.binance.vision/data/spot/daily/depth"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_DEPTH = 20  # levels captured in snapshot file
DOWNLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "binance"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def daterange(start_date: dt.date, end_date: dt.date):
    """
    Yield every date from start_date to end_date inclusive.
    Args:
        start_date: The start date
        end_date: The end date
    """
    for n in range((end_date - start_date).days + 1):
        yield start_date + dt.timedelta(n)


def build_url(symbol: str, depth: int, date: dt.date) -> str:
    """
    Build the URL for a depth snapshot from Binance.
    Args:
        symbol: The trading pair symbol, e.g. BTCUSDT
        depth: The depth of the order book, e.g. 20
        date: The date of the snapshot
    """
    # NOTE: They don't have historical depth data here, visit BASE URL and explore options when getting data from Binance
    return f"{BASE_URL}/{symbol.upper()}/{symbol.upper()}-depth-{depth}-{date.isoformat()}.zip"



def download_file(url: str, out_path: Path, chunk_size: int = 1 << 20):
    """
    Stream-download a file with a progress bar.
    Args:
        url: The URL to the file
        out_path: The path to the output file
        chunk_size: The size of the chunks to download
    """
    resp = requests.get(url, stream=True, timeout=30)

    if resp.status_code == 404:
        return False  # file not found
    resp.raise_for_status()

    # Calculate progress
    total = int(resp.headers.get("Content-Length", 0))
    progress = tqdm(
        resp.iter_content(chunk_size=chunk_size),
        total=total // chunk_size + 1,
        unit="MB",
        desc=out_path.name,
        leave=False,
    )

    # Write the file to the output path
    with out_path.open("wb") as f:
        for chunk in progress:
            if chunk:
                f.write(chunk)
    return True


def maybe_unzip(zip_path: Path):
    """
    Unzip the .zip into a .gz for NDJSON and remove original zip to save space.
    Args:
        zip_path: The path to the zip file
    """
    if not zip_path.exists():
        return
    
    target_gz = zip_path.with_suffix(".jsonl.gz")
    # Check if the file already exists
    if target_gz.exists():
        zip_path.unlink(missing_ok=True)
        return  # already processed
    
    # Binance archives contain a single .json file inside the zip
    import zipfile

    # Unzip the file and write to the target path
    with zipfile.ZipFile(zip_path) as zf, gzip.open(target_gz, "wb") as gz_out:
        json_name = zf.namelist()[0]
        with zf.open(json_name) as json_f:
            shutil.copyfileobj(json_f, gz_out)
    zip_path.unlink(missing_ok=True)


def download_range(symbol: str, depth: int, start: dt.date, end: dt.date):
    """
    Download a range of depth snapshots from Binance and save them to the data directory.
    Args:
        symbol: The trading pair symbol, e.g. BTCUSDT
        depth: The depth of the order book, e.g. 20
        start: The start date of the range
        end: The end date of the range
    """
    downloaded_files: List[Path] = []
    
    # Loop through the dates and download the files
    for day in daterange(start, end):
        url = build_url(symbol, depth, day)
        out_path = DOWNLOAD_DIR / f"{symbol}_{depth}_{day.isoformat()}.zip"

        # Check if the file already exists
        if out_path.exists():
            print(f"✔ Already have {out_path.name}, skipping")
            continue # Skip if the file already exists
        print(f"⇩ Fetching {url}")

        # Download the file
        ok = download_file(url, out_path)

        # error handling
        if not ok:
            print(f"✖ {day.isoformat()} not available (404)")
            continue

        # Add the file to the list of downloaded files
        downloaded_files.append(out_path)
        maybe_unzip(out_path)
    print(f"Done. {len(downloaded_files)} files downloaded.")


def parse_args(argv):
    """
    Parse the command line arguments.
    Args:
        argv: The command line arguments
    """
    p = argparse.ArgumentParser(description="Download Binance depth snapshots (daily)")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Trading pair symbol, e.g. BTCUSDT")
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH, choices=[5, 10, 20, 50, 100])
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    return p.parse_args(argv)


# Main function to run the script
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end)
    download_range(args.symbol, args.depth, start_date, end_date) 