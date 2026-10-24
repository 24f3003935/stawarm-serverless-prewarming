"""Download the public traces into data/raw/.

    python scripts/data/fetch_traces.py azure2019               # Azure Functions 2019 (minute-level pilot only)
    python scripts/data/fetch_traces.py azure2021               # Azure Functions 2021 invocation trace
    python scripts/data/fetch_traces.py huawei                  # Huawei Cloud Region 2: cold starts, runtimes
    python scripts/data/fetch_traces.py huawei --requests       # + Region 2 day-30 request tables (large)

Downloads resume if interrupted: run the same command again.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

RAW = Path("data/raw")

AZURE2019_URL = ("https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/"
                 "azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz")
AZURE2021_URL = ("https://github.com/Azure/AzurePublicDataset/raw/master/data/"
                 "AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar")      # 18 MB; redirects to raw.githubusercontent.com

# Google Drive file ids from https://github.com/sir-lab/data-release/blob/main/README_data_release_2025.md
HUAWEI_R2 = {
    "cold_starts_region2": "1qjnBBH_yEOnxAG5vFPQwx6xyITZrrP8L",
    "timeseries_region2": "1iZk-n_5IT7j3M_pHpRzhwSPTuJ2vxvlR",
    "runtimes_triggers_region2": "1dUg_yxeLR5OyldivPZGW5djYF31166EG",
}
HUAWEI_R2_REQUESTS = [
    "1clZwg5z6vAIF38K3NSkAxuzASsa5GN4B", "1S-MpmXdRSxiADKMXKlDnVdHM5Hlw2cKv", "1o4_KES_thVoC-P98uneeVbGrQH1G3VqM",
    "1jcFk5XBhmkwiz9EH8c69WW0BGDJ1UFeQ", "1PqeJp9DrpkD3UHHbvizPOCfNGptpRgBI", "1eT9n9yQZUeKFAyIblS0ks1XqQ_g2OJZl",
    "1SG7tHcuflIE-EE8AoAp99aMVqEF0vJL8", "1mCFakKPn8crcNrzFjmBttIJY7-TW3c3r", "1lW4N3SrN_z5BgjkHjU2QtDolROs8YhP5",
    "1TBD_1GysKZyGUaFll0qUz5FPR1gVjX0B",
]


def download(url: str, dest: Path) -> Path:
    """Stream a file to disk with a progress line; resumes a partial download when the server allows it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "stawarm-fetch"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:                      # already complete
            print(f"  already downloaded: {dest}")
            return dest
        raise
    total = resp.headers.get("Content-Length")
    total = int(total) + have if total else None
    mode = "ab" if resp.status == 206 else "wb"
    done = have if mode == "ab" else 0
    with open(dest, mode) as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dest.name}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
            else:
                print(f"\r  {dest.name}: {done / 1e6:,.0f} MB", end="", flush=True)
    print()
    return dest


def find_7zip() -> str | None:
    for c in ["7z", "7za", r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"]:
        if shutil.which(c) or Path(c).exists():
            return shutil.which(c) or c
    return None


def extract_with_7zip(archive: Path, out: Path) -> None:
    exe = find_7zip()
    if exe is None:
        sys.exit(f"7-Zip not found. Install it from https://www.7-zip.org, or extract {archive} by hand into {out}.")
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([exe, "x", str(archive), f"-o{out}", "-y"], check=True, stdout=subprocess.DEVNULL)


def fetch_azure2019() -> None:
    out = RAW / "azure2019"
    if any(out.glob("invocations_per_function_md.anon.d*.csv")):
        print("Azure 2019 already present:", out)
        return
    arc = download(AZURE2019_URL, RAW / "az2019.tar.xz")
    print("  extracting (about 2 GB)...")
    with tarfile.open(arc, "r:xz") as t:
        t.extractall(out, filter="data")      # safe extraction; silences the Python 3.14 notice
    arc.unlink()
    print("done:", len(list(out.glob("*.csv"))), "CSV files in", out)


def fetch_azure2021() -> None:
    out = RAW / "azure2021"
    if any(out.rglob("*.txt")) or any(out.rglob("*.csv")):
        print("Azure 2021 already present:", out)
        return
    arc = RAW / "azure2021.rar"
    try:
        download(AZURE2021_URL, arc)
    except Exception as e:  # GitHub LFS sometimes refuses scripted downloads
        sys.exit(f"Download failed ({e}). Open https://github.com/Azure/AzurePublicDataset/blob/master/"
                 "AzureFunctionsInvocationTrace2021.md in a browser, download the .rar, save it as "
                 f"{arc}, then run this command again.")
    if arc.stat().st_size < 1_000_000:
        sys.exit(f"{arc} is too small to be the trace (Git LFS pointer?). Download it in a browser instead.")
    print("  extracting...")
    extract_with_7zip(arc, out)
    print("done:", [p.name for p in out.rglob("*") if p.is_file()])


def extract_archives(folder: Path) -> None:
    """Extract the archives in a download folder unless it already holds extracted data files."""
    files = [p for p in folder.rglob("*") if p.is_file()]
    if any(p.suffix in (".csv", ".parquet") or p.name.endswith(".csv.gz") for p in files):
        return
    for p in files:
        if p.name.endswith(".tar.gz") or p.suffix == ".tar":
            print("  extracting", p.name)
            with tarfile.open(p) as t:
                t.extractall(folder, filter="data")
        elif p.suffix == ".zip":
            print("  extracting", p.name)
            shutil.unpack_archive(str(p), str(folder))
        elif p.suffix in (".7z", ".rar"):
            print("  extracting", p.name)
            extract_with_7zip(p, folder)


def fetch_huawei(requests: bool) -> None:
    try:
        import gdown
    except ImportError:
        sys.exit("Please run:  pip install gdown   (it downloads Google Drive files)")
    out = RAW / "huawei"
    out.mkdir(parents=True, exist_ok=True)
    items = list(HUAWEI_R2.items())
    if requests:
        items += [(f"requests_region2_part{k+1}", fid) for k, fid in enumerate(HUAWEI_R2_REQUESTS)]
    failed = []
    for name, fid in items:
        target = out / name
        if target.exists() and any(target.iterdir()):
            print(f"  {name}: already present")
        else:
            target.mkdir(exist_ok=True)
            print(f"  {name}")
            try:
                # a trailing path separator tells gdown to keep the original file name inside this folder
                path = gdown.download(id=fid, output=os.path.join(str(target), ""), quiet=False)
            except Exception as e:  # Google Drive refuses scripted downloads once a file's daily quota is used
                print(f"  failed: {type(e).__name__}")
                path = None
            if path is None:
                failed.append((fid, target))
                continue
        extract_archives(target)
    if failed:
        lines = [f"  https://drive.google.com/file/d/{fid}  ->  save into {target}" for fid, target in failed]
        sys.exit(f"{len(failed)} file(s) could not be downloaded (Google Drive may be rate-limiting). Try again "
                 "later, or download them in a browser and save each into the folder shown (archives are "
                 "extracted automatically when this command is run again):\n" + "\n".join(lines))
    print("done:", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    sub.add_parser("azure2019")
    sub.add_parser("azure2021")
    h = sub.add_parser("huawei"); h.add_argument("--requests", action="store_true")
    args = ap.parse_args()
    if args.what == "azure2019":
        fetch_azure2019()
    elif args.what == "azure2021":
        fetch_azure2021()
    else:
        fetch_huawei(args.requests)


if __name__ == "__main__":
    main()
