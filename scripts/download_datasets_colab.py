from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path

import requests

PNEUMA_REPO = "https://github.com/EPFL-ENAC/pNEUMA.git"
PNEUMA_URL = "https://zenodo.org/records/10491409/files/pNEUMA_dataset.zip?download=1"
PNEUMA_ARCHIVE = "pNEUMA_dataset.zip"
CICV5G_REPO = "https://github.com/zxr805/CICV5G.git"
SEE_V2X_REPO = "https://github.com/UCR-CISL/SEE-V2X.git"
SEE_V2X_OUTDOOR_ID = "133L1r_C9y08jPSyzL64_28AVGkpXAzek"
SEE_V2X_ARCHIVE = "outdoor_parkinglot.tar.gz"


def run(command, cwd=None):
    command = [str(x) for x in command]
    print("$", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def clone_or_update(url: str, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if (target / ".git").exists():
        print("Repository already exists:", target)
        try:
            run(["git", "-C", target, "pull", "--ff-only"])
        except subprocess.CalledProcessError:
            print("Pull failed; retaining existing clone.")
        return
    if target.exists() and any(target.iterdir()):
        print("Existing non-empty path retained:", target)
        return
    run(["git", "clone", "--depth", "1", url, target])


def request_with_retries(url: str, attempts: int = 6):
    last = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                url,
                headers={
                    "Range": "bytes=0-0",
                    "User-Agent": "Mozilla/5.0 CA-DTNet-Colab/2.0",
                },
                timeout=180,
                allow_redirects=True,
                stream=True,
            )
            if response.status_code in {429,500,502,503,504}:
                raise requests.HTTPError(f"Temporary HTTP {response.status_code}")
            response.raise_for_status()
            response.close()
            return
        except Exception as exc:
            last = exc
            if attempt == attempts:
                break
            wait = min(60, 2 ** attempt)
            print(f"Request attempt {attempt}/{attempts} failed: {exc}; retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Official pNEUMA endpoint is unavailable.") from last


def selective_pneuma(output_dir: Path, count: int):
    from remotezip import RemoteZip
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.csv"))
    if count > 0 and len(existing) >= count:
        print("Required pNEUMA files already exist.")
        return existing[:count]
    request_with_retries(PNEUMA_URL)
    with RemoteZip(PNEUMA_URL, timeout=240, initial_buffer_size=8*1024*1024) as archive:
        members = [
            item.filename for item in archive.infolist()
            if item.filename.lower().endswith(".csv") and not item.filename.endswith("/")
        ]
        selected = members[:count] if count > 0 else members
        if not selected:
            raise RuntimeError("No CSV members found in pNEUMA archive.")
        outputs = []
        for name in selected:
            output = output_dir / Path(name).name
            print("pNEUMA:", name, "->", output)
            if not output.exists() or output.stat().st_size == 0:
                with archive.open(name) as source, output.open("wb") as target:
                    shutil.copyfileobj(source, target, length=16*1024*1024)
            outputs.append(output)
        return outputs


def full_pneuma_fallback(output_dir: Path, cache_dir: Path, count: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / PNEUMA_ARCHIVE
    required = int(15.8 * 1024**3 * 1.25)
    if shutil.disk_usage(cache_dir).free < required:
        raise RuntimeError("Not enough local Colab space for full pNEUMA fallback.")
    run([
        "aria2c", "--continue=true", "--max-tries=8", "--retry-wait=10",
        "--timeout=180", "--connect-timeout=60", "--split=8",
        "--max-connection-per-server=8", "--file-allocation=none",
        "--dir", cache_dir, "--out", PNEUMA_ARCHIVE, PNEUMA_URL,
    ])
    outputs = []
    with zipfile.ZipFile(archive_path) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith('.csv')]
        selected = members[:count] if count > 0 else members
        for name in selected:
            output = output_dir / Path(name).name
            if not output.exists() or output.stat().st_size == 0:
                with archive.open(name) as source, output.open('wb') as target:
                    shutil.copyfileobj(source, target, length=16*1024*1024)
            outputs.append(output)
    archive_path.unlink(missing_ok=True)
    return outputs


def prepare_pneuma(output_dir: Path, cache_dir: Path, count: int):
    try:
        return selective_pneuma(output_dir, count)
    except Exception as exc:
        print("Selective pNEUMA extraction failed:", repr(exc))
        print("Trying full local archive fallback.")
        return full_pneuma_fallback(output_dir, cache_dir, count)


def safe_target(base: Path, member_name: str):
    base = base.resolve()
    target = (base / member_name).resolve()
    if not str(target).startswith(str(base)):
        raise RuntimeError(f"Unsafe archive member: {member_name}")
    return target


def prepare_see_v2x(output_dir: Path, cache_dir: Path):
    import gdown
    existing = sorted(output_dir.rglob('rx_*_tx_*.csv'))
    if existing:
        print("SEE-V2X receiver files already exist:", len(existing))
        return existing
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / SEE_V2X_ARCHIVE
    downloaded = gdown.download(
        id=SEE_V2X_OUTDOOR_ID,
        output=str(archive_path),
        quiet=False,
        resume=True,
    )
    if downloaded is None or not archive_path.exists():
        raise RuntimeError("SEE-V2X Outdoor ParkingLot download failed.")
    extracted = []
    with tarfile.open(archive_path, mode='r:gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = Path(member.name).name
            lower = name.lower()
            keep = (lower.startswith('rx_') and lower.endswith('.csv')) or lower == 'parameters.csv'
            if not keep:
                continue
            relative = Path(*Path(member.name).parts[-4:])
            output = safe_target(output_dir, str(relative))
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists() or output.stat().st_size == 0:
                source = archive.extractfile(member)
                if source is None:
                    continue
                with source, output.open('wb') as target:
                    shutil.copyfileobj(source, target, length=16*1024*1024)
            extracted.append(output)
    archive_path.unlink(missing_ok=True)
    if not extracted:
        raise RuntimeError("No SEE-V2X receiver CSVs extracted.")
    print("Extracted SEE-V2X files:", len(extracted))
    return extracted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='/content/drive/MyDrive/CA_DTNet')
    parser.add_argument('--cache-root', default='/content/ca_dtnet_cache')
    parser.add_argument('--pneuma-count', type=int, default=2)
    parser.add_argument('--skip-pneuma', action='store_true')
    parser.add_argument('--skip-cicv5g', action='store_true')
    parser.add_argument('--skip-see-v2x', action='store_true')
    args = parser.parse_args()
    project = Path(args.project_root)
    raw = project / 'data' / 'raw'
    vendor = project / 'vendor'
    cache = Path(args.cache_root)
    for folder in (raw, vendor, cache):
        folder.mkdir(parents=True, exist_ok=True)
    clone_or_update(PNEUMA_REPO, vendor/'pNEUMA')
    clone_or_update(SEE_V2X_REPO, vendor/'SEE-V2X')
    if not args.skip_cicv5g:
        clone_or_update(CICV5G_REPO, raw/'cicv5g')
    if not args.skip_pneuma:
        prepare_pneuma(raw/'pneuma', cache/'pneuma', args.pneuma_count)
    if not args.skip_see_v2x:
        prepare_see_v2x(raw/'see_v2x', cache/'see_v2x')
    print('Dataset preparation complete:', raw)


if __name__ == '__main__':
    main()
