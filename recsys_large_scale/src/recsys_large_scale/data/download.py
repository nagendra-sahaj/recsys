import shutil
import urllib.request
import zipfile
from pathlib import Path

from recsys_large_scale.utils.logging import get_logger

logger = get_logger(__name__)


def download_and_extract(url: str, raw_dir: str | Path) -> Path:
    """Download a zip from `url` and extract it into `raw_dir`.

    Skips the download if `raw_dir` already has content, so this is safe to re-run.
    Returns the path to the extracted top-level folder (if the zip has exactly one),
    else `raw_dir` itself.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if any(raw_dir.iterdir()):
        logger.info(f"{raw_dir} is not empty, skipping download (delete it to re-download)")
        return _single_subdir_or_self(raw_dir)

    zip_path = raw_dir / "download.zip"
    logger.info(f"Downloading {url} -> {zip_path}")

    with urllib.request.urlopen(url) as response, open(zip_path, "wb") as out_file:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        while chunk := response.read(chunk_size):
            out_file.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100 * downloaded / total
                print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)
        print()

    logger.info(f"Extracting {zip_path} -> {raw_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    zip_path.unlink()

    return _single_subdir_or_self(raw_dir)


def _single_subdir_or_self(raw_dir: Path) -> Path:
    subdirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and not any(p.is_file() for p in raw_dir.iterdir()):
        return subdirs[0]
    return raw_dir


def clear_raw_dir(raw_dir: str | Path) -> None:
    raw_dir = Path(raw_dir)
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
