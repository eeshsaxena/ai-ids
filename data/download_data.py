"""Download and verify NSL-KDD dataset."""

import os
import requests
import yaml
from pathlib import Path
from tqdm import tqdm


def download_file(url: str, dest: str) -> None:
    """Download a file from url with progress bar."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        desc=os.path.basename(dest),
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def download_nsl_kdd(config_path: str = "config/config.yaml") -> None:
    """Download NSL-KDD train and test datasets."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    files = [
        (cfg["data"]["nsl_kdd_train_url"], cfg["data"]["train_path"]),
        (cfg["data"]["nsl_kdd_test_url"], cfg["data"]["test_path"]),
    ]

    for url, dest in files:
        if Path(dest).exists():
            print(f"[✓] Already exists: {dest}")
        else:
            print(f"[↓] Downloading {dest}...")
            download_file(url, dest)
            print(f"[✓] Saved: {dest}")

    print("\n[✓] NSL-KDD dataset ready.")


if __name__ == "__main__":
    download_nsl_kdd()
