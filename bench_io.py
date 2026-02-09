import re
from pathlib import Path
from typing import List


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return cleaned or "model"


def find_input_files(input_dir: Path) -> List[Path]:
    files = [p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    return sorted(files)
