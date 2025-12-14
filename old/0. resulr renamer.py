import json
import shutil
from pathlib import Path

RAW_DIR = Path("data/raw")
NEW_DIR = Path("data/new")
NEW_DIR.mkdir(parents=True, exist_ok=True)

for subdir in RAW_DIR.iterdir():
    if not subdir.is_dir():
        continue

    src = subdir / "result.json"
    if not src.exists():
        continue

    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    name = data.get("name")
    if not name:
        continue

    safe_name = "".join(c for c in name if c not in r'\/:*?"<>|')
    dst = NEW_DIR / f"{safe_name}.json"

    shutil.copyfile(src, dst)