"""Build train/val/test folder hierarchies for torchvision.datasets.ImageFolder.

Output layout:
    data/folders/train/{class_name}/<image>.jpg
    data/folders/val/{class_name}/<image>.jpg
    data/folders/test/{class_name}/<image>.jpg

Train/val: stratified 80/20 from the official `annotations/trainval.txt`.
Test: copied verbatim from the official `annotations/test.txt`.
"""
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOT_DIR = REPO_ROOT / "data" / "Data" / "annotations"
IMAGES_DIR = REPO_ROOT / "data" / "Data" / "images"
OUT_DIR = REPO_ROOT / "data" / "folders"


def class_from_name(name: str) -> str:
    """`Abyssinian_100` → `Abyssinian`."""
    return name.rsplit("_", 1)[0]


def parse(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split()[0]
        rows.append((name, class_from_name(name)))
    return rows


def copy_to_folders(rows: list[tuple[str, str]], out_split_dir: Path) -> None:
    for name, cls in rows:
        dst_dir = out_split_dir / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(IMAGES_DIR / f"{name}.jpg", dst_dir / f"{name}.jpg")


def summarize(name: str, rows: list[tuple[str, str]]) -> None:
    h = Counter(cls for _, cls in rows)
    n_cat = sum(v for c, v in h.items() if c[0].isupper())
    n_dog = sum(v for c, v in h.items() if c[0].islower())
    print(f"  {name:5s} {len(rows):5d} images | {len(h):2d} classes "
          f"(per-class min/mean/max = {min(h.values())}/{sum(h.values())//len(h)}/{max(h.values())}) "
          f"| species: {n_cat} cat / {n_dog} dog")


def main() -> None:
    trainval = parse(ANNOT_DIR / "trainval.txt")
    test_rows = parse(ANNOT_DIR / "test.txt")
    print(f"trainval.txt: {len(trainval)} images, {len({c for _, c in trainval})} classes")
    print(f"test.txt:     {len(test_rows)} images")

    classes = [c for _, c in trainval]
    train_rows, val_rows = train_test_split(
        trainval,
        test_size=0.20,
        random_state=SEED,
        stratify=classes,
    )

    print()
    print("=== copying ===")
    print(f"  train ({len(train_rows)}) → {OUT_DIR / 'train'}")
    copy_to_folders(train_rows, OUT_DIR / "train")
    print(f"  val   ({len(val_rows)}) → {OUT_DIR / 'val'}")
    copy_to_folders(val_rows, OUT_DIR / "val")
    print(f"  test  ({len(test_rows)}) → {OUT_DIR / 'test'}")
    copy_to_folders(test_rows, OUT_DIR / "test")

    print()
    print("=== sizes ===")
    summarize("train", train_rows)
    summarize("val", val_rows)
    summarize("test", test_rows)


if __name__ == "__main__":
    main()
