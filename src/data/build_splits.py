"""Build reproducible train/val/test splits from Oxford-IIIT Pet.

Test split is taken verbatim from the official `annotations/test.txt`.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOT_DIR = REPO_ROOT / "data" / "Data" / "annotations"
OUT_DIR = REPO_ROOT / "splits"

FIELDS = ["name", "class_id", "species"]


def parse(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, class_id, species, _breed = line.split()
        rows.append({"name": name, "class_id": int(class_id), "species": int(species)})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: r["name"])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows_sorted)


def class_histogram(rows: list[dict]) -> Counter:
    return Counter(r["class_id"] for r in rows)


def main() -> None:
    trainval = parse(ANNOT_DIR / "trainval.txt")
    test_rows = parse(ANNOT_DIR / "test.txt")
    print(f"trainval.txt: {len(trainval)} images, {len(set(r['class_id'] for r in trainval))} classes")
    print(f"test.txt:     {len(test_rows)} images")

    train_rows, val_rows = train_test_split(
        trainval,
        test_size=0.20,
        random_state=SEED,
        stratify=[r["class_id"] for r in trainval],
    )
    write_csv(OUT_DIR / "train.csv", train_rows)
    write_csv(OUT_DIR / "val.csv", val_rows)
    write_csv(OUT_DIR / "test.csv", test_rows)

    print()
    print("=== sizes ===")
    for name, r in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        h = class_histogram(r)
        n_cat = sum(1 for x in r if x["species"] == 1)
        n_dog = sum(1 for x in r if x["species"] == 2)
        print(f"  {name:6s} {len(r):5d} images | {len(h):2d} classes "
              f"(per-class min/mean/max = {min(h.values())}/{sum(h.values())//len(h)}/{max(h.values())}) "
              f"| species: {n_cat} cat / {n_dog} dog")

    print()
    print("=== integrity ===")
    tn = {r["name"] for r in train_rows}
    vn = {r["name"] for r in val_rows}
    sn = {r["name"] for r in test_rows}
    print(f"  train ∩ val:  {len(tn & vn)} (must be 0)")
    print(f"  train ∩ test: {len(tn & sn)} (must be 0)")
    print(f"  val ∩ test:   {len(vn & sn)} (must be 0)")


if __name__ == "__main__":
    main()
