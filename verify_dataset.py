#!/usr/bin/env python3
"""Verify poultry dataset: count per class, total, split stats."""
import json, sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# Try common locations
candidates = [
    DATA_DIR / "images",
    DATA_DIR / "raw",
    DATA_DIR,
    ROOT / "paper",  # fallback
]
# Actually dataset after download should be data/images/<split>/<class> or data/raw/<class>
# We'll recursively find all images

CLASS_NAMES = ["cocci", "healthy", "ncd", "salmo"]
ALT_NAMES = {
    "coccidiosis": "cocci",
    "healthy": "healthy",
    "newcastle": "ncd",
    "salmonella": "salmo",
    "salmo": "salmo",
    "ncd": "ncd",
}

def count_images(root: Path):
    counts = Counter()
    total = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpg",".jpeg",".png",".bmp"):
            # parent dir name as class hint
            cls = p.parent.name.lower()
            # normalize
            cls_norm = ALT_NAMES.get(cls, cls)
            counts[cls_norm] += 1
            total += 1
    return counts, total

def main():
    print(f"Scanning {DATA_DIR} ...")
    if not DATA_DIR.exists():
        print(f"DATA_DIR not found: {DATA_DIR}")
        sys.exit(1)
    counts, total = count_images(DATA_DIR)
    print(f"Total images found: {total}")
    for k,v in sorted(counts.items()):
        pct = 100*v/total if total else 0
        print(f"  {k}: {v} ({pct:.1f}%)")
    # Also try to find split dirs
    for sub in ["train","val","test","images/train","images/val","images/test"]:
        p = DATA_DIR / sub
        if p.exists():
            c,t = count_images(p)
            print(f"\n[{sub}] total={t} -> {dict(c)}")

    out = {
        "total": total,
        "per_class": dict(counts),
        "per_class_pct": {k: round(100*v/total,2) if total else 0 for k,v in counts.items()},
        "scanned_root": str(DATA_DIR),
    }
    out_path = ROOT / "results" / "dataset_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path,"w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")

if __name__ == "__main__":
    main()
