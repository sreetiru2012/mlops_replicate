import argparse
from pathlib import Path
import random
import shutil

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    images = []
    for label_dir in in_dir.iterdir():
        for img in label_dir.glob("*"):
            images.append((img, label_dir.name))

    random.shuffle(images)

    n = len(images)
    n_train = int(n * args.train)
    n_val = int(n * args.val)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:]
    }

    for split_name, items in splits.items():
        for img, label in items:
            dest = out_dir / split_name / label / img.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest)

    print("Split complete.")

if __name__ == "__main__":
    main()