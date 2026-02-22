import argparse
from pathlib import Path
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def preprocess_image(in_path: Path, out_path: Path, size: int = 224):
    with Image.open(in_path) as im:
        im = im.convert("RGB")
        im = im.resize((size, size))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path, format="JPEG")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--size", type=int, default=224)
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    for class_folder in raw_dir.iterdir():
        if not class_folder.is_dir():
            continue

        label = 0 if class_folder.name.lower() == "cat" else 1

        for img in class_folder.glob("*"):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue

            out_path = out_dir / str(label) / img.name
            try:
                preprocess_image(img, out_path, args.size)
            except:
                continue

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()