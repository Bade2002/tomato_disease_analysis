import os
from pathlib import Path
from PIL import Image

DATASET_PATH = r"C:\Users\prajw\Downloads\archive\PlantVillage\dataset_split"  # <-- Update this path


def verify_dataset(base_dir):
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: Path '{base_path}' does not exist!")
        return

    splits = [d for d in base_path.iterdir() if d.is_dir()]
    print(f"Dataset root: {base_path}")
    print(f"Found splits: {[s.name for s in splits]}\n")

    total_images = 0
    corrupted_images = 0

    for split in sorted(splits):
        print(f"=== Split: {split.name} ===")
        classes = [c for c in split.iterdir() if c.is_dir()]
        split_total = 0

        for cls in sorted(classes):
            files = [f for f in cls.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            count = len(files)
            split_total += count
            print(f"  └── {cls.name:<30}: {count:>5} images")

            # Quick integrity check on the first 5 images per class
            for file_path in files[:5]:
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                except Exception as e:
                    print(f"      [CORRUPT FILE] {file_path.name}: {e}")
                    corrupted_images += 1

        print(f"  Total for '{split.name}': {split_total} images\n")
        total_images += split_total

    print(f"=====================================")
    print(f"Total Dataset Images Found: {total_images}")
    print(f"Corrupted samples detected: {corrupted_images}")


if __name__ == "__main__":
    verify_dataset(DATASET_PATH)