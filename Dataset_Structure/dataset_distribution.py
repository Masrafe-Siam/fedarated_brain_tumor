import os
import random
import shutil
from pathlib import Path
import math

# --- CONFIG ---
MAIN_DIR = r"D:\Masrafe\Coding\Git_Hub_code\ml_project\fedarated_brain_tumor\Dataset_2\Training\pituitary_tumor"      # folder with images
OUTPUT_BASE = r"D:\Masrafe\Coding\Git_Hub_code\ml_project\fedarated_brain_tumor\test"   # parent folder where set1..set4 will be created
SET_NAMES = ["set1", "set2", "set3", "set4"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
SEED = None   # None for true randomness

# -------------

def list_images(dir_path):
    """List images in the given folder."""
    return [
        f.name for f in Path(dir_path).iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ]

def ensure_dirs():
    Path(OUTPUT_BASE).mkdir(parents=True, exist_ok=True)
    for name in SET_NAMES:
        Path(OUTPUT_BASE, name).mkdir(parents=True, exist_ok=True)

def move_batch(batch_files, src_dir, dst_dir):
    for fname in batch_files:
        shutil.move(str(Path(src_dir, fname)), str(Path(dst_dir, fname)))

def main():
    if SEED is not None:
        random.seed(SEED)

    ensure_dirs()

    total_imgs = list_images(MAIN_DIR)
    total_count = len(total_imgs)

    if total_count < 4:
        raise ValueError("Not enough images to split into 4 sets.")

    # Calculate size for first 3 sets
    batch_size = math.floor(total_count / 4)

    for i, set_name in enumerate(SET_NAMES):
        remaining = list_images(MAIN_DIR)

        if i < len(SET_NAMES) - 1:
            k = batch_size
        else:
            # last set gets all remaining images
            k = len(remaining)

        chosen = random.sample(remaining, k)
        dst = Path(OUTPUT_BASE, set_name)
        move_batch(chosen, MAIN_DIR, dst)

        print(f"Moved {len(chosen)} images to {dst} | Remaining: {len(list_images(MAIN_DIR))}")

    print("Done ✅")

if __name__ == "__main__":
    main()
