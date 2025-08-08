import os
import random
import shutil
from pathlib import Path

# --- CONFIG ---
MAIN_DIR = r"D:\Masrafe\Coding\Git_Hub_code\ml_project\fedarated_brain_tumor\Dataset_2\pituitary_tumor"       # folder with all images
OUTPUT_BASE = r"D:\Masrafe\Coding\Git_Hub_code\ml_project\fedarated_brain_tumor\test"     # folder where train/test/val will be saved
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
SEED = None   # None = different every run, or set to an int for reproducibility
MOVE_FILES = True  # True = move images, False = copy images

# -------------

def list_images(dir_path):
    return [
        f for f in Path(dir_path).iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ]

def ensure_dirs():
    for sub in ["train", "test", "val"]:
        Path(OUTPUT_BASE, sub).mkdir(parents=True, exist_ok=True)

def transfer_files(files, dst_dir):
    for file_path in files:
        if MOVE_FILES:
            shutil.move(str(file_path), str(Path(dst_dir, file_path.name)))
        else:
            shutil.copy2(str(file_path), str(Path(dst_dir, file_path.name)))

def main():
    if SEED is not None:
        random.seed(SEED)

    ensure_dirs()

    all_images = list_images(MAIN_DIR)
    total = len(all_images)
    if total < 10:
        raise ValueError("Not enough images to split.")

    random.shuffle(all_images)

    train_end = int(0.8 * total)
    test_end = train_end + int(0.1 * total)

    train_files = all_images[:train_end]
    test_files = all_images[train_end:test_end]
    val_files = all_images[test_end:]

    transfer_files(train_files, Path(OUTPUT_BASE, "train"))
    transfer_files(test_files, Path(OUTPUT_BASE, "test"))
    transfer_files(val_files, Path(OUTPUT_BASE, "val"))

    print(f"Total images: {total}")
    print(f"Train: {len(train_files)}, Test: {len(test_files)}, Val: {len(val_files)}")
    print("Done ✅")

if __name__ == "__main__":
    main()
