# import os
# import sys
# import cv2
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
# from collections import OrderedDict
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# PROJECT_ROOT = r"D:\Masrafe\Coding\Git_Hub_code\ml_project\fedarated_brain_tumor"
# DATA_DIR     = r"D:\Masrafe\Coding\Git_Hub_code\ml_project\fedarated_brain_tumor\Dataset_1"
# IMG_SIZE     = (224, 224)
# DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# sys.path.append(PROJECT_ROOT)
# from models.model_factory import get_model  

# def medical_preprocess_gray(image: np.ndarray) -> np.ndarray:
#     """Apply CLAHE + resize + normalization for CT scans."""
#     image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     image = clahe.apply(image)
#     image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
#     return image.astype(np.uint8)

# transform = A.Compose([
#     A.Resize(IMG_SIZE[1], IMG_SIZE[0]),
#     A.Normalize(mean=[0.5], std=[0.5]),
#     ToTensorV2()
# ])

# def prepare_tensor(img_path: str):
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise ValueError(f"Cannot load image: {img_path}")
#     img = medical_preprocess_gray(img)
#     tensor = transform(image=img)["image"].unsqueeze(0).to(DEVICE)
#     return tensor, img

# def load_model(model_path: str, num_classes: int):
#     """Load checkpoint safely into model."""
#     model = get_model("customcnn", num_classes=num_classes, pretrained=False, dropout_rate=0.5).to(DEVICE)
#     ckpt = torch.load(model_path, map_location=DEVICE)

#     if "model_state_dict" in ckpt:
#         state_dict = ckpt["model_state_dict"]
#     elif "state_dict" in ckpt:
#         state_dict = ckpt["state_dict"]
#     else:
#         state_dict = ckpt

#     clean_sd = OrderedDict()
#     for k, v in state_dict.items():
#         clean_sd[k.replace("module.", "", 1) if k.startswith("module.") else k] = v

#     model.load_state_dict(clean_sd, strict=False)
#     model.eval()
#     return model

# def predict_and_save(model, img_path: str, class_names, out_dir: str):
#     tensor, disp_img = prepare_tensor(img_path)

#     with torch.no_grad():
#         outputs = model(tensor)
#         probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
#         pred_idx = int(np.argmax(probs))
#         pred_class = class_names[pred_idx]
#         conf = probs[pred_idx] * 100

#     # Save output with prediction title
#     plt.imshow(disp_img, cmap="gray")
#     plt.title(f"Predicted = {pred_class} ({conf:.2f}%)")
#     plt.axis("off")

#     os.makedirs(out_dir, exist_ok=True)
#     base_name = os.path.basename(img_path)
#     save_path = os.path.join(out_dir, base_name)
#     plt.savefig(save_path, bbox_inches="tight")
#     plt.close()

#     print(f"Saved prediction image: {save_path}")
#     return pred_class, conf, probs

# def main():
#     model_path = r"D:\Masrafe\Coding\Git_Hub_code\ml_project\fedarated_brain_tumor\Result\FLResult\fl_results_20251008_205502\last_global_model.pth"
#     img_path   = r"D:\Masrafe\Coding\Git_Hub_code\ml_project\fedarated_brain_tumor\Dataset_2\glioma_tumor\glioma_tumor_6.jpg"
#     out_dir    = r"D:\Masrafe\Coding\Git_Hub_code\ml_project\fedarated_brain_tumor\predicted_outputs"

#     class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

#     model = load_model(model_path, num_classes=len(class_names))
#     pred_class, conf, probs = predict_and_save(model, img_path, class_names, out_dir)

#     print("\nClass probabilities:")
#     for i, c in enumerate(class_names):
#         print(f"  {c}: {probs[i]*100:.2f}%")


# if __name__ == "__main__":
#     main()


from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict
import logging
from typing import List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- CONFIG ---
PROJECT_ROOT = Path(r"D:\Masrafe\Coding\Git_Hub_code\ml_project\fedarated_brain_tumor")
IMG_SIZE = (224, 224)  # (width, height)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ensure project root on path for local imports
sys.path.append(str(PROJECT_ROOT))
from models.model_factory import get_model  # unchanged import


# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    model_path: Path
    input_path: Path         # single image OR folder
    out_dir: Path
    class_names: List[str]
    img_size: Tuple[int, int] = IMG_SIZE
    device: torch.device = DEVICE


def make_transform(img_size: Tuple[int, int]) -> A.Compose:
    w, h = img_size
    return A.Compose(
        [
            A.Resize(h, w),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ]
    )


def medical_preprocess_gray(image: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("medical_preprocess_gray expects a single-channel grayscale image")
    image = cv2.resize(image, img_size, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return image.astype(np.uint8)


def load_model_safe(model_path: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    logger.info("Instantiating model and loading weights from %s", model_path)
    model = get_model("customcnn", num_classes=num_classes, pretrained=False, dropout_rate=0.5).to(device)

    ckpt = torch.load(str(model_path), map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    clean_sd = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace("module.", "", 1) if k.startswith("module.") else k
        clean_sd[new_k] = v

    model.load_state_dict(clean_sd, strict=False)
    model.eval()
    logger.info("Model loaded (eval mode).")
    return model


class Predictor:
    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.transform = make_transform(cfg.img_size)
        self.model = load_model_safe(cfg.model_path, num_classes=len(cfg.class_names), device=cfg.device)

    def prepare_tensor(self, img_path: Path) -> Tuple[torch.Tensor, np.ndarray]:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        proc = medical_preprocess_gray(img, img_size=self.cfg.img_size)
        tensor = self.transform(image=proc)["image"].unsqueeze(0).to(self.cfg.device)
        return tensor, proc

    def predict_single(self, img_path: Path) -> Tuple[str, float, np.ndarray, np.ndarray]:
        tensor, disp_img = self.prepare_tensor(img_path)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
            idx = int(np.argmax(probs))
            pred_class = self.cfg.class_names[idx]
            conf = float(probs[idx] * 100.0)
        return pred_class, conf, probs, disp_img

    def save_prediction_image(self, disp_img: np.ndarray, pred_class: str, conf: float, save_path: Path) -> None:
        plt.imshow(disp_img, cmap="gray")
        plt.title(f"Predicted = {pred_class} ({conf:.2f}%)")
        plt.axis("off")
        plt.savefig(str(save_path), bbox_inches="tight")
        plt.close()
        logger.info("Saved prediction image: %s", save_path)

    def predict_and_save(self, img_path: Path) -> Tuple[str, float, np.ndarray]:
        pred_class, conf, probs, disp_img = self.predict_single(img_path)
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.cfg.out_dir / img_path.name
        self.save_prediction_image(disp_img, pred_class, conf, save_path)
        return pred_class, conf, probs

    def predict_folder(self, folder_path: Path, recursive: bool = False) -> None:
        if not folder_path.exists() or not folder_path.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {folder_path}")

        # collect image files
        if recursive:
            files = [p for p in folder_path.rglob("*") if p.suffix.lower() in self.SUPPORTED_EXTS]
        else:
            files = [p for p in folder_path.iterdir() if p.suffix.lower() in self.SUPPORTED_EXTS]

        files = sorted(files)
        if not files:
            logger.warning("No supported image files found in %s", folder_path)
            return

        summary = []
        logger.info("Found %d images in %s (recursive=%s). Processing...", len(files), folder_path, recursive)

        for idx, p in enumerate(files, 1):
            try:
                pred_class, conf, probs = self.predict_and_save(p)
                print(f"[{idx}/{len(files)}] {p.name} -> {pred_class} ({conf:.2f}%)")
                summary.append((p.name, pred_class, conf, probs))
            except Exception as e:
                logger.exception("Failed to process %s: %s", p, e)

        # final summary
        print("\nBatch prediction summary:")
        for name, pred_class, conf, probs in summary:
            print(f"  {name}: {pred_class} ({conf:.2f}%)")
        logger.info("Completed batch processing of %d images.", len(summary))


def main():
    cfg = Config(
        model_path=Path(r"D:\Masrafe\Coding\Git_Hub_code\ml_project\fedarated_brain_tumor\Result\FLResult\fl_results_20251008_205502\last_global_model.pth"),
        input_path=Path(r"D:\Masrafe\Coding\Git_Hub_code\ml_project\fedarated_brain_tumor\Dataset_2"),  # can be an image or a folder
        out_dir=Path(r"D:\Masrafe\Coding\Git_Hub_code\ml_project\fedarated_brain_tumor\predicted_outputs"),
        class_names=["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"],
        img_size=IMG_SIZE,
        device=DEVICE,
    )

    predictor = Predictor(cfg)

    if cfg.input_path.is_dir():
        # process all images in folder (non-recursive). Set recursive=True if you want recursion.
        predictor.predict_folder(cfg.input_path, recursive=False)
    else:
        pred_class, conf, probs = predictor.predict_and_save(cfg.input_path)
        print(f"\nPrediction: {pred_class} ({conf:.2f}%)\n")
        print("Class probabilities:")
        for i, name in enumerate(cfg.class_names):
            print(f"  {name}: {probs[i]*100:.2f}%")

if __name__ == "__main__":
    main()

