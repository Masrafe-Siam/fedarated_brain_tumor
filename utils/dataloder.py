import os
import torch
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Dataset ====================
class CTScanDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, subset: str = 'train', samples: List[Tuple[str,int]] = None):
        """
        Args:
            data_dir: Root directory containing class folders
            transform: Albumentations transform pipeline
            subset: 'train', 'val', 'test', or 'full'
            samples: Optional preselected samples
        """
        self.data_dir = data_dir
        self.transform = transform
        self.subset = subset

        # Dynamically generate class mapping from folder names
        class_names = sorted([
            d for d in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

        # Load samples
        if samples is not None:
            self.samples = samples
        else:
            self.samples = self._load_samples()

        # Calculate class weights
        self.class_weights = self._calculate_class_weights()
        logger.info(f"Loaded {len(self.samples)} samples for {subset} set")
        self._print_class_distribution()

    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and their corresponding labels"""
        samples = []
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_path) or class_name not in self.class_to_idx:
                continue
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    img_path = os.path.join(class_path, img_name)
                    samples.append((img_path, class_idx))
        return samples

    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced datasets"""
        class_counts = [0] * len(self.class_to_idx)
        for _, label in self.samples:
            class_counts[label] += 1

        total_samples = len(self.samples)
        weights = []
        for count in class_counts:
            if count == 0:
                weights.append(0.0)
                logger.warning("⚠️ One or more classes have 0 samples — check your dataset.")
            else:
                weights.append(total_samples / (len(self.class_to_idx) * count))
        return torch.FloatTensor(weights)

    def _print_class_distribution(self):
        """Print class distribution for debugging"""
        class_counts = [0] * len(self.class_to_idx)
        for _, label in self.samples:
            class_counts[label] += 1
        logger.info("Class distribution:")
        for idx, count in enumerate(class_counts):
            logger.info(f"  {self.idx_to_class[idx]}: {count} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = self._load_and_preprocess_image(img_path)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, label

    def _load_and_preprocess_image(self, img_path: str) -> np.ndarray:
        try:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {img_path}")
            image = self._apply_medical_preprocessing(image)
            return image
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            return np.zeros((224, 224), dtype=np.uint8)

    def _apply_medical_preprocessing(self, image: np.ndarray) -> np.ndarray:
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return image.astype(np.uint8)

# ==================== Transforms ====================
def get_medical_transforms(image_size: Tuple[int, int] = (224, 224), subset: str = 'train') -> A.Compose:
    if subset == 'train':
        transform = A.Compose([
            A.Resize(*image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.0, rotate_limit=0, p=0.5),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(*image_size),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])
    return transform

# ==================== DataLoader creation ====================
def create_data_loaders(data_dir: str, 
                        batch_size: int = 32, 
                        train_split: float = 0.8,
                        val_split: float = 0.1,
                        test_split: float = 0.1,
                        image_size: Tuple[int, int] = (224, 224),
                        num_workers: int = 1,
                        pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:

    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"

    train_transform = get_medical_transforms(image_size, 'train')
    val_transform = get_medical_transforms(image_size, 'val')
    test_transform = get_medical_transforms(image_size, 'test')

    full_dataset = CTScanDataset(data_dir, transform=None, subset='full')

    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    # Split indices with stratification
    train_indices, temp_indices = train_test_split(
        range(total_size),
        train_size=train_size,
        stratify=[full_dataset.samples[i][1] for i in range(total_size)],
        random_state=42
    )

    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_size,
        stratify=[full_dataset.samples[i][1] for i in temp_indices],
        random_state=42
    )

    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]
    test_samples = [full_dataset.samples[i] for i in test_indices]

    train_dataset = CTScanDataset(data_dir, transform=train_transform, subset='train', samples=train_samples)
    val_dataset = CTScanDataset(data_dir, transform=val_transform, subset='val', samples=val_samples)
    test_dataset = CTScanDataset(data_dir, transform=test_transform, subset='test', samples=test_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    logger.info(f"Data splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader

# ==================== Visualization ====================
def visualize_batch(data_loader: DataLoader, num_samples: int = 8):
    batch_images, batch_labels = next(iter(data_loader))

    n_cols = min(4, num_samples)
    n_rows = (num_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.ravel()

    class_names = data_loader.dataset.idx_to_class

    for i in range(min(num_samples, len(batch_images))):
        img = batch_images[i].squeeze()
        if len(img.shape) == 3 and img.shape[0] == 1:
            img = img.squeeze(0)
        img = img * 0.5 + 0.5  # Denormalize
        img = torch.clamp(img, 0, 1)
        label = batch_labels[i].item()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {class_names[label]}')
        axes[i].axis('off')

    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()

# ==================== Class weights ====================
def get_class_weights(data_loader: DataLoader) -> torch.Tensor:
    num_classes = len(data_loader.dataset.class_to_idx)
    class_counts = torch.zeros(num_classes)
    total_samples = 0

    for _, labels in data_loader:
        for label in labels:
            class_counts[label] += 1
            total_samples += 1

    class_counts = torch.where(class_counts == 0, torch.tensor(1.0), class_counts)
    weights = total_samples / (num_classes * class_counts)
    return weights

# ==================== Main ====================
if __name__ == "__main__":
    DATA_DIR = r"D:\Masrafe\Coding\Git_Hub_code\ml_project\fedarated_brain_tumor\Dataset_1"  # Replace with your path

    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=DATA_DIR,
        batch_size=32,
        image_size=(224, 224),
        num_workers=0,  # Windows safe
        pin_memory=False
    )

    visualize_batch(train_loader)
    weights = get_class_weights(train_loader)
    logger.info(f"Class Weights: {weights}")
