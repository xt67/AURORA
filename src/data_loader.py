import os
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Callable, List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class MedicalImageDataset(Dataset):
    """Dataset class for medical images (X-rays, MRI).

    Args:
        root_dir: Root directory containing class folders.
        transform: Transform to apply to images.
        target_classes: List of class names to load. None for all.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_classes: Optional[List[str]] = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []

        # Deterministic: sort class directories by name
        self.classes = sorted([d for d in self.root_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
        self.class_to_idx = {c.name: i for i, c in enumerate(self.classes)}

        if target_classes:
            self.classes = [c for c in self.classes if c.name in target_classes]


        skipped_files = []
        for class_dir in self.classes:
            # Deterministic: sort image paths by name
            img_paths = sorted([p for p in class_dir.iterdir() if p.is_file()], key=lambda x: x.name)
            for img_path in img_paths:
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    try:
                        img = Image.open(img_path)
                        img.verify()
                        self.images.append(img_path)
                        self.labels.append(self.class_to_idx[class_dir.name])
                    except Exception as e:
                        skipped_files.append((str(img_path), str(e)))

        if skipped_files:
            logger.warning(f"Skipped {len(skipped_files)} corrupted files")
            log_path = self.root_dir.parent / "skipped_files.log"
            with open(log_path, "w") as f:
                f.writelines([f"{p[0]}: {p[1]}\n" for p in skipped_files])

        if len(self) == 0:
            raise RuntimeError(f"No images found in {root_dir}. Check dataset path and structure.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        with Image.open(img_path) as source:
            image = to_grayscale_8bit(source)

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


def get_transforms(
    image_size: int = 128,
    is_training: bool = True,
    normalize_range: Tuple[float, float] = (-1, 1),
    augmentation_mode: str = "conventional",
    for_classifier: bool = False
) -> transforms.Compose:
    """Create image transforms.

    Args:
        image_size: Target image size.
        is_training: Whether for training (with augmentation).
        normalize_range: Range for normalization (used for GAN: [-1, 1]).
        augmentation_mode: "none" | "conventional" | "synthetic" - controls stochastic augmentations.
        for_classifier: If True, use ImageNet normalization for pretrained ResNet.

    Returns:
        Composed transforms.
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
    ]

    # Only apply stochastic augmentations during training AND when mode permits
    if augmentation_mode not in ("none", "conventional", "synthetic"):
        raise ValueError("Unknown augmentation_mode")
    if is_training and augmentation_mode == "conventional":
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])

    transform_list.append(transforms.ToTensor())

    if for_classifier:
        # Convert grayscale to 3-channel for pretrained ResNet
        transform_list.append(repeat_grayscale_channels)
        # ImageNet normalization for pretrained ResNet-18
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    else:
        # GAN normalization: grayscale [-1, 1]
        transform_list.append(
            transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
        )

    return transforms.Compose(transform_list)


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 128,
    num_workers: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    target_classes: Optional[List[str]] = None,
    augmentation_mode: str = "conventional",
    for_classifier: bool = False,
    manifest_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders.

    Args:
        data_dir: Directory containing class folders.
        batch_size: Batch size for dataloaders.
        image_size: Target image size.
        num_workers: Number of worker processes.
        train_ratio: Ratio of training data.
        val_ratio: Ratio of validation data.
        seed: Random seed for splitting.
        target_classes: Classes to include.
        augmentation_mode: "none" | "conventional" - controls stochastic augmentations on train set.
        for_classifier: If True, use ImageNet normalization (3-channel) for pretrained ResNet.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("Split ratios must be positive and sum to less than one")
    # For GAN: grayscale [-1,1]; for Classifier: 3-channel ImageNet normalization
    train_transform = get_transforms(
        image_size, is_training=True, augmentation_mode=augmentation_mode, for_classifier=for_classifier
    )
    val_transform = get_transforms(
        image_size, is_training=False, augmentation_mode="none", for_classifier=for_classifier
    )

    # Use one dataset instance for metadata/splitting (transform unused for split).
    base_dataset = MedicalImageDataset(data_dir, None, target_classes)

    # Stratified split by class label so the minority (TB) class is represented
    # in val/test. (No patient IDs available in this dataset -> image-level split.)
    from collections import defaultdict
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(base_dataset.labels):
        label_to_indices[label].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for label, idxs in label_to_indices.items():
        idxs = sorted(idxs)
        g = torch.Generator().manual_seed(seed + int(label))
        perm = torch.randperm(len(idxs), generator=g).tolist()
        shuffled = [idxs[i] for i in perm]
        n = len(shuffled)
        tr_n = int(train_ratio * n)
        va_n = int(val_ratio * n)
        train_idx.extend(shuffled[:tr_n])
        val_idx.extend(shuffled[tr_n:tr_n + va_n])
        test_idx.extend(shuffled[tr_n + va_n:])

    # Separate dataset instances per split avoid transform leakage across splits.
    train_full = MedicalImageDataset(data_dir, train_transform, target_classes)
    val_full = MedicalImageDataset(data_dir, val_transform, target_classes)
    test_full = MedicalImageDataset(data_dir, val_transform, target_classes)

    train_len = len(train_idx)
    val_len = len(val_idx)
    test_len = len(test_idx)

    train_dataset = torch.utils.data.Subset(train_full, train_idx)
    val_dataset = torch.utils.data.Subset(val_full, val_idx)
    test_dataset = torch.utils.data.Subset(test_full, test_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    # Persist split manifest with actual indices and filenames for reproducibility
    split_manifest = {
        "train": {"count": train_len, "indices": train_idx},
        "val": {"count": val_len, "indices": val_idx},
        "test": {"count": test_len, "indices": test_idx},
        "seed": seed,
        "stratified": True,
        "deduplicated": False,
        "protocol": "exploratory-image-level-random-split",
        "filenames": {
            "train": [base_dataset.images[i].name for i in train_idx],
            "val": [base_dataset.images[i].name for i in val_idx],
            "test": [base_dataset.images[i].name for i in test_idx],
        }
    }
    # Never silently overwrite the canonical study split during loader creation.
    if manifest_path is not None:
        with open(manifest_path, "x") as f:
            json.dump(split_manifest, f, indent=2)

    # Verify disjoint splits
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    assert len(train_set & val_set) == 0, "Train/val overlap!"
    assert len(train_set & test_set) == 0, "Train/test overlap!"
    assert len(val_set & test_set) == 0, "Val/test overlap!"

    class_counts = {}
    for c in train_full.classes:
        i = train_full.class_to_idx[c.name]
        class_counts[c.name] = sum(1 for label in train_full.labels if label == i)
    logger.info(f"Dataset class distribution: {class_counts}")

    if any(c < 50 for c in class_counts.values()):
        logger.warning(f"Some classes have fewer than 50 samples. Consider oversampling.")

    return train_loader, val_loader, test_loader


def to_grayscale_8bit(image: Image.Image) -> Image.Image:
    """Reject implicit clipping of high-bit-depth medical pixel intensities."""
    if image.mode in ("I", "F") or image.mode.startswith("I;16"):
        raise ValueError("High-bit-depth input requires an explicitly approved intensity conversion")
    return image.convert("L")


def repeat_grayscale_channels(image: torch.Tensor) -> torch.Tensor:
    """Pickleable transform for Windows DataLoader workers."""
    return image.repeat(3, 1, 1) if image.shape[0] == 1 else image


def grayscale_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """Convert grayscale image to 3-channel RGB.

    Args:
        image: Grayscale tensor of shape (B, 1, H, W).

    Returns:
        RGB tensor of shape (B, 3, H, W).
    """
    return image.repeat(1, 3, 1, 1)
