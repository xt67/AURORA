import torch
import torch.nn as nn

try:
    from torchvision import models as tv_models
except ImportError as e:  # pragma: no cover - environment dependent
    raise ImportError(
        "torchvision is required for the classifier. "
        "Install it with: pip install torchvision"
    ) from e


class MedicalImageClassifier(nn.Module):
    """ResNet-18 based classifier for medical images.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(MedicalImageClassifier, self).__init__()

        try:
            weights = "IMAGENET1K_V1" if pretrained else None
            self.model = tv_models.resnet18(weights=weights)
        except TypeError:
            # Older torchvision API
            self.model = tv_models.resnet18(pretrained=pretrained)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both grayscale (1-channel) and RGB (3-channel) inputs
        # For 1-channel input (GAN-style), repeat to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)


def create_classifier(num_classes: int = 2, pretrained: bool = True) -> MedicalImageClassifier:
    """Create and return classifier model."""
    return MedicalImageClassifier(num_classes=num_classes, pretrained=pretrained)


def get_class_weights(dataset, num_classes: int = 2) -> torch.Tensor:
    """Compute class weights inversely proportional to class frequency.

    Handles:
    - torch.utils.data.Subset wrapping a dataset with `.labels` or `.targets`
    - torch.utils.data.Subset wrapping a TensorDataset (has `.tensors[1]` for targets)
    - TensorDataset directly (has `.tensors[1]` for targets)
    - Datasets exposing `.labels` or `.targets`
    - Fall back to iterating (may be slow but robust).
    """
    from collections import Counter

    labels = []
    
    # Subset wrapping a dataset with labels
    if hasattr(dataset, "indices"):
        base_ds = dataset.dataset
        # Subset of TensorDataset
        if hasattr(base_ds, "tensors") and len(base_ds.tensors) > 1:
            labels = [base_ds.tensors[1][i].item() for i in dataset.indices]
        # Subset of dataset with labels
        elif hasattr(base_ds, "labels") and base_ds.labels is not None:
            labels = [base_ds.labels[i] for i in dataset.indices]
        # Subset of dataset with targets
        elif hasattr(base_ds, "targets") and base_ds.targets is not None:
            labels = [base_ds.targets[i] for i in dataset.indices]
    # TensorDataset directly
    elif hasattr(dataset, "tensors") and len(dataset.tensors) > 1:
        labels = dataset.tensors[1].tolist()
    # Direct labels/targets attribute
    elif hasattr(dataset, "labels") and dataset.labels is not None:
        labels = list(dataset.labels)
    elif hasattr(dataset, "targets") and dataset.targets is not None:
        labels = list(dataset.targets)
    else:
        # Fall back to iterating (may be slow but robust).
        for _, label in dataset:
            labels.append(label)

    counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        weights.append(total / (num_classes * c))
    return torch.tensor(weights, dtype=torch.float)