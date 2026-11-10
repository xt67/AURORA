import torch
import numpy as np
from pathlib import Path
from pytorch_fid.fid_score import calculate_fid_given_paths


def compute_fid(real_images_path: str, fake_images_path: str, device: str = "cuda", batch_size: int = 50) -> float:
    """Compute FID score between real and generated images.
    
    Args:
        real_images_path: Path to real images folder.
        fake_images_path: Path to generated images folder.
        device: Device to run computation on.
        batch_size: Batch size for Inception v3.
    
    Returns:
        FID score.
    """
    paths = [real_images_path, fake_images_path]
    fid_score = calculate_fid_given_paths(paths, batch_size, device, dims=2048)
    return fid_score


def compute_fid_from_tensors(real_images: torch.Tensor, fake_images: torch.Tensor, device: str = "cuda", input_range: tuple = (-1, 1)) -> float:
    """Compute FID score from image tensors.
    
    Args:
        real_images: Tensor of real images (B, C, H, W).
        fake_images: Tensor of fake images (B, C, H, W).
        device: Device to run on.
    
    Returns:
        FID score.
    """
    import tempfile
    
    from torchvision.utils import save_image
    low, high = input_range
    if high <= low:
        raise ValueError("input_range must be increasing")
    for images in (real_images, fake_images):
        if len(images) < 2 or not torch.isfinite(images).all():
            raise ValueError("FID requires at least two finite images in each set")
        if images.min() < low or images.max() > high:
            raise ValueError("Images are outside input_range")
    with tempfile.TemporaryDirectory() as folder:
        real_path, fake_path = Path(folder) / "real", Path(folder) / "fake"
        real_path.mkdir()
        fake_path.mkdir()
        for images, destination in ((real_images, real_path), (fake_images, fake_path)):
            for i, img in enumerate(images):
                save_image((img.detach().cpu() - low) / (high - low), destination / f"{i}.png")
        return compute_fid(str(real_path), str(fake_path), device)
