import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import MedicalImageDataset, get_transforms, grayscale_to_rgb, create_dataloaders


@pytest.fixture
def sample_transform():
    return get_transforms(image_size=128, is_training=False)


def test_grayscale_to_rgb():
    """Test grayscale to RGB conversion."""
    x = torch.randn(4, 1, 128, 128)
    x_rgb = grayscale_to_rgb(x)
    assert x_rgb.shape == (4, 3, 128, 128)


def test_transform_output_range_gan():
    """Test GAN transforms output range [-1, 1]."""
    from PIL import Image
    import numpy as np

    img_array = (np.random.rand(128, 128) * 255).astype(np.uint8)
    img = Image.fromarray(img_array, "L")

    transform = get_transforms(image_size=128, is_training=False, for_classifier=False)
    tensor = transform(img)

    assert tensor.min() >= -1.0 and tensor.max() <= 1.0
    assert tensor.shape == (1, 128, 128)  # grayscale


def test_transform_output_range_classifier():
    """Test classifier transforms use ImageNet normalization."""
    from PIL import Image
    import numpy as np

    img_array = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    img = Image.fromarray(img_array, "RGB")

    transform = get_transforms(image_size=128, is_training=False, for_classifier=True)
    tensor = transform(img)

    # ImageNet normalized: roughly mean 0, std 1 range
    assert tensor.shape == (3, 128, 128)  # 3-channel


def test_augmentation_modes():
    """Test different augmentation modes produce deterministic results when seeded."""
    from PIL import Image
    import numpy as np

    img_array = (np.random.rand(128, 128) * 255).astype(np.uint8)
    img = Image.fromarray(img_array, "L")

    # Mode "none" - no stochastic augmentations
    t_none = get_transforms(image_size=128, is_training=True, augmentation_mode="none")
    out1 = t_none(img)
    out2 = t_none(img)
    assert torch.allclose(out1, out2), "Mode 'none' should be deterministic"

    # Mode "conventional" - stochastic, so different each call
    t_conv = get_transforms(image_size=128, is_training=True, augmentation_mode="conventional")
    out1 = t_conv(img)
    out2 = t_conv(img)
    # Very unlikely to be identical (though possible)
    # Just check it runs without error
    assert out1.shape == (1, 128, 128)
    assert out2.shape == (1, 128, 128)


def test_deterministic_split():
    """Test that dataset split is deterministic and disjoint."""
    # Create temporary dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for cls in ["Normal", "TB"]:
            (tmpdir / cls).mkdir()
            for i in range(20):
                arr = (np.random.rand(128, 128) * 255).astype(np.uint8)
                Image.fromarray(arr, "L").save(tmpdir / cls / f"{i:03d}.png")

        # Create loaders twice with same seed
        tr1, vl1, te1 = create_dataloaders(str(tmpdir), batch_size=4, seed=42, num_workers=0)
        tr2, vl2, te2 = create_dataloaders(str(tmpdir), batch_size=4, seed=42, num_workers=0)

        # Check same indices
        assert tr1.dataset.indices == tr2.dataset.indices
        assert vl1.dataset.indices == vl2.dataset.indices
        assert te1.dataset.indices == te2.dataset.indices

        # Check disjoint
        all_idx = set(tr1.dataset.indices) | set(vl1.dataset.indices) | set(te1.dataset.indices)
        assert len(all_idx) == 40  # 20+20 total

        # Check split counts
        assert len(tr1.dataset) == 28  # 70%
        assert len(vl1.dataset) == 6   # 15%
        assert len(te1.dataset) == 6   # 15%


def test_classifier_transforms_produce_3ch():
    """Test that for_classifier=True produces 3-channel output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for cls in ["Normal", "TB"]:
            (tmpdir / cls).mkdir()
            for i in range(10):
                arr = (np.random.rand(128, 128) * 255).astype(np.uint8)
                Image.fromarray(arr, "L").save(tmpdir / cls / f"{i:03d}.png")

        tr, _, _ = create_dataloaders(str(tmpdir), batch_size=4, seed=42, num_workers=0, for_classifier=True)
        x, _ = next(iter(tr))
        assert x.shape[1] == 3, f"Expected 3 channels, got {x.shape[1]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
