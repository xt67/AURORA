import pytest
import torch
import numpy as np
import yaml
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.model import MedicalImageClassifier, get_class_weights
from src.classifier.train_cls import train_classifier, evaluate_classifier
from src.data_loader import create_dataloaders
from src.utils.seed import set_seed
from PIL import Image


def test_classifier_forward_1ch():
    """Test classifier handles 1-channel input."""
    model = MedicalImageClassifier(num_classes=2, pretrained=False)
    x = torch.randn(4, 1, 128, 128)
    out = model(x)
    assert out.shape == (4, 2)


def test_classifier_forward_3ch():
    """Test classifier handles 3-channel input."""
    model = MedicalImageClassifier(num_classes=2, pretrained=False)
    x = torch.randn(4, 3, 128, 128)
    out = model(x)
    assert out.shape == (4, 2)


def test_class_weights_subset():
    """Test get_class_weights works with Subset."""
    from torch.utils.data import Subset, TensorDataset
    
    # Create a simple dataset
    X = torch.randn(20, 1, 128, 128)
    y = torch.tensor([0]*15 + [1]*5)  # imbalanced
    ds = TensorDataset(X, y)
    subset = Subset(ds, list(range(20)))
    
    weights = get_class_weights(subset, num_classes=2)
    assert weights.shape == (2,)
    # Class 0 has 15, class 1 has 5 -> weights should be inverse
    # total=20, n_classes=2, class0=15, class1=5
    # weight0 = 20/(2*15) = 0.667, weight1 = 20/(2*5) = 2.0
    assert weights[0] < weights[1]  # minority class gets higher weight
    assert abs(weights[0] - 20/(2*15)) < 0.01
    assert abs(weights[1] - 20/(2*5)) < 0.01


def test_different_seeds_produce_different_initializations():
    """Test that different seeds produce different model initializations."""
    # Note: With pretrained=True, weights are same but training order differs
    # With pretrained=False, initialization differs
    model1 = MedicalImageClassifier(num_classes=2, pretrained=False)
    set_seed(42)
    model2 = MedicalImageClassifier(num_classes=2, pretrained=False)
    set_seed(123)
    model3 = MedicalImageClassifier(num_classes=2, pretrained=False)
    
    # Check that weights differ
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.allclose(p1, p2):
            break
    else:
        pytest.fail("Models with different seeds should have different weights")
    
    for p1, p3 in zip(model1.parameters(), model3.parameters()):
        if not torch.allclose(p1, p3):
            break
    else:
        pytest.fail("Models with different seeds should have different weights")


def test_amp_training_loop():
    """Test that AMP training loop runs without error on CPU."""
    # Use project outputs directory to avoid Windows temp cleanup issues
    tmpdir = Path("outputs/test_amp_training")
    if tmpdir.exists():
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    tmpdir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create minimal config
        config = {
            "seed": 42,
            "num_classes": 2,
            "pretrained": False,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 1,
            "early_stop_patience": 5,
            "use_amp": True,
            "data_dir": str(tmpdir),
        }
        
        import yaml
        config_path = tmpdir / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Create dummy data
        for cls in ["Normal", "TB"]:
            (tmpdir / cls).mkdir()
            for i in range(10):
                arr = (np.random.rand(128, 128) * 255).astype(np.uint8)
                Image.fromarray(arr, "L").save(tmpdir / cls / f"{i:03d}.png")
        
        import torch
        
        tr, vl, te = create_dataloaders(str(tmpdir), batch_size=4, seed=42, num_workers=0, for_classifier=True)
        
        # This should run without error on CPU with AMP (which will be disabled)
        model = train_classifier(
            str(config_path), tr, vl, torch.device("cpu"),
            output_dir=str(tmpdir / "out"),
            seed=42,
        )
        
        # Check best checkpoint was loaded
        assert model is not None
        metrics = evaluate_classifier(model, te, torch.device("cpu"))
        assert "accuracy" in metrics
        assert "recall" in metrics
        assert "auprc" in metrics
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])