from io import BytesIO
import json

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.gan.train_gan import train_gan
from src.gan.discriminator import Discriminator
from src.gan.inference import load_generator, generate_images, png_grid
from src.data_loader import MedicalImageDataset, create_dataloaders, get_transforms
from src.evaluate.metrics import compute_metrics


def test_loggers_are_isolated_per_experiment(tmp_path):
    from src.utils.logger import setup_logger
    first = setup_logger("isolation", str(tmp_path / "one"))
    second = setup_logger("isolation", str(tmp_path / "two"))
    first.info("first-run")
    second.info("second-run")
    for logger in (first, second):
        for handler in logger.handlers:
            handler.flush()
    assert "second-run" not in next((tmp_path / "one").glob("*.log")).read_text()
    assert "second-run" in next((tmp_path / "two").glob("*.log")).read_text()


@pytest.mark.parametrize("truth,pred,prob", [([], [], []), ([0], [2], [.2]), ([0], [0], [float('nan')]), ([0], [0, 1], [.2]), ([0], [0], [2.])])
def test_metrics_reject_invalid_inputs(truth, pred, prob):
    with pytest.raises(ValueError):
        compute_metrics(truth, pred, prob)


@pytest.mark.parametrize("loss_type", ["bce", "wgan-gp"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_training_checkpoint_inference_png(tmp_path, loss_type, device):
    config = dict(seed=42, z_dim=8, ngf=2, ndf=2, lr_g=0.0002, lr_d=0.0002,
                  beta1=0.5, beta2=0.999, epochs=1, loss_type=loss_type, use_amp=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    loader = DataLoader(TensorDataset(torch.rand(2, 1, 128, 128) * 2 - 1, torch.ones(2)), batch_size=2)
    _, critic = train_gan(str(path), loader, torch.device(device), str(tmp_path / "run"))
    if loss_type == "wgan-gp":
        assert not any(isinstance(m, (torch.nn.Sigmoid, torch.nn.BatchNorm2d)) for m in critic.modules())
    model, loaded = load_generator(tmp_path / "run/models/generator_latest.pth", device)
    first = generate_images(model, 2, loaded["z_dim"], 7)
    assert torch.equal(first, generate_images(model, 2, loaded["z_dim"], 7))
    assert torch.isfinite(first).all()
    payload = png_grid(first)
    with Image.open(BytesIO(payload)) as image:
        assert image.format == "PNG"
        image.verify()
    assert (tmp_path / "run/samples/epoch_1.png").is_file()


def test_critic_scores_independent_of_other_batch_members():
    critic = Discriminator(ndf=2, loss_type="wgan-gp").train()
    x = torch.randn(1, 1, 128, 128)
    assert torch.allclose(critic(x), critic(torch.cat([x, torch.randn_like(x)]))[:1], atol=1e-6)


def test_loader_preserves_manifest_and_minority_labels(tmp_path):
    data = tmp_path / "images"
    for label in ("Normal", "Tuberculosis"):
        (data / label).mkdir(parents=True)
        for index in range(10):
            Image.fromarray(np.full((16, 16), index, dtype=np.uint8)).save(data / label / f"{index}.png")
    manifest = tmp_path / "splits.json"
    manifest.write_text('{"canonical": true}')
    tr, _, _ = create_dataloaders(str(data), batch_size=100)
    assert len(tr) == 1
    assert json.loads(manifest.read_text()) == {"canonical": True}
    minority = MedicalImageDataset(str(data), target_classes=["Tuberculosis"])
    assert set(minority.labels) == {1}


def test_synthetic_mode_does_not_add_conventional_augmentation():
    transform = get_transforms(augmentation_mode="synthetic")
    image = Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16))
    assert torch.equal(transform(image), transform(image))


def test_one_class_evaluation_has_fixed_confusion_matrix():
    metrics = compute_metrics(np.array([0, 0]), np.array([0, 0]), np.array([0.1, 0.2]))
    assert metrics["confusion_matrix"] == [[2, 0], [0, 0]]
    assert metrics["specificity"] == 1
    assert metrics["roc_auc"] is None


def test_fid_denormalizes_pixels_and_cleans_up_on_error(monkeypatch):
    from pathlib import Path
    import src.evaluate.fid as fid
    folders = []
    def fail_after_export(real_path, fake_path, device):
        folders.extend([Path(real_path), Path(fake_path)])
        with Image.open(Path(real_path) / "0.png") as image:
            pixels = np.array(image)
            assert pixels.min() == 0
            assert pixels.max() == 255
            assert (pixels == 128).any()
        raise RuntimeError("Expected scoring failure")
    monkeypatch.setattr(fid, "compute_fid", fail_after_export)
    images = torch.tensor([-1., 0., 1.]).view(1, 1, 1, 3).repeat(2, 1, 1, 1)
    with pytest.raises(RuntimeError, match="Expected scoring failure"):
        fid.compute_fid_from_tensors(images, images, "cpu")
    assert folders and all(not folder.exists() for folder in folders)


def test_classifier_consumes_all_synthetic_batches(tmp_path, monkeypatch):
    import src.classifier.train_cls as trainer
    seen = []
    class TinyClassifier(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.linear = torch.nn.Linear(1, 2)
        def forward(self, images):
            if self.training:
                seen.append(images[-1].item())
            return self.linear(images)
    monkeypatch.setattr(trainer, "MedicalImageClassifier", TinyClassifier)
    config = tmp_path / "classifier.yaml"
    config.write_text(yaml.safe_dump(dict(seed=42, epochs=1, pretrained=False, use_amp=False)))
    real = DataLoader(TensorDataset(torch.zeros(6, 1), torch.tensor([0, 1, 0, 1, 0, 1])), batch_size=2)
    synthetic = DataLoader(TensorDataset(torch.tensor([[1.], [2.]]), torch.ones(2, dtype=torch.long)), batch_size=1)
    trainer.train_classifier(str(config), real, real, torch.device("cpu"), use_synthetic=True,
                             synthetic_loader=synthetic, output_dir=str(tmp_path / "classifier"))
    assert seen == [1., 2., 1.]


@pytest.mark.parametrize("page", ["Overview", "Generator", "Results"])
def test_demo_empty_state(page, monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("AURORA_CHECKPOINT", str(tmp_path / "missing.pth"))
    monkeypatch.setenv("AURORA_RESULTS", str(tmp_path / "missing.json"))
    app = AppTest.from_file("app/streamlit_app.py").run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
    assert not app.success
