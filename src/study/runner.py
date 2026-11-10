"""Matched-update study runner; the CLI always enforces the publication gate."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, WeightedRandomSampler
import yaml

from src.classifier.model import MedicalImageClassifier
from src.data_loader import get_transforms, to_grayscale_8bit
from src.evaluate.metrics import compute_metrics
from src.utils.seed import set_seed
from .data import budget_rows, sha256, validate_manifest
from .gate import publication_preflight


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


class ManifestDataset(Dataset):
    def __init__(self, rows, root, classifier=True, augment=False):
        self.rows = rows
        self.root = Path(root)
        self.labels = [r["label"] for r in rows]
        self.transform = get_transforms(is_training=augment, augmentation_mode="conventional" if augment else "none", for_classifier=classifier)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as image:
            return self.transform(to_grayscale_8bit(image)), row["label"]


def select_threshold(labels, probabilities):
    """Maximize validation F1; ties choose the highest threshold."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if len(np.unique(labels)) != 2:
        raise ValueError("Threshold selection requires both validation classes")
    compute_metrics(labels, (probabilities >= .5).astype(int), probabilities)
    candidates = np.unique(np.r_[0., probabilities, 1.])
    scores = [(compute_metrics(labels, (probabilities >= t).astype(int))["f1"], float(t)) for t in candidates]
    return max(scores)[1]


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    truth, probabilities = [], []
    loss_sum = 0.
    for images, labels in loader:
        outputs = model(images.to(device))
        if not torch.isfinite(outputs).all():
            raise FloatingPointError("Non-finite evaluation logits")
        loss_sum += torch.nn.functional.cross_entropy(outputs, labels.to(device), reduction="sum").item()
        truth.extend(labels.tolist())
        probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().tolist())
    if not truth:
        raise ValueError("Evaluation partition is empty")
    return np.array(truth), np.array(probabilities), loss_sum / len(truth)


def evaluate_predictions(rows, labels, probabilities, threshold):
    result = {"overall": compute_metrics(labels, (probabilities >= threshold).astype(int), probabilities), "cohorts": {}}
    for cohort in sorted({r["cohort"] for r in rows}):
        mask = np.array([r["cohort"] == cohort for r in rows])
        result["cohorts"][cohort] = compute_metrics(labels[mask], (probabilities[mask] >= threshold).astype(int), probabilities[mask])
    return result


def make_train_loader(dataset, arm, updates, batch_size, seed):
    rng = torch.Generator().manual_seed(seed)
    count = updates * batch_size
    if arm == "A1":
        labels = np.array(dataset.labels)
        counts = np.bincount(labels, minlength=2)
        if min(counts) == 0:
            raise ValueError("Oversampling requires both classes")
        # Draw with replacement; inverse class frequency yields equal expected
        # class counts, without pretending to add independent real observations.
        sampler = WeightedRandomSampler([1. / counts[y] for y in labels], count, replacement=True, generator=rng)
    else:
        sampler = RandomSampler(dataset, replacement=True, num_samples=count, generator=rng)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)


def train_fixed_updates(dataset, val_dataset, cfg, arm, seed, device, out):
    updates, batch_size = cfg["updates"], cfg["batch_size"]
    if updates < 1 or batch_size < 2 or cfg["eval_every"] < 1:
        raise ValueError("Positive update/evaluation counts and batch_size >= 2 required")
    set_seed(seed)
    model = MedicalImageClassifier(pretrained=cfg.get("pretrained", True)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    amp = bool(cfg.get("use_amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    train = make_train_loader(dataset, arm, updates, batch_size, seed)
    val = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    best_loss = float("inf")
    history = []
    for step, (images, labels) in enumerate(train, 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            loss = torch.nn.functional.cross_entropy(model(images.to(device)), labels.to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite classifier loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Non-finite classifier gradient; run failed")
        scaler.step(optimizer)
        scaler.update()
        if step % cfg["eval_every"] == 0 or step == updates:
            _, _, val_loss = predict(model, val, device)
            history.append({"step": step, "training_loss": loss.item(), "validation_loss": val_loss})
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({"state_dict": model.state_dict(), "step": step, "config": cfg}, out / "classifier.pth")
    model.load_state_dict(torch.load(out / "classifier.pth", map_location=device, weights_only=True)["state_dict"])
    write_json(out / "training_history.json", history)
    labels, probabilities, _ = predict(model, val, device)
    threshold = select_threshold(labels, probabilities)
    return model, threshold


def augmentation_count(train_rows, config):
    ratio = float(config.get("synthetic_ratio", 1.))
    if not np.isfinite(ratio) or ratio < 0:
        raise ValueError("Synthetic ratio must be finite and nonnegative")
    positives = sum(r["label"] == 1 for r in train_rows)
    negatives = sum(r["label"] == 0 for r in train_rows)
    return int(np.ceil(max(0, negatives - positives) * ratio))


def duplicate_positive_rows(train_rows, count, seed):
    """Match G1's added count without claiming new independent observations."""
    positives = [r for r in train_rows if r["label"] == 1]
    if count and not positives:
        raise ValueError("Duplicate control requires positive training images")
    rng = np.random.default_rng(seed)
    return [positives[i] for i in rng.integers(0, len(positives), size=count)] if count else []


def add_synthetic(train_rows, root, config, seed, device, out):
    from src.gan.train_gan import train_gan
    from src.gan.inference import generate_images
    from torchvision.utils import save_image
    positives = [r for r in train_rows if r["label"] == 1]
    # The generator receives ONLY the selected real positive training rows.
    generator_data = ManifestDataset(positives, root, classifier=False)
    gan_config = dict(config["generator"], seed=seed)
    path = out / "generator.yaml"
    path.write_text(yaml.safe_dump(gan_config))
    set_seed(seed)
    loader = DataLoader(generator_data, batch_size=gan_config.get("batch_size", 16), shuffle=True, num_workers=0)
    generator, _ = train_gan(str(path), loader, device, str(out / "generator"))
    generator.eval()
    count = augmentation_count(train_rows, config)
    folder = out / "synthetic"
    folder.mkdir()
    synthetic = []
    for start in range(0, count, 16):
        images = generate_images(generator, min(16, count - start), gan_config["z_dim"], seed + start)
        for offset, image in enumerate(images):
            target = folder / f"synthetic_{start + offset:06d}.png"
            save_image((image.cpu() + 1) / 2, target)
            synthetic.append({"id": target.stem, "path": target.name, "label": 1, "cohort": "synthetic"})
    write_json(out / "generator_provenance.json", {"training_ids": [r["id"] for r in positives],
               "training_sha256": [r["sha256"] for r in positives], "seed": seed, "synthetic_count": count,
               "checkpoint_sha256": sha256(out / "generator/models/generator_latest.pth"), "selection": "last epoch, fixed configuration"})
    # Return separate datasets so no synthetic path is interpreted as a real row.
    return ManifestDataset(synthetic, folder), count


def run_one(config, manifest, arm, fraction, seed, out, publication=False):
    if arm not in ("A0", "A1", "A2", "A3", "G1"):
        raise ValueError("Unsupported arm; diffusion is not silently substituted")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if config.get("device", "auto") == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["device"])
    rows = manifest["rows"]
    selected = budget_rows(rows, fraction, config.get("budget_seed", 42))
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    real = ManifestDataset(selected, config["data_root"], augment=arm == "A2")
    dataset = real
    synthetic_count = 0
    duplicate_rows = []
    try:
        memory = None
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; select cpu or auto in a newly approved config")
            cuda_index = device.index if device.index is not None else torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(cuda_index)
            memory = {"free_bytes_at_start": free, "total_bytes": total}
            if free < 512 * 1024 * 1024:
                raise RuntimeError("Less than 512 MiB CUDA memory is free; close competing GPU applications")
            torch.cuda.reset_peak_memory_stats(device)
        if arm == "G1":
            synthetic, synthetic_count = add_synthetic(selected, config["data_root"], config, seed, device, out)
            dataset = torch.utils.data.ConcatDataset([real, synthetic])
        elif arm == "A3":
            duplicate_rows = duplicate_positive_rows(selected, augmentation_count(selected, config), seed)
            dataset = ManifestDataset(selected + duplicate_rows, config["data_root"])
        cfg = config["classifier"]
        val = ManifestDataset(val_rows, config["data_root"])
        model, threshold = train_fixed_updates(dataset, val, cfg, arm, seed, device, out)
        test = DataLoader(ManifestDataset(test_rows, config["data_root"]), batch_size=cfg["batch_size"], num_workers=0)
        labels, probs, _ = predict(model, test, device)
        predictions = [{"id": row["id"], "group": row["group"], "cohort": row["cohort"], "label": int(label), "probability": float(prob)}
                       for row, label, prob in zip(test_rows, labels, probs)]
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "unavailable"
        result = {"status": "measured", "publication": publication, "arm": arm, "budget": fraction, "seed": seed,
                  "threshold": threshold, "threshold_rule": "maximum validation F1; highest threshold breaks ties",
                  "metrics": evaluate_predictions(test_rows, labels, probs, threshold), "predictions": predictions,
                  "training_ids": [r["id"] for r in selected], "synthetic_count": synthetic_count,
                  "duplicate_count": len(duplicate_rows), "duplicate_source_ids": [r["id"] for r in duplicate_rows],
                  "updates": cfg["updates"], "batch_size": cfg["batch_size"], "config": config,
                  "manifest_digest": __import__("hashlib").sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                  "checkpoint_sha256": sha256(out / "classifier.pth"), "commit": commit,
                  "runtime_seconds": time.perf_counter() - started, "python": platform.python_version(),
                  "torch": str(torch.__version__), "device": str(device),
                  "cuda_memory": {**memory, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)} if memory else None,
                  "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()}
        write_json(out / "result.json", result)
        return result
    except Exception as exc:
        write_json(out / "failure.json", {"arm": arm, "budget": fraction, "seed": seed,
                   "error": str(exc), "publication": publication, "runtime_seconds": time.perf_counter() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config, manifest = publication_preflight(args.config, args.approval)
    if args.check_only:
        print("Publication gate passed; no training launched.")
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "frozen_manifest.json", manifest)
    write_json(out / "run_provenance.json", {"config_sha256": sha256(args.config), "approval_sha256": sha256(args.approval),
               "manifest_sha256": sha256(config["manifest"]), "protocol_sha256": sha256(config["protocol"])})
    (out / "config.yaml").write_text(Path(args.config).read_text())
    (out / "approval.json").write_text(Path(args.approval).read_text(encoding="utf-8"), encoding="utf-8")
    (out / "protocol.md").write_text(Path(config["protocol"]).read_text(encoding="utf-8"), encoding="utf-8")
    for fraction in config["budgets"]:
        for seed in config["seeds"]:
            for arm in config["arms"]:
                run_one(config, manifest, arm, fraction, seed, out / f"{arm}_b{fraction}_s{seed}", publication=True)


if __name__ == "__main__":
    main()
