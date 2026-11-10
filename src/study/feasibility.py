"""Approved two-epoch v2 DCGAN feasibility only; no classifier or holdout loader."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import platform
import time

from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.data_loader import get_transforms, to_grayscale_8bit
from src.gan.train_gan import train_gan
from .v2_gate import feasibility_preflight, digest, object_digest


class ApprovedImageReader:
    """Application-level allowlist; reject all nonselected IDs before any file read."""
    def __init__(self, plan):
        self.root = plan.root
        self.ids = plan.source_ids
        self.sources = {r.id: r for r in plan.sources}
        if len(self.sources) != 52 or tuple(r.id for r in plan.sources) != self.ids:
            raise ValueError("Exactly the approved 52 ordered source IDs are required")
        for row in plan.sources:
            if row.split != "train" or row.label != 1 or row.path != row.id:
                raise ValueError("Holdout/non-TB or altered source path rejected")
        self.read_ids = []
        self.transform = get_transforms(image_size=128, is_training=False, augmentation_mode="none", for_classifier=False)

    def read(self, image_id):
        if image_id not in self.sources:
            raise ValueError("Holdout or unapproved source access rejected before image IO")
        row = self.sources[image_id]
        path = self.root / row.path
        resolved = path.resolve()
        # Reject traversal and symlink/junction substitutions before opening bytes.
        if not resolved.is_relative_to(self.root) or resolved != path.absolute():
            raise ValueError("Source path escape or redirection rejected")
        payload = path.read_bytes()
        if digest(payload) != row.sha256:
            raise ValueError("Selected image hash mismatch")
        with Image.open(BytesIO(payload)) as image:
            tensor = self.transform(to_grayscale_8bit(image))
        if tensor.shape != (1, 128, 128) or not torch.isfinite(tensor).all():
            raise ValueError("Invalid selected image tensor")
        self.read_ids.append(image_id)
        return tensor

    def preload(self):
        images = torch.stack([self.read(identity) for identity in self.ids])
        # Training sees only in-memory tensors. No paths, root, manifest or
        # validation/test image loader is passed to the GAN.
        return TensorDataset(images, torch.ones(52, dtype=torch.long))


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def run_feasibility(config_path, approval_path, output=None, check_only=False):
    started = time.perf_counter()
    plan = feasibility_preflight(config_path, approval_path)
    out = Path(output).resolve() if output is not None else None
    if not check_only:
        if out is None:
            raise ValueError("New output directory required")
        if out.exists() or out.is_relative_to(plan.root):
            raise ValueError("Output must be new and outside the original dataset")
    reader = ApprovedImageReader(plan)
    dataset = reader.preload()
    if check_only:
        return {"status": "preflight_passed_no_training", "source_count": len(dataset),
                "source_ids": list(plan.source_ids), "effective_config": plan.effective_config,
                "bindings": plan.bindings, "device": plan.device}
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(plan.device)
    memory = None
    provenance = {"scope": "v2_feasibility_only", "publication": False, "epochs": 2, "seed": 42,
                  "source_ids": list(plan.source_ids), "source_sha256": {r.id: r.sha256 for r in plan.sources},
                  "actual_image_read_ids": reader.read_ids, "source_count": len(dataset),
                  "bindings": plan.bindings, "commit": plan.bindings["implementation_commit"],
                  "configuration_sha256": plan.bindings["hashes"]["config"],
                  "effective_config": plan.effective_config, "approval_sha256": digest(plan.approval_bytes),
                  "device": str(device), "python": platform.python_version(), "torch": str(torch.__version__),
                  "classifier_training": False, "holdout_evaluation": False}
    try:
        for key, payload in plan.artifacts.items():
            (out / f"input_{key}.snapshot").write_bytes(payload)
        (out / "approval.json").write_bytes(plan.approval_bytes)
        effective_path = out / "effective_generator.yaml"
        effective_path.write_text(yaml.safe_dump(plan.effective_config), encoding="utf-8")
        provenance["effective_yaml_sha256"] = digest(effective_path.read_bytes())
        _write(out / "provenance.json", provenance)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Approved CUDA device unavailable; no silent CPU fallback")
            torch.cuda.synchronize(device)
            free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
            memory = {"free_bytes_at_start": free, "total_bytes": total,
                      "device_name": torch.cuda.get_device_name(device)}
            torch.cuda.reset_peak_memory_stats(device)
        loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False, num_workers=0,
                            generator=torch.Generator().manual_seed(42))
        if len(loader) != 4 or len(dataset) != 52:
            raise ValueError("Expected exactly four batches per epoch from 52 sources")
        training_started = time.perf_counter()
        train_gan(str(effective_path), loader, device, str(out / "gan"))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        training_seconds = time.perf_counter() - training_started
        checkpoint_path = out / "gan/models/generator_latest.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("epoch") != 2 or checkpoint.get("config") != plan.effective_config:
            raise ValueError("Trainer did not complete the exact two-epoch effective configuration")
        if object_digest(plan.effective_config) != plan.bindings["effective_config_sha256"]:
            raise ValueError("Effective configuration changed during execution")
        report = {**provenance, "status": "completed_feasibility_not_publication",
                  "runtime_seconds": time.perf_counter() - started, "training_seconds": training_seconds,
                  "gpu_memory": memory, "gpu_memory_note": "not applicable: approved CPU" if memory is None else None,
                  "batches_per_epoch": 4, "classifier_updates": 0,
                  "checkpoint_sha256": digest(checkpoint_path.read_bytes()),
                  "limitations": "Two epochs check execution only; no medical quality, clinical utility or publication result is established."}
        _write(out / "feasibility_result.json", report)
        return report
    except Exception as exc:
        if memory is not None:
            memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        _write(out / "failure.json", {**provenance, "status": "failed", "error": str(exc),
                                     "runtime_seconds": time.perf_counter() - started, "gpu_memory": memory})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--output")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only and not args.output:
        parser.error("--output is required for an approved run")
    report = run_feasibility(args.config, args.approval, args.output, args.check_only)
    print(json.dumps({k: report[k] for k in ("status", "source_count")}, indent=2))


if __name__ == "__main__":
    main()
