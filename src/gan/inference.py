"""Shared inference and real PNG export for the demo and command line."""
from io import BytesIO
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from .generator import Generator


def load_generator(path, device="cpu"):
    device = torch.device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "netG_state" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Checkpoint requires netG_state and config; legacy weights lack architecture metadata")
    config = checkpoint["config"]
    model = Generator(config["z_dim"], config.get("ngf", 64), 1).to(device)
    model.load_state_dict(checkpoint["netG_state"])
    model.eval()
    return model, config


@torch.inference_mode()
def generate_images(model, count, z_dim, seed=42):
    if not 1 <= count <= 256:
        raise ValueError("count must be between 1 and 256")
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    return model(torch.randn(count, z_dim, 1, 1, generator=rng, device=device))


def png_grid(images):
    stream = BytesIO()
    grid = make_grid(images.detach().cpu(), nrow=4, normalize=True, value_range=(-1, 1))
    save_image(grid, stream, format="PNG")
    return stream.getvalue()


def main():
    import argparse
    import hashlib
    import json
    parser = argparse.ArgumentParser(description="Export explicitly synthetic images from an AURORA checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory must be empty to avoid mixing experiment artifacts")
    model, config = load_generator(args.checkpoint, args.device)
    out.mkdir(parents=True, exist_ok=True)
    for start in range(0, args.count, 32):
        images = generate_images(model, min(32, args.count - start), config["z_dim"], args.seed + start)
        for offset, image in enumerate(images):
            save_image((image.cpu() + 1) / 2, out / f"synthetic_{start + offset:06d}.png")
    manifest = {"synthetic": True, "count": args.count, "seed": args.seed,
                "batch_size": 32, "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                "config": config, "clinical_validation": False}
    (out / "generation_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
