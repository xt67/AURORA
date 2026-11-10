"""Image diversity/nearest-neighbour diagnostics and optional Inception FID/KID."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import pairwise_distances
from src.data_loader import to_grayscale_8bit


def image_features(folder):
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if len(paths) < 2:
        raise ValueError("At least two images per directory are required")
    vectors = []
    for path in paths:
        with Image.open(path) as image:
            vectors.append(np.asarray(to_grayscale_8bit(image).resize((64, 64)), dtype=np.float32).ravel() / 255.)
    return paths, np.array(vectors)


def nearest_neighbours(real, synthetic, chunk_size=64):
    distances, indices = [], []
    for start in range(0, len(synthetic), chunk_size):
        matrix = pairwise_distances(synthetic[start:start + chunk_size], real) / np.sqrt(real.shape[1])
        index = matrix.argmin(axis=1)
        indices.extend(index.tolist())
        distances.extend(matrix[np.arange(len(index)), index].tolist())
    return distances, indices


def polynomial_mmd(real, fake):
    real, fake = np.asarray(real, dtype=np.float64), np.asarray(fake, dtype=np.float64)
    if real.ndim != 2 or fake.ndim != 2 or real.shape[1] != fake.shape[1] or min(len(real), len(fake)) < 2:
        raise ValueError("KID needs at least two matching-dimensional feature vectors per set")
    dim = real.shape[1]
    rr, ff, rf = (real @ real.T / dim + 1) ** 3, (fake @ fake.T / dim + 1) ** 3, (real @ fake.T / dim + 1) ** 3
    return float((rr.sum() - np.trace(rr)) / (len(real) * (len(real) - 1)) +
                 (ff.sum() - np.trace(ff)) / (len(fake) * (len(fake) - 1)) - 2 * rf.mean())


def diagnostics(real_folder, fake_folder, seed=42, inception=False, device="cpu"):
    real_paths, real = image_features(real_folder)
    fake_paths, fake = image_features(fake_folder)
    distances, indices = nearest_neighbours(real, fake)
    rng = np.random.default_rng(seed)
    sample = fake[rng.choice(len(fake), min(1000, len(fake)), replace=False)]
    pairwise = pairwise_distances(sample) / np.sqrt(fake.shape[1])
    result = {"real_count": len(real), "synthetic_count": len(fake), "seed": seed,
              "feature_space": "64x64 grayscale pixels in [0,1]; RMSE distances",
              "mean_nearest_training_rmse": float(np.mean(distances)),
              "diversity_mean_pairwise_rmse": float(pairwise[np.triu_indices(len(sample), 1)].mean()),
              "neighbours": [{"synthetic": fake_paths[i].name, "reference": real_paths[j].name, "rmse": distances[i]} for i, j in enumerate(indices)],
              "limitations": "Pixel proximity is an exploratory memorization screen, not a privacy guarantee or clinical assessment. Use training images as the reference for memorization; use a separately declared validation reference for fidelity."}
    if inception:
        from pytorch_fid.inception import InceptionV3
        from pytorch_fid.fid_score import get_activations
        from .fid import compute_fid
        model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
        a = get_activations([str(p) for p in real_paths], model, batch_size=16, dims=2048, device=device, num_workers=0)
        b = get_activations([str(p) for p in fake_paths], model, batch_size=16, dims=2048, device=device, num_workers=0)
        size = min(1000, len(a), len(b))
        estimates = [polynomial_mmd(a[rng.choice(len(a), size, replace=False)], b[rng.choice(len(b), size, replace=False)]) for _ in range(20)]
        result.update(fid=compute_fid(real_folder, fake_folder, device, batch_size=16),
                      kid_mean=float(np.mean(estimates)), kid_subset_std=float(np.std(estimates, ddof=1)),
                      kid_subset_size=size, kid_subsets=20, inception_features="pytorch-fid 2048-dimensional pool3")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True)
    parser.add_argument("--synthetic", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inception", action="store_true", help="May download pretrained Inception weights; compute FID/KID")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = diagnostics(args.real, args.synthetic, inception=args.inception, device=args.device)
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
