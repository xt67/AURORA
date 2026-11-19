"""Approved v2 grid: fit all models before any test evaluation; never overwrite fits."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.utils import save_image
import yaml

from src.classifier.model import MedicalImageClassifier
from src.data_loader import get_transforms, to_grayscale_8bit, repeat_grayscale_channels
from src.gan.train_gan import train_gan
from src.gan.inference import generate_images
from src.evaluate.generative import nearest_neighbours
from .data import budget_rows, sha256
from .publication_gate import preflight
from .runner import make_train_loader, select_threshold, predict, evaluate_predictions, duplicate_positive_rows, write_json
from src.utils.seed import set_seed
from .v2_gate import digest, object_digest


class ImageReader:
    """Hash-check and decode only explicitly allowed identities/partitions."""
    def __init__(self, root, rows, partition, positive_only=False):
        self.root = Path(root).resolve()
        self.rows = {r['id']: r for r in rows}
        self.read_ids = []
        if len(self.rows) != len(rows) or not rows:
            raise ValueError('Nonempty unique image allowlist required')
        for r in rows:
            if r['split'] != partition or (positive_only and r['label'] != 1) or r['id'] != r['path']:
                raise ValueError('Holdout, label or identity mismatch in image allowlist')

    def read(self, identity):
        if identity not in self.rows:
            raise ValueError('Unapproved/holdout image access rejected before IO')
        r = self.rows[identity]
        path = self.root / r['path']
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root) or resolved != path.absolute():
            raise ValueError('Image path escape/redirection rejected')
        raw = path.read_bytes()
        if digest(raw) != r['sha256']:
            raise ValueError('Image hash mismatch')
        with Image.open(BytesIO(raw)) as im:
            image = to_grayscale_8bit(im).copy()
        self.read_ids.append(identity)
        return image


class ClassifierDataset(Dataset):
    def __init__(self, images, labels, augment_tb=False):
        self.labels = list(labels)
        self.images = images
        self.augment_tb = augment_tb
        self.deterministic = get_transforms(is_training=False, augmentation_mode='none', for_classifier=True)
        self.augmented = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomAffine(degrees=5, translate=(.05, .05)),
            transforms.ColorJitter(brightness=.1, contrast=.1),
            transforms.ToTensor(), repeat_grayscale_channels,
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        self.cached = [self.deterministic(im) for im in images]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        tensor = self.augmented(self.images[i]) if self.augment_tb and self.labels[i] == 1 else self.cached[i]
        return tensor, self.labels[i]


def load_dataset(root, rows, partition, augment_tb=False):
    reader = ImageReader(root, rows, partition)
    return ClassifierDataset([reader.read(r['id']) for r in rows], [r['label'] for r in rows], augment_tb), reader.read_ids


def generator_dataset(root, rows):
    reader = ImageReader(root, rows, 'train', positive_only=True)
    images = [reader.read(r['id']) for r in rows]
    transform = get_transforms(is_training=False, augmentation_mode='none')
    tensor = torch.stack([transform(im) for im in images])
    return TensorDataset(tensor, torch.ones(len(images), dtype=torch.long)), images, reader.read_ids


def image_diagnostics(real_images, fake_images, source_ids):
    def features(images):
        return np.stack([np.asarray(im.resize((64, 64)), dtype=np.float32).ravel() / 255. for im in images])
    real, fake = features(real_images), features(fake_images)
    distances, indices = nearest_neighbours(real, fake)
    from sklearn.metrics import pairwise_distances
    pairwise = pairwise_distances(fake) / np.sqrt(fake.shape[1])
    hashes = [digest(im.tobytes()) for im in fake_images]
    real_hashes = {digest(im.resize((128, 128), Image.Resampling.BILINEAR).tobytes()) for im in real_images}
    return {'real_training_count': len(real), 'synthetic_count': len(fake),
            'feature_space': '64x64 grayscale pixels [0,1], Pillow default bicubic diagnostic resize; RMSE',
            'mean_nearest_training_rmse': float(np.mean(distances)),
            'min_nearest_training_rmse': float(np.min(distances)),
            'diversity_mean_pairwise_rmse': float(pairwise[np.triu_indices(len(fake), 1)].mean()),
            'synthetic_duplicate_count': len(hashes) - len(set(hashes)),
            'exact_matches_to_resized_training': sum(h in real_hashes for h in hashes),
            'neighbours': [{'synthetic_index': i, 'training_id': source_ids[j], 'rmse': distances[i]} for i, j in enumerate(indices)],
            'limitations': 'Exploratory pixel diagnostics; no clinical fidelity assessment or privacy guarantee. No FID/KID computed.'}


def fit_generator(plan, selected, seed, device, out):
    positives = [r for r in selected if r['label'] == 1]
    data, images, read_ids = generator_dataset(plan.root, positives)
    config = {k: v for k, v in plan.config['generator'].items() if k not in ('architecture', 'selection')}
    config.update(seed=seed, image_size=128, label_smoothing=.9)
    (out / 'generator.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    loader = DataLoader(data, batch_size=16, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed), drop_last=False)
    start = time.perf_counter()
    generator, discriminator = train_gan(str(out / 'generator.yaml'), loader, device, str(out / 'generator'))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    checkpoint = out / 'generator/models/generator_latest.pth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state['epoch'] != 200 or state['config'] != config:
        raise ValueError('Generator did not complete the frozen 200-epoch configuration')
    generator.eval()
    folder = out / 'synthetic'
    folder.mkdir()
    fake_images, hashes = [], {}
    for start in range(0, len(positives), 16):
        batch = generate_images(generator, min(16, len(positives) - start), config['z_dim'], seed + start)
        for offset, tensor in enumerate(batch):
            path = folder / f'synthetic_{start + offset:06d}.png'
            save_image((tensor.cpu() + 1) / 2, path)
            with Image.open(path) as im:
                fake_images.append(im.convert('L').copy())
            hashes[path.name] = sha256(path)
    diagnostics = image_diagnostics(images, fake_images, read_ids)
    write_json(out / 'generator_diagnostics.json', diagnostics)
    write_json(out / 'generator_provenance.json', {
        'source_ids': read_ids, 'source_sha256': {r['id']: r['sha256'] for r in positives},
        'actual_image_read_ids': read_ids, 'effective_config': config, 'config_sha256': object_digest(config),
        'checkpoint_sha256': sha256(checkpoint), 'synthetic_sha256': hashes,
        'source_count': len(positives), 'synthetic_count': len(fake_images), 'runtime_seconds': elapsed,
        'holdout_images_read': False, 'selection': 'fixed final epoch, all generated samples retained'})
    del generator, discriminator, state, data
    torch.cuda.empty_cache()
    return ClassifierDataset(fake_images, [1] * len(fake_images))


def train_classifier(dataset, validation, cfg, arm, seed, device, out):
    """Fixed successful updates; low AMP scale prevents initial FP16 overflow."""
    set_seed(seed)
    model = MedicalImageClassifier(pretrained=cfg['pretrained']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    amp = bool(cfg['use_amp'] and device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=cfg['amp_initial_scale'])
    train = make_train_loader(dataset, arm, cfg['updates'], cfg['batch_size'], seed)
    val = DataLoader(validation, batch_size=cfg['batch_size'], num_workers=0)
    history, best_loss = [], float('inf')
    for step, (images, labels) in enumerate(train, 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            loss = torch.nn.functional.cross_entropy(model(images.to(device)), labels.to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite classifier loss at update {step}')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f'Non-finite classifier gradient at update {step}; no skipped optimizer updates allowed')
        scaler.step(optimizer)
        scaler.update()
        if step % cfg['eval_every'] == 0 or step == cfg['updates']:
            _, _, val_loss = predict(model, val, device)
            history.append({'step': step, 'training_loss': loss.item(), 'validation_loss': val_loss, 'amp_scale': scaler.get_scale()})
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({'state_dict': model.state_dict(), 'step': step, 'config': cfg}, out / 'classifier.pth')
            write_json(out / 'training_history.json', history)
            print(f'CLASSIFIER {out.name}: {step}/{cfg["updates"]} updates', flush=True)
    model.load_state_dict(torch.load(out / 'classifier.pth', map_location=device, weights_only=True)['state_dict'])
    labels, probabilities, _ = predict(model, val, device)
    return model, select_threshold(labels, probabilities)


def fit_one(plan, arm, fraction, seed, out):
    out.mkdir(exist_ok=False)
    start = time.perf_counter()
    selected = budget_rows(plan.manifest['rows'], fraction, 42)
    device = torch.device('cuda')
    try:
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        real, read_ids = load_dataset(plan.root, selected, 'train', arm == 'C')
        added = sum(r['label'] == 1 for r in selected)
        duplicates = []
        if arm == 'B':
            duplicates = duplicate_positive_rows(selected, added, seed)
            index = {r['id']: i for i, r in enumerate(selected)}
            dataset = ClassifierDataset(real.images + [real.images[index[r['id']]] for r in duplicates], real.labels + [1] * added)
        elif arm == 'D':
            synthetic = fit_generator(plan, selected, seed, device, out)
            dataset = ConcatDataset([real, synthetic])
        elif arm in ('A', 'C'):
            dataset = real
        else:
            raise ValueError('Unknown condition')
        val_rows = [r for r in plan.manifest['rows'] if r['split'] == 'val']
        val, val_read_ids = load_dataset(plan.root, val_rows, 'val')
        cfg = {**plan.config['classifier'], 'use_amp': plan.config['classifier']['use_amp_on_cuda'],
               'amp_initial_scale': plan.config['publication_settings']['classifier_amp_initial_scale']}
        model, threshold = train_classifier(dataset, val, cfg, arm, seed, device, out)
        labels, probabilities, val_loss = predict(model, DataLoader(val, batch_size=16), device)
        torch.cuda.synchronize()
        record = {'status': 'fitted_test_not_evaluated', 'arm': arm, 'budget': fraction, 'seed': seed,
                  'publication': True, 'threshold': threshold, 'validation_loss': val_loss,
                  'validation_predictions': [{'id': r['id'], 'label': int(y), 'probability': float(p)} for r, y, p in zip(val_rows, labels, probabilities)],
                  'training_ids': read_ids, 'validation_read_ids': val_read_ids,
                  'duplicate_count': len(duplicates), 'duplicate_source_ids': [r['id'] for r in duplicates],
                  'synthetic_count': added if arm == 'D' else 0, 'updates': 1000, 'batch_size': 16,
                  'bindings': plan.bindings, 'config': plan.config,
                  'manifest_digest': object_digest(plan.manifest), 'checkpoint_sha256': sha256(out / 'classifier.pth'),
                  'commit': plan.bindings['implementation']['commit'], 'runtime_seconds': time.perf_counter() - start,
                  'cuda_memory': {'free_bytes_at_start': free, 'total_bytes': total,
                                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                  'peak_reserved_bytes': torch.cuda.max_memory_reserved()},
                  'hardware': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
                  'python': platform.python_version(), 'pretrained_weights_sha256': plan.bindings['pretrained_weights_sha256']}
        write_json(out / 'fit.json', record)
        del model, dataset, real, val
        torch.cuda.empty_cache()
        print(f'FIT COMPLETE {out.name}: {record["runtime_seconds"]:.1f}s', flush=True)
    except Exception as exc:
        write_json(out / 'failure.json', {'error': str(exc), 'arm': arm, 'budget': fraction,
                  'seed': seed, 'bindings': plan.bindings, 'runtime_seconds': time.perf_counter() - start})
        raise


def verified_fit(folder, plan):
    if (folder / 'failure.json').exists() or not (folder / 'fit.json').is_file():
        raise ValueError(f'Incomplete/failed fit retained; no automatic overwrite: {folder}')
    fit = json.loads((folder / 'fit.json').read_text())
    if fit['bindings'] != plan.bindings or fit['checkpoint_sha256'] != sha256(folder / 'classifier.pth'):
        raise ValueError('Completed fit provenance/checkpoint mismatch')
    if fit['arm'] == 'D':
        provenance = json.loads((folder / 'generator_provenance.json').read_text())
        expected = plan.bindings['generator_ids'][str(fit['budget'])]
        if provenance['source_ids'] != expected or provenance['synthetic_count'] != len(expected):
            raise ValueError('Generator source/count mismatch')
        if provenance['checkpoint_sha256'] != sha256(folder / 'generator/models/generator_latest.pth'):
            raise ValueError('Generator checkpoint mismatch')
        for name, value in provenance['synthetic_sha256'].items():
            if sha256(folder / 'synthetic' / name) != value:
                raise ValueError('Synthetic image hash mismatch')
    return fit


def run(config, approval, output, resume=False, check_only=False):
    plan = preflight(config, approval)
    if not torch.cuda.is_available():
        raise RuntimeError('Approved CUDA unavailable; no fallback')
    if check_only:
        print('Publication metadata gate passed; no training or image IO.', flush=True)
        return
    out = Path(output).resolve()
    if out.is_relative_to(plan.root):
        raise ValueError('Output must be outside originals')
    if resume:
        if json.loads((out / 'bindings.json').read_text()) != plan.bindings:
            raise ValueError('Resume bindings mismatch')
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / 'bindings.json', plan.bindings)
        write_json(out / 'frozen_manifest.json', plan.manifest)
        write_json(out / 'config.json', plan.config)
        (out / 'approval.json').write_bytes(Path(approval).read_bytes())
        for key, payload in plan.artifacts.items():
            (out / (key + '.snapshot')).write_bytes(payload)
    # Bound CPU overhead consistently; no change to optimizer or batch counts.
    torch.set_num_threads(4)
    folders = []
    for fraction in plan.config['budgets']:
        for seed in plan.config['seeds']:
            for arm in plan.config['conditions']:
                folder = out / f'{arm}_b{fraction}_s{seed}'
                folders.append(folder)
                if folder.exists():
                    verified_fit(folder, plan)
                else:
                    print(f'FITTING {folder.name}', flush=True)
                    fit_one(plan, arm, fraction, seed, folder)
    fits = [verified_fit(folder, plan) for folder in folders]
    if len(fits) != 36:
        raise ValueError('Entire grid must finish before test evaluation')
    # The test image bytes are first read only after ALL checkpoint selections.
    test_rows = [r for r in plan.manifest['rows'] if r['split'] == 'test']
    test, read_ids = load_dataset(plan.root, test_rows, 'test')
    for folder, fit in zip(folders, fits):
        if (folder / 'result.json').exists():
            previous = json.loads((folder / 'result.json').read_text())
            if previous['bindings'] != plan.bindings or previous['checkpoint_sha256'] != fit['checkpoint_sha256']:
                raise ValueError('Existing evaluation differs from frozen fit')
            continue
        model = MedicalImageClassifier(pretrained=False).cuda()
        state = torch.load(folder / 'classifier.pth', map_location='cuda', weights_only=True)
        model.load_state_dict(state['state_dict'])
        labels, probabilities, _ = predict(model, DataLoader(test, batch_size=16), torch.device('cuda'))
        result = {**fit, 'status': 'measured', 'test_read_ids': read_ids,
                  'predictions': [{'id': r['id'], 'group': r['group'], 'cohort': r['cohort'], 'label': int(y), 'probability': float(p)} for r, y, p in zip(test_rows, labels, probabilities)],
                  'metrics': evaluate_predictions(test_rows, labels, probabilities, fit['threshold'])}
        write_json(folder / 'result.json', result)
        del model, state
        torch.cuda.empty_cache()
    write_json(out / 'completed.json', {'classifier_count': 36, 'generator_count': 9, 'bindings': plan.bindings})
    print('COMPLETE: 36 classifier results and nine generators. Analysis pending.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    run(args.config, args.approval, args.output, args.resume, args.check_only)


if __name__ == '__main__':
    main()
