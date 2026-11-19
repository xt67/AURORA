"""Separate v2 publication gate; historical v1 and feasibility gates stay intact."""
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess

import torch
import yaml

from . import v2_gate
from .data import budget_rows, sha256
from .v2_gate import digest, object_digest

SCOPE = 'aurora_v2_publication'
FLAGS = ('dataset_approved', 'split_approved', 'protocol_approved', 'publication_run_approved')
FILES = tuple(sorted(set(v2_gate.IMPLEMENTATION_FILES + (
    'src/study/publication_gate.py', 'src/study/publication.py',
    'src/study/publication_analysis.py', 'src/study/runner.py', 'src/study/analysis.py',
    'src/classifier/model.py', 'src/evaluate/metrics.py', 'src/evaluate/generative.py',
    'src/gan/inference.py',
))))


def implementation():
    root = v2_gate.ROOT
    changed = subprocess.check_output(['git', 'status', '--porcelain', '--', *FILES], cwd=root, text=True)
    if changed.strip():
        raise ValueError('Publication implementation must be committed')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    return {'commit': commit, 'sha256': object_digest({p: sha256(root / p) for p in FILES})}


@dataclass(frozen=True)
class PublicationPlan:
    config: dict
    manifest: dict
    root: Path
    bindings: dict
    artifacts: dict


def plan(config_path):
    raw = Path(config_path).read_bytes()
    cfg = yaml.safe_load(raw)
    if cfg.get('scope') != SCOPE or cfg.get('schema_version') != 2 or cfg.get('device') != 'cuda':
        raise ValueError('Explicit v2 CUDA publication config required')
    if cfg.get('diagnostics') != 'training_pixel_neighbours_and_diversity' or cfg.get('bootstrap_repetitions') != 1000:
        raise ValueError('Unexpected diagnostic/analysis settings')
    # Reuse the full provenance reconstruction, not any feasibility approval.
    base = v2_gate._plan(cfg['proposal'], 'cuda')
    design = yaml.safe_load(base.artifacts['config'])
    expected_classifier = dict(architecture='resnet18', pretrained=True, optimizer='adam', updates=1000,
        eval_every=100, batch_size=16, lr=.0001, weight_decay=.0001, use_amp_on_cuda=True,
        loss='unweighted_cross_entropy', checkpoint_selection='minimum_validation_loss',
        threshold_selection='maximum_validation_f1_highest_threshold_tiebreak')
    expected_aug = dict(target_label='tuberculosis', horizontal_flip_probability=0., rotation_degrees=5,
                        translation_fraction=.05, brightness_jitter=.1, contrast_jitter=.1)
    if design['classifier'] != expected_classifier or design['conventional_augmentation'] != expected_aug:
        raise ValueError('Publication classifier/augmentation differs from specified design')
    if design['conditions'] != dict(A='real_only', B='duplicate_real_tb', C='conventional_augmentation_tb_only', D='dcgan_synthetic_tb'):
        raise ValueError('Exactly the four specified conditions required')
    if design['budgets'] != [.25, .5, 1.] or design['seeds'] != [42, 123, 999]:
        raise ValueError('Exactly the planned budgets and seeds required')
    if design['added_tb_count'] != dict(basis='selected_real_tb_training_count', multiplier=1., applies_to=['B', 'D']):
        raise ValueError('B and D must add one image per selected TB source')
    if (design['generator']['epochs'], design['generator']['checkpoint_every'], design['generator']['selection']) != (200, 10, 'last_fixed_epoch'):
        raise ValueError('Fixed 200-epoch final-checkpoint generator required')
    manifest = json.loads(base.artifacts['split_manifest'])
    selections = {str(b): budget_rows(manifest['rows'], b, 42) for b in design['budgets']}
    positives = [{r['id'] for r in rows if r['label'] == 1} for rows in selections.values()]
    if list(map(len, positives)) != [52, 105, 210] or not positives[0] < positives[1] < positives[2]:
        raise ValueError('Nested TB budgets mismatch')
    weight_file = Path(torch.hub.get_dir()) / 'checkpoints/resnet18-f37072fd.pth'
    weight_hash = sha256(weight_file)
    if not weight_hash.startswith('f37072fd'):
        raise ValueError('ImageNet ResNet18 checkpoint hash mismatch')
    artifacts = {'publication_config': raw, 'publication_protocol': Path(cfg['protocol']).read_bytes(),
                 **{'proposal_' + k: v for k, v in base.artifacts.items()}}
    bindings = {'hashes': {k: digest(v) for k, v in artifacts.items()},
                'training_ids': {b: [r['id'] for r in rows] for b, rows in selections.items()},
                'generator_ids': {b: [r['id'] for r in rows if r['label'] == 1] for b, rows in selections.items()},
                'selection_sha256': object_digest(selections), 'implementation': implementation(),
                'pretrained_weights_sha256': weight_hash}
    return PublicationPlan({**design, 'publication_settings': cfg}, manifest, base.root, bindings, artifacts)


def draft(config_path):
    p = plan(config_path)
    return {'schema_version': 2, 'scope': SCOPE, **{f: False for f in FLAGS},
            'approved_by': '', 'approved_at': '', 'authorization_basis': '', **p.bindings}


def preflight(config_path, approval_path):
    if not Path(approval_path).is_file():
        raise ValueError('Publication approval missing')
    a = json.loads(Path(approval_path).read_text(encoding='utf-8'))
    if a.get('scope') != SCOPE or a.get('schema_version') != 2:
        raise ValueError('Separate v2 publication approval scope required')
    if any(a.get(f) is not True for f in FLAGS):
        raise ValueError('Explicit dataset/split/protocol/publication approval required')
    if not a.get('approved_by', '').strip() or not a.get('authorization_basis', '').strip():
        raise ValueError('Researcher and actual authorization basis required')
    if datetime.fromisoformat(a.get('approved_at', '').replace('Z', '+00:00')).tzinfo is None:
        raise ValueError('Timezone-qualified approval time required')
    p = plan(config_path)
    for key, value in p.bindings.items():
        if a.get(key) != value:
            raise ValueError(f'Publication approval mismatch: {key}')
    return p
