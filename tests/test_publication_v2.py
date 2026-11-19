"""Publication scope, hash/source boundaries and TB-only augmentation tests."""
from copy import deepcopy
import json
from pathlib import Path

from PIL import Image
import pytest
import torch
import yaml

from test_v2_feasibility import approved_fixture
from src.study import publication_gate as gate
from src.study import publication as runner
from src.study.data import sha256


@pytest.fixture
def publication_fixture(approved_fixture, tmp_path, monkeypatch):
    proposal, _, _, _, manifest = approved_fixture
    monkeypatch.setattr(gate, 'implementation', lambda: {'commit': 'c' * 40, 'sha256': 'd' * 64})
    real_hash = gate.sha256
    monkeypatch.setattr(gate, 'sha256', lambda p: 'f37072fd' + '0' * 56 if Path(p).name == 'resnet18-f37072fd.pth' else real_hash(p))
    cfg = yaml.safe_load(Path('configs/study_publication_v2.yaml').read_text())
    cfg['proposal'] = str(proposal)
    path = tmp_path / 'publication.yaml'
    path.write_text(yaml.safe_dump(cfg))
    approval = gate.draft(path)
    assert all(approval[k] is False for k in gate.FLAGS)
    approval.update({k: True for k in gate.FLAGS})
    approval.update(approved_by='TEST ONLY', approved_at='2026-11-19T12:00:00+05:30', authorization_basis='Artificial fixture')
    auth = tmp_path / 'publication_approval.json'
    auth.write_text(json.dumps(approval))
    return path, auth, approval, manifest


def test_publication_gate_nested_sources(publication_fixture):
    cfg, auth, _, manifest = publication_fixture
    plan = gate.preflight(cfg, auth)
    ids = [set(x) for x in plan.bindings['generator_ids'].values()]
    assert list(map(len, ids)) == [52, 105, 210]
    assert ids[0] < ids[1] < ids[2]
    holdout = {r['id'] for r in manifest['rows'] if r['split'] != 'train'}
    assert not ids[2] & holdout


@pytest.mark.parametrize('field', ['scope', *gate.FLAGS, 'hashes', 'generator_ids', 'training_ids', 'implementation', 'pretrained_weights_sha256'])
def test_publication_rejects_missing_or_changed_approval(publication_fixture, field):
    cfg, auth, approval, _ = publication_fixture
    approval[field] = False if field in gate.FLAGS else 'wrong'
    auth.write_text(json.dumps(approval))
    with pytest.raises(ValueError):
        gate.preflight(cfg, auth)


def test_missing_approval_and_changed_config(publication_fixture, tmp_path):
    cfg, auth, _, _ = publication_fixture
    with pytest.raises(ValueError, match='missing'):
        gate.preflight(cfg, tmp_path / 'absent.json')
    with cfg.open('a') as stream:
        stream.write('\n# changed input bytes\n')
    with pytest.raises(ValueError, match='mismatch'):
        gate.preflight(cfg, auth)


def test_generator_rejects_holdout_before_image_io(tmp_path, monkeypatch):
    row = dict(id='cohort/x.png', path='cohort/x.png', split='val', label=1, sha256='0' * 64)
    monkeypatch.setattr(Path, 'read_bytes', lambda p: pytest.fail('No image IO permitted'))
    with pytest.raises(ValueError, match='Holdout'):
        runner.generator_dataset(tmp_path, [row])
    row['split'] = 'train'
    reader = runner.ImageReader(tmp_path, [row], 'train', True)
    with pytest.raises(ValueError, match='before IO'):
        reader.read('test/unknown.png')


def test_generator_hash_checked_and_tensor_only(tmp_path):
    image = tmp_path / 'x.png'
    Image.new('L', (16, 16), 100).save(image)
    row = dict(id='x.png', path='x.png', split='train', label=1, sha256=sha256(image))
    data, _, ids = runner.generator_dataset(tmp_path, [row])
    assert isinstance(data, torch.utils.data.TensorDataset)
    assert data.tensors[0].shape == (1, 1, 128, 128)
    assert ids == ['x.png']
    image.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        runner.generator_dataset(tmp_path, [row])


def test_conventional_only_tb_no_flip_and_duplicate_counts():
    ds = runner.ClassifierDataset([Image.new('L', (8, 8), 50)] * 2, [0, 1], True)
    assert not any('Flip' in type(t).__name__ for t in ds.augmented.transforms)
    calls = []
    ds.augmented = lambda im: calls.append(True) or torch.ones(3, 128, 128)
    ds[0]
    assert calls == []
    ds[1]
    assert calls == [True]
    for count in [52, 105, 210]:
        rows = [dict(id=str(i), label=1) for i in range(count)] + [dict(id='normal', label=0)]
        added = runner.duplicate_positive_rows(rows, count, 42)
        assert len(added) == count and all(r['label'] == 1 for r in added)


def test_diagnostics_detect_identical_images():
    image = Image.new('L', (128, 128), 100)
    d = runner.image_diagnostics([image, image], [image, image], ['a', 'b'])
    assert d['synthetic_duplicate_count'] == 1
    assert d['exact_matches_to_resized_training'] == 2
    assert d['mean_nearest_training_rmse'] == pytest.approx(0, abs=1e-6)
    assert d['diversity_mean_pairwise_rmse'] == pytest.approx(0, abs=1e-6)


def test_test_images_not_opened_when_training_fails(publication_fixture, tmp_path, monkeypatch):
    cfg, auth, _, _ = publication_fixture
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(runner, 'load_dataset', lambda *args: pytest.fail('No evaluation image IO before fitting'))
    def failed_fit(*args):
        raise RuntimeError('artificial training failure')
    monkeypatch.setattr(runner, 'fit_one', failed_fit)
    with pytest.raises(RuntimeError, match='artificial training failure'):
        runner.run(cfg, auth, tmp_path / 'run')
    assert not (tmp_path / 'run/completed.json').exists()


def test_check_only_does_not_train_or_read_images(publication_fixture, tmp_path, monkeypatch):
    cfg, auth, _, _ = publication_fixture
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(runner, 'fit_one', lambda *args: pytest.fail('No training'))
    monkeypatch.setattr(runner.ImageReader, 'read', lambda *args: pytest.fail('No images'))
    runner.run(cfg, auth, tmp_path / 'not_created', check_only=True)
    assert not (tmp_path / 'not_created').exists()
