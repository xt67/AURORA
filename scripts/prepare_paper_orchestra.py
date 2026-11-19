"""Build a sanitized Paper Orchestra input bundle from measured local artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise SystemExit('Refusing to overwrite an existing input bundle')
    run = ROOT / 'outputs/feasibility_v2/run_01'
    r = json.loads((run / 'feasibility_result.json').read_text())
    assert r['status'] == 'completed_feasibility_not_publication'
    assert r['epochs'] == 2 and r['source_count'] == 52
    assert r['actual_image_read_ids'] == r['source_ids']
    assert r['classifier_training'] is False and r['holdout_evaluation'] is False
    out.mkdir(parents=True)
    # Explicit aggregate allowlist: no source IDs, medical images or approval identity.
    keys = ['status', 'epochs', 'seed', 'source_count', 'runtime_seconds', 'training_seconds',
            'gpu_memory', 'commit', 'configuration_sha256', 'checkpoint_sha256', 'effective_config']
    evidence = {key: r[key] for key in keys}
    import numpy as np
    evidence['generator_epoch_losses'] = np.load(run / 'gan/g_losses.npy').tolist()
    evidence['discriminator_epoch_losses'] = np.load(run / 'gan/d_losses.npy').tolist()
    evidence['holdout_image_reads'] = 0
    evidence['publication_experiments_completed'] = False
    manuscript = (ROOT / 'paper/professor_review/manuscript.md').read_text(encoding='utf-8-sig')
    (out / 'idea_sparse.md').write_text(
        '# Writing brief\nWrite a protocol and execution-feasibility paper, not an effectiveness paper. '
        'Author: Rayan Rahman; Marathwada Mitra Mandal Institute of Technology. '
        'Preserve missing declarations; never invent metrics, approvals, novelty or clinical review. '
        'Keep planned methods in future tense. Verify every citation. Supplied figures are measured or schematic.\n\n'
        + manuscript, encoding='utf-8')
    (out / 'experimental_log.md').write_text(
        '# Verified aggregate execution evidence\nOnly one two-epoch run has completed. '
        'No classifier results, quality scores, external validation or clinical review exist.\n\n```json\n'
        + json.dumps(evidence, indent=2) + '\n```\n', encoding='utf-8')
    (out / 'figures').mkdir()
    for name in ['feasibility-losses.png', 'aurora-workflow.png']:
        shutil.copyfile(ROOT / 'docs/images' / name, out / 'figures' / name)
    figures = [
        {'name': 'feasibility-losses.png', 'caption': 'Measured batch-mean generator and discriminator losses for two epochs, 52 real TB training images, seed 42. Not evidence of convergence or clinical image quality.'},
        {'name': 'aurora-workflow.png', 'caption': 'Schematic of automated image selection, fixed image-level split, completed feasibility run and proposed unexecuted four-condition comparison. Counts are images, not unique patients.'},
    ]
    (out / 'figures/info.json').write_text(json.dumps(figures, indent=2), encoding='utf-8')
    hashes = {str(p.relative_to(out)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in out.rglob('*') if p.is_file()}
    (out / 'input_hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    print(f'Prepared aggregate-only input bundle: {out}')


if __name__ == '__main__':
    main()
