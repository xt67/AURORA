"""Export all measured v2 results, paired contrasts and generator diagnostics."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .analysis import summarize, METRICS
from .runner import write_json
from .data import sha256
from .v2_gate import object_digest


def analyze(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('Analysis output must be new')
    done = json.loads((root / 'completed.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    manifest = json.loads((root / 'frozen_manifest.json').read_text())
    rows, diagnostics = [], []
    for budget, seed, arm in itertools.product(config['budgets'], config['seeds'], config['conditions']):
        folder = root / f'{arm}_b{budget}_s{seed}'
        result = json.loads((folder / 'result.json').read_text())
        if result['bindings'] != done['bindings'] or result['manifest_digest'] != object_digest(manifest):
            raise ValueError('Result provenance differs from completed run')
        if result['checkpoint_sha256'] != sha256(folder / 'classifier.pth'):
            raise ValueError('Classifier checkpoint hash mismatch')
        if (result['arm'], result['budget'], result['seed']) != (arm, budget, seed):
            raise ValueError('Result identity differs from configured grid')
        if result['publication'] is not True or result['status'] != 'measured':
            raise ValueError('Unmeasured or fixture result')
        expected_test = [(r['id'], r['label']) for r in manifest['rows'] if r['split'] == 'test']
        if [(r['id'], r['label']) for r in result['predictions']] != expected_test:
            raise ValueError('Real test identity/label mismatch')
        rows.append(result)
        if arm == 'D':
            diag = json.loads((folder / 'generator_diagnostics.json').read_text())
            provenance = json.loads((folder / 'generator_provenance.json').read_text())
            if provenance['checkpoint_sha256'] != sha256(folder / 'generator/models/generator_latest.pth'):
                raise ValueError('Generator checkpoint hash mismatch')
            diagnostics.append({'budget': budget, 'seed': seed,
                **{k: v for k, v in diag.items() if k != 'neighbours'},
                'runtime_seconds': provenance['runtime_seconds']})
    print('Computing all per-model and cohort bootstrap intervals...', flush=True)
    summary = summarize(rows, bootstrap=1000, baseline='A')
    comparisons = [r for r in summary['paired'] if r['arm'] == 'D']
    by_key = {(r['arm'], r['budget'], r['seed']): r for r in rows}
    for budget in config['budgets']:
        deltas = {k: [] for k in METRICS}
        for seed in config['seeds']:
            b, d = [by_key[(arm, budget, seed)] for arm in ['B', 'D']]
            if b['duplicate_count'] != d['synthetic_count'] or b['training_ids'] != d['training_ids']:
                raise ValueError('B/D counts or real sources differ')
            for key in METRICS:
                deltas[key].append(d['metrics']['overall'][key] - b['metrics']['overall'][key])
        rng = np.random.default_rng(42)
        comparisons.append({'arm': 'D', 'baseline': 'B', 'budget': budget,
            'paired_seeds': config['seeds'], 'metrics': {k: {
                'mean_difference': float(np.mean(v)),
                'seed_bootstrap_ci95': np.quantile(np.mean(rng.choice(v, size=(1000, len(v)), replace=True), axis=1), [.025, .975]).tolist()
            } for k, v in deltas.items()}})
    report = {'status': 'measured', 'schema_version': 2, 'run_count': 36, 'generator_count': 9,
              'experiments': summary['aggregate'], 'per_run': summary['per_run'],
              'comparisons': comparisons, 'generator_diagnostics': diagnostics,
              'bindings': done['bindings'], 'limitations': summary['limitations'],
              'runtime_seconds_total_fits': sum(r['runtime_seconds'] for r in rows),
              'max_allocated_gpu_bytes': max(r['cuda_memory']['peak_allocated_bytes'] for r in rows)}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'paper_results.json', report)
    lines = ['# AURORA measured four-condition comparison', '',
             'All 36 classifier fits and nine 200-epoch DCGAN fits completed. Values are mean ± sample standard deviation across three seeds.', '',
             '| Condition | TB budget | Recall | AUPRC | F1 | Specificity | AUROC |',
             '|---|---|---|---|---|---|---|']
    for row in report['experiments']:
        values = [f'{row["metrics"][k]["mean"]:.4f} ± {row["metrics"][k]["std"]:.4f}' for k in METRICS]
        lines.append(f'| {row["arm"]} | {row["budget"]:.0%} | ' + ' | '.join(values) + ' |')
    lines += ['', 'A: real only; B: repeated real TB; C: conventional TB-only transforms; D: DCGAN synthetic TB.', '',
              '## Paired recall differences', '', '| Comparison | TB budget | Mean difference | Three-seed bootstrap interval |', '|---|---|---|---|']
    for row in comparisons:
        metric = row['metrics']['recall']
        lines.append(f'| D − {row["baseline"]} | {row["budget"]:.0%} | {metric["mean_difference"]:.4f} | {metric["seed_bootstrap_ci95"]} |')
    lines += ['', report['limitations'], '',
              'These are image-level experiments on an automatically screened subset. Missing patient identifiers, selection bias, unvalidated labels and small cohorts limit generalization. Pixel diagnostics do not validate clinical fidelity or privacy. No significance or universal superiority is claimed.', '']
    (output / 'results_table.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Analysis complete: {output}', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.run, args.output)
