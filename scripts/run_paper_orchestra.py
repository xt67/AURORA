"""Preflight or explicitly execute the pinned local Paper Orchestra CLI."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PIN = 'd9bae997e793a015ed38bb1605be9eb58c00284f'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--writer-model', default='gemini-3.1-pro-preview')
    p.add_argument('--reflection-model', default='gemini-3.1-pro-preview')
    p.add_argument('--execute', action='store_true', help='Actually make external API calls')
    a = p.parse_args()
    repo = ROOT / 'outputs/tools/paper-orchestra'
    issues = []
    if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != PIN:
        issues.append('Upstream revision differs from reviewed pin')
    for command in ['pdflatex', 'bibtex']:
        if not shutil.which(command):
            issues.append(f'Missing {command} on PATH')
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv(repo / '.env')
        # Upstream citation parsing also uses Gemini even with OpenAI writing models.
        if not (os.getenv('GEMINI_API_KEY') or (os.getenv('VERTEX_AI_PROJECT') and os.getenv('VERTEX_AI_LOCATION'))):
            issues.append('Missing Gemini/Vertex credential (also used by citation parser)')
        if any('gpt' in m.lower() or m.lower().startswith(('o1', 'o3')) for m in [a.writer_model, a.reflection_model]) and not os.getenv('OPENAI_API_KEY'):
            issues.append('Missing OPENAI_API_KEY for selected writing model')
    except ImportError:
        issues.append('Python dependencies are not installed; use the isolated environment')
    for name in ['idea_sparse.md', 'experimental_log.md', 'input_hashes.json']:
        if not (a.inputs / name).is_file():
            issues.append(f'Missing input {name}')
    if (a.inputs / 'input_hashes.json').is_file():
        for name, expected in json.loads((a.inputs / 'input_hashes.json').read_text()).items():
            source = (a.inputs / name).resolve()
            if not source.is_relative_to(a.inputs.resolve()):
                issues.append('Input manifest contains an escaping path')
            elif not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                issues.append(f'Input hash mismatch: {name}')
    if a.output.exists():
        issues.append('Output already exists; choose a new directory')
    if issues:
        print('\n'.join(issues))
        raise SystemExit(2)
    if not a.execute:
        print('Preflight passed. No API request made; --execute is required.')
        return
    command = [sys.executable, str(repo / 'paper_writing_cli.py'), '--raw_materials_dir', str(a.inputs.resolve()),
               '--latex_template_dir', str(ROOT / 'paper/orchestra/template'), '--output_dir', str(a.output.resolve()),
               '--writer_model_name', a.writer_model, '--reflection_model_name', a.reflection_model,
               '--research_cutoff', '2026-11', '--use_plotting', 'false']
    subprocess.run(command, cwd=repo, check=True)
    if not (a.output / 'final_paper.pdf').is_file():
        raise SystemExit('Pipeline returned without final_paper.pdf; inspect logs, do not claim success')
    print('Generated candidate PDF. Scientific and citation review still required.')


if __name__ == '__main__':
    main()
