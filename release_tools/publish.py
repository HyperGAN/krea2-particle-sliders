#!/usr/bin/env python3
"""Validate a release, publish it to the Hub, and verify remote hashes/readbacks."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from huggingface_hub import HfApi, ModelCard, hf_hub_download
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
import torch
from release_tools.lora import comfy_name, digest, read_factors, write_json

REPO = 'ntc-ai/krea2-particle-sliders'


def release_files(folder):
    return [p for p in sorted(folder.rglob('*')) if p.is_file()
            and p.relative_to(folder).parts[0] not in ('.cache', '.git', '.gitattributes')]


def validate(folder):
    torch.set_num_threads(4)
    catalog = json.loads((folder / 'catalog.json').read_text())
    assert [e['id'] for e in catalog['sliders']] == ['final-boss', 'eldritch']
    expected_weights, expected_images, expected_comfy = set(), set(), set()
    for entry in catalog['sliders']:
        name = entry['id']
        source = ROOT / f'outputs/{name}-krea2-bbox/checkpoint-0400/{name}-krea2-bbox.safetensors'
        assert digest(source) == entry['original']['parent_sha256']
        source_factors = read_factors(source)
        assert entry['distill']['parent_sha256'] == entry['original']['sha256']['native']
        assert entry['recommended_strength'] == 1
        for kind, rank in [('original', 16), ('distill', 8)]:
            export = entry[kind]
            assert export['rank'] == rank and export['projections'] == 128
            assert math.isfinite(export['alpha']) and export['alpha'] > 0
            assert export['recommended_strength'] == 1
            for fmt, filename in export['files'].items():
                assert digest(folder / filename) == export['sha256'][fmt], filename
                expected_weights.add(filename)
            native = folder / export['files']['native']
            factors = read_factors(native)
            peft_path = folder / export['files']['peft']
            peft = read_factors(peft_path)
            comfy = load_file(str(folder / export['files']['comfyui']))
            expected_comfy.add(export['files']['comfyui'])
            config = json.loads(peft_path.with_name('adapter_config.json').read_text())
            assert config['r'] == rank and config['lora_alpha'] == export['alpha']
            with safe_open(native, framework='pt') as handle:
                metadata = handle.metadata()
            config = json.loads(metadata['lora_adapter_metadata'])
            assert config['transformer.r'] == rank and config['transformer.lora_alpha'] == export['alpha']
            assert metadata['parent_sha256'] == export['parent_sha256']
            assert factors.keys() == source_factors.keys() == peft.keys() and len(factors) == 128
            assert len(comfy) == 128 * 3
            for module, (down, up) in factors.items():
                assert down.shape[0] == up.shape[1] == rank
                assert torch.isfinite(down).all() and torch.isfinite(up).all()
                prefix = 'diffusion_model.' + comfy_name(module)
                assert float(comfy[prefix + '.alpha']) == export['alpha']
                for a, b, c in [(down, peft[module][0], comfy[prefix + '.lora_down.weight']),
                                (up, peft[module][1], comfy[prefix + '.lora_up.weight'])]:
                    assert torch.equal(a, b) and torch.equal(a, c), module
                if kind == 'original':
                    assert all(torch.equal(a, b) for a, b in zip((down, up), source_factors[module]))
        cases = {case['case'] for case in entry['comparisons']}
        assert {'knight', 'heldout-bridge', 'fruit-control'} <= cases <= {'knight', 'heldout-bridge', 'fruit-control', 'street-photo'}
        for case in entry['comparisons']:
            assert [s['format'] for s in case['samples']] == ['original', 'distill', 'off']
            records = []
            for sample in case['samples']:
                expected_images.add(sample['image'])
                with Image.open(folder / sample['image']) as image:
                    image.load()
                    assert image.size == (768, 768)
                record = json.loads((folder / sample['metadata']).read_text())
                records.append(record)
                kind = sample['format']
                export = entry['distill' if kind == 'distill' else 'original']
                assert record['format'] == kind
                assert record['strength'] == sample['strength'] == (0 if kind == 'off' else 1)
                assert record['alpha'] == export['alpha'] and record['rank'] == export['rank']
                assert record['adapter'] == export['files']['native']
                assert record['adapter_sha256'] == export['sha256']['native']
                assert record['model_revision'] == catalog['revision']
            for key in ('prompt', 'seed', 'width', 'height', 'steps', 'guidance', 'mu'):
                assert len({r[key] for r in records}) == 1, (name, key)
            assert records[0]['steps'] == 8 and records[0]['guidance'] == 0 and records[0]['mu'] == 1.15
            with Image.open(folder / case['asset']) as image:
                image.load()
                assert image.size == (1152, 432)
        report = json.loads((folder / f'evidence/{name}/distillation.json').read_text())
        assert report['student_rank'] == 8 and report['training_records'] == 60 and report['development_records'] == 16
        assert len(report['projections']) == 128
        assert report['selected'] == min(report['alpha_candidates'], key=lambda c: c['relative_full_edit_mse'])
        assert report['distilled_alpha'] == entry['distill']['alpha']
        assert all(math.isfinite(c[k]) for c in report['alpha_candidates']
                   for k in ('relative_full_edit_mse', 'edit_cosine', 'student_to_teacher_rms'))
        replays = json.loads((folder / f'evidence/{name}/alpha-replay.json').read_text())['checks']
        off = [r for r in replays if r['format'] == 'off']
        assert len(off) == 3 and all(r['pixel_identical'] and r['pixel_mae'] == 0 for r in off)
    assert expected_weights == {str(p.relative_to(folder)) for p in folder.rglob('*.safetensors')}
    assert len(expected_weights) == 12
    assert len(expected_images) == 3 * sum(len(e['comparisons']) for e in catalog['sliders'])
    assert expected_images == {str(p.relative_to(folder)) for p in (folder / 'samples').rglob('*.png')}
    comfy = json.loads((folder / 'validation/comfyui.json').read_text())
    assert comfy['passed'] and {r['file'] for r in comfy['files']} == expected_comfy
    for result in comfy['files']:
        path = folder / result['file']
        assert result['sha256'] == digest(path)
        assert result['native_sha256'] == digest(path.parent.parent / 'native' / path.name)
    assert all(r['patches'] == 128 and r['max_relative_matrix_error'] < 1e-6
               and all(r[k] for k in ('native_comfy_matrices_equal', 'custom_node_applies_weights',
                                      'native_node_alpha_equal', 'clone_isolation', 'zero_bypass')) for r in comfy['files'])
    card = (folder / 'README.md').read_text()
    assert card.index('## Samples') < card.index('## Downloads') < card.index('## Distillation and alpha')
    for name in re.findall(r'https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/([^)?\s]+)', card):
        # These three are generated below from the committed source tree.
        if name not in ('source.zip', 'source-provenance.json', 'release-manifest.json'):
            assert (folder / name).is_file(), ('Broken release link', name)
    for path in release_files(folder):
        assert path.suffix not in ('.log', '.pt', '.pyc'), ('Unexpected runtime artifact', path)
        assert not path.is_symlink(), ('Release files must be self-contained', path)
    return catalog


def verify_remote(path, remote):
    assert remote.size == path.stat().st_size, ('Remote size mismatch', path)
    if remote.lfs:
        assert remote.lfs.sha256 == digest(path), ('Remote SHA256 mismatch', path)
    else:
        data = path.read_bytes()
        expected = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        assert remote.blob_id == expected, ('Remote Git blob mismatch', path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=ROOT / 'artifacts/release')
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--expected-parent', help='Inspected Hub commit required for an additive update')
    options = parser.parse_args()
    folder = options.folder.resolve()
    catalog = validate(folder)
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(), 'Commit source before publishing'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    files = [p for p in subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0') if p]
    assert all(not name.startswith(('artifacts/', 'outputs/', 'models/', 'samples/')) for name in files)
    write_json(folder / 'source-provenance.json', dict(
        repository='https://github.com/HyperGAN/krea2-particle-sliders', commit=commit,
        files={name: dict(bytes=(ROOT / name).stat().st_size, sha256=digest(ROOT / name)) for name in files}))
    with zipfile.ZipFile(folder / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(ROOT / name, 'krea2-particle-sliders/' + name)
    manifest = dict(source_commit=commit, files={str(path.relative_to(folder)):
        dict(bytes=path.stat().st_size, sha256=digest(path))
        for path in release_files(folder) if path.name != 'release-manifest.json'})
    write_json(folder / 'release-manifest.json', manifest)
    ModelCard.load(folder / 'README.md').validate()
    print(f"Validated {len(manifest['files'])} release files; source {commit}", flush=True)
    if not options.publish:
        return
    api = HfApi()
    api.create_repo(REPO, repo_type='model', private=False, exist_ok=True)
    before = api.model_info(REPO, files_metadata=True)
    assert not before.private, 'Expected a public release repository'
    existing = {f.rfilename: f for f in before.siblings}
    if set(existing) - {'.gitattributes'}:
        assert options.expected_parent == before.sha, 'Supply the inspected release parent before updating'
        assert set(existing) - {'.gitattributes'} <= {str(p.relative_to(folder)) for p in release_files(folder)}
        for name, remote in existing.items():
            if name.startswith(('weights/', 'distilled/', 'samples/', 'assets/', 'evidence/')):
                verify_remote(folder / name, remote)
    elif options.expected_parent:
        assert options.expected_parent == before.sha, 'Release parent changed'
    result = api.upload_folder(repo_id=REPO, repo_type='model', folder_path=folder,
        parent_commit=before.sha, ignore_patterns=['.cache/**', '.git/**', '.gitattributes'],
        commit_message='Publish Final Boss and Eldritch with rank-8 distills, calibrated alpha and matched samples')
    revision = result.oid
    remote = {f.rfilename: f for f in api.model_info(REPO, revision=revision, files_metadata=True).siblings}
    paths = release_files(folder)
    for path in paths:
        verify_remote(path, remote[str(path.relative_to(folder))])
    readbacks = ['README.md'] + [filename for e in catalog['sliders'] for kind in ('original', 'distill')
                                for filename in e[kind]['files'].values()]
    for name in readbacks:
        downloaded = Path(hf_hub_download(REPO, name, revision=revision))
        assert digest(downloaded) == digest(folder / name), ('Downloaded SHA256 mismatch', name)
    report = dict(repo=REPO, commit=revision, source_commit=commit, previous_commit=before.sha,
                  verified_files=len(paths), downloaded_readbacks=len(readbacks),
                  originals_unchanged=True, existing_release_artifacts_preserved=True)
    write_json(folder.parent / 'publication.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
