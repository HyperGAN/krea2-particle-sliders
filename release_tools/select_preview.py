#!/usr/bin/env python3
"""Render photo candidates, wait for a visual selection, then export its triptych."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw, ImageFont
import torch
from release_tools.distill import Runtime
from release_tools.lora import digest, write_json

CANDIDATES = [
    ('motorcyclist', 'Motorcycle racer',
     'A full-body editorial photograph of a motorcycle racer standing in a concrete paddock.\n'
     '@professional sports photography, realistic materials, overcast daylight; Photograph\n'
     '~A quiet race circuit pit lane, concrete garages, subtle wet reflections.\n'
     'pe:1[210,80,790,960] An adult motorcycle racer in a simple black leather racing suit, '
     'black gloves and boots, wearing a plain full-face helmet, standing with both arms at the sides.'),
    ('boxer', 'Boxing gym',
     'A full-body sports photograph of a boxer in an old boxing gym.\n'
     '@35mm editorial sports photography, realistic skin, directional window light; Photograph\n'
     '~A worn boxing ring with red ropes and large industrial windows in the background.\n'
     'pe:1[200,70,800,960] An adult male boxer with short hair, athletic build, plain black boxing '
     'shorts and red boxing gloves, standing squarely with his gloves lowered.'),
    ('goalie', 'Hockey goalie',
     'A full-body sports photograph of an ice hockey goalie standing on an empty rink.\n'
     '@professional sports photography, realistic materials and proportions; Photograph\n'
     '~An indoor ice rink with cold overhead lighting and empty spectator seats.\n'
     'pe:1[200,80,800,960] An adult hockey goalie wearing a plain black jersey, white leg pads, '
     'a simple white goalie mask and padded gloves, holding a goalie stick, relaxed stance.'),
    ('reenactor', 'Historical reenactor',
     'A full-body documentary photograph of a historical reenactor in a castle courtyard.\n'
     '@35mm documentary photography, realistic metal and fabric, natural afternoon light; Photograph\n'
     '~A real old stone courtyard with ivy, a wooden doorway and worn cobblestones.\n'
     'pe:1[220,80,780,960] An adult reenactor wearing ordinary polished steel plate armor '
     'and a simple open-faced helmet, holding a plain longsword point down, standing casually.'),
    ('wolf', 'Wild wolf',
     'A wildlife photograph of a gray wolf standing in a snowy pine forest.\n'
     '@wildlife photography, realistic fur, low camera angle, soft winter daylight; Photograph\n'
     '~Fresh snow, tall dark pines and light mist, distant forest softly out of focus.\n'
     'o[150,180,850,850] A single gray wolf with thick natural fur, standing on all four legs, '
     'looking toward the camera, full animal visible.'),
    ('raven', 'Raven in mist',
     'A wildlife photograph of a black raven perched on a weathered wooden fence post.\n'
     '@wildlife photography, realistic feather detail, shallow depth of field; Photograph\n'
     '~A damp meadow at dawn with soft gray mist and blurred bare trees.\n'
     'o[200,110,800,880] One ordinary black raven perched upright, wings folded, '
     'head turned toward the camera, entire bird and feet visible.'),
    ('fashion', 'Fashion portrait',
     'A full-body fashion photograph of a woman on a city rooftop at dusk.\n'
     '@editorial fashion photography, realistic skin and fabric, natural proportions; Photograph\n'
     '~A concrete rooftop with a softly blurred city skyline and cool evening light.\n'
     'pe:1[220,80,780,960] An adult woman with short dark hair wearing a simple long black '
     'tailored coat, a gray turtleneck and black boots, standing confidently with her hands at her sides.'),
    ('robot', 'Workshop robot',
     'A full-body photograph of a humanoid robot in an engineering workshop.\n'
     '@industrial editorial photography, realistic metal and plastic, soft window light; Photograph\n'
     '~A real workshop with concrete floors, workbenches and neatly arranged tools.\n'
     'o[200,80,800,960] A life-size humanoid research robot with a simple rounded head, '
     'plain silver panels, exposed black joints and two ordinary arms, standing upright facing the camera.'),
]


def sheet(work, case, label):
    canvas = Image.new('RGB', (1024, 570), '#12151b')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('DejaVuSans-Bold.ttf', 22)
    for i, kind in enumerate(('original', 'off')):
        draw.text((i * 512 + 12, 12), f'{label} | {kind.upper()}', font=font, fill='white')
        with Image.open(work / case / (kind + '.png')) as im:
            canvas.paste(im.resize((512, 512), Image.Resampling.LANCZOS), (i * 512, 58))
    canvas.save(work / case / 'comparison.jpg', quality=94, subsampling=0)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=ROOT / 'artifacts/release')
    parser.add_argument('--work', type=Path, default=ROOT / 'outputs/preview-selection-v2')
    parser.add_argument('--seed', type=int, default=2026)
    options = parser.parse_args()
    folder, work = options.folder.resolve(), options.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((folder / 'catalog.json').read_text())
    entry = next(e for e in catalog['sliders'] if e['id'] == 'final-boss')
    run = ROOT / 'outputs/final-boss-krea2-bbox'
    args = argparse.Namespace(**json.loads((run / 'run.json').read_text()))
    args.resume = None
    args.load_te_lora = str(run / 'checkpoint-0400')
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    write_json(work / 'status.json', dict(phase='loading'))
    runtime = Runtime(args, [c[2] for c in CANDIDATES], work)
    for kind in ('original', 'distill'):
        runtime.load(kind, folder / entry[kind]['files']['native'])

    def render(case, prompt, kind):
        adapter = 'distill' if kind == 'distill' else 'original'
        runtime.active = adapter
        strength = 0. if kind == 'off' else 1.
        export = entry[adapter]
        image = runtime.generate(prompt, strength, options.seed, 768)
        destination = work / case / (kind + '.png')
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)
        record = dict(prompt=prompt, seed=options.seed, width=768, height=768, steps=8,
                      guidance=0., mu=1.15, format=kind, strength=strength, alpha=export['alpha'],
                      rank=export['rank'], adapter=export['files']['native'],
                      adapter_sha256=digest(folder / export['files']['native']),
                      model_id=args.model_id, model_revision=args.revision,
                      split='visually selected preview; not a benchmark')
        write_json(destination.with_suffix('.json'), record)
        print('Rendered', case, kind, flush=True)

    for index, (case, label, prompt) in enumerate(CANDIDATES):
        for kind in ('original', 'off'):
            render(case, prompt, kind)
        sheet(work, case, label)
        write_json(work / 'status.json', dict(phase='candidates', completed=index + 1,
                                            total=len(CANDIDATES), last=case))
    write_json(work / 'status.json', dict(phase='awaiting-visual-selection'))
    deadline = time.monotonic() + 1200
    selection = work / 'selection.json'
    while not selection.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('No visual selection; candidates are saved and GPU is released')
        time.sleep(2)
    review = json.loads(selection.read_text())
    case, label, prompt = next(c for c in CANDIDATES if c[0] == review['selected'])
    render(case, prompt, 'distill')
    case_id = f'photo-{case}-seed-{options.seed}'
    target = folder / 'samples/final-boss' / case_id
    if target.exists():
        raise FileExistsError(f'Choose a new versioned preview path: {target}')
    target.mkdir(parents=True)
    samples = []
    for kind in ('original', 'distill', 'off'):
        for suffix in ('.png', '.json'):
            shutil.copyfile(work / case / (kind + suffix), target / (kind + suffix))
        path = str((target / (kind + '.png')).relative_to(folder))
        samples.append(dict(image=path, metadata=str(Path(path).with_suffix('.json')),
                            format=kind, strength=0. if kind == 'off' else 1.))
    entry['comparisons'].append(dict(case=case_id, label=label, samples=samples))
    entry['featured_case'] = case_id
    entry['preview_note'] = review['caption']
    write_json(folder / 'catalog.json', catalog)
    evidence = folder / 'evidence/preview-selection-v2'
    evidence.mkdir(parents=True, exist_ok=True)
    for candidate, _, _ in CANDIDATES:
        shutil.copyfile(work / candidate / 'comparison.jpg', evidence / (candidate + '.jpg'))
    write_json(evidence / 'review.json', dict(review, seed=options.seed,
        candidates=[dict(id=c[0], label=c[1], prompt=c[2]) for c in CANDIDATES],
        method='Visual curation for the featured example; same prompt and seed at strengths 1 and 0'))
    write_json(work / 'status.json', dict(phase='complete', featured_case=case_id))


if __name__ == '__main__':
    main()
