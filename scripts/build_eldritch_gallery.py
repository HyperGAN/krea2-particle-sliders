#!/usr/bin/env python3
"""Add Eldritch to the existing gallery, with live status until verified."""
import json
import os
from pathlib import Path

from serve_final_boss import PAGE, ROOT, RUN, build_gallery

ELDRITCH = ROOT / 'outputs/eldritch-krea2-bbox'
GALLERY = RUN / 'gallery/eldritch'

PENDING = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Eldritch · training</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#101115;color:#eeeef2}
body{max-width:1100px;margin:auto;padding:40px 24px}a{color:#8bd6c1}h1{font-size:56px;margin-bottom:12px}
p{color:#bdc4c9;line-height:1.6}progress{width:100%;height:20px;accent-color:#8bd6c1}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}img{width:100%;border-radius:12px}
figcaption{padding:12px 0}small{color:#a7afb8}#preview{margin-top:32px}
</style></head><body>
<nav aria-label="Sliders"><a href="/">Final Boss</a> · <a href="/eldritch/">Eldritch</a></nav>
<h1>Eldritch</h1><p>Chitin armor, clusters of eyes, curling tendrils, and alien silhouettes.</p>
<p>Training on GPU 0 · Krea2 Turbo BBox · rank 16 · 400 updates</p>
<h2 id="status" aria-live="polite">Loading training status…</h2>
<progress id="progress" max="400" value="0"></progress>
<p id="detail">This page updates automatically. Final comparisons and weights will appear after verification.</p>
<section id="preview" hidden><h2>Early preview · 50 updates</h2>
<p>Same prompt and seed. These are intermediate renders; training continues.</p>
<div class="pair"><figure><img id="base" alt="Base model cathedral knight"><figcaption>Base model · strength 0</figcaption></figure>
<figure><img id="edit" alt="Early Eldritch cathedral knight"><figcaption>Eldritch · strength 1</figcaption></figure></div></section>
<script>
async function update(){
 try{
  const response=await fetch('status.json',{cache:'no-store'});if(!response.ok)throw Error('Status unavailable');
  const state=await response.json();const step=state.step||0;document.querySelector('#progress').value=step;
  const labels={loading:'Loading the model',teacher_cache:'Preparing training targets',training:`Training · ${step} / ${state.total_steps} updates`,sampling:'Rendering comparisons',complete:'Training complete · checking results',failed:'Run stopped · investigating'};
  document.querySelector('#status').textContent=labels[state.phase]||state.phase;
  if(state.phase==='sampling')document.querySelector('#detail').textContent=`${state.samples_completed||0} of 14 comparisons rendered. The final pair is 1536px.`;
  if(step>=50){
   for(const [id,scale] of [['base','0'],['edit','1']]){const image=document.getElementById(id);if(!image.getAttribute('src')){image.onload=()=>{document.querySelector('#preview').hidden=false};image.onerror=()=>image.removeAttribute('src');image.src=`preview-scale-${scale}.png`;}}
  }
  const ready=await fetch('ready.json',{cache:'no-store'});if(ready.ok)location.reload();
 }catch(error){document.querySelector('#detail').textContent='Waiting for a fresh status update…';}
}
update();setInterval(update,15000);
</script></body></html>'''


def link(source, target, *, symbolic=False):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    if symbolic:
        target.symlink_to(source)
    else:
        os.link(source, target)


def main():
    GALLERY.mkdir(parents=True, exist_ok=True)
    build_gallery()  # Keep the root page's navigation in sync.
    link(ELDRITCH / 'status.json', GALLERY / 'status.json', symbolic=True)
    for scale in (0, 1):
        link(ELDRITCH / f'preview-step-0050-scale-{scale}.png',
             GALLERY / f'preview-scale-{scale}.png', symbolic=True)
    if not (ELDRITCH / 'verification.json').exists():
        page = PENDING
    else:
        metadata = json.loads((ELDRITCH / 'samples/metadata.json').read_text())
        if len(metadata['samples']) != 14:
            raise ValueError('The gallery requires all 14 verified samples')
        for sample in metadata['samples']:
            link(ELDRITCH / 'samples' / sample['file'], GALLERY / 'samples' / sample['file'])
        for name in ('samples/metadata.json', 'grid.png', 'RESULTS.md'):
            link(ELDRITCH / name, GALLERY / name)
        link(ELDRITCH / 'checkpoint-0400/eldritch-krea2-bbox.safetensors',
             GALLERY / 'eldritch-krea2-bbox.safetensors')
        page = PAGE.replace('Final Boss', 'Eldritch').replace('final-boss-krea2-bbox.safetensors', 'eldritch-krea2-bbox.safetensors')
        page = page.replace('<a href="/">Eldritch</a>', '<a href="/">Final Boss</a>')
        page = page.replace('Ordinary armor becomes something worth a final encounter.',
                            'Organic armor and corrupted silhouettes.')
        page = page.replace('Eldritch: warrior with a crown, enormous spiked armor and a glowing greatsword',
                            'Held-out warrior rendered with Eldritch at strength 1')
        page = page.replace('The fruit remains a fruit bowl, but its bowl and rendering style become more painterly at higher strengths.',
                            'At these strengths the effect is strongest in the armor; extra eyes and tentacles remain weak. The fruit control stays recognizable, with painterly style drift at strength 1.')
        probe_metadata = ELDRITCH / 'strength-probes/metadata.json'
        if probe_metadata.exists() and (ELDRITCH / 'strength-probes/grid.png').exists():
            probes = json.loads(probe_metadata.read_text())['samples']
            if any(sample.get('teacher') for sample in probes):
                for sample in probes:
                    link(ELDRITCH / 'strength-probes' / sample['file'], GALLERY / 'strength-probes' / sample['file'])
                for name in ('grid.png', 'metadata.json'):
                    link(ELDRITCH / 'strength-probes' / name, GALLERY / 'strength-probes' / name)
                extra = '''<details open><summary>Stronger corruption · strengths 1, 1.5 and 2</summary>
<p class="note">Cathedral knight and held-out bridge, using the same neutral prompts and seed 42. Start around 1–1.5: 1.5 adds curling appendages and more alien armor; 2 becomes more fragmented. Extra eyes remain weak.</p>
<a href="strength-probes/grid.png" target="_blank" rel="noopener"><img src="strength-probes/grid.png" alt="Knight and bridge comparisons at strengths 1, 1.5 and 2" style="aspect-ratio:auto"></a>
<p class="note"><a href="strength-probes/knight-positive-teacher-seed-42.png" target="_blank" rel="noopener">Target-prompt reference without the LoRA</a> · <a href="strength-probes/metadata.json">Exact prompts and settings</a></p></details>'''
                page = page.replace('<div class="footer">', extra + '<div class="footer">')
                page = page.replace('1 is the strongest tested setting.', '1 is the top setting in the main comparison; additional checks at 1.5 and 2 are below.')
        page = page.replace('__SAMPLES__', json.dumps(metadata['samples']).replace('<', '\\u003c'))
    temporary = GALLERY / 'index.html.tmp'
    temporary.write_text(page)
    temporary.replace(GALLERY / 'index.html')
    if (ELDRITCH / 'verification.json').exists():
        (GALLERY / 'ready.json').write_text('{"ready":true}\n')
    print(GALLERY)


if __name__ == '__main__':
    main()
