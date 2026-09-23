#!/usr/bin/env python3
"""Serve the completed Final Boss gallery on all interfaces."""
import argparse
import html
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/final-boss-krea2-bbox'

PAGE = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Final Boss · Krea2 Turbo BBox</title>
<style>
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif;background:#101115;color:#eeeef2}
*{box-sizing:border-box}body{margin:0}main{max-width:1440px;margin:auto;padding:40px 24px 64px}
a{color:#e9b16b;text-decoration:none}a:hover{text-decoration:underline}
.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#c89f70}
h1{font-size:clamp(36px,5vw,64px);letter-spacing:-.05em;margin:10px 0}p{color:#b7b9c2;line-height:1.6}
.top{display:flex;align-items:center;justify-content:space-between;gap:24px;flex-wrap:wrap}
.button{display:inline-block;background:#e5ac66;color:#191510;padding:13px 20px;border-radius:9px;font-weight:650}
.button:hover{background:#f3c282;text-decoration:none}.chips{display:flex;gap:8px;flex-wrap:wrap;margin:20px 0 30px}
.chips span{font-size:13px;border:1px solid #383940;border-radius:20px;padding:7px 12px;color:#c7c8d0}
h2{font-size:23px;font-weight:600;letter-spacing:-.025em;margin:36px 0 10px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.triptych{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
figure{margin:0;overflow:hidden;border:1px solid #34353b;border-radius:12px;background:#1b1c22}
figcaption{display:flex;justify-content:space-between;align-items:center;padding:12px 15px;font-size:14px}
figcaption span{font-size:12px;color:#aeb0bc}img{width:100%;display:block;aspect-ratio:1;object-fit:contain}
label{color:#c7c8d0;font-size:14px}select{background:#25262d;color:#f0f0f5;border:1px solid #494b56;border-radius:8px;padding:10px 14px;font:inherit;margin:12px 0 20px 8px}
details{margin:18px 0;border:1px solid #34353b;border-radius:10px;padding:15px}summary{cursor:pointer;color:#c7c8d0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.7;color:#c0c4cf;font-size:13px}
.note{font-size:14px;max-width:920px}.footer{border-top:1px solid #34353b;margin-top:32px;padding-top:20px;font-size:13px;color:#999ca7}
@media(max-width:650px){main{padding:26px 14px}.pair{gap:8px}.triptych{grid-template-columns:1fr}figcaption{padding:10px;font-size:12px}figcaption span{font-size:10px}.top .button{width:100%;text-align:center}}
</style></head><body><main>
<nav aria-label="Sliders"><a href="/">Final Boss</a> · <a href="/eldritch/">Eldritch</a></nav>
<div class="top"><div><div class="eyebrow">Krea2 Turbo BBox · trained slider</div><h1>Final Boss</h1>
<p>Ordinary armor becomes something worth a final encounter.</p></div>
<a class="button" href="final-boss-krea2-bbox.safetensors" download>Download LoRA · 77 MB</a></div>
<div class="chips"><span>GPU 0 · RTX A6000</span><span>400 updates</span><span>Rank 16</span><span>8 steps · guidance 0 · mu 1.15</span></div>
<h2>A new scene, the same seed</h2>
<p class="note">1536px · seed 1234 · the bridge prompt was held out from training. Click either image to open the full-resolution file.</p>
<div class="pair"><figure><figcaption>Base model <span>Strength 0</span></figcaption><a href="samples/showcase-scale-0-seed-1234.png" target="_blank" rel="noopener"><img src="samples/showcase-scale-0-seed-1234.png" alt="Base model: steel-armored warrior guarding a volcanic bridge" width="1536" height="1536"></a></figure>
<figure><figcaption>Final Boss <span>Strength 1</span></figcaption><a href="samples/showcase-scale-1-seed-1234.png" target="_blank" rel="noopener"><img src="samples/showcase-scale-1-seed-1234.png" alt="Final Boss: warrior with a crown, enormous spiked armor and a glowing greatsword" width="1536" height="1536"></a></figure></div>
<h2>Explore the strength</h2><p class="note">Saved renders at 768px, all with seed 42. The prompt stays the same across each row.</p>
<label for="scene">Scene</label><select id="scene"><option value="heldout-bridge">Volcanic bridge · held out</option><option value="knight">Cathedral knight</option><option value="cave-warrior">Cave warrior · prose prompt</option><option value="fruit-control">Fruit bowl · control</option></select>
<div class="triptych" id="comparisons"></div>
<details><summary>Exact prompt for this scene</summary><pre id="prompt"></pre></details>
<p class="note">Strength 0.5 gives a milder edit; 1 is the strongest tested setting. The fruit remains a fruit bowl, but its bowl and rendering style become more painterly at higher strengths.</p>
<div class="footer">Base: <a href="https://huggingface.co/jimmycarter/krea2-turbo-bbox" target="_blank" rel="noopener">jimmycarter/krea2-turbo-bbox</a> · epoch-14-step-73184<br>
<a href="grid.png" target="_blank" rel="noopener">Full comparison grid</a> · <a href="samples/metadata.json" target="_blank" rel="noopener">Prompts and sample metadata</a> · <a href="RESULTS.md" target="_blank" rel="noopener">Run notes</a></div>
</main><script>
const samples=__SAMPLES__;
const scene=document.querySelector('#scene');
function render(){
 const chosen=samples.filter(s=>s.file.startsWith(scene.value+'-')&&s.resolution===768).sort((a,b)=>a.scale-b.scale);
 const container=document.querySelector('#comparisons');container.replaceChildren();
 for(const sample of chosen){
  const figure=document.createElement('figure');const caption=document.createElement('figcaption');
  caption.textContent=sample.scale===0?'Base model':'Final Boss';const tag=document.createElement('span');tag.textContent='Strength '+sample.scale;caption.append(tag);
  const link=document.createElement('a');link.href='samples/'+sample.file;link.target='_blank';link.rel='noopener';
  const image=document.createElement('img');image.src=link.href;image.alt=scene.options[scene.selectedIndex].text+' at strength '+sample.scale;image.width=768;image.height=768;image.loading='lazy';
  link.append(image);figure.append(caption,link);container.append(figure);
 }
 document.querySelector('#prompt').textContent=chosen[0].prompt;
}
scene.addEventListener('change',render);render();
</script></body></html>'''


def build_gallery():
    gallery = RUN / 'gallery'
    (gallery / 'samples').mkdir(parents=True, exist_ok=True)
    metadata = json.loads((RUN / 'samples/metadata.json').read_text())
    links = [(RUN / 'samples' / s['file'], gallery / 'samples' / s['file']) for s in metadata['samples']]
    links += [(RUN / 'samples/metadata.json', gallery / 'samples/metadata.json'),
              (RUN / 'grid.png', gallery / 'grid.png'),
              (RUN / 'RESULTS.md', gallery / 'RESULTS.md'),
              (RUN / 'checkpoint-0400/final-boss-krea2-bbox.safetensors', gallery / 'final-boss-krea2-bbox.safetensors')]
    for source, target in links:
        if not target.exists():
            os.link(source, target)
    data = json.dumps(metadata['samples']).replace('<', '\\u003c')
    page = PAGE.replace('__SAMPLES__', data)
    release = ROOT / 'artifacts/release'
    photo = release / 'samples/final-boss/street-photo'
    if all((photo / f'{kind}.png').exists() for kind in ('original', 'distill', 'off')):
        cards = []
        (gallery / 'street-photo').mkdir(exist_ok=True)
        for kind in ('original', 'distill', 'off'):
            for suffix in ('.png', '.json'):
                target = gallery / 'street-photo' / (kind + suffix)
                if not target.exists():
                    os.link(photo / (kind + suffix), target)
            strength = 0 if kind == 'off' else 1
            cards.append(f'<figure><figcaption>{kind.title()} <span>Strength {strength}</span></figcaption>'
                         f'<a href="street-photo/{kind}.png" target="_blank" rel="noopener">'
                         f'<img src="street-photo/{kind}.png" alt="Rainy Tokyo street photograph: {kind}" '
                         'width="768" height="768"></a></figure>')
        record = json.loads((photo / 'original.json').read_text())
        featured = '<h2>A photograph, the same seed</h2><p class="note">AI-generated rainy Tokyo street photo · '
        featured += '768px · seed 4242. Original and rank-8 Distill use calibrated strength 1. '
        featured += 'The effect is subtler here than in the armored examples.</p>'
        featured += '<div class="triptych">' + ''.join(cards) + '</div>'
        featured += '<details><summary>Exact photographic prompt</summary><pre>' + html.escape(record['prompt']) + '</pre></details>'
        start, end = page.index('<h2>A new scene'), page.index('<h2>Explore the strength')
        page = page[:start] + featured + page[end:]
        page = page.replace('Ordinary armor becomes something worth a final encounter.',
                            'The Final Boss slider on a regular street photograph.')
        page = page.replace('<a class="button" href="final-boss-krea2-bbox.safetensors" download>Download LoRA · 77 MB</a>',
                            '<a class="button" href="https://huggingface.co/ntc-ai/krea2-particle-sliders">Original + Distill downloads</a>')
    (gallery / 'index.html').write_text(page)
    return gallery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8780)
    args = parser.parse_args()
    gallery = build_gallery()
    handler = partial(SimpleHTTPRequestHandler, directory=str(gallery))
    server = ThreadingHTTPServer(('0.0.0.0', args.port), handler)
    print(f'Final Boss gallery listening on 0.0.0.0:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
