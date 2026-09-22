# Prompting the bbox turbo finetune

Source of truth: [PROMPTING.md on `jimmycarter/krea2-turbo-bbox`](https://huggingface.co/jimmycarter/krea2-turbo-bbox/blob/main/PROMPTING.md). This page is a short local pointer. It is not a copy of the Hub card.

The finetune samples in **8 steps** with CFG off (`guidance_scale=0.0`, `mu=1.15` in diffusers; CFG 1.0 in Comfy). See [README.md](README.md).

About **90%** of its training used a compact grounding DSL: plain text, one element per line, boxes on a **0–1000** grid. The other 10% was ordinary prose, which still works. There is no separate API flag. Both go in the same `prompt` string.

Slider **training** rows in `configs/krea2/` stay bare UNI captions so they match the particle-sliders prompt schema. Use the DSL at inference when you care about layout.

## Coordinates

```text
[x0, y0, x1, y1]
```

X is first. Origin is top-left, and y increases downward. `[0,0,1000,1000]` is the whole canvas, whatever the pixel size. `p[500,0,1000,500]` is the top-right quadrant. A swapped axis still generates an image; it just puts everything on the wrong side of the diagonal.

## Tags

| Tag | Meaning |
|---|---|
| *(first line, no sigil)* | One sentence for the whole image |
| `@` | Style, then medium: `@clean line art, flat colors; Digital illustration` |
| `~` | Background |
| `p` | Panel. Put panels first |
| `o` | Object |
| `t` | Verbatim text, in quotes, then an optional style note |
| `pe` | Person. Optional `pe:<id>` |
| `ac` | Anime character. Optional `ac:<id>` |
| `fc` | Furry / anthro character. Optional `fc:<id>` |

Reuse the same small integer id (`ac:1`, `fc:2`) to mean the same character across panels. The id is local to the prompt. Describe the character again at each box; the id only says "same one".

Keep a speech bubble's box inside its panel. The whole prompt is encoded into a fixed 512-token window, so a long prompt drops the tail first, which is usually your text lines.

## Examples

Prose:

```text
a cat sitting on a windowsill at sunset
```

Grounded four-panel page (x-first boxes):

```text
A four-panel comic in which an anime handywoman and an anthropomorphic tabby cat assemble a flat-pack shelf and fail completely.
@clean line art, flat colors, light cel shading; Digital illustration
~A bare apartment living room with cardboard packaging and loose screws on a pale wood floor.
p[0,0,500,500] Top-left panel: the handywoman holding up the instruction sheet, confident.
p[500,0,1000,500] Top-right panel: the cat batting a single screw off the floor.
p[0,500,500,1000] Bottom-left panel: a lopsided half-built shelf wobbling.
p[500,500,1000,1000] Bottom-right panel: the shelf collapsed, the cat sitting smugly on the wreckage.
ac:1[60,80,340,470] A young woman with short dark hair in blue overalls, holding an instruction sheet, confident grin.
fc:2[560,90,880,470] An anthropomorphic tabby cat in a tool belt, one paw raised mid-swat, innocent expression.
t[80,40,420,120]"SOME ASSEMBLY REQUIRED." black text in a rounded speech bubble
t[540,530,960,610]"IT HAS FIVE LEGS NOW." black text in a rounded speech bubble
```

Same character across a strip:

```text
A three-panel vertical comic strip following one barista through a shift.
@soft line art, flat colors; Digital illustration
~A small café counter with a brass espresso machine.
p[0,0,1000,333] Top panel: the barista takes an order.
p[0,333,1000,666] Middle panel: the barista pulling shots.
p[0,666,1000,1000] Bottom panel: the barista slumped at closing time.
ac:1[80,40,400,320] A barista with short auburn hair, smiling.
ac:1[560,370,900,650] The same barista, sleeves rolled, brow furrowed.
ac:1[100,700,420,980] The same barista slumped over the counter.
```

The Hub doc has the full convention list and a small Python builder. Prefer that file over extending this one.
