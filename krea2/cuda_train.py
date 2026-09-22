"""CUDA UNI LoRA: neutral captions learn the frozen positive velocity.

Teacher targets share the student's exact latent state and timestep. States
come from both neutral and positive frozen 8-step trajectories. Scale zero
disables the adapter exactly; no negative teacher or CFG is used.
"""
from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer
from PIL import Image, ImageDraw

from krea2.grounding import check_token_budget, validate_grounding
from krea2.live import load_krea2_bbox_pipeline

PRESERVE = (
    'a quiet mountain lake at sunrise',
    'a red bicycle parked beside a brick wall',
    'a domestic cat asleep on a windowsill',
)
HELDOUT = ('A lone warrior guarding a volcanic bridge.\n'
           '@dramatic lighting, detailed fantasy game art; Digital illustration\n'
           '~A stone bridge above a glowing lava river.\n'
           'pe:1[220,80,780,950] An adult warrior in steel armor, holding a sword, standing guard.')


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


class CudaUNI:
    def __init__(self, args, texts, out):
        self.args, self.out = args, out
        self.device = torch.device(f'cuda:{args.device}')
        self.dtype = torch.bfloat16
        self.pipe = load_krea2_bbox_pipeline(
            model_id=args.model_id, subfolder=args.transformer_subfolder,
            skeleton=args.skeleton_model, transformer=args.transformer,
            allow_hub=args.allow_hub, revision=args.revision,
            skeleton_revision=args.skeleton_revision, device=str(self.device))
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.transformer.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False).eval()
        self.pipe.vae.requires_grad_(False).eval()
        self.pipe.vae.enable_tiling()
        self.pipe.text_encoder.to(self.device)
        self.embeddings = {}
        token_counts = {}
        for index, text in enumerate(dict.fromkeys(texts)):
            validate_grounding(text)
            token_counts[text] = check_token_budget(
                self.pipe.tokenizer, text, self.pipe.prompt_template_encode_prefix)
            with torch.no_grad():
                embeds, mask = self.pipe.get_text_hidden_states(text, 512, self.device)
            self.embeddings[text] = (embeds.cpu(), mask.cpu())
            del embeds, mask
            print(f'Encoded prompt {index + 1}: {token_counts[text]}/507 tokens', flush=True)
        write_json(out / 'prompt_token_counts.json', token_counts)
        # All prompts are cached; release the 4B encoder before backpropagation.
        self.pipe.text_encoder.to('cpu')
        self.pipe.text_encoder = None
        gc.collect()
        torch.cuda.empty_cache()
        if args.resume or args.load_te_lora:
            self.model = PeftModel.from_pretrained(
                self.pipe.transformer, args.resume or args.load_te_lora,
                is_trainable=not bool(args.load_te_lora))
        else:
            self.model = get_peft_model(self.pipe.transformer, LoraConfig(
                r=args.rank, lora_alpha=args.rank,
                target_modules=['to_q', 'to_k', 'to_v', 'to_out.0']))
        self.pipe.transformer = self.model
        from krea2.attention import install_large_image_attention
        install_large_image_attention(self.model)
        self.model.get_base_model().enable_gradient_checkpointing()
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        print(f'Trainable LoRA parameters: {sum(p.numel() for p in self.params):,}', flush=True)
        self.position_ids = {}

    @contextlib.contextmanager
    def scale(self, value):
        if value == 0:
            with self.model.disable_adapter():
                yield
        else:
            for module in self.model.modules():
                if isinstance(module, BaseTunerLayer):
                    module.set_scale('default', value)
            try:
                yield
            finally:
                for module in self.model.modules():
                    if isinstance(module, BaseTunerLayer):
                        module.set_scale('default', 1.0)

    def embeds(self, text):
        return tuple(t.to(self.device) for t in self.embeddings[text])

    def forward(self, text, z, timestep):
        embeds, mask = self.embeds(text)
        grid = self.args.resolution // (self.pipe.vae_scale_factor * self.pipe.patch_size)
        key = (embeds.shape[1], grid)
        if key not in self.position_ids:
            self.position_ids[key] = self.pipe.prepare_position_ids(*key, grid, self.device)
        return self.model(
            hidden_states=z.to(self.device, self.dtype),
            encoder_hidden_states=embeds,
            encoder_attention_mask=mask,
            timestep=(timestep.to(self.device) / self.pipe.scheduler.config.num_train_timesteps)
                .expand(z.shape[0]).to(self.dtype),
            position_ids=self.position_ids[key], return_dict=False)[0].float()

    @torch.no_grad()
    def trajectory(self, neutral, positive, seed, positive_rollout=False):
        scheduler = self.pipe.scheduler.from_config(self.pipe.scheduler.config)
        scheduler.set_timesteps(
            sigmas=np.linspace(1., 1 / self.args.sample_steps, self.args.sample_steps),
            device=self.device, mu=self.args.mu)
        scheduler.set_begin_index(0)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        z = self.pipe.prepare_latents(
            1, self.model.config.in_channels // self.pipe.patch_size ** 2,
            self.args.resolution, self.args.resolution, self.dtype, self.device, generator)
        records = []
        with self.scale(0):
            for timestep in scheduler.timesteps:
                v0 = self.forward(neutral, z, timestep)
                vp = self.forward(positive, z, timestep) if positive != neutral else v0
                records.append(dict(prompt=neutral, z=z.cpu(), t=timestep.cpu(),
                                    baseline=v0.cpu(), teacher=vp.cpu()))
                velocity = vp if positive_rollout else v0
                z = scheduler.step(velocity.to(self.dtype), timestep, z, return_dict=False)[0].to(self.dtype)
        return records

    @torch.no_grad()
    def generate(self, prompt, scale, seed, resolution):
        self.model.eval()
        embeds, mask = self.embeds(prompt)
        with self.scale(scale):
            # Keep the VAE off GPU until the transformer has finished sampling.
            z = self.pipe(
                prompt_embeds=embeds, prompt_embeds_mask=mask,
                height=resolution, width=resolution,
                num_inference_steps=self.args.sample_steps,
                guidance_scale=0.0, output_type='latent',
                generator=torch.Generator(device=self.device).manual_seed(seed)).images
        self.pipe.vae.to(self.device)
        try:
            latents = self.pipe._unpack_latents(z, resolution, resolution).to(self.pipe.vae.dtype)
            mean = torch.tensor(self.pipe.vae.config.latents_mean, device=self.device,
                                dtype=latents.dtype).view(1, -1, 1, 1, 1)
            std = torch.tensor(self.pipe.vae.config.latents_std, device=self.device,
                               dtype=latents.dtype).view(1, -1, 1, 1, 1)
            pixels = self.pipe.vae.decode(latents * std + mean, return_dict=False)[0][:, :, 0]
            result = self.pipe.image_processor.postprocess(pixels, output_type='pil')[0]
        finally:
            self.pipe.vae.to('cpu')
            torch.cuda.empty_cache()
        return result

    def checkpoint(self, step, optimizer, rng):
        dest = self.out / f'checkpoint-{step:04d}'
        dest.mkdir(exist_ok=True)
        self.model.save_pretrained(dest)
        state = get_peft_model_state_dict(self.model)
        state = {k.removeprefix('base_model.model.'): v for k, v in state.items()}
        self.pipe.save_lora_weights(dest, transformer_lora_layers=state,
                                    weight_name=f'{self.args.name}.safetensors')
        torch.save(dict(step=step, optimizer=optimizer.state_dict(), rng=rng.getstate()),
                   dest / 'training_state.pt')
        write_json(self.out / 'latest_checkpoint.json', dict(step=step, path=str(dest.resolve())))
        print(f'Saved {dest}', flush=True)
        return dest


def make_grid(images, scales, dest, title):
    tile = 384
    grid = Image.new('RGB', (len(scales) * tile, len(images) * (tile + 32)), 'white')
    draw = ImageDraw.Draw(grid)
    for row, variants in enumerate(images):
        for col, img in enumerate(variants):
            grid.paste(img.resize((tile, tile)), (col * tile, row * (tile + 32) + 32))
            draw.text((col * tile + 8, row * (tile + 32) + 8),
                      f'{title[row]} | scale {scales[col]}', fill='black')
    grid.save(dest)


def train_cuda(args, prompts, meta):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for --live')
    if args.sample_guidance != 0 or args.sample_steps != 8 or args.mu != 1.15:
        raise ValueError('Live turbo-bbox training uses exactly 8 steps, CFG 0, mu=1.15')
    if args.resolution % 16 or args.steps < 1 or args.cache_seeds < 1 or args.save_every < 1:
        raise ValueError('Invalid resolution, step count, cache seeds or save interval')
    out = Path(args.save_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'train.jsonl').exists() and not (args.resume or args.load_te_lora):
        raise ValueError(f'{out} already has training; use --resume or a new directory')
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision('high')
    rng = random.Random(args.seed)
    control = args.control_prompt or meta.control_prompt
    verify = [prompts[0].neutral, prompts[4].neutral if len(prompts) > 4 else prompts[-1].neutral,
              HELDOUT, control]
    verify_names = ['knight', 'cave-warrior', 'heldout-bridge', 'fruit-control']
    texts = [text for p in prompts for text in (p.target, p.neutral, p.positive, p.negative) if text]
    texts += list(PRESERVE) + verify
    prompt_hash = hashlib.sha256(json.dumps(texts).encode()).hexdigest()
    manifest = dict(
        **vars(args), prompts_sha256=prompt_hash,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        gpu=torch.cuda.get_device_name(args.device),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        versions={n: importlib.metadata.version(n) for n in
                  ('torch', 'diffusers', 'transformers', 'peft', 'safetensors')},
        objective='adapter(neutral, scale=1) -> frozen positive velocity on shared states',
        teacher_trajectories='alternating neutral and positive, frozen, 8 steps, mu=1.15',
        preservation=list(PRESERVE), fruit_is_verify_only=True,
        scale_zero='adapter disabled exactly', minus_teacher=False)
    root = Path(__file__).resolve().parents[1]
    snapshot = out / 'source'
    source_paths = list((root / 'krea2').glob('*.py')) + [
        root / 'scripts/train_krea2.py', *sorted((root / 'scripts').glob('train_*_gpu0.sh')),
        root / args.prompts_file, root / args.config_file]
    source_hashes = {}
    for source in source_paths:
        if not source.is_file():
            continue
        relative = source.relative_to(root) if source.is_relative_to(root) else Path(source.name)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest['source_files_sha256'] = source_hashes
    write_json(out / 'run.json', manifest)
    write_json(out / 'prompts.json', [vars(p) for p in prompts])
    status = dict(phase='loading', step=0, total_steps=args.steps)
    write_json(out / 'status.json', status)
    started = time.time()
    try:
        backend = CudaUNI(args, texts, out)
        if args.load_te_lora:
            start_step = args.steps
        else:
            cache_path = out / 'teacher_cache.pt'
            status['phase'] = 'teacher_cache'
            write_json(out / 'status.json', status)
            if cache_path.exists():
                cache = torch.load(cache_path, map_location='cpu', weights_only=False)
                expected = dict(prompts_sha256=prompt_hash, resolution=args.resolution,
                                revision=args.revision, seeds=args.cache_seeds, seed=args.seed)
                if cache['key'] != expected:
                    raise ValueError('Teacher cache does not match this run')
            else:
                edit, preserve = [], []
                for index, prompt in enumerate(prompts):
                    if prompt.target != prompt.neutral:
                        raise ValueError('Live UNI currently requires target == neutral')
                    for seed_index in range(args.cache_seeds):
                        seed = args.seed + 1000 * (index + 1) + seed_index
                        edit.extend(backend.trajectory(prompt.neutral, prompt.positive, seed,
                                                       positive_rollout=bool(seed_index % 2)))
                        print(f'Teacher cache: row {index + 1}/{len(prompts)}, seed {seed_index + 1}/{args.cache_seeds}', flush=True)
                for index, text in enumerate(PRESERVE):
                    preserve.extend(backend.trajectory(text, text, args.seed + 90000 + index))
                cache = dict(edit=edit, preserve=preserve, key=dict(
                    prompts_sha256=prompt_hash, resolution=args.resolution,
                    revision=args.revision, seeds=args.cache_seeds, seed=args.seed))
                torch.save(cache, cache_path)
            print(f'Teacher cache ready: {len(cache["edit"])} edit states, {len(cache["preserve"])} preservation states', flush=True)
            optimizer = torch.optim.AdamW(backend.params, lr=args.lr, weight_decay=0.01)
            start_step = 0
            if args.resume:
                saved = torch.load(Path(args.resume) / 'training_state.pt', map_location='cpu', weights_only=False)
                optimizer.load_state_dict(saved['optimizer'])
                rng.setstate(saved['rng'])
                start_step = saved['step']
            status['phase'] = 'training'
            with (out / 'train.jsonl').open('a', buffering=1) as log:
                for step in range(start_step + 1, args.steps + 1):
                    tick = time.time()
                    holding = step % 5 == 0
                    record = rng.choice(cache['preserve' if holding else 'edit'])
                    optimizer.zero_grad(set_to_none=True)
                    backend.model.train()
                    pred = backend.forward(record['prompt'], record['z'], record['t'])
                    teacher = record['teacher'].to(backend.device)
                    baseline = record['baseline'].to(backend.device)
                    raw_loss = (pred - teacher).square().mean()
                    loss = raw_loss * (args.hold_weight if holding else 1.)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite loss at step {step}')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(backend.params, 1., error_if_nonfinite=True)
                    if step == start_step + 1 and norm.item() == 0:
                        raise RuntimeError('No gradient reached the LoRA')
                    optimizer.step()
                    stats = dict(step=step, kind='preserve' if holding else 'edit',
                                 loss=loss.item(), raw_loss=raw_loss.item(),
                                 baseline_gap=(baseline - teacher).square().mean().item(),
                                 adapter_delta=(pred.detach() - baseline).square().mean().item(),
                                 grad_norm=norm.item(), seconds=time.time() - tick,
                                 gpu_peak_gib=torch.cuda.max_memory_allocated() / 1024 ** 3)
                    log.write(json.dumps(stats) + '\n')
                    status.update(step=step, elapsed_seconds=time.time() - started, last=stats)
                    write_json(out / 'status.json', status)
                    print(f'Step {step}/{args.steps} {stats["kind"]}: loss={stats["loss"]:.6f} grad={stats["grad_norm"]:.4f} {stats["seconds"]:.1f}s', flush=True)
                    del pred, teacher, baseline, raw_loss, loss
                    if step % args.save_every == 0 or step == args.steps:
                        backend.checkpoint(step, optimizer, rng)
                        if step == args.save_every:
                            preview = []
                            for scale in (0., 1.):
                                image = backend.generate(verify[0], scale, args.sample_seed, args.sample_resolution)
                                image.save(out / f'preview-step-{step:04d}-scale-{scale:g}.png')
                                preview.append(image)
                            make_grid([preview], (0., 1.), out / f'preview-step-{step:04d}.png', ['knight'])
                del optimizer
        status['phase'] = 'sampling'
        status['training_elapsed_seconds'] = time.time() - started
        write_json(out / 'status.json', status)
        samples = out / 'samples'
        samples.mkdir(exist_ok=True)
        variants = []
        sample_meta = []
        for label, text in zip(verify_names, verify):
            row = []
            for scale in (0., .5, 1.):
                image = backend.generate(text, scale, args.sample_seed, args.sample_resolution)
                filename = f'{label}-scale-{scale:g}-seed-{args.sample_seed}.png'
                image.save(samples / filename)
                row.append(image)
                sample_meta.append(dict(file=filename, prompt=text, scale=scale,
                                        seed=args.sample_seed, resolution=args.sample_resolution))
                write_json(samples / 'metadata.json', dict(samples=sample_meta, steps=8, guidance=0., mu=1.15))
                status.update(samples_completed=len(sample_meta), elapsed_seconds=time.time() - started)
                write_json(out / 'status.json', status)
                print(f'Sample: {filename}', flush=True)
            variants.append(row)
        make_grid(variants, (0., .5, 1.), out / 'grid.png', verify_names)
        write_json(samples / 'metadata.json', dict(samples=sample_meta, steps=8, guidance=0., mu=1.15))
        if args.final_resolution:
            for scale in (0., 1.):
                print(f'Showcase: {args.final_resolution}px, scale {scale:g}', flush=True)
                image = backend.generate(HELDOUT, scale, 1234, args.final_resolution)
                filename = f'showcase-scale-{scale:g}-seed-1234.png'
                image.save(samples / filename)
                sample_meta.append(dict(file=filename, prompt=HELDOUT, scale=scale,
                                        seed=1234, resolution=args.final_resolution))
                write_json(samples / 'metadata.json', dict(samples=sample_meta, steps=8, guidance=0., mu=1.15))
                status.update(samples_completed=len(sample_meta), elapsed_seconds=time.time() - started)
                write_json(out / 'status.json', status)
        write_json(samples / 'metadata.json', dict(samples=sample_meta, steps=8, guidance=0., mu=1.15))
        status.update(phase='complete', step=args.steps, elapsed_seconds=time.time() - started)
        write_json(out / 'status.json', status)
        print(f'Complete: {out}', flush=True)
    except BaseException as exc:
        status.update(phase='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(out / 'status.json', status)
        raise
    return out / 'status.json'
