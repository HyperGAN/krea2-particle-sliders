"""CUDA conditional GAN LoRA using the installed shared particle-sliders API.

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
import re
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, PeftModel
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
        if self.pipe.scheduler.config.num_train_timesteps != 1000:
            raise ValueError('qualified Krea2 native response time requires1000 training timesteps')
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
        if args.load_te_lora:
            self.model = PeftModel.from_pretrained(
                self.pipe.transformer, args.load_te_lora, is_trainable=False)
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
        # Every native consumer (including release/probe helpers) gets the same
        # new-checkpoint identity gate; historical PEFT folders remain labeled unknown.
        if args.load_te_lora and (Path(args.load_te_lora) / 'adapter_identity.json').exists():
            self._loaded_frozen_identity = self.frozen_identity()
            self.loaded_adapter_steps = validate_adapter_identity(args.load_te_lora,
                dict(run_contract(args, []), frozen_model_sha256=self._loaded_frozen_identity))

    def frozen_identity(self):
        """Content identity of the actual staged frozen parameters and buffers."""
        result = hashlib.sha256()
        for label, module in (('transformer', self.model), ('vae', self.pipe.vae)):
            for name, value in list(module.named_parameters()) + list(module.named_buffers()):
                if '.lora_' in name:
                    continue
                result.update(f'{label}.{name}:{tuple(value.shape)}:{value.dtype}'.encode())
                # Stream one tensor at a time; no full-model clone is retained.
                data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
                result.update(data.numpy().tobytes())
        return result.hexdigest()

    def trainable_parameters(self):
        return self.params

    @contextlib.contextmanager
    def scale(self, value):
        # Shared context keeps alpha/rank and heterogeneous layer scales intact.
        from particle_sliders.pg09_backends import lora_scale_context
        if not hasattr(self, '_scale_context'):
            self._scale_context = lora_scale_context([('transformer', self.model)])
        with self._scale_context(value):
            yield

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
                records.append(dict(prompt=neutral, positive=positive, z=z.cpu(), t=timestep.cpu(),
                                    baseline=v0.cpu(), teacher=vp.cpu()))
                velocity = vp if positive_rollout else v0
                z = scheduler.step(velocity.to(self.dtype), timestep, z, return_dict=False)[0].to(self.dtype)
        return records

    @torch.no_grad()
    def generate(self, prompt, scale, seed, resolution):
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            return self._generate(prompt, scale, seed, resolution)
        finally:
            for module, training in modes:
                module.training = training

    def _generate(self, prompt, scale, seed, resolution):
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

    def checkpoint(self, step, game):
        game_state, exported = game.state_dict(), game.export_state_dict()
        if game_state['completed_steps'] != step:
            raise ValueError('checkpoint step differs from shared game')
        dest = self.out / f'checkpoint-{step:04d}'
        dest.mkdir(exist_ok=False)
        self.model.save_pretrained(dest)
        state = get_peft_model_state_dict(self.model)
        state = {k.removeprefix('base_model.model.'): v for k, v in state.items()}
        self.pipe.save_lora_weights(dest, transformer_lora_layers=state,
                                    weight_name=f'{self.args.name}.safetensors',
                                    transformer_lora_adapter_metadata=json.loads((dest / 'adapter_config.json').read_text()))
        torch.save(game_state, dest / 'training_state.pt')
        torch.save(exported, dest / 'gan_adapter.pt')
        from krea2.pg09 import file_digest
        write_json(dest / 'adapter_identity.json', dict(
            schema='krea2-gan-adapter-v1', completed_steps=step,
            model_identity=game_state.get('config', {}).get('backend', {}).get('model_identity'),
            files={p.name: file_digest(p) for p in dest.iterdir() if p.is_file()}))
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


def source_commit():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def validate_args(args):
    if args.resume and args.load_te_lora:
        raise ValueError('--resume and --load_te_lora are mutually exclusive')
    if args.sample_guidance != 0 or args.sample_steps != 8 or args.mu != 1.15:
        raise ValueError('Live turbo-bbox uses exactly 8 steps, CFG 0, mu=1.15')
    if (args.resolution <= 0 or args.resolution % 16 or args.steps < 1 or
            args.cache_seeds < 1 or args.save_every < 1 or args.rank < 1 or
            args.sample_resolution <= 0 or args.sample_resolution % 16 or
            args.final_resolution < 0 or args.final_resolution % 16 or
            not 2 <= args.gan_batch_size <= 512 or not np.isfinite(args.lr) or args.lr <= 0):
        raise ValueError('Invalid resolution, steps, cache, rank, batch size or learning rate')
    if args.hold_weight != .1:
        raise ValueError('--hold_weight is a historical MSE option; GAN preservation uses balanced4:1 sampling')
    if not all(re.fullmatch(r'[0-9a-fA-F]{40}', value or '') for value in
               (args.revision, args.skeleton_revision)):
        raise ValueError('exact frozen transformer and skeleton revisions required')


def validate_adapter_identity(path, current_contract):
    from krea2.pg09 import file_digest
    path = Path(path)
    sidecar = path / 'adapter_identity.json'
    if not sidecar.exists():
        return None  # Historical PEFT adapter; no GAN training/resume claim.
    data = json.loads(sidecar.read_text())
    identity = data.get('model_identity')
    if data.get('schema') != 'krea2-gan-adapter-v1' or not isinstance(identity, dict):
        raise ValueError('invalid GAN adapter provenance')
    for key in ('model_id', 'transformer_subfolder', 'revision', 'skeleton', 'skeleton_revision',
                'local_transformer', 'local_skeleton', 'rank', 'frozen_model_sha256'):
        if identity.get(key) != current_contract.get(key):
            raise ValueError('loaded adapter frozen model differs: ' + key)
    for name, checksum in data['files'].items():
        if Path(name).name != name or file_digest(path / name) != checksum:
            raise ValueError('loaded adapter checksum differs')
    step = data.get('completed_steps')
    if type(step) is not int or step < 1:
        raise ValueError('invalid loaded adapter training step')
    return step


def run_contract(args, prompts):
    from krea2.pg09 import file_digest
    from particle_sliders.pg09 import PG_VERSION
    from krea2.card import resolve_source
    plan = resolve_source(args.model_id, subfolder=args.transformer_subfolder,
                          transformer=args.transformer, skeleton=args.skeleton_model)
    def local_files(candidate):
        path = Path(candidate).resolve()
        files = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
        if not files:
            raise ValueError('local model files required')
        return {str(p.relative_to(path) if path.is_dir() else p.name): file_digest(p) for p in files}
    local = local_files(plan['transformer_path']) if plan['source'] != 'hub_subfolder' else None
    skeleton_local = local_files(args.skeleton_model) if Path(args.skeleton_model).exists() else None
    return dict(schema='krea2-conditional-gan-v1', model_id=args.model_id,
        transformer_subfolder=args.transformer_subfolder, revision=args.revision,
        skeleton=args.skeleton_model, skeleton_revision=args.skeleton_revision, local_transformer=local,
        local_skeleton=skeleton_local,
        resolution=args.resolution, seeds=args.cache_seeds, seed=args.seed,
        sample_steps=args.sample_steps, guidance=args.sample_guidance, mu=args.mu,
        rank=args.rank, steps=args.steps, lr=args.lr, batch_size=args.gan_batch_size, preset=args.gan_preset,
        prompts=[vars(p) for p in prompts], preservation=list(PRESERVE), dtype='bfloat16; adapter=float32',
        versions={n: importlib.metadata.version(n) for n in
                  ('torch', 'diffusers', 'transformers', 'peft', 'safetensors', 'particle-sliders-core')},
        particlegan=PG_VERSION, source_sha256={p.name: file_digest(p) for p in
            sorted(Path(__file__).parent.glob('*.py'))}, objective='g_adv;d_adv+public_penalty', particles=None)


def train_cuda(args, prompts, meta, formulation):
    validate_args(args)
    from krea2.shared import validate_shared_runtime
    validate_shared_runtime()
    for prompt in prompts:
        if prompt.action != 'enhance' or prompt.target != prompt.neutral:
            raise ValueError('live Krea2 GAN supports enhance with target == neutral only')
    contract = run_contract(args, prompts)
    out = Path(args.save_dir).resolve()
    if args.load_te_lora and out.exists() and any(out.iterdir()):
        raise ValueError('sample-only output requires a new empty directory')
    if not args.resume and not args.load_te_lora and (out / 'run.json').exists():
        raise ValueError('existing run provenance requires --resume or a new directory')
    cache_path = out / 'teacher_cache.pt'
    if cache_path.exists() and not args.load_te_lora:
        cached = torch.load(cache_path, map_location='cpu', weights_only=True)
        key = dict(cached.get('key', {}))
        key.pop('frozen_model_sha256', None)
        if key != contract:
            raise ValueError('teacher cache source/model/prompt contract differs')
    continuation = None
    if args.resume:
        checkpoint = Path(args.resume).resolve()
        source_run = checkpoint.parent
        previous = json.loads((source_run / 'run.json').read_text())
        if previous.get('training_contract') != contract:
            raise ValueError('resume source/model/prompt/objective contract differs; use a new cohort')
        state = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=True)
        if state.get('schema') != 'conditional-pg09-v1':
            raise ValueError('legacy MSE checkpoints cannot resume the GAN objective')
        if type(state.get('completed_steps')) is not int or not 0 <= state['completed_steps'] <= args.steps:
            raise ValueError('invalid GAN checkpoint step')
        records = [json.loads(line) for line in (source_run / 'train.jsonl').read_text().splitlines()]
        prefix = records[:state['completed_steps']]
        if [r['step'] for r in prefix] != list(range(1, state['completed_steps'] + 1)):
            raise ValueError('training log prefix differs from resume checkpoint')
        if out == source_run and len(records) != state['completed_steps']:
            raise ValueError('updates after checkpoint remain in log; resume into a new --save_dir to preserve them')
        if out != source_run:
            if out.exists() and any(out.iterdir()):
                raise ValueError('continuation output must be a new empty directory')
            continuation = dict(source=source_run, records=prefix, checkpoint=str(checkpoint))
    elif (out / 'train.jsonl').exists() and not args.load_te_lora:
        raise ValueError(f'{out} already has training; use --resume or a new directory')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for --live; resource owner admission required')
    out.mkdir(parents=True, exist_ok=True)
    if continuation is not None:
        shutil.copyfile(continuation['source'] / 'teacher_cache.pt', out / 'teacher_cache.pt')
        (out / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in continuation['records']))
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision('high')
    control = args.control_prompt or meta.control_prompt
    verify = [prompts[0].neutral, prompts[4].neutral if len(prompts) > 4 else prompts[-1].neutral,
              HELDOUT, control]
    verify_names = ['knight', 'cave-warrior', 'heldout-bridge', 'fruit-control']
    texts = [text for p in prompts for text in (p.target, p.neutral, p.positive, p.negative) if text]
    texts += list(PRESERVE) + verify
    prompt_hash = hashlib.sha256(json.dumps(texts).encode()).hexdigest()
    manifest = dict(
        **vars(args), prompts_sha256=prompt_hash, training_contract=contract,
        continued_from=(continuation['checkpoint'] if continuation is not None else None),
        source_commit=source_commit(),
        gpu=torch.cuda.get_device_name(args.device),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        versions={n: importlib.metadata.version(n) for n in
                  ('torch', 'diffusers', 'transformers', 'peft', 'safetensors')},
        objective='shared ConditionalPG09 adapter GAN: frozen positive velocity targets',
        optimized_mse=False, particles=None, preservation_sampling='4:1 balanced edit:preserve',
        teacher_trajectories='alternating neutral and positive, frozen, 8 steps, mu=1.15',
        preservation=list(PRESERVE), fruit_is_verify_only=True,
        scale_zero='adapter disabled exactly', minus_teacher=False,
        formulation=formulation,
        loaded_adapter=(dict(path=str(Path(args.load_te_lora).resolve()),
            files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in Path(args.load_te_lora).iterdir() if p.is_file()},
            provenance='verified GAN sidecar' if (Path(args.load_te_lora) / 'adapter_identity.json').exists()
                       else 'legacy PEFT; training objective and step unverified')
            if args.load_te_lora else None))
    root = Path(__file__).resolve().parents[1]
    snapshot = out / ('sample_source' if args.load_te_lora else 'source')
    source_paths = list((root / 'krea2').glob('*.py')) + [
        root / 'krea2/shared.lock.json', root / 'requirements.txt', root / 'pyproject.toml',
        root / 'scripts/train_krea2.py', *sorted((root / 'scripts').glob('train_*_gpu0.sh')),
        root / args.prompts_file, root / args.config_file]
    source_hashes = {}
    for source in source_paths:
        if not source.is_file():
            continue
        relative = source.relative_to(root) if source.is_relative_to(root) else Path(source.name)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError('source snapshot differs; use a new cohort')
        shutil.copyfile(source, target)
        source_hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest['source_files_sha256'] = source_hashes
    manifest_path = out / ('sample_run.json' if args.load_te_lora else 'run.json')
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    write_json(out / 'prompts.json', [vars(p) for p in prompts])
    status = dict(phase='loading', step=0, total_steps=args.steps)
    write_json(out / 'status.json', status)
    started = time.time()
    try:
        backend = CudaUNI(args, texts, out)
        frozen_identity = getattr(backend, '_loaded_frozen_identity', None) or backend.frozen_identity()
        cache_contract = dict(contract, frozen_model_sha256=frozen_identity)
        completed_steps = None
        if args.load_te_lora:
            completed_steps = validate_adapter_identity(args.load_te_lora, cache_contract)
            status['loaded_adapter_steps'] = completed_steps
            status['training_performed'] = False
        else:
            cache_path = out / 'teacher_cache.pt'
            status['phase'] = 'teacher_cache'
            write_json(out / 'status.json', status)
            if cache_path.exists():
                cache = torch.load(cache_path, map_location='cpu', weights_only=True)
                expected = cache_contract
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
                cache = dict(edit=edit, preserve=preserve, key=cache_contract)
                torch.save(cache, cache_path)
            print(f'Teacher cache ready: {len(cache["edit"])} edit states, {len(cache["preserve"])} preservation states', flush=True)
            from krea2.pg09 import build_game, file_digest
            game = build_game(backend, cache, cache_sha256=file_digest(cache_path),
                model_identity=cache_contract, total_steps=args.steps, seed=args.seed,
                batch_size=args.gan_batch_size, preset=args.gan_preset, lr=args.lr)
            if args.resume:
                game.load_state_dict(torch.load(Path(args.resume) / 'training_state.pt',
                                                map_location='cpu', weights_only=True))
            write_json(out / 'gan_protocol.json', game.state_dict()['config'])
            status['phase'] = 'training'
            with (out / 'train.jsonl').open('a', buffering=1) as log:
                while game.completed_steps < args.steps:
                    tick = time.time()
                    stats = game.step()
                    step = game.completed_steps
                    stats.update(seconds=time.time() - tick,
                        gpu_peak_gib=torch.cuda.max_memory_allocated() / 1024 ** 3)
                    log.write(json.dumps(stats) + '\n')
                    status.update(step=step, elapsed_seconds=time.time() - started, last=stats)
                    write_json(out / 'status.json', status)
                    print(f'Step {step}/{args.steps}: g_adv={stats["g_adv"]:.6f} d_adv={stats["d_adv"]:.6f}', flush=True)
                    if step % args.save_every == 0 or step == args.steps:
                        backend.checkpoint(step, game)
                        if step == args.save_every:
                            preview = []
                            for scale in (0., 1.):
                                image = backend.generate(verify[0], scale, args.sample_seed, args.sample_resolution)
                                image.save(out / f'preview-step-{step:04d}-scale-{scale:g}.png')
                                preview.append(image)
                            make_grid([preview], (0., 1.), out / f'preview-step-{step:04d}.png', ['knight'])
            completed_steps = game.completed_steps
            del game
            if backend.frozen_identity() != frozen_identity:
                raise RuntimeError('frozen model changed during training')
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
        status.update(phase='complete', step=completed_steps, elapsed_seconds=time.time() - started)
        write_json(out / 'status.json', status)
        print(f'Complete: {out}', flush=True)
    except BaseException as exc:
        status.update(phase='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(out / 'status.json', status)
        raise
    return out / 'status.json'
