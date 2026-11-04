"""Krea2 product binding; shared particle-sliders owns the entire GAN update.

Frozen native trajectory targets remain on CPU. Only selected latent/context rows
move to the staged model. This is an adapter GAN, with no routed particle bank.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch

from particle_sliders.pg09 import ConditionalPG09, validate_pg09_domain
from particle_sliders.pg09_backends import PreparedResponse, ResponseRow
from particle_sliders.pg09_images import (
    ImageResponseAdapter, image_condition_features, image_lora_parameters,
)
from particle_sliders.field_critic import FieldCritic


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def tensor_identity(value):
    tensor = value.detach().cpu().contiguous()
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                sha256=hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


class Krea2Rows:
    """Frozen cached positive velocity on native Krea spatial fields.

    Balanced row sampling contains exactly four edited rows per preservation
    row over a full cycle. Historical MSE hold_weight is not an optimized term.
    Critic context uses shared full native conditioning projections, never IDs.
    """
    def __init__(self, native, cache, *, cache_sha256):
        self.native, self.device = native, native.device
        self.edit, self.preserve = cache['edit'], cache['preserve']
        if not self.edit or not self.preserve:
            raise ValueError('nonempty edited and preservation response pools required')
        self.contexts = {}
        for pool in (self.edit, self.preserve):
            for record in pool:
                for text in (record['prompt'], record['positive']):
                    if text not in self.contexts:
                        embeds, mask = native.embeddings[text]
                        self.contexts[text] = dict(
                            embeds=embeds.detach().cpu().clone(), mask=mask.detach().cpu().clone(),
                            position_ids=native.pipe.prepare_position_ids(
                                embeds.shape[1], native.args.resolution //
                                (native.pipe.vae_scale_factor * native.pipe.patch_size),
                                native.args.resolution // (native.pipe.vae_scale_factor * native.pipe.patch_size),
                                'cpu'))
        self.response_shape = tuple(self._unpack(self.edit[0]['z']).shape)
        validate_pg09_domain(self.response_shape, response_domain='continuous_field')
        self.edit_slots = math.lcm(len(self.edit), 4 * len(self.preserve))
        self.preserve_slots = self.edit_slots // 4
        # Project each immutable context pair once. Native 4B encoder outputs can
        # be large; retaining/reprojecting an encoding per trajectory state would
        # multiply both host memory and work by the number of sampled timesteps.
        pairs = list(dict.fromkeys((r['prompt'], r['positive'])
                                  for pool in (self.edit, self.preserve) for r in pool))
        context_rows = []
        for neutral, positive in pairs:
            row = ResponseRow(self.contexts[neutral], self.contexts[positive], self.contexts[neutral],
                              torch.zeros(1, 1, 1, 1, 1), timestep=torch.tensor(0.))
            context_rows.append(PreparedResponse(row, row.latent, row.latent))
        projected = image_condition_features(context_rows).cpu()
        templates = {pair: projected[i] for i, pair in enumerate(pairs)}
        features, identities = [], []
        energy, count = 0., 0
        for holding, pool in ((False, self.edit), (True, self.preserve)):
            for record in pool:
                if (not torch.isfinite(record['t']).all() or record['t'].numel() != 1
                        or not 0 <= float(record['t']) <= 1000):
                    raise ValueError('invalid native cached timestep')
                fields = [self._unpack(record[k]).float() for k in ('z', 'teacher', 'baseline')]
                if any(tuple(t.shape) != self.response_shape or not torch.isfinite(t).all() for t in fields):
                    raise ValueError('invalid native cached response field')
                if holding and (record['prompt'] != record['positive'] or not torch.equal(fields[1], fields[2])):
                    raise ValueError('preservation must target its exact frozen neutral response')
                if not holding:
                    delta = fields[1].double() - fields[2].double()
                    energy += float(delta.square().sum())
                    count += delta.numel()
                feature = templates[(record['prompt'], record['positive'])].clone()
                feature[0] = float(record['t']) / 1000.
                features.append(feature)
                identities.append({k: tensor_identity(record[k]) for k in ('z', 't', 'teacher', 'baseline')})
        self.features = torch.stack(features)
        self.condition_width = self.features.shape[1]
        self.response_scale = math.sqrt(energy / count)
        if not math.isfinite(self.response_scale) or self.response_scale <= 0:
            raise ValueError('nonzero finite training edit contrast required')
        self._identity = dict(schema='krea2-native-response-cache-v1', cache_sha256=cache_sha256,
            cache_key=deepcopy(cache['key']), tensors=identities,
            contexts={p: {k: tensor_identity(v) for k, v in c.items()} for p, c in self.contexts.items()},
            features=tensor_identity(self.features), response_shape=list(self.response_shape),
            conditioning_policy='shared full-native projections once per immutable context pair; native time/1000',
            mixture='balanced cycles: edit4:preserve1', edit_slots=self.edit_slots,
            preserve_slots=self.preserve_slots, teacher='frozen positive velocity',
            scale=1., training_cfg='direct conditional (native pipeline guidance_scale=0)',
            optimized_mse=False, particles=None)

    def _unpack(self, value):
        return self.native.pipe._unpack_latents(value, self.native.args.resolution, self.native.args.resolution)

    def _row(self, source, latent, *, device):
        def context(text):
            return {k: v.to(device).clone() for k, v in self.contexts[text].items()}
        neutral, positive = context(source['prompt']), context(source['positive'])
        # guidance=1 means direct conditional in the shared mathematical CFG convention.
        # negative is neutral and is never evaluated; there is no negative teacher.
        return ResponseRow(neutral, positive, neutral, latent,
                           timestep=source['t'].to(device).clone(), scale=1., guidance=1.)

    def __len__(self):
        return self.edit_slots + self.preserve_slots

    def _address(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        if index < self.edit_slots:
            return self.edit[index % len(self.edit)], index % len(self.edit)
        index = (index - self.edit_slots) % len(self.preserve)
        return self.preserve[index], len(self.edit) + index

    def __getitem__(self, index):
        source, _ = self._address(index)
        row = self._row(source, self._unpack(source['z']).to(self.device).clone(), device=self.device)
        return PreparedResponse(row, self._unpack(source['teacher']).float().detach().clone(),
                                self._unpack(source['baseline']).float().detach().clone())

    def conditions(self, indices):
        return torch.stack([self.features[self._address(int(i))[1]] for i in indices])

    def identity(self):
        return deepcopy(self._identity)


def build_game(native, cache, *, cache_sha256, model_identity, total_steps, seed=7,
               batch_size=2, preset='bispf_optimized', lr=5e-5):
    """Bind staged native components; never load weights or start a device job."""
    from krea2.shared import validate_shared_runtime
    validate_shared_runtime()
    rows = Krea2Rows(native, cache, cache_sha256=cache_sha256)
    adapter = ImageResponseAdapter('krea2', native.model, image_lora_parameters(native.model),
        pipeline=native.pipe, scheduler=native.pipe.scheduler, model_identity=dict(model_identity, frozen_cache=rows.identity()))
    native.params = list(adapter.network.parameters())
    with torch.random.fork_rng(devices=[native.device.index or 0] if native.device.type == 'cuda' else []):
        torch.manual_seed(seed)
        critic = FieldCritic(rows.response_shape[1], rows.condition_width).to(native.device)
    game = ConditionalPG09(adapter, rows, critic, particles=None,
        response_domain='continuous_field', response_scale=rows.response_scale,
        preset=preset, batch_size=batch_size, seed=seed, total_steps=total_steps,
        recipe_overrides={'lr': lr}, sampling='balanced')
    return game
