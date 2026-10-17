"""Resolved shared conditional GAN identity and a plainly labeled CPU diagnostic.

Historical winning_formulation()/particle-gmix-1600-v2 stamps did not execute
Krea2's live updates. They must not be used as proof of a GAN training path.
"""
from importlib.metadata import version


def formulation_record(preset="bispf_optimized"):
    from krea2.shared import validate_shared_runtime
    runtime = validate_shared_runtime()
    return dict(shared_runtime=runtime, api='particle_sliders.pg09.ConditionalPG09',
                particlegan='0.9.1', preset=preset,
                generator_arm='adapter_gan', particles=None,
                objective='g_adv;d_adv+public_penalty', optimized_mse=False,
                quality_qualified=False)


def cpu_gan_step():
    """Execute the shared runner on CPU fixture tensors; no Krea weights trained."""
    from krea2.shared import validate_shared_runtime
    validate_shared_runtime()
    import torch
    from torch import nn
    from particle_sliders.pg09 import ConditionalPG09
    from particle_sliders.pg09_backends import AdapterParameters, ResponseRow
    from particle_sliders.pg09_readout_critic import ConditionalReadoutCritic
    from contextlib import nullcontext

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        weight = nn.Parameter(torch.zeros(1))
        backend = _DiagnosticBackend(AdapterParameters([('fixture', weight)], lambda scale: nullcontext()))
        rows = [backend.prepare(ResponseRow(torch.tensor(float(i)), torch.tensor(float(i) + .2),
                torch.tensor(0.), torch.zeros(1, 1, 2, 2), timestep=torch.tensor(500.))) for i in range(2)]
        game = ConditionalPG09(backend, rows, ConditionalReadoutCritic(4, 1),
            conditions=torch.tensor([[0.], [1.]]), particles=None,
            response_domain='continuous_field', total_steps=2, batch_size=2, seed=7)
        stats = game.step()
    return dict(stats, diagnostic_only=True, trained_product_weights=False,
                common_api='particle_sliders.pg09.ConditionalPG09',
                particlegan=version('particlegan'), optimized_mse=False)


# Keep inspectable classes at module scope: shared provenance binds source bytes.
from particle_sliders.pg09_backends import FrozenResponseAdapter


class _DiagnosticBackend(FrozenResponseAdapter):
    family = 'krea2_cpu_diagnostic'
    time_coordinates = 'fixture only'

    def _velocity(self, condition, latent, timestep):
        return latent + condition + self.network.weights[0] * self.network.scale
