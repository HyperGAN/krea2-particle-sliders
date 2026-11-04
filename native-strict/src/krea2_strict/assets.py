"""Pinned local native assets; metadata inspection does not import ML libraries."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def expected():
    return json.loads(Path(__file__).with_name("assets.json").read_text())


def inspect_assets(transformer, skeleton, *, full_hashes=False):
    manifest = expected()
    roots = dict(turbo=Path(transformer).resolve(strict=True), skeleton=Path(skeleton).resolve(strict=True))
    for role, root in roots.items():
        for name, identity in manifest[role]["files"].items():
            path = root / name
            if not path.is_file() or path.stat().st_size != identity["bytes"]:
                raise ValueError("strict native asset path/size differs: " + role + "/" + name)
            if full_hashes or not name.endswith(".safetensors"):
                if digest(path) != identity["sha256"]:
                    raise ValueError("strict native asset SHA256 differs: " + role + "/" + name)
        # No competing weight variants can redirect from_pretrained's choice.
        for folder in ({"epoch-14-step-73184/transformer"} if role == "turbo" else {"text_encoder", "vae"}):
            actual = {str(p.relative_to(root)) for p in (root / folder).iterdir() if p.is_file()}
            admitted = {n for n in manifest[role]["files"] if str(Path(n).parent) == folder}
            if actual != admitted:
                raise ValueError("unadmitted native loader files in " + folder)
    return dict(schema="krea2-strict-native-assets-v1", sources=manifest,
                full_weight_hashes_verified=full_hashes)


def require_args(args):
    """Only the fully pinned local transformer/skeleton route is admitted."""
    manifest = expected()
    if (args.model_id != manifest["turbo"]["repo"]
            or args.revision != manifest["turbo"]["revision"]
            or args.skeleton_revision != manifest["skeleton"]["revision"]
            or args.transformer_subfolder != "epoch-14-step-73184/transformer"
            or not args.transformer or not Path(args.transformer).is_dir()
            or not Path(args.skeleton_model).is_dir()):
        raise ValueError("strict native route requires the pinned local transformer and skeleton directories")
    return inspect_assets(args.transformer, args.skeleton_model)


def memory_budget():
    return dict(transformer_bf16_weight_gib=23.87924685329199,
        encoder_bf16_weight_gib=8875715136 / 2**30,
        transformer_and_encoder_overlap_in_inherited_loader=True,
        process_cap_gib=36, global_free_floor_gib=5, admission_free_gib=41,
        measured_peak=None, qualification="Unmeasured. Weights alone do not establish native training fit.",
        missing=["LoRA parameters/gradients/optimizer", "critic/activations", "math-SDPA workspaces",
                 "text encoder and VAE loading/staging peak", "CUDA context/allocator"])
