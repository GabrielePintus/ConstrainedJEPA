"""Convert Hugging Face LeWM checkpoints into stable-worldmodel object checkpoints.

The Hugging Face mirrors contain a `config.json` and `weights.pt` for each
environment. stable_worldmodel.policy.AutoCostModel expects a serialized object
named `<run_name>_object.ckpt`, so this script performs that one-time conversion.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/constrained-jepa-matplotlib")

DEFAULT_ENVS = ("pusht", "tworoom", "cube", "reacher")


def remap_encoder_keys(state_dict):
    """Map older Transformers ViT key names to the current installed names."""
    replacements = (
        ("encoder.encoder.layer.", "encoder.layers."),
        (".attention.attention.query.", ".attention.q_proj."),
        (".attention.attention.key.", ".attention.k_proj."),
        (".attention.attention.value.", ".attention.v_proj."),
        (".attention.output.dense.", ".attention.o_proj."),
        (".intermediate.dense.", ".mlp.fc1."),
        (".output.dense.", ".mlp.fc2."),
    )

    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in replacements:
            new_key = new_key.replace(old, new)
        remapped[new_key] = value
    return remapped


def stablewm_home() -> Path:
    return Path(os.environ.get("STABLEWM_HOME", "artifacts/stablewm")).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=stablewm_home(),
        help="Artifact root containing hf/<env> folders and receiving <env>/lewm_object.ckpt.",
    )
    parser.add_argument(
        "--env",
        choices=DEFAULT_ENVS,
        action="append",
        help="Environment to convert. May be repeated. Defaults to all downloaded envs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_object.ckpt files.",
    )
    return parser.parse_args()


def load_dependencies():
    try:
        import torch
        import stable_pretraining as spt
        from constrained_jepa.lewm import Embedder, LeWM, MLP, Predictor
    except ImportError as exc:
        raise SystemExit(
            "Missing conversion dependencies. Install them with:\n"
            '  uv pip install -e ".[lewm]"\n'
            "or install torch, hydra-core, omegaconf, stable-pretraining, and stable-worldmodel."
        ) from exc

    return spt, torch, Embedder, LeWM, MLP, Predictor


def convert_one(root: Path, env_name: str, *, overwrite: bool) -> Path:
    spt, torch, Embedder, LeWM, MLP, Predictor = load_dependencies()

    src = root / "hf" / env_name
    cfg_path = src / "config.json"
    weights_path = src / "weights.pt"
    out = root / env_name / "lewm_object.ckpt"

    if not cfg_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Expected {cfg_path} and {weights_path}. Download {env_name} first."
        )

    if out.exists() and not overwrite:
        print(f"skip {env_name}: {out} already exists")
        return out

    cfg = json.loads(cfg_path.read_text())

    encoder_cfg = dict(cfg["encoder"])
    encoder_cfg.pop("_target_", None)
    encoder = spt.backbone.utils.vit_hf(**encoder_cfg)

    predictor_cfg = dict(cfg["predictor"])
    predictor_cfg.pop("_target_", None)

    action_encoder_cfg = dict(cfg["action_encoder"])
    action_encoder_cfg.pop("_target_", None)

    def make_mlp(section: str):
        mlp_cfg = dict(cfg[section])
        mlp_cfg.pop("_target_", None)
        mlp_cfg.pop("norm_fn", None)
        return MLP(**mlp_cfg, norm_fn=torch.nn.BatchNorm1d)

    model = LeWM(
        encoder=encoder,
        predictor=Predictor(**predictor_cfg),
        action_encoder=Embedder(**action_encoder_cfg),
        projector=make_mlp("projector"),
        pred_proj=make_mlp("pred_proj"),
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = remap_encoder_keys(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, out)
    print(f"converted {env_name}: {out}")
    return out


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    envs = tuple(args.env or DEFAULT_ENVS)

    for env_name in envs:
        convert_one(root, env_name, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
