"""LeWM model components needed to load released checkpoints.

These classes mirror the public LeWM architecture used by the Hugging Face
checkpoint configs. They let us serialize object checkpoints without depending
on an import path that is currently missing from the PyPI stable-worldmodel
package.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def _detach_clone(value):
    return value.detach().clone() if torch.is_tensor(value) else value


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        return self.to_out(rearrange(out, "b h t d -> b t (h d)"))


class ConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.layers = nn.ModuleList(
            [ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()

    def forward(self, x, c=None):
        x = self.input_proj(x)
        if c is not None:
            c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)
        x = self.norm(x)
        return self.output_proj(x)


class Embedder(nn.Module):
    def __init__(self, input_dim=10, smoothed_dim=10, emb_dim=10, mlp_scale=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.float()
        x = self.patch_embed(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.embed(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, norm_fn=nn.LayerNorm, act_fn=nn.GELU):
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
        )

    def forward(self, x, c):
        t = x.size(1)
        x = x + self.pos_embedding[:, :t]
        x = self.dropout(x)
        return self.transformer(x, c)


class LeWM(nn.Module):
    def __init__(self, encoder, predictor, action_encoder, projector=None, pred_proj=None):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        pixels = info["pixels"].float()
        batch = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        emb = self.projector(output.last_hidden_state[:, 0])
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=batch)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=emb.size(0))

    def rollout(self, info, action_sequence, history_size: int = 3):
        assert "pixels" in info, "pixels not in info_dict"
        history = info["pixels"].size(2)
        batch, samples, horizon = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [history, horizon - history], dim=2)
        info["action"] = act_0

        init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        init = self.encode(init)
        emb = info["emb"] = init["emb"].unsqueeze(1).expand(batch, samples, -1, -1)
        init = {k: _detach_clone(v) for k, v in init.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        for t in range(horizon - history):
            act_emb = self.action_encoder(act)
            pred_emb = self.predict(emb[:, -history_size:], act_emb[:, -history_size:])[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)
            act = torch.cat([act, act_future[:, t : t + 1, :]], dim=1)

        act_emb = self.action_encoder(act)
        pred_emb = self.predict(emb[:, -history_size:], act_emb[:, -history_size:])[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=batch, s=samples)
        return info

    def criterion(self, info_dict):
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
        return F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))

    def get_cost(self, info_dict, action_candidates):
        assert "goal" in info_dict, "goal not in info_dict"
        device = next(self.parameters()).device
        for key in list(info_dict.keys()):
            if torch.is_tensor(info_dict[key]):
                info_dict[key] = info_dict[key].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for key in info_dict:
            if key.startswith("goal_"):
                goal[key[len("goal_") :]] = goal.pop(key)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)
