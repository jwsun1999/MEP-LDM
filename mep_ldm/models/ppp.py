"""Canonical differentiable PPP model and versioned checkpoint interface.

The single-input calibrators and product-form mixed-box operator are shared
by calibration, evaluation, and diffusion training.
"""
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def fractal_dimension_soft_mixed_blocks(
    p_phase: torch.Tensor,
    bins: int,
    min_size: int,
    alpha_any: float,
    beta_all: float,
) -> torch.Tensor:
    """
    平滑版混合块计数分形维数。

    输入 p_phase 是某一相的连续概率场，例如孔隙概率场或固体概率场。
    输出为每个样本对应的软分形维数估计值。
    """
    _, _, z_size, y_size, x_size = p_phase.shape
    device = p_phase.device

    max_size = min(z_size, y_size, x_size) // 2
    if max_size < min_size:
        max_size = min_size * 2

    box_sizes = np.round(
        np.logspace(math.log10(min_size), math.log10(max_size), bins)
    ).astype(int)
    box_sizes = np.unique(box_sizes)
    box_sizes = [int(d) for d in box_sizes if d >= 1]

    log_counts_list = []
    log_sizes_list = []

    for box_size in box_sizes:
        kz = min(box_size, z_size)
        ky = min(box_size, y_size)
        kx = min(box_size, x_size)
        volume = float(kz * ky * kx)

        pooled = F.avg_pool3d(
            p_phase,
            kernel_size=(kz, ky, kx),
            stride=(kz, ky, kx),
            padding=0,
        )
        sum_phase = pooled * volume
        sum_other = volume - sum_phase

        soft_any = 1.0 - torch.exp(-alpha_any * sum_phase)
        soft_all = 1.0 - torch.exp(-beta_all * sum_other)
        soft_mix = soft_any * soft_all

        n_d = soft_mix.sum(dim=(1, 2, 3, 4)).clamp_min(1e-8)
        log_counts_list.append(torch.log(n_d))
        log_sizes_list.append(math.log(float(box_size)))

    y_mat = torch.stack(log_counts_list, dim=1)
    x_vec = torch.tensor(log_sizes_list, device=device).view(1, -1)

    x_centered = x_vec - x_vec.mean(dim=1, keepdim=True)
    y_centered = y_mat - y_mat.mean(dim=1, keepdim=True)
    denom = (x_centered * x_centered).sum(dim=1, keepdim=True).clamp_min(1e-12)

    slope = (x_centered * y_centered).sum(dim=1) / denom.squeeze(1)
    return -slope


@dataclass
class SurrogateConfig:
    tv_epsilon: float = 1e-6
    invert_logits: bool = True
    clamp_prob: float | None = 1e-4
    box_bins: int = 10
    box_min_size: int = 2
    alpha_any: float = 5.0
    beta_all: float = 5.0


class PhysPropSurrogate(nn.Module):
    """
    可微物性预测器主体。

    输出：
        phi : 孔隙度
        S_v : 比表面积预测值
        D   : 分形维数预测值
    """

    def __init__(self, cfg: Optional[SurrogateConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else SurrogateConfig()
        self.register_buffer("_dev", torch.empty(0), persistent=False)

        # These calibrators can be trained separately and attached at inference time.
        self.sv_calib: Optional[nn.Module] = None
        self.d_calib: Optional[nn.Module] = None

    def extract_features(self, values: torch.Tensor, *, input_type="logits"):
        if values.ndim != 5 or values.shape[1] != 1:
            raise ValueError("PPP expects [B, 1, Z, Y, X].")
        if input_type == "logits":
            signed = -values if self.cfg.invert_logits else values
            pore_prob = torch.sigmoid(signed)
        elif input_type == "solid_probability":
            pore_prob = 1.0 - values
        elif input_type == "pore_probability":
            pore_prob = values
        else:
            raise ValueError(f"Unknown PPP input_type: {input_type!r}")
        if self.cfg.clamp_prob is not None:
            eps = self.cfg.clamp_prob
            pore_prob = pore_prob.clamp(eps, 1.0 - eps)
        return {
            "phi": pore_prob.mean(dim=(1, 2, 3, 4)),
            "S_v": self._compute_Sv_raw_from_p(pore_prob),
            "D": fractal_dimension_soft_mixed_blocks(
                pore_prob, bins=self.cfg.box_bins, min_size=self.cfg.box_min_size,
                alpha_any=self.cfg.alpha_any, beta_all=self.cfg.beta_all,
            ),
        }

    def forward(self, values: torch.Tensor, *, input_type="logits"):
        raw = self.extract_features(values, input_type=input_type)
        return {
            "phi": raw["phi"],
            "S_v": self._calibrate_Sv(raw["S_v"]),
            "D": self.d_calib(raw["D"].view(-1, 1)) if self.d_calib is not None else raw["D"],
        }

    def _compute_Sv_raw_from_p(self, p: torch.Tensor) -> torch.Tensor:
        """用孔隙概率场梯度近似比表面积原始值。"""
        p_pad = F.pad(p, (1, 1, 1, 1, 1, 1), mode="replicate")

        px = (p_pad[:, :, 1:-1, 1:-1, 2:] - p_pad[:, :, 1:-1, 1:-1, :-2]) * 0.5
        py = (p_pad[:, :, 1:-1, 2:, 1:-1] - p_pad[:, :, 1:-1, :-2, 1:-1]) * 0.5
        pz = (p_pad[:, :, 2:, 1:-1, 1:-1] - p_pad[:, :, :-2, 1:-1, 1:-1]) * 0.5

        grad_mag = torch.sqrt(px * px + py * py + pz * pz + self.cfg.tv_epsilon)
        numerator = grad_mag.sum(dim=(1, 2, 3, 4))
        denominator = p.sum(dim=(1, 2, 3, 4)).clamp_min(1e-12)
        return numerator / denominator

    def _calibrate_Sv(self, sv_raw: torch.Tensor) -> torch.Tensor:
        """仅基于解析算子得到的 S_v 特征进行校准，不引入孔隙度。"""
        if self.sv_calib is not None:
            return self.sv_calib(sv_raw.view(-1, 1))

        return sv_raw


class SmallMLP(nn.Module):
    """用于 S_v 二次校准和 D 校准的小型 MLP。"""

    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if y.shape[-1] == 1:
            y = y.squeeze(-1)
        return y


CHECKPOINT_FORMAT = "mep-ldm-ppp"
CHECKPOINT_VERSION = 1
ARCHITECTURE = "single-input-mlp32-product-mixed-box"


def save_ppp(model, path):
    """Save only tensors and primitive metadata; no pickled model objects."""
    expected = SmallMLP(in_dim=1).state_dict()
    for name in ("sv_calib", "d_calib"):
        module = getattr(model, name)
        if not isinstance(module, SmallMLP):
            raise ValueError(f"Attach a trained SmallMLP as {name} before saving.")
        state = module.state_dict()
        if state.keys() != expected.keys() or any(state[k].shape != v.shape for k, v in expected.items()):
            raise ValueError("PPP v1 requires single-input MLPs with 32 hidden units.")
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "cfg": asdict(model.cfg),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }, path)


def load_ppp(path, *, device="cpu", freeze=True, legacy_cfg=None):
    """Load a calibrated PPP strictly, without disabling input gradients.

    Legacy {sv_calib, d_calib, args} checkpoints require an explicit legacy_cfg
    because the old files did not record analytical-operator settings. The older
    multi-input surrogate is a different model and is intentionally rejected.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a PPP checkpoint dictionary.")
    if checkpoint.get("format") == CHECKPOINT_FORMAT:
        if checkpoint.get("version") != CHECKPOINT_VERSION or checkpoint.get("architecture") != ARCHITECTURE:
            raise ValueError("Unsupported PPP checkpoint version or architecture.")
        cfg = SurrogateConfig(**checkpoint["cfg"])
        state = checkpoint["state_dict"]
    elif {"sv_calib", "d_calib"}.issubset(checkpoint):
        if not isinstance(legacy_cfg, SurrogateConfig):
            raise ValueError("Legacy calibration checkpoint: supply legacy_cfg=SurrogateConfig(...) with verified original settings, then save_ppp().")
        cfg = legacy_cfg
        state = {f"sv_calib.{k}": v for k, v in checkpoint["sv_calib"].items()}
        state.update({f"d_calib.{k}": v for k, v in checkpoint["d_calib"].items()})
    else:
        raise ValueError("Incompatible legacy surrogate checkpoint. Multi-input calibrators and the old mixed-box operator cannot be relabeled as PPP v1; retrain or retain the legacy experiment separately.")
    model = PhysPropSurrogate(cfg)
    model.sv_calib = SmallMLP(in_dim=1)
    model.d_calib = SmallMLP(in_dim=1)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    if freeze:
        model.requires_grad_(False)
    return model


def load_surrogate_model(checkpoint_path, **kwargs):
    """Compatibility name for callers; delegates to the canonical loader."""
    return load_ppp(checkpoint_path, **kwargs)
