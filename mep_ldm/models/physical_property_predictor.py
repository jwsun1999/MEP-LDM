# -*- coding: utf-8 -*-
"""
Differentiable physical-property predictor for MEP-LDM.

The predictor combines analytic operators with lightweight MLP calibrators:
porosity is computed by global averaging, specific surface area is estimated
from finite-difference gradients, and fractal dimension is estimated with a
soft mixed-block counting operator.

Supported data sources:
- TIFF folders with data_path/train/*.tif and data_path/val/*.tif.
- The MEP-LDM binary datamodule loaders.
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import math
import os
import random
from .ppp import (PhysPropSurrogate, SurrogateConfig, SmallMLP,
                  fractal_dimension_soft_mixed_blocks, save_ppp)
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
# 0. Optional dependency: MEP-LDM binary datamodule
# -----------------------------------------------------------------------------
try:
    from mep_ldm.data.binary_datamodule import get_binary_datamodule
except ImportError:  # pragma: no cover - only used when running the module standalone
    get_binary_datamodule = None

# -----------------------------------------------------------------------------
# 1. 基础工具
# -----------------------------------------------------------------------------
def get_device() -> torch.device:
    """返回当前训练设备。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_global_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def probs_to_logits(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """将 [0, 1] 概率值转换为 logits。"""
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log(1.0 - x)

def get_logits_from_batch(batch: Tuple[torch.Tensor, dict]) -> torch.Tensor:
    """
    从 dataloader batch 中取出图像并转为 logits。

    支持：
    - 3D: [B, 1, Z, Y, X]
    - 2D: [B, 1, H, W]，会自动扩展为 [B, 1, 1, H, W]
    """
    x, _ = batch
    if x.ndim == 4:
        x = x.unsqueeze(2)

    assert x.ndim == 5 and x.shape[1] == 1, (
        f"Expect x [B,1,Z,Y,X], got {tuple(x.shape)}"
    )
    return probs_to_logits(x)

# -----------------------------------------------------------------------------
# 2. 标签解析与 TIFF 数据集
# -----------------------------------------------------------------------------
def parse_truth_from_filename(filename: str) -> Dict[str, float]:
    """
    从文件名解析物性真值标签。

    期望文件名格式：
        phi_0.113_D_2.160_Sv_0.4302.tif

    解析规则：
        parts[1] -> phi
        parts[3] -> D
        parts[5] -> S_v
    """
    name_no_ext = os.path.splitext(filename)[0]
    parts = name_no_ext.split("_")

    try:
        phi_val = float(parts[1])
        d_val = float(parts[3])
        sv_val = float(parts[5])
    except (IndexError, ValueError) as exc:
        print(f"Error parsing filename '{filename}': {exc}. Using zeros.")
        return {"phi": 0.0, "D": 0.0, "S_v": 0.0}

    return {"phi": phi_val, "D": d_val, "S_v": sv_val}

def _detect_truth_mapping_from_keys(keys: Iterable[str]) -> Dict[str, Optional[str]]:
    """将不同命名方式的标签键统一映射到 phi / S_v / D。"""
    lower_to_real = {k.lower(): k for k in keys}
    mapping: Dict[str, Optional[str]] = {"phi": None, "S_v": None, "D": None}

    for key in ("phi", "porosity"):
        if key in lower_to_real:
            mapping["phi"] = lower_to_real[key]
            break

    for key in (
        "s_v",
        "sv",
        "surface_area_density",
        "specific_surface",
        "specific_surface_area",
    ):
        if key in lower_to_real:
            mapping["S_v"] = lower_to_real[key]
            break

    for key in ("d", "fractal_dim", "fractal_dimension"):
        if key in lower_to_real:
            mapping["D"] = lower_to_real[key]
            break

    return mapping

def get_truth_from_batch(batch: Tuple[torch.Tensor, dict]) -> Dict[str, Optional[torch.Tensor]]:
    """从 batch 的标签字典中取出标准化后的 phi / S_v / D。"""
    _, y = batch
    assert isinstance(y, dict), "Expect (x, y_dict) from dataloader."

    mapping = _detect_truth_mapping_from_keys(y.keys())
    out: Dict[str, Optional[torch.Tensor]] = {}

    for std_key in ["phi", "S_v", "D"]:
        real_key = mapping.get(std_key)
        if real_key is None:
            out[std_key] = None
            continue

        value = y[real_key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.float32)
        if value.ndim == 0:
            value = value[None]
        out[std_key] = value.float()

    return out

class TiffDataset(Dataset):
    """从 train/val 文件夹读取 3D TIFF，并从文件名解析物性真值。"""

    def __init__(self, root_dir: str, split: str = "train", dim: int = 3):
        self.split_dir = os.path.join(root_dir, split)
        self.files = sorted(glob.glob(os.path.join(self.split_dir, "*.tif")))
        if len(self.files) == 0:
            self.files = sorted(glob.glob(os.path.join(self.split_dir, "*.tiff")))

        self.dim = dim
        print(f"[TiffDataset] Found {len(self.files)} images in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, float | str]]:
        path = self.files[idx]
        filename = os.path.basename(path)

        try:
            img = tifffile.imread(path)
        except Exception as exc:
            print(f"Error reading {path}: {exc}")
            return torch.zeros(1, 1, 1, 1), {}

        # 兼容 0/255 与 0/1 两类 TIFF；最终强制二值化。
        if img.max() > 1:
            img = img / 255.0
        img = (img > 0.5).astype(np.float32)

        img_tensor = torch.from_numpy(img).unsqueeze(0)
        truth_dict = parse_truth_from_filename(filename)
        truth_dict["id"] = filename

        return img_tensor, truth_dict

def get_tiff_dataloaders(
    data_path: str,
    batch_size: int,
    dimension: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """构建 TIFF 训练集和验证集 dataloader。"""
    train_ds = TiffDataset(data_path, split="train", dim=dimension)
    val_ds = TiffDataset(data_path, split="val", dim=dimension)

    # 注意：如果 TIFF 尺寸不一致，batch_size 需要设置为 1，或额外定义 collate_fn。
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader

# -----------------------------------------------------------------------------
# 3. 可微解析物性算子
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# 4. 校准网络
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 5. 校准器训练
# -----------------------------------------------------------------------------
def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
    r = pred - target
    abs_r = torch.abs(r)
    quad = 0.5 * r * r
    lin = delta * (abs_r - 0.5 * delta)
    return torch.where(abs_r <= delta, quad, lin).mean()

def train_sv_mlp_calibrator(
    sv_raw_tr: torch.Tensor,
    sv_true_tr: torch.Tensor,
    sv_raw_val: torch.Tensor,
    sv_true_val: torch.Tensor,
    lr: float = 1e-3,
    epochs: int = 7000,
    patience: int = 200,
    delta: float = 0.05,
    hidden: int = 32,
) -> SmallMLP:
    """训练 S_v 二次校准器：只使用解析算子 S_v 特征，不使用孔隙度。"""
    device = get_device()

    x_train = sv_raw_tr.view(-1, 1).to(device)
    x_val = sv_raw_val.view(-1, 1).to(device)

    y_train = sv_true_tr.view(-1).to(device)
    y_val = sv_true_val.view(-1).to(device)

    model = SmallMLP(in_dim=1, hidden=hidden, out_dim=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for _ in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        loss = huber_loss(model(x_train), y_train, delta)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = huber_loss(model(x_val), y_val, delta)

        if val_loss.item() + 1e-9 < best_loss:
            best_loss = val_loss.item()
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model

def train_d_calibrator(
    d_soft_pore_tr: torch.Tensor,
    d_true_tr: torch.Tensor,
    d_soft_pore_val: torch.Tensor,
    d_true_val: torch.Tensor,
    lr: float = 1e-3,
    epochs: int = 2000,
    patience: int = 200,
    delta: float = 0.05,
    hidden: int = 32,
) -> SmallMLP:
    device = get_device()

    x_train = d_soft_pore_tr.view(-1, 1).to(device)
    x_val = d_soft_pore_val.view(-1, 1).to(device)

    y_train = d_true_tr.view(-1).to(device)
    y_val = d_true_val.view(-1).to(device)

    model = SmallMLP(in_dim=1, hidden=hidden, out_dim=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for _ in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        loss = huber_loss(model(x_train), y_train, delta)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = huber_loss(model(x_val), y_val, delta)

        if val_loss.item() + 1e-9 < best_loss:
            best_loss = val_loss.item()
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model

# -----------------------------------------------------------------------------
# 6. 评估、缓存与预测
# -----------------------------------------------------------------------------
def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    y_true = y_true.double().flatten()
    y_pred = y_pred.double().flatten()

    mae = (y_true - y_pred).abs().mean().item()
    rmse = torch.sqrt(((y_true - y_pred) ** 2).mean()).item()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum().clamp(min=1e-12)
    r2 = 1.0 - (ss_res / ss_tot).item()
    std = (y_true - y_pred).std(unbiased=False).item()

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "STD": std}

def _try_get_sample_ids_from_y(y: dict, batch_size: int) -> Optional[list[str]]:
    if not isinstance(y, dict):
        return None

    candidate_keys = ["id", "sample_id", "name", "fname", "filename"]
    lower_to_real = {k.lower(): k for k in y.keys()}
    hit = None

    for key in candidate_keys:
        if key in lower_to_real:
            hit = lower_to_real[key]
            break

    if hit is None:
        return None

    value = y[hit]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    try:
        value = list(value)
    except Exception:
        return None

    if len(value) != batch_size:
        return None

    return [str(item) for item in value]

@torch.no_grad()
def collect_eval_cache(
    dataloader: DataLoader,
    surrogate_cfg: SurrogateConfig,
) -> Dict[str, torch.Tensor | list[str]]:
    """
    预计算并缓存解析特征和真值标签。

    缓存字段：
        sample_ids, phi_raw, Sv_raw, D_soft_solid, D_soft_pore,
        phi_true, Sv_true, D_true
    """
    device = get_device()
    surrogate = PhysPropSurrogate(cfg=surrogate_cfg).to(device).eval()

    sample_ids: list[str] = []
    phi_raw_list = []
    sv_raw_list = []
    d_pore_list = []
    phi_true_list = []
    sv_true_list = []
    d_true_list = []

    global_idx = 0
    print("Collecting cache...")

    for batch in dataloader:
        x, y = batch
        batch_size = x.shape[0]

        ids = _try_get_sample_ids_from_y(y, batch_size)
        if ids is None:
            ids = [f"{global_idx + i}" for i in range(batch_size)]
        global_idx += batch_size
        sample_ids.extend(ids)

        values = x.to(device).float()
        if values.ndim == 4:
            values = values.unsqueeze(2)
        features = surrogate.extract_features(values, input_type="solid_probability")
        phi_raw, sv_raw, d_soft_pore = features["phi"], features["S_v"], features["D"]

        truth = get_truth_from_batch(batch)
        sv_true = truth["S_v"].reshape(-1).float().to(device)
        d_true = truth["D"].reshape(-1).float().to(device)

        if truth["phi"] is not None:
            phi_true = truth["phi"].reshape(-1).float().to(device)
        else:
            x_for_phi = x.float()
            if x_for_phi.ndim == 4:
                x_for_phi = x_for_phi.unsqueeze(2)
            phi_true = (1.0 - x_for_phi).mean(dim=(1, 2, 3, 4)).reshape(-1).to(device)

        phi_raw_list.append(phi_raw.cpu())
        sv_raw_list.append(sv_raw.cpu())
        d_pore_list.append(d_soft_pore.cpu())
        phi_true_list.append(phi_true.cpu())
        sv_true_list.append(sv_true.cpu())
        d_true_list.append(d_true.cpu())

    return {
        "sample_ids": sample_ids,
        "phi_raw": torch.cat(phi_raw_list, dim=0),
        "Sv_raw": torch.cat(sv_raw_list, dim=0),
        "D_soft_pore": torch.cat(d_pore_list, dim=0),
        "phi_true": torch.cat(phi_true_list, dim=0),
        "Sv_true": torch.cat(sv_true_list, dim=0),
        "D_true": torch.cat(d_true_list, dim=0),
    }

@torch.no_grad()
def predict_from_cache(
    cache: Dict[str, torch.Tensor | list[str]],
    sv_mlp: nn.Module,
    d_calib: nn.Module,
) -> Dict[str, torch.Tensor]:
    """基于缓存特征和校准器得到 phi / S_v / D 预测值。"""
    device = get_device()

    phi_raw = cache["phi_raw"].to(device)
    sv_raw = cache["Sv_raw"].to(device)
    d_pore = cache["D_soft_pore"].to(device)

    sv_mlp = sv_mlp.to(device).eval()
    d_calib = d_calib.to(device).eval()

    sv_pred = sv_mlp(sv_raw.view(-1, 1)).view(-1)
    d_pred = d_calib(d_pore.view(-1, 1)).view(-1)

    return {"phi": phi_raw.cpu(), "S_v": sv_pred.cpu(), "D": d_pred.cpu()}


# -----------------------------------------------------------------------------
# 7. CSV 导出
# -----------------------------------------------------------------------------
def append_stability_rows(
    cache: Dict[str, torch.Tensor | list[str]],
    y_pred: Dict[str, torch.Tensor],
    out_csv: str,
    split_name: str,
    run_id: int,
    append: bool,
) -> None:
    mode = "a" if append else "w"
    fieldnames = ["Target", "Split", "RunID", "MAE", "RMSE", "R2", "STD"]

    with open(out_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append:
            writer.writeheader()

        writer.writerow(
            {
                "Target": "porosity",
                "Split": split_name,
                "RunID": run_id,
                **regression_metrics(cache["phi_true"], y_pred["phi"]),
            }
        )
        writer.writerow(
            {
                "Target": "Sv",
                "Split": split_name,
                "RunID": run_id,
                **regression_metrics(cache["Sv_true"], y_pred["S_v"]),
            }
        )
        writer.writerow(
            {
                "Target": "fractal_dim",
                "Split": split_name,
                "RunID": run_id,
                **regression_metrics(cache["D_true"], y_pred["D"]),
            }
        )

def export_predictions_long_from_cache(
    cache: Dict[str, torch.Tensor | list[str]],
    y_pred: Dict[str, torch.Tensor],
    out_csv: str,
    split_name: str,
    run_id: int,
    append: bool,
) -> None:
    mode = "a" if append else "w"
    fieldnames = ["SampleID", "Split", "Target", "True", "Pred", "RunID"]
    sample_ids = cache["sample_ids"]

    with open(out_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append:
            writer.writeheader()

        for i, sample_id in enumerate(sample_ids):
            writer.writerow(
                {
                    "SampleID": sample_id,
                    "Split": split_name,
                    "Target": "porosity",
                    "True": float(cache["phi_true"][i]),
                    "Pred": float(y_pred["phi"][i]),
                    "RunID": run_id,
                }
            )
            writer.writerow(
                {
                    "SampleID": sample_id,
                    "Split": split_name,
                    "Target": "Sv",
                    "True": float(cache["Sv_true"][i]),
                    "Pred": float(y_pred["S_v"][i]),
                    "RunID": run_id,
                }
            )
            writer.writerow(
                {
                    "SampleID": sample_id,
                    "Split": split_name,
                    "Target": "fractal_dim",
                    "True": float(cache["D_true"][i]),
                    "Pred": float(y_pred["D"][i]),
                    "RunID": run_id,
                }
            )

def export_plot_csv_for_notebook(
    pred_long_csv: str,
    stability_csv: str,
    out_csv: str,
    model_name_map: Optional[Dict[str, str]] = None,
) -> None:
    """把逐样本预测结果和多 run 稳定性指标整理成 notebook 画图用 CSV。"""
    if model_name_map is None:
        model_name_map = {
            "porosity": "porosity",
            "Sv": "Sv",
            "fractal_dim": "fractal_dim",
        }

    fieldnames = [
        "Data_Type",
        "Model",
        "Split",
        "True_Yield",
        "Predicted_Yield",
        "Run_ID",
        "R2",
        "RMSE",
        "STD",
        "MAE",
    ]

    def map_model(target_name: str) -> str:
        return model_name_map.get(target_name, target_name)

    with open(out_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        if os.path.exists(pred_long_csv):
            with open(pred_long_csv, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    writer.writerow(
                        {
                            "Data_Type": "Prediction",
                            "Model": map_model(row.get("Target", "")),
                            "Split": row.get("Split", ""),
                            "True_Yield": row.get("True", ""),
                            "Predicted_Yield": row.get("Pred", ""),
                            "Run_ID": "",
                            "R2": "",
                            "RMSE": "",
                            "STD": "",
                            "MAE": "",
                        }
                    )

        if os.path.exists(stability_csv):
            with open(stability_csv, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    writer.writerow(
                        {
                            "Data_Type": "Stability",
                            "Model": map_model(row.get("Target", "")),
                            "Split": row.get("Split", ""),
                            "True_Yield": "",
                            "Predicted_Yield": "",
                            "Run_ID": row.get("RunID", ""),
                            "R2": row.get("R2", ""),
                            "RMSE": row.get("RMSE", ""),
                            "STD": row.get("STD", ""),
                            "MAE": row.get("MAE", ""),
                        }
                    )

    print(f"[OK] Exported notebook plot CSV to: {out_csv}")

# -----------------------------------------------------------------------------
# 8. 数据加载与实验主流程
# -----------------------------------------------------------------------------
def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """根据 args.loader 构建训练集与验证集 dataloader。"""
    if args.loader == "tiff":
        print(f"Loading TIFF data from {args.data_path}...")
        return get_tiff_dataloaders(
            args.data_path,
            batch_size=args.batch_size,
            dimension=args.dimension,
            num_workers=args.num_workers,
        )

    if get_binary_datamodule is None:
        raise ImportError("MEP-LDM datamodule is unavailable; use --loader tiff")

    cfg_dm = {
        "path": args.data_path,
        "dimension": args.dimension,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "cache_dir": args.cache_dir,
        "loader": args.loader,
        "val_split": args.val_split,
    }
    datamodule = get_binary_datamodule(args.data_path, cfg_dm, stride=None)
    datamodule.setup()
    return datamodule.train_dataloader(), datamodule.val_dataloader()

def run_one_calibration(
    train_cache: Dict[str, torch.Tensor | list[str]],
    val_cache: Dict[str, torch.Tensor | list[str]],
    args: argparse.Namespace,
) -> Tuple[nn.Module, nn.Module, Dict[str, torch.Tensor]]:
    """完成一次 S_v MLP 校准和 D MLP 校准。"""

    sv_mlp = train_sv_mlp_calibrator(
        train_cache["Sv_raw"],
        train_cache["Sv_true"],
        val_cache["Sv_raw"],
        val_cache["Sv_true"],
        epochs=args.sv_mlp_epochs,
        patience=args.sv_mlp_patience,
    )

    d_calib = train_d_calibrator(
        train_cache["D_soft_pore"],
        train_cache["D_true"],
        val_cache["D_soft_pore"],
        val_cache["D_true"],
        epochs=args.d_mlp_epochs,
        patience=args.d_mlp_patience,
    )

    y_pred_val = predict_from_cache(val_cache, sv_mlp, d_calib)

    return sv_mlp, d_calib, y_pred_val

def run_experiment(args: argparse.Namespace) -> None:
    train_loader, val_loader = build_dataloaders(args)
    os.makedirs(args.output_dir, exist_ok=True)

    surrogate_cfg = SurrogateConfig()
    train_cache = collect_eval_cache(train_loader, surrogate_cfg)
    val_cache = collect_eval_cache(val_loader, surrogate_cfg)

    export_run_id = min(args.export_run_id, args.num_runs - 1)
    stability_csv = os.path.join(args.output_dir, f"stability_{args.out_prefix}.csv")
    pred_long_csv = os.path.join(args.output_dir, f"predictions_long_{args.out_prefix}.csv")
    plot_csv = os.path.join(args.output_dir, f"predictions_{args.out_prefix}.csv")
    weights_path = os.path.join(
        args.output_dir,
        f"physical_property_predictor_{args.out_prefix}.pt",
    )

    first_write_done = False
    exported_pred_long = False

    for run_id in range(args.num_runs):
        seed = args.base_seed + run_id
        set_global_seed(seed)

        sv_mlp, d_calib, y_pred_val = run_one_calibration(
            train_cache,
            val_cache,
            args,
        )

        append_stability_rows(
            val_cache,
            y_pred_val,
            stability_csv,
            split_name="Test",
            run_id=run_id,
            append=first_write_done,
        )
        first_write_done = True

        if run_id == export_run_id and not exported_pred_long:
            y_pred_train = predict_from_cache(train_cache, sv_mlp, d_calib)
            export_predictions_long_from_cache(
                train_cache,
                y_pred_train,
                pred_long_csv,
                split_name="Train",
                run_id=run_id,
                append=False,
            )
            export_predictions_long_from_cache(
                val_cache,
                y_pred_val,
                pred_long_csv,
                split_name="Test",
                run_id=run_id,
                append=True,
            )
            model = PhysPropSurrogate(surrogate_cfg)
            model.sv_calib = sv_mlp
            model.d_calib = d_calib
            save_ppp(model, weights_path)
            exported_pred_long = True

        print(f"[Run {run_id:03d}] done. (seed={seed})")

    if exported_pred_long:
        export_plot_csv_for_notebook(pred_long_csv, stability_csv, plot_csv)

    print(f"[OK] Saved outputs with prefix: {args.out_prefix}")

# -----------------------------------------------------------------------------
# 9. 命令行入口
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Data and loader configuration.
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
    )
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--loader",
        type=str,
        default="tiff",
        choices=["eleven_sandstones", "porespy", "tiff"],
    )
    parser.add_argument("--val_split", type=float, default=0.3)

    # Multi-seed calibration stability configuration.
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--export_run_id", type=int, default=0)
    parser.add_argument("--out_prefix", type=str, default="property_predictor")
    parser.add_argument("--output_dir", type=str, default="outputs/property_predictor")

    # Calibrator training configuration.
    parser.add_argument("--sv_mlp_epochs", type=int, default=7000)
    parser.add_argument("--sv_mlp_patience", type=int, default=200)
    parser.add_argument("--d_mlp_epochs", type=int, default=7000)
    parser.add_argument("--d_mlp_patience", type=int, default=200)

    return parser

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
