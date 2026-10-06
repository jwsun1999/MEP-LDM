# test_property_predictor.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import json
from mep_ldm.models.ppp import load_surrogate_model
from typing import Dict, Optional, Tuple
import numpy as np
from torch.utils.data import DataLoader
import math  # Used by the fractal dimension estimator.

# ------------------------------
# 1. Core model definitions
# ------------------------------




def probs_to_logits(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log(1.0 - x)

def get_logits_from_batch(batch) -> torch.Tensor:
    x = batch
    assert x.ndim == 5 and x.shape[1] == 1, f"Expect x [B,1,Z,Y,X], got {tuple(x.shape)}"
    return probs_to_logits(x)
def _get_logits_from_batch(batch) -> torch.Tensor:
    x,_ = batch
    assert x.ndim == 5 and x.shape[1] == 1, f"Expect x [B,1,Z,Y,X], got {tuple(x.shape)}"
    return probs_to_logits(x)

def _detect_truth_mapping_from_keys(keys) -> Dict[str,str]:
    lk = {k.lower(): k for k in keys}
    m = {"phi": None, "S_v": None, "D": None}
    for c in ("phi","porosity"):
        if c in lk: m["phi"] = lk[c]; break
    for c in ("s_v","sv","surface_area_density","specific_surface","specific_surface_area"):
        if c in lk: m["S_v"] = lk[c]; break
    for c in ("d","fractal_dim","fractal_dimension"):
        if c in lk: m["D"] = lk[c]; break
    return m

def get_truth_from_batch(batch) -> Dict[str, Optional[torch.Tensor]]:
    x, y = batch
    assert isinstance(y, dict), "Expect (x, y_dict) from dataloader."
    mapping = _detect_truth_mapping_from_keys(y.keys())
    out = {}
    for std_key in ["phi","S_v","D"]:
        realk = mapping.get(std_key, None)
        if realk is None:
            out[std_key] = None
        else:
            t = y[realk]
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t, dtype=torch.float32)
            if t.ndim == 0:
                t = t[None]
            out[std_key] = t.float()
    return out



# ------------------------------
# 2. Evaluation metrics
# ------------------------------

def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    y_true = y_true.double().flatten()
    y_pred = y_pred.double().flatten()
    mae = (y_true - y_pred).abs().mean().item()
    rmse = torch.sqrt(((y_true - y_pred)**2).mean()).item()
    ss_res = ((y_true - y_pred)**2).sum()
    ss_tot = ((y_true - y_true.mean())**2).sum().clamp(min=1e-12)
    r2 = 1.0 - (ss_res / ss_tot).item()
    mape = ((y_true - y_pred).abs() / y_true.abs().clamp(min=1e-12)).mean().item()
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

# ------------------------------
# 3. Model loading
# ------------------------------


# ------------------------------
# 4. Evaluation entry point
# ------------------------------

@torch.no_grad()
def evaluate_model(
    checkpoint_path: str,
    data_path: str,
    cache_dir: str | None = None,
    val_split: float = 0.0,
    batch_size: int = 2,
    image_size: int = 128,
    dimension: int = 3,
    loader: str = "eleven_sandstones",
    num_workers: int = 4
):
    # Load the surrogate model.
    model = load_surrogate_model(checkpoint_path)

    # Build the datamodule configuration.
    cfg_dm = {
        "path": data_path,
        "dimension": dimension,
        "image_size": image_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "loader": loader,
        "val_split": val_split,
    }

    if cache_dir:
        cfg_dm["cache_dir"] = cache_dir
        cfg_dm["cache_format"] = "npy"
        cfg_dm["cache_save_features"] = True

    # Create the datamodule.
    try:
        from mep_ldm.data.binary_datamodule import get_binary_datamodule
        dm = get_binary_datamodule(data_path, cfg_dm, stride=None)
        dm.setup(stage='test')
        dataloader = dm.val_dataloader()
    except ImportError as e:
        raise ImportError(f"Could not import get_binary_datamodule: {e}\nCheck the package path or PYTHONPATH.")

    # Run inference and metric evaluation.
    pred = {"phi": [], "S_v": [], "D": []}
    gt = {"phi": [], "S_v": [], "D": []}

    print("Starting inference...")
    device = next(model.parameters()).device

    for batch in dataloader:
        values = batch[0].to(device).float()
        out = model(values, input_type="solid_probability")
        truth = get_truth_from_batch(batch)

        for k in pred.keys():
            if truth[k] is not None:
                pred[k].append(out[k].cpu())
                gt[k].append(truth[k].cpu())

    # Aggregate metric results.
    report = {}
    for k in pred.keys():
        if len(pred[k]) > 0:
            y_pred = torch.cat(pred[k], 0)
            y_true = torch.cat(gt[k], 0)
            report[k] = regression_metrics(y_true, y_pred)
            print(f"\n{k.upper()} prediction metrics:")
            for metric, value in report[k].items():
                print(f"  {metric}: {value:.6f}")

    # Save the evaluation report.
    with open("inference_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved evaluation report to inference_report.json")

    return report

# ------------------------------
# 5. Command line interface
# ------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a physical-property surrogate model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained surrogate checkpoint.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the source data or cached dataset.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional cache directory.")
    parser.add_argument("--val_split", type=float, default=0.0, help="Validation split used when building a cache.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--loader", type=str, default="eleven_sandstones", choices=["eleven_sandstones", "porespy"])
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        cache_dir=args.cache_dir,
        val_split=args.val_split,
        batch_size=args.batch_size,
        image_size=args.image_size,
        dimension=args.dimension,
        loader=args.loader,
        num_workers=args.num_workers
    )
