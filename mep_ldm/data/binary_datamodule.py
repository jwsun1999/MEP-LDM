from typing import Any

from torch.utils.data import DataLoader
import lightning as L
from pathlib import Path
from .binary_datasets import (VoxelToSlicesDataset, VoxelToSubvoxelDataset,
                              SequenceOfVoxelsToSlicesDataset, SequenceOfVoxelsToSubvoxelDataset,
                              VoxelToSubvoxelSequentialDataset,
                              load_binary_from_eleven_sandstones, load_porespy_generated)
from mep_ldm.features import feature_extractors
import os
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple


# def _collate_only_x(batch):
#     # batch: list of x or list of (x, y)
#     if isinstance(batch[0], (tuple, list)):
#         xs = [b[0] for b in batch]
#     else:
#         xs = batch
#     return torch.stack(xs, dim=0)

class BinaryVoxelDataModule(L.LightningDataModule):
    def __init__(self, data_path: str | Path | list[str | Path] = '',
                 cfg: dict[str, Any] = {},
                 stride: None | int = None):
        super().__init__()
        self.cfg = cfg
        if data_path == '':
            data_path = cfg.get('path', '')
            if data_path == '':
                raise ValueError("data_path or cfg.path must be provided")
        self.data_path = data_path
        # The default here comes from eleven sandstones
        self.voxel_size_um = (self.cfg.get("voxel_size_um", 2.25) *
                              self.cfg.get("voxel_downscale_factor", 1))
        self.stride = stride

    def setup(self, stage=None):
        cache_dir = self.cfg.get("cache_dir", None)
        if cache_dir:
            meta_path = os.path.join(cache_dir, "meta.json")
            if not os.path.exists(meta_path):
                print(f"[INFO] No cache found in {cache_dir}, building cache from raw voxels...")
                voxels = self.load_voxels()
                build_disk_cache(voxels, self.cfg)  # Build the cache from raw voxels.
            else:
                print(f"[INFO] Loading preprocessed dataset from {cache_dir}")

            # Use the .npy-backed dataset. Add a raw-backed dataset here if needed.
            feature_extractor = self.get_feature_extractor()
            common_kwargs = dict(
                center=self.cfg.get('center', False),
                invert=self.cfg.get('invert', False),
                feature_extractor=feature_extractor,
                transform=self.cfg.get('transform', None),
            )
            self.train_dataset = CachedVoxelNPYDataset(cache_dir, split="train", **common_kwargs)
            self.val_dataset   = CachedVoxelNPYDataset(cache_dir, split="val",   **common_kwargs)
            return

        # Fallback path when no cache_dir is configured.
        print("[INFO] No cache_dir provided; generating on-the-fly dataset from raw voxels...")
        voxels = self.load_voxels()
        if isinstance(voxels, list):
            dataset_class = (SequenceOfVoxelsToSubvoxelDataset
                             if self.cfg['dimension'] == 3
                             else SequenceOfVoxelsToSlicesDataset)
        else:
            dataset_class = (VoxelToSubvoxelDataset
                             if self.cfg['dimension'] == 3
                             else VoxelToSlicesDataset)

        feature_extractor = self.get_feature_extractor()

        if self.stride is not None:
            dataset_args = {'subslice': self.cfg['image_size']}
            self.train_dataset = VoxelToSubvoxelSequentialDataset(self.stride, voxels,
                                        dataset_size=self.cfg['training_dataset_size'],
                                        **dataset_args)
            self.val_dataset   = VoxelToSubvoxelSequentialDataset(self.stride, voxels,
                                        dataset_size=self.cfg['validation_dataset_size'],
                                        **dataset_args)
        else:
            dataset_args = {
                'subslice': self.cfg['image_size'],
                'voxel_downscale_factor': self.cfg['voxel_downscale_factor'],
                'feature_extractor': feature_extractor,
                'center': self.cfg.get('center', False),
                'invert': self.cfg.get('invert', False),
                'transform': self.cfg.get('transform', None),
            }
            self.train_dataset = dataset_class(voxels, dataset_size=self.cfg['training_dataset_size'], **dataset_args)
            self.val_dataset   = dataset_class(voxels, dataset_size=self.cfg['validation_dataset_size'], **dataset_args)

    def load_voxels(self):
        loader = self.cfg.get('loader', 'eleven_sandstones')
        if isinstance(self.data_path, (str, Path)):
            if loader == 'eleven_sandstones':
                return load_binary_from_eleven_sandstones(self.data_path)
            elif loader == 'porespy':
                return load_porespy_generated(self.data_path)
        elif isinstance(self.data_path, list):
            return [self.load_single_voxel(path) for path in self.data_path]
        raise ValueError(f"Unsupported data path or loader: {self.data_path}, {loader}")

    def load_single_voxel(self, path):
        loader = self.cfg.get('loader', 'eleven_sandstones')
        if loader == 'eleven_sandstones':
            return load_binary_from_eleven_sandstones(path)
        elif loader == 'porespy':
            return load_porespy_generated(path)
        raise ValueError(f"Unsupported loader: {loader}")

    def get_feature_extractor(self):
        feature_config = self.cfg.get('feature_extractor')
        if not feature_config:
            return None
        if isinstance(feature_config, str):
            extractor_kwargs = self.cfg.get('feature_extractor_kwargs', {})
            return feature_extractors.make_feature_extractor(
                feature_config,
                **extractor_kwargs
            )
        elif isinstance(feature_config, list):
            extractor_names = feature_config
            extractor_kwargs = self.cfg.get('feature_extractor_kwargs', {})
            return feature_extractors.make_composite_feature_extractor(
                extractor_names, extractor_kwargs)
        else:
            raise ValueError(f"Unsupported feature extractor configuration: {feature_config}")



    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg['batch_size'],
                          shuffle=True,
                          num_workers=self.cfg['num_workers'],
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg['batch_size'],
                          shuffle=False,
                          num_workers=self.cfg['num_workers'],
                          persistent_workers=True)


def get_binary_datamodule(data_path: str | Path, cfg: dict[str, Any],
                          stride: None | int = None) -> BinaryVoxelDataModule:  # noqa: C901
    return BinaryVoxelDataModule(data_path, cfg, stride)

class CachedVoxelNPYDataset(Dataset):
    """
    Read sharded .npy files from cache_dir/{split} with lazy memmap loading.
    Directory layout:
      cache_dir/
        meta.json
        train/
          shard_000.npy
          shard_001.npy
          ...
        val/
          shard_000.npy
          ...
    Each .npy shape: (N_i, 1, D, H, W) or (N_i, 1, H, W).
    dtype: uint8 or float32. uint8 is preferred for binary {0, 1} data.
    """
    def __init__(self, cache_dir: str, split: str = "train",
                 center: bool = False, invert: bool = False,
                 feature_extractor=None, transform=None):
        super().__init__()
        self.split = split
        self.cache_dir = os.path.join(cache_dir, split)
        meta_path = os.path.join(cache_dir, "meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        assert meta["format"] == "npy", "Cached format mismatch: expected npy"
        self.shape_per_item = tuple(meta["shape_per_item"])  # e.g. [1, 64, 64, 64]
        self.dtype = np.dtype(meta["dtype"])
        self.files = [os.path.join(self.cache_dir, f)
                      for f in sorted(os.listdir(self.cache_dir))
                      if f.endswith(".npy")]
        assert len(self.files) > 0, f"No cached .npy files found in {self.cache_dir}"

        self.feat_paths: List[str|None] = []
        for p in self.files:
            base = os.path.splitext(p)[0]
            featp = base + ".feat.npz"
            self.feat_paths.append(featp if os.path.exists(featp) else None)

        # Preload shard lengths without reading the full arrays into memory.
        self.shard_sizes = []
        self.mmaps = []  # Lazy-open each shard on first access.
        self.feat_handles: List[object|None|bool] = []
        total = 0
        # for path in self.files:
        for j, path in enumerate(self.files):
            # Read only the header to get the shape.
            arr = np.load(path, mmap_mode='r')
            n = arr.shape[0]
            self.shard_sizes.append(n)
            self.mmaps.append(None)
            if self.feat_paths[j] is None:
                self.feat_handles.append(False)
            else:
                self.feat_handles.append(None)
            total += n
        self.total = total

        # Post-processing options.
        self.center = center
        self.invert = invert
        self.feature_extractor = feature_extractor
        self.transform = transform

        # Cumulative offsets used for global-to-shard index mapping.
        self.offsets = []
        s = 0
        for n in self.shard_sizes:
            self.offsets.append((s, s + n))
            s += n

    def __len__(self):
        return self.total

    def _ensure_mmap(self, shard_idx):
        if self.mmaps[shard_idx] is None:
            self.mmaps[shard_idx] = np.load(self.files[shard_idx], mmap_mode='r')
        return self.mmaps[shard_idx]

    def _ensure_feat(self, shard_idx):
        """Open .feat.npz if present and return a dict-like handle or None."""
        h = self.feat_handles[shard_idx]
        if h is False:
            return None
        if h is None:
            self.feat_handles[shard_idx] = np.load(self.feat_paths[shard_idx])
        return self.feat_handles[shard_idx]

    def __getitem__(self, idx):
        # Locate the shard that contains this global index.
        for i, (st, ed) in enumerate(self.offsets):
            if st <= idx < ed:
                local_idx = idx - st
                arr = self._ensure_mmap(i)
                x = arr[local_idx]  # np.ndarray, shape like (1, D, H, W) or (1, H, W)
                # Convert to torch and apply post-processing.
                x = torch.from_numpy(x.astype(np.float32))
                if self.invert:
                    x = 1.0 - x
                if self.center:
                    x = 2.0 * x - 1.0
                if self.transform:
                    x = self.transform(x)

                # 1. Prefer precomputed offline features when present.
                fh = self._ensure_feat(i)
                if fh is not None:
                    y = {k: torch.from_numpy(fh[k][local_idx]) for k in fh.files}
                    return x,y
                # 2. Otherwise compute features online.
                if self.feature_extractor:
                    y = self.feature_extractor(x)
                    return x,y
                return x
        raise IndexError("Index out of range")

# --------------------------------------
# Sliding-window patch extraction and disk-cache construction.
# --------------------------------------
def _iter_sliding_starts(vol_shape, subshape, stride):
    # Supports both 3D and 2D volumes.
    starts_axes = [list(range(0, dim - sub + 1, stride))
                   for dim, sub in zip(vol_shape, subshape)]
    if len(starts_axes) == 3:
        for z in starts_axes[0]:
            for y in starts_axes[1]:
                for x in starts_axes[2]:
                    yield (z, y, x)
    else:
        for y in starts_axes[0]:
            for x in starts_axes[1]:
                yield (y, x)

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def build_disk_cache(voxels, cfg: dict):
    """
    Build a local npy/raw cache from cfg and write meta.json.
    Key configuration:
      cfg['image_size']           # Patch size, either (D, H, W) or int.
      cfg['dimension']            # 2 or 3.
      cfg['cache_dir']            # Output root directory.
      cfg['cache_format']         # 'npy' or 'raw'; defaults to 'npy'.
      cfg['cache_stride']         # Sliding-window stride; defaults to image_size // 2.
      cfg['val_split']            # Validation split ratio; defaults to 0.1.
      cfg['shard_size']           # Number of patches per shard; defaults to 1024.
      cfg['voxel_downscale_factor'], cfg['invert'], cfg['center'], and cfg['transform']
          are applied only when reading for training. Cache raw binary {0, 1} data when possible.
    """
    cache_dir = cfg["cache_dir"]
    fmt = cfg.get("cache_format", "npy")  # 'npy' | 'raw'
    val_split = float(cfg.get("val_split", 0.1))
    shard_size = int(cfg.get("shard_size", 1024))
    dim = int(cfg["dimension"])
    subslice = cfg["image_size"]
    if isinstance(subslice, int):
        subshape = (subslice, subslice, subslice) if dim == 3 else (subslice, subslice)
    else:
        subshape = tuple(subslice)
    stride = int(cfg.get("cache_stride", subshape[-1] // 2))

    _ensure_dir(cache_dir)
    train_dir = os.path.join(cache_dir, "train")
    val_dir = os.path.join(cache_dir, "val")
    _ensure_dir(train_dir)
    _ensure_dir(val_dir)

    # Build a feature extractor using the same configuration logic as get_feature_extractor.
    feat_cfg = cfg.get('feature_extractor', None)
    feat_kwargs = cfg.get('feature_extractor_kwargs', {})
    feature_extractor = None
    feature_names: List[str] = []
    feature_shapes: Dict[str, List[int]] = {}
    if feat_cfg:
        if isinstance(feat_cfg, str):
            feature_extractor = feature_extractors.make_feature_extractor(feat_cfg, **feat_kwargs)
            # Expected feature keys.
            feature_names = feature_extractors.EXTRACTORS_RETURN_KEYS_MAP.get(feat_cfg, [])
        elif isinstance(feat_cfg, list):
            feature_extractor = feature_extractors.make_composite_feature_extractor(feat_cfg, feat_kwargs)
            # Merge feature keys.
            for name in feat_cfg:
                feature_names.extend(feature_extractors.EXTRACTORS_RETURN_KEYS_MAP.get(name, []))
        else:
            raise ValueError(f"Unsupported feature_extractor in cfg: {feat_cfg}")
        # Whether to compute and save features during cache construction.
        cache_save_features = bool(cfg.get('cache_save_features', True))
    else:
        cache_save_features = False


    # Normalize voxels to a list.
    vox_list = voxels if isinstance(voxels, list) else [voxels]

    # Store patch indices instead of patch arrays to keep memory usage low.
    indices = []  # [(voxel_idx, (start...)), ...]
    for v_idx, v in enumerate(vox_list):
        shape = v.shape  # (D,H,W) or (H,W)
        for starts in _iter_sliding_starts(shape, subshape, stride):
            indices.append((v_idx, starts))

    # Shuffle and split.
    rng = np.random.default_rng(1234)
    rng.shuffle(indices)
    n_total = len(indices)
    n_val = int(n_total * val_split)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    def _dump_shards(split_indices, out_dir):
        n = len(split_indices)
        n_shards = math.ceil(n / shard_size)
        written = 0
        for si in range(n_shards):
            chunk = split_indices[si*shard_size:(si+1)*shard_size]
            batch = []
            # If features are cached, keep one accumulator list per key.
            feat_accum: Dict[str, List[np.ndarray]] = {k: [] for k in
                                                       feature_names} if cache_save_features and feature_extractor else {}
            for (v_idx, starts) in chunk:
                v = vox_list[v_idx]
                if dim == 3:
                    z, y, x = starts
                    d, h, w = subshape
                    crop = v[z:z+d, y:y+h, x:x+w][None, ...]  # (1, D, H, W)
                else:
                    y, x = starts
                    h, w = subshape
                    crop = v[y:y+h, x:x+w][None, ...]         # (1, H, W)
                # Store binary 0/1 data compactly as uint8.
                batch.append(crop.astype(np.uint8))
                # Compute and accumulate features.
                if cache_save_features and feature_extractor:
                    xt = torch.from_numpy(crop.astype(np.float32))
                    feats = feature_extractor(xt)  # dict[str, tensor]
                    # Record feature shapes from the first sample.
                    if not feature_shapes:
                        for k in feature_names:
                            val = feats[k]
                            if torch.is_tensor(val):
                                vnp0 = val.detach().cpu().numpy()
                            else:
                                vnp0 = np.asarray(val)
                            if vnp0.ndim == 0:
                                vnp0 = vnp0.reshape(1)
                            feature_shapes[k] = list(vnp0.shape)
                    for k in feature_names:
                        val = feats[k]
                        if torch.is_tensor(val):
                            vnp = val.detach().cpu().numpy()
                        else:
                            vnp = np.asarray(val)

                        # Ensure at least 1D so stack/concat works for scalar features.
                        if vnp.ndim == 0:
                            vnp = vnp.reshape(1)

                        # Use a consistent dtype across cached features.
                        vnp = vnp.astype(np.float32, copy=False)

                        feat_accum[k].append(vnp)
            if len(batch) == 0:
                continue
            arr = np.stack(batch, axis=0)  # (N_i, 1, ...)
            if fmt == "npy":
                np.save(os.path.join(out_dir, f"shard_{si:03d}.npy"), arr)
                if cache_save_features and feature_extractor:
                    # Stack each key as (N_i, *key_shape).
                    save_dict = {k: np.stack(vs, axis=0) for k, vs in feat_accum.items()}
                    np.savez_compressed(os.path.join(out_dir, f"shard_{si:03d}.feat.npz"), **save_dict)
            elif fmt == "raw":
                raw_path = os.path.join(out_dir, f"shard_{si:03d}.raw")
                arr.tofile(raw_path)
                # Save a matching .shape.json file.
                with open(os.path.join(out_dir, f"shard_{si:03d}.shape.json"), "w") as f:
                    json.dump({"shape": list(arr.shape), "dtype": "uint8"}, f)
            written += arr.shape[0]
        return n

    n_train = _dump_shards(train_indices, train_dir)
    n_val = _dump_shards(val_indices, val_dir)

    # Write meta.json.
    sample_shape = (1,) + tuple(subshape)
    meta = {
        "format": fmt,
        "dtype": "uint8",
        "shape_per_item": list(sample_shape),
        "dimension": dim,
        "feature_names": feature_names,
        "feature_shapes": feature_shapes,
        "train_count": n_train,
        "val_count": n_val,
        "stride": stride,
        "val_split": val_split,
        "shard_size": shard_size
    }
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
