from __future__ import annotations

from typing import Any, Callable, Literal
import functools
import numpy as np
import torch
import porespy as ps
from . import surface_area
KwargsType = dict[str, Any]


AVAILABLE_EXTRACTORS = [
    "porosity",
    "surface_area_density_from_slice",
    "surface_area_density_from_voxel",
    "surface_area_density_from_voxel_slice",
    "fractal_dimension_porespy_from_voxel",
]


EXTRACTORS_RETURN_KEYS_MAP = {
    "porosity": ["porosity"],
    "surface_area_density_from_slice": ["surface_area_density"],
    "surface_area_density_from_voxel": ["surface_area_density"],
    "surface_area_density_from_voxel_slice": ["surface_area_density"],
    "fractal_dimension_porespy_from_voxel": ["fractal_dimension"],
}


def extract_surface_area_density_base(data, voxel_size: float = 1.0):
    # Use the first channel and convert solid occupancy to pore occupancy.
    data_np = (1 - data[0].long()).numpy()
    surface_areas = surface_area.region_surface_areas(data_np, voxel_size=voxel_size)

    pore_surface_area = np.sum(surface_areas)
    pore_voxels = (data_np > 0.5).sum()
    sa_value = pore_surface_area / pore_voxels
    sa = torch.tensor([sa_value], dtype=torch.float)

    return {
        "surface_area_density": sa,
    }


def extract_surface_area_density_from_slice(data, voxel_size: float = 1.0):
    return extract_surface_area_density_base(data, voxel_size=voxel_size)


def extract_surface_area_density_from_voxel(data, voxel_size: float = 1.0):
    return extract_surface_area_density_base(data, voxel_size=voxel_size)


def extract_surface_area_density_from_voxel_slice(data,
                                                  voxel_size: float = 1.0,
                                                  axis: int = 0):
    center = data.shape[axis + 1] // 2
    slice_data = data.select(axis + 1, center).unsqueeze(0)
    return extract_surface_area_density_base(slice_data, voxel_size=voxel_size)


def extract_fractal_dimension_porespy_from_voxel(data, bins: int = 10):
    # Original convention: 0 is matrix and 1 is pore.
    im = data[0] == 0

    if isinstance(bins, int):
        min_size = 2
        max_size = min(im.shape) // 2
        if max_size < min_size:
            max_size = min_size * 2
        box_sizes = np.round(np.logspace(np.log10(min_size), np.log10(max_size), bins)).astype(int)
        box_sizes = np.unique(box_sizes)
    else:
        box_sizes = np.asarray(bins)

    bc = ps.metrics.boxcount(im=im, bins=box_sizes)
    mask = np.array(bc.count) > 0
    x = np.log(np.array(bc.size)[mask])
    y = np.log(np.array(bc.count)[mask])
    global_slope, _ = np.polyfit(x, y, 1)
    Df = -global_slope
    Df = round(Df, 2)
    fractal_dimension = torch.tensor([Df], dtype=torch.float32)

    return {
        "fractal_dimension": fractal_dimension,
    }


def extract_porosity(data):
    porosity = ps.metrics.porosity(1 - data)
    return {"porosity": porosity}


def composite_feature_extractor(x, extractors: list[Callable]):
    """Run multiple extractors and merge their outputs."""
    data = {}
    for extractor in extractors:
        data.update(extractor(x))
    return data


def extract_composite(extractors: list[Callable]):
    return functools.partial(composite_feature_extractor, extractors=extractors)


def make_feature_extractor(extractor_name: str,
                           input_type: Literal["binary", "edt"] = "binary",
                           **kwargs):
    if extractor_name not in AVAILABLE_EXTRACTORS:
        available = ", ".join(AVAILABLE_EXTRACTORS)
        raise ValueError(f"Extractor {extractor_name} not available. Available: {available}")

    extractor_fn = globals()[f"extract_{extractor_name}"]
    extractor_fn_partial = functools.partial(extractor_fn, **kwargs)
    if input_type == "edt":
        return lambda x: extractor_fn_partial(x > 0)
    return extractor_fn_partial


def make_composite_feature_extractor(extractor_names: list[str],
                                     extractor_kwargs: dict[str, KwargsType] | None = None):
    if extractor_kwargs is None:
        extractor_kwargs = {}

    extractors = []
    for name in extractor_names:
        args = extractor_kwargs.get(name, {})
        extractors.append(make_feature_extractor(name, **args))
    return extract_composite(extractors)
