from typing import Any

import yaml
import pathlib
import os

import torch

import mep_ldm.data
import mep_ldm.features
import mep_ldm.models
from .mep_ldm_trainer import MEPLDMTrainer
from .mep_ldm_vae_trainer import MEPLDMVAETrainer


KwargsType = dict[str, Any]
ConditionType = str | dict[str, torch.Tensor] | torch.Tensor


def _resolve_data_path(cfg: dict[str, Any],
                       data_path: str | pathlib.Path | None) -> str | pathlib.Path:
    if data_path is not None:
        return data_path
    configured_path = cfg.get('data', {}).get('path')
    if configured_path:
        return configured_path
    raise ValueError(
        "A dataset path is required. Pass --datapath on the command line "
        "or set data.path in the YAML configuration."
    )


def mep_ldm_train(cfg_path: str | pathlib.Path,
               data_path: str | pathlib.Path | None = None,
               checkpoint_path: str | pathlib.Path | None = None,
               fast_dev_run: bool = False,
               load_on_fit: bool = False
               ) -> MEPLDMTrainer:
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data_path = _resolve_data_path(cfg, data_path)
    datamodule = mep_ldm.data.get_binary_datamodule(data_path, cfg['data'])
    datamodule.setup()
    models = mep_ldm.models.get_model(cfg['model'])

    filename = os.path.basename(cfg_path)
    filename = filename.split('.')[0]
    basepath = pathlib.Path(cfg_path).parent.parent.parent
    folder = basepath/'savedmodels/experimental'/filename
    cfg['output']['folder'] = folder

    trainer = MEPLDMTrainer(
        models,
        cfg['training'],
        cfg['output'],
        load=checkpoint_path,
        fast_dev_run=fast_dev_run,
        load_on_fit=load_on_fit)
    trainer.train(datamodule)


def mep_ldm_vae_train(cfg_path, data_path=None, checkpoint_path=None, fast_dev_run=False):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data_path = _resolve_data_path(cfg, data_path)
    datamodule = mep_ldm.data.get_binary_datamodule(data_path, cfg['data'])
    datamodule.setup()
    filename = os.path.basename(cfg_path)
    filename = filename.split('.')[0]
    basepath = pathlib.Path(cfg_path).parent.parent.parent
    folder = basepath/'savedmodels/experimental'/filename
    cfg['output']['folder'] = folder

    trainer = MEPLDMVAETrainer(
        cfg['model'],
        cfg['training'],
        cfg['output'],
        cfg['data'],
        load=checkpoint_path,
        fast_dev_run=fast_dev_run)
    trainer.train(datamodule)


def mep_ldm_load(cfg_path, checkpoint_path, load_data=False, data_path=None, image_size: int | None = None):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    res = dict()
    models = mep_ldm.models.get_model(cfg['model'])
    trainer = MEPLDMTrainer(
        models,
        cfg['training'],
        cfg['output'],
        load=checkpoint_path,
        data_config=cfg['data'])
    res['trainer'] = trainer
    if load_data:
        data_path = _resolve_data_path(cfg, data_path)
        if image_size is not None:
            cfg['data']['image_size'] = image_size
        datamodule = mep_ldm.data.get_binary_datamodule(data_path, cfg['data'])
        datamodule.setup()
        res['datamodule'] = datamodule
    else:
        res['datamodule'] = None
    return res


def mep_ldm_vae_load(cfg_path, checkpoint_path, load_data=False, data_path=None, image_size=None):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    res = dict()
    trainer = MEPLDMVAETrainer(
        cfg['model'],
        cfg['training'],
        cfg['output'],
        cfg['data'],
        load=checkpoint_path)
    res['trainer'] = trainer
    if load_data:
        data_path = _resolve_data_path(cfg, data_path)
        datamodule = mep_ldm.data.get_binary_datamodule(data_path, cfg['data'])
        if image_size is not None:
            datamodule.cfg['image_size'] = image_size
        datamodule.setup()
        res['datamodule'] = datamodule
    else:
        res['datamodule'] = None
    return res
