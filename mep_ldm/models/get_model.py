from typing import Any

import mep_ldm_core.models
import mep_ldm_core.models.nets.autoencoderldm3d


from . import embedder


def get_conditional_embedding(conditional_embedding=None,
                              conditional_embedding_args=None,
                              dembed=64):
    if conditional_embedding_args is None:
        conditional_embedding_args = {}
    if conditional_embedding is None:
        return None
    elif isinstance(conditional_embedding, str):
        return get_single_embedding(conditional_embedding,
                                    conditional_embedding_args,
                                    dembed)
    elif isinstance(conditional_embedding, list):
        embedders = []
        for embedding_type in conditional_embedding:
            args = conditional_embedding_args.get(embedding_type, {})
            embedders.append(get_single_embedding(embedding_type, args, dembed))
        return embedder.CompositeEmbedder(embedders)
    else:
        raise ValueError(f"Unsupported conditional_embedding type: {conditional_embedding}")


def get_single_embedding(embedding_type, embedding_kwargs, dembed):
    # Get the embedding class from its name
    # From the embedder module
    embedding_fn = getattr(embedder, f'get_{embedding_type}', None)
    if embedding_fn is None:
        raise ValueError(f"Embedding type {embedding_type} not found")

    # Create the embedding instance
    embed = embedding_fn(dembed, **embedding_kwargs)
    return embed


def get_model(cfg: dict[str, Any]) -> dict[str, Any]:
    """
        Returns a dict with keys 'model' and 'autoencoder'.
        'model' contains a PUNetG or PUNetGCond model
        'autoencoder' contains an autoencoder model or None
    """
    model_type = cfg['type']
    items = dict()
    if model_type == 'PUNetG':

        # Create PUNetGConfig
        config_params = cfg.get('config', {})
        punetg_config = mep_ldm_core.models.PUNetGConfig(**config_params)

        # Create PUNetG
        model_params = cfg.get('params', {})
        conditional_embedding = model_params.pop('conditional_embedding', None)  # noqa: F841
        conditional_embedding_kwargs = model_params.pop('conditional_embedding_kwargs', None)  # noqa: F841
        channel_conditional_items = model_params.pop('channel_conditional_items', None)  # noqa: F841
        dembed = config_params.get('model_channels', 64)
        embed = get_conditional_embedding(conditional_embedding,
                                          conditional_embedding_kwargs,
                                          dembed)
        channel_conditional_items = model_params.pop('channel_conditional_items', None)

        if channel_conditional_items:
            raise NotImplementedError("Channel conditional items are not implemented in get_model")
            model = mep_ldm_core.models.PUNetGCond(punetg_config,
                                              conditional_embedding=embed,
                                              channel_conditional_items=channel_conditional_items,
                                              **model_params)
        else:
            model = mep_ldm_core.models.PUNetG(punetg_config,
                                          conditional_embedding=embed,
                                          **model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    items['model'] = model

    # Load autoencoder
    autoencoder_cfg = cfg.get('autoencoder', {})
    if autoencoder_cfg:
        autoencoder_type = autoencoder_cfg['type']
        if autoencoder_type == 'AutoencoderKL':
            checkpoint_path = autoencoder_cfg.get('checkpoint_path')
            if not checkpoint_path:
                raise ValueError("model.autoencoder.checkpoint_path must be set to a local checkpoint path.")
            lossconfig = mep_ldm_core.models.nets.autoencoderldm3d.lossconfig(
                kl_weight=autoencoder_cfg.get('kl_weight', 1e-4)
            )
            # ddconfig = mep_ldm_core.models.nets.autoencoderldm3d.ddconfig(
            #     resolution=autoencoder_cfg['resolution'],
            #     has_mid_attn=autoencoder_cfg.get('has_mid_attn', True)
            # )
            ddconfig_params = autoencoder_cfg.get('config', {})
            vae_config = mep_ldm_core.models.nets.autoencoderldm3d.ddconfig(**ddconfig_params)
            # ddconfig = mep_ldm_core.models.nets.autoencoderldm3d.ddconfig(
            #     ch=autoencoder_cfg.get('ch', 64),
            #     in_channels=autoencoder_cfg.get('in_channels', 1),
            #     out_ch=autoencoder_cfg.get('out_ch', 1),
            #     ch_mult=autoencoder_cfg.get('ch_mult', [1, 2, 2]),
            #     num_res_blocks=autoencoder_cfg.get('num_res_blocks', 2),
            #     attn_resolutions=autoencoder_cfg.get('attn_resolutions', []),
            #     resolution=autoencoder_cfg['resolution'],
            #     z_channels=autoencoder_cfg.get('z_channels', 4),
            #     double_z=autoencoder_cfg.get('double_z', True),
            #     dropout=autoencoder_cfg.get('dropout', 0.0),
            #     has_mid_attn=autoencoder_cfg.get('has_mid_attn', True)
            # )
            vae_module = mep_ldm_core.models.nets.autoencoderldm3d.AutoencoderKL.load_from_checkpoint(
                checkpoint_path,
                ddconfig=vae_config,
                lossconfig=lossconfig
            )
            vae_module.eval()
        else:
            raise ValueError(f"Unsupported autoencoder type: {autoencoder_type}")
    else:
        vae_module = None

    items['autoencoder'] = vae_module
    return items


def get_autoencoder(config: dict[str, Any]):
    """
    Load an autoencoder model based on the provided configuration.
    
    Args:
        config: Configuration dictionary for the autoencoder
        
    Returns:
        dict: Dictionary containing the autoencoder model
    """
    items = {}
    
    if config:
        autoencoder_type = config['type']
        if autoencoder_type == 'AutoencoderKL':
            checkpoint_path = config.get('checkpoint_path')
            if not checkpoint_path:
                raise ValueError("autoencoder.checkpoint_path must be set to a local checkpoint path.")
            lossconfig = mep_ldm_core.models.nets.autoencoderldm3d.lossconfig(
                kl_weight=config.get('kl_weight', 1e-4)
            )
            ddconfig = mep_ldm_core.models.nets.autoencoderldm3d.ddconfig(
                resolution=config['resolution'],
                has_mid_attn=config.get('has_mid_attn', False)
            )
            vae_module = mep_ldm_core.models.nets.autoencoderldm3d.AutoencoderKL.load_from_checkpoint(
                checkpoint_path,
                ddconfig=ddconfig,
                lossconfig=lossconfig
            )
            vae_module.eval()
        else:
            raise ValueError(f"Unsupported autoencoder type: {autoencoder_type}")
    else:
        vae_module = None

    items['autoencoder'] = vae_module
    return items
