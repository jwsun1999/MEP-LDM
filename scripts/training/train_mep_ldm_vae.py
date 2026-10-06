import pathlib
import click

import mep_ldm.features
import mep_ldm.models
import mep_ldm.trainers


@click.command()
@click.option('--cfgpath', type=click.Path(exists=True), required=True,
              help='Path to the YAML configuration file.')
@click.option('--datapath', type=click.Path(exists=True), default=None,
              help='Path to the data file (e.g., Estaillades_1000c_3p31136um.raw). '
                   'If not provided, the data will be loaded from the configuration file.')
@click.option('--checkpoint_path', type=str, default='best',
              help='Path to checkpoint for resuming training. '
                   'Use "best" for the best checkpoint, or provide a specific path. Default is None.')
@click.option('--fast_dev_run', is_flag=True, default=False,
              help='If set, disables saving and trains only a few steps.')
def train(datapath, cfgpath, checkpoint_path, fast_dev_run):
    """
    Train a pore generation model using the specified data and configuration.
    """
    if datapath is not None:
        datapath = pathlib.Path(datapath)
    cfgpath = pathlib.Path(cfgpath)

    # Run training
    mep_ldm.trainers.mep_ldm_vae_train(cfgpath,
                                    datapath,
                                    checkpoint_path=checkpoint_path,
                                    fast_dev_run=fast_dev_run)

    click.echo("Training completed.")


if __name__ == '__main__':
    train()
