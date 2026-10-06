# flake8: noqa

from .binary_datasets import (load_binary_from_eleven_sandstones,
                              load_porespy_generated,
                              get_standard_binary_transforms,
                              VoxelToSlicesDataset,
                              SequenceOfVoxelsToSlicesDataset,
                              VoxelToSubvoxelDataset,
                              SequenceOfVoxelsToSubvoxelDataset,
                              VoxelToSubvoxelSequentialDataset)

try:
    from .binary_datamodule import get_binary_datamodule
except ModuleNotFoundError as exc:
    if exc.name != "lightning":
        raise

    def get_binary_datamodule(*args, **kwargs):
        raise ModuleNotFoundError(
            "lightning is required to use get_binary_datamodule"
        ) from exc
