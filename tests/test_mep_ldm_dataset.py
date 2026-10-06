import numpy as np
import torch

import mep_ldm.data


def test_mep_ldm_dataset():
    voxel = np.random.randn(100, 100, 100) > 0

    dataset = mep_ldm.data.VoxelToSubvoxelDataset(voxel, subslice=10)
    assert (dataset[0].shape == torch.Size([1, 10, 10, 10]))

    dataset = mep_ldm.data.VoxelToSubvoxelDataset(voxel, subslice=[10, 20, 30])
    assert (dataset[0].shape == torch.Size([1, 10, 20, 30]))

    dataset = mep_ldm.data.VoxelToSlicesDataset(voxel, image_size=10)
    assert (dataset[0].shape == torch.Size([1, 10, 10]))

    dataset = mep_ldm.data.SequenceOfVoxelsToSlicesDataset([voxel, voxel],
                                                           image_size=10)
    assert (dataset[0].shape == torch.Size([1, 10, 10]))

    dataset = mep_ldm.data.SequenceOfVoxelsToSubvoxelDataset(
        [voxel, voxel],
        subslice=[10, 20, 30])
    assert (dataset[0].shape == torch.Size([1, 10, 20, 30]))

    def feature_extractor(x):
        return x.mean()

    dataset = mep_ldm.data.SequenceOfVoxelsToSlicesDataset(
        [voxel, voxel],
        image_size=10,
        feature_extractor=feature_extractor
    )

    x, y = dataset[0]
    assert (x.shape == torch.Size([1, 10, 10]))
    assert (torch.isclose(y, x.mean()))

    dataset = mep_ldm.data.VoxelToSubvoxelDataset(voxel,
                                                  subslice=12,
                                                  voxel_downscale_factor=5)
    assert (dataset[0].shape == torch.Size([1, 12, 12, 12]))


if __name__ == "__main__":
    test_mep_ldm_dataset()
    print("All tests passed")
