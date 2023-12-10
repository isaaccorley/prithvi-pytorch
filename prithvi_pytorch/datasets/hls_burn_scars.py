import glob
import os
from typing import Any, Callable, Optional

import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from torch.utils.data import Dataset
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datasets.utils import percentile_normalization
from torchgeo.transforms import AugmentationSequential


class HLSBurnScars(Dataset):
    bands = ["B02", "B03" "B04", "B8A", "B11" "B12"]
    directories = {"train": "training", "val": "validation"}

    def __init__(
        self, root: str, split: str = "train", transforms: Optional[Callable] = None
    ):
        assert split in ["train", "val"]
        self.transforms = transforms
        directory = os.path.join(root, self.directories[split])
        self.images = sorted(glob.glob(f"{directory}/*merged.tif"))
        self.masks = sorted(glob.glob(f"{directory}/*mask.tif"))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = self.load_image(index)
        mask = self.load_mask(index)
        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def load_image(self, index: int) -> torch.Tensor:
        path = self.images[index]
        with rasterio.open(path) as dataset:
            image = dataset.read()
            image = torch.from_numpy(image)
            return image

    def load_mask(self, index: int) -> torch.Tensor:
        path = self.masks[index]
        with rasterio.open(path) as dataset:
            mask = dataset.read(indexes=1)
            mask = torch.from_numpy(mask).to(torch.long)
            mask = torch.clip(mask, 0, 1)
            return mask

    def plot(
        self,
        sample: dict[str, torch.Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        image = sample["image"][:3].numpy()  # get BGR channels & convert to numpy
        image = np.rollaxis(image, 0, 3)  # CHW -> HWC
        image = image[..., ::-1]  # bgr to rgb
        image = percentile_normalization(
            image, axis=(0, 1)
        )  # normalize for visualizing

        ncols = 1
        show_mask = "mask" in sample
        show_predictions = "prediction" in sample

        if show_mask:
            mask = sample["mask"].numpy()
            ncols += 1

        if show_predictions:
            prediction = sample["prediction"].numpy()
            ncols += 1

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 8, 8))
        if not isinstance(axs, np.ndarray):
            axs = [axs]
        axs[0].imshow(image)
        axs[0].axis("off")
        if show_titles:
            axs[0].set_title("Image")

        if show_mask:
            axs[1].imshow(mask, interpolation="none")
            axs[1].axis("off")
            if show_titles:
                axs[1].set_title("Label")

        if show_predictions:
            axs[2].imshow(prediction, interpolation="none")
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig


class HLSBurnScarsDataModule(NonGeoDataModule):
    # mean/std taken from https://github.com/NASA-IMPACT/hls-foundation-os/blob/main/configs/burn_scars.py
    mean = torch.tensor(
        [
            0.033349706741586264,
            0.05701185520536176,
            0.05889748132001316,
            0.2323245113436119,
            0.1972854853760658,
            0.11944914225186566,
        ]
    )
    std = torch.tensor(
        [
            0.02269135568823774,
            0.026807560223070237,
            0.04004109844362779,
            0.07791732423672691,
            0.08708738838140137,
            0.07241979477437814,
        ]
    )

    image_size = (512, 512)

    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            HLSBurnScars, batch_size=batch_size, num_workers=num_workers, **kwargs
        )

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["image", "mask"],
        )
        self.val_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=["image", "mask"]
        )
        self.test_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=["image", "mask"]
        )

    def setup(self, stage: str) -> None:
        if stage in ["fit", "validate"]:
            self.train_dataset = HLSBurnScars(split="train", **self.kwargs)
            self.val_dataset = HLSBurnScars(split="val", **self.kwargs)
        if stage in ["test"]:
            # HLS Burn Scars has no test set
            self.test_dataset = HLSBurnScars(split="val", **self.kwargs)
