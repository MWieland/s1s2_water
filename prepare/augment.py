import albumentations as al
import logging
import numpy as np
import tifffile as tiff

from pathlib import Path


def aug_seq_radiometry():
    return al.Compose([al.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.1, 0.1), p=1.0)])


def aug_seq_geometry(xsize, ysize):
    return al.Compose(
        [
            al.RandomRotate90(p=1.0),
            al.RandomScale(scale_limit=(0.9, 1.1), interpolation=1, p=0.8),
            al.RandomCrop(height=ysize, width=xsize, p=1.0),
            al.Flip(p=0.8),
        ]
    )


def run(img_dir, msk_dir, naugs=1, slope_in_bands=False):
    logging.info("Augmenting samples")

    img_files = sorted(Path(img_dir).glob("*.tif"))
    msk_files = sorted(Path(msk_dir).glob("*.tif"))

    for i in range(len(img_files)):
        img = tiff.imread(img_files[i])
        msk = tiff.imread(msk_files[i])

        augmentation_seq_radiometry = aug_seq_radiometry()
        augmentation_seq_geometry = aug_seq_geometry(xsize=img.shape[0], ysize=img.shape[1])

        for n in range(naugs):
            if slope_in_bands is True:
                # augment radiometry (exclude dem band for this)
                content = {"image": img[:, :, :-1], "mask": msk}
                augmented = augmentation_seq_radiometry(**content)
                img_aug, msk_aug = augmented["image"], augmented["mask"]
                img_aug = np.append(img_aug, img[:, :, -1:], axis=2)
            else:
                content = {"image": img, "mask": msk}
                augmented = augmentation_seq_radiometry(**content)
                img_aug, msk_aug = augmented["image"], augmented["mask"]

            # augment geometry (include slope for this)
            content = {"image": img_aug, "mask": msk_aug}
            augmented = augmentation_seq_geometry(**content)
            img_aug, msk_aug = augmented["image"], augmented["mask"]

            tiff.imsave(Path(img_dir) / f"{img_files[i].stem}_aug_{n}.tif", img_aug, planarconfig="contig")
            tiff.imsave(Path(msk_dir) / f"{msk_files[i].stem}_aug_{n}.tif", msk_aug)

