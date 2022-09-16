import albumentations as al
import logging
import numpy as np
import tifffile as tiff

from pathlib import Path


def aug_seq_radiometry():
    """defines augmentation sequence for general case (radiometric transformations)
    NOTE: Albumentations supports two data types that describe the intensity of pixels: - np.uint8, an unsigned 8-bit
    integer that can define values between 0 and 255. - np.float32, a floating-point number with single precision.
    For np.float32 input, Albumentations expects that value will lie in the range between 0.0 and 1.0.
    """
    return al.Compose([al.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.1, 0.1), p=1.0)])


def aug_seq_geometry(xsize, ysize):
    """defines augmentation sequence for general case (geometric transformations)
    NOTE: Albumentations supports two data types that describe the intensity of pixels: - np.uint8, an unsigned 8-bit
    integer that can define values between 0 and 255. - np.float32, a floating-point number with single precision.
    For np.float32 input, Albumentations expects that value will lie in the range between 0.0 and 1.0.
    """
    return al.Compose(
        [
            al.RandomRotate90(p=1.0),
            al.RandomScale(scale_limit=(0.9, 1.1), interpolation=1, p=0.8),
            al.RandomCrop(height=ysize, width=xsize, p=1.0),
            al.Flip(p=0.8),
        ]
    )


def run(
    img_dir,
    msk_dir,
    fda_ref_dir=None,
    naugs=1,
    dem_in_bands=False,
    seed=0,
):
    """Augments training data.
    """
    # get input files
    img_files = sorted(Path(img_dir).glob("*.tif"))
    msk_files = sorted(Path(msk_dir).glob("*.tif"))
    if fda_ref_dir:
        fda_ref_files = sorted(Path(fda_ref_dir).glob("*.tif"))

    for i in range(len(img_files)):
        # load image and mask
        img = tiff.imread(img_files[i])
        msk = tiff.imread(msk_files[i])

        # define augmentation sequences
        augmentation_seq_radiometry = aug_seq_radiometry()
        augmentation_seq_geometry = aug_seq_geometry(xsize=img.shape[0], ysize=img.shape[1])

        # TODO: support n_augs<1 to apply augmentation only to a random subset of images
        for n in range(len(naugs)):
            if dem_in_bands is True:
                # augment radiometry (exclude dem band for this)
                # NOTE: we do not want dem band (slope) values to be augmented - only image bands
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

            # save results
            tiff.imsave(
                img_dir + str(img_files[i].stem) + "_aug" + str(n) + ".tif",
                img_aug,
                planarconfig="contig",
            )
            tiff.imsave(msk_dir + str(msk_files[i].stem) + "_aug" + str(n) + ".tif", msk_aug)

            if fda_ref_dir:
                # load random reference image for augmentation
                # NOTE: reference image needs to be of same shape and scaled like this image
                fda_ref_file = random.choice(fda_ref_files)
                img_ref = tiff.imread(fda_ref_file)

                # define augmentation sequence
                augmentation_seq_fda = al.Compose([al.FDA([img_ref], p=1, beta_limit=0.001, read_fn=lambda x: x)])

                if dem_in_bands is True:
                    # apply FDA augmentation to target image (exclude dem band for this)
                    img_aug = aug_fda(image=img[:, :, :-1])["image"]
                    img_aug = np.append(img_aug, img[:, :, -1:], axis=2)
                else:
                    img_aug = aug_fda(image=img)["image"]

                # save augmented image and mask
                tiff.imsave(
                    img_dir + str(img_files[i].stem)[:-4] + "aug_fda" + str(n) + ".tif",
                    img_aug,
                    planarconfig="contig",
                )
                tiff.imsave(
                    msk_dir + str(msk_files[i].stem)[:-4] + "aug_fda" + str(n) + ".tif",
                    msk_aug,
                )
