import logging
import numpy as np
import sys
import tifffile as tiff
import tqdm

from pathlib import Path
from prepare import utils
from ukis_pysat.raster import Image


def run(
    data_dir, sensor="s1", tile_shape=(256, 256), img_bands_idx=[0, 1], dem=False, out_dir=None, seed=0, num_threads=1,
):
    if sensor == "s1":
        scale_min_max = (0, 400)
    elif sensor == "s2":
        scale_min_max = (0, 10000)
    else:
        raise NotImplementedError(f"Sensor {str(sensor)} not supported ['s1', 's2']")

    if out_dir is None:
        out_dir = data_dir + "/data/tiles/"
    os.makedirs(out_dir + "train/", exist_ok=True)
    os.makedirs(out_dir + "train/img/", exist_ok=True)
    os.makedirs(out_dir + "train/msk/", exist_ok=True)
    os.makedirs(out_dir + "test/", exist_ok=True)
    os.makedirs(out_dir + "test/img/", exist_ok=True)
    os.makedirs(out_dir + "test/msk/", exist_ok=True)
    os.makedirs(out_dir + "val/", exist_ok=True)
    os.makedirs(out_dir + "val/img/", exist_ok=True)
    os.makedirs(out_dir + "val/msk/", exist_ok=True)

    """
    # use samples metadata to split dataset into train, val and test subsets (this also shuffles the samples in each subset)
    X_train, y_train = utils.split_samples(samples_file, img_key, msk_key, "split", "train", exclude_substr, seed)
    X_val, y_val = utils.split_samples(samples_file, img_key, msk_key, "split", "val", exclude_substr, seed)
    X_test, y_test = utils.split_samples(samples_file, img_key, msk_key, "split", "test", exclude_substr, seed)
    logging.info(
        f"Training images: {str(len(X_train))}, Validation images: {str(len(X_val))}, Test images: {str(len(X_test))}"
    )

    logging.info("Splitting training samples")
    sys.stdout.flush()
    for i, file in tqdm.tqdm(enumerate(X_train), total=len(X_train)):
        # load training image and mask
        img = Image(data=img_dir + X_train[i], dimorder="last")
        msk = Image(data=msk_dir + y_train[i], dimorder="last")

        # scale image to range [0,1] using simple min max scaler
        img_scaled = utils.scale_min_max(img.arr, min=scale_min_max[0], max=scale_min_max[1])[:, :, img_bands_idx]

        if dem:
            if os.path.isfile(dem_dir + X_train[i].replace(sensor, dem)[:-7] + "slope.tif"):
                # load slope, align to image and add to bandstack if it exists
                slope_img = Image(data=dem_dir + X_train[i].replace(sensor, dem)[:-7] + "slope.tif", dimorder="last")
                slope_img.warp(resampling_method=2, dst_crs=img.dataset.crs, target_align=img, num_threads=num_threads)
                img_scaled = np.append(img_scaled, slope_img.arr, axis=2)
            else:
                # skip this sample entirely
                logging.warning(f"No dem file found for this sample. Skipping {X_train[i]} entirely.")
                continue

        # tile image and mask
        img_tiles = utils.tile_array(img_scaled, xsize=tile_shape[0], ysize=tile_shape[1], overlap=0.0, padding=False)
        msk_tiles = utils.tile_array(msk.arr, xsize=tile_shape[0], ysize=tile_shape[1], overlap=0.0, padding=False)

        for j in range(len(img_tiles)):
            if exclude_nodata is not None:
                # skip tiles that cover nodata values
                if exclude_nodata in img_tiles[j, :, :, 0]:
                    continue

            # save image and mask tile
            tiff.imsave(
                out_dir + "train/img/" + X_train[i][:-4] + "_tile" + str(j) + ".tif",
                img_tiles[j, :, :, :],
                planarconfig="contig",
            )
            tiff.imsave(out_dir + "train/msk/" + y_train[i][:-4] + "_tile" + str(j) + ".tif", msk_tiles[j, :, :, :])

    logging.info("Splitting validation samples")
    sys.stdout.flush()
    for i, file in tqdm.tqdm(enumerate(X_val), total=len(X_val)):
        # load validation image and mask
        img = Image(data=img_dir + X_val[i], dimorder="last")
        msk = Image(data=msk_dir + y_val[i], dimorder="last")

        # scale image to range [0,1] using simple min max scaler
        img_scaled = utils.scale_min_max(img.arr, min=scale_min_max[0], max=scale_min_max[1])[:, :, img_bands_idx]

        if dem:
            if os.path.isfile(dem_dir + X_val[i].replace(sensor, dem)[:-7] + "slope.tif"):
                # load slope, align to image and add to bandstack if it exists
                slope_img = Image(data=dem_dir + X_val[i].replace(sensor, dem)[:-7] + "slope.tif", dimorder="last")
                slope_img.warp(resampling_method=2, dst_crs=img.dataset.crs, target_align=img, num_threads=4)
                img_scaled = np.append(img_scaled, slope_img.arr, axis=2)
            else:
                # skip this sample entirely
                logging.warning(f"No dem file found for this sample. Skipping {X_val[i]} entirely.")
                continue

        # tile image and mask
        img_tiles = utils.tile_array(img_scaled, xsize=tile_shape[0], ysize=tile_shape[1], overlap=0.0, padding=False)
        msk_tiles = utils.tile_array(msk.arr, xsize=tile_shape[0], ysize=tile_shape[1], overlap=0.0, padding=False)

        for j in range(len(img_tiles)):
            if exclude_nodata is not None:
                # skip tiles that cover nodata values
                if exclude_nodata in img_tiles[j, :, :, 0]:
                    continue

            # save image and mask tile
            tiff.imsave(
                out_dir + "val/img/" + X_val[i][:-4] + "_tile" + str(j) + ".tif",
                img_tiles[j, :, :, :],
                planarconfig="contig",
            )
            tiff.imsave(out_dir + "val/msk/" + y_val[i][:-4] + "_tile" + str(j) + ".tif", msk_tiles[j, :, :, :])

    logging.info("Splitting testing samples")
    sys.stdout.flush()
    for i, file in tqdm.tqdm(enumerate(X_test), total=len(X_test)):
        # load test image and mask
        img = Image(data=img_dir + X_test[i], dimorder="last")
        msk = Image(data=msk_dir + y_test[i], dimorder="last")

        # scale image to range [0,1] using simple min max scaler
        img_scaled = utils.scale_min_max(img.arr, min=scale_min_max[0], max=scale_min_max[1])[:, :, img_bands_idx]

        if dem:
            if os.path.isfile(dem_dir + X_test[i].replace(sensor, dem)[:-7] + "slope.tif"):
                # load slope, align to image and add to bandstack if it exists
                slope_img = Image(data=dem_dir + X_test[i].replace(sensor, dem)[:-7] + "slope.tif", dimorder="last")
                slope_img.warp(resampling_method=2, dst_crs=img.dataset.crs, target_align=img, num_threads=4)
                img_scaled = np.append(img_scaled, slope_img.arr, axis=2)
            else:
                # skip this sample entirely
                logging.warning(f"No dem file found for this sample. Skipping {X_test[i]} entirely.")
                continue
        # tile image and mask
        img_tiles = utils.tile_array(img_scaled, xsize=tile_shape[0], ysize=tile_shape[1], overlap=0.0, padding=False)
        msk_tiles = utils.tile_array(msk.arr, xsize=tile_shape[0], ysize=tile_shape[1], overlap=0.0, padding=False)

        for j in range(len(img_tiles)):
            if exclude_nodata is not None:
                # skip tiles that cover nodata values
                if exclude_nodata in img_tiles[j, :, :, 0]:
                    continue

            # save image and mask tile
            tiff.imsave(
                out_dir + "test/img/" + X_test[i][:-4] + "_tile" + str(j) + ".tif",
                img_tiles[j, :, :, :],
                planarconfig="contig",
            )
            tiff.imsave(out_dir + "test/msk/" + y_test[i][:-4] + "_tile" + str(j) + ".tif", msk_tiles[j, :, :, :])
     """
