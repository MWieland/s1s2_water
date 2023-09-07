import json
import logging
import numpy as np
import sys
import tifffile as tiff
import tqdm

from pathlib import Path
from prepare.utils import scale_min_max, tile_array
from pystac_client import Client
from ukis_pysat.raster import Image


def run(data_dir, out_dir, sensor="s1", tile_shape=(256, 256), img_bands_idx=[0, 1], slope=False, exclude_nodata=False):
    logging.info("Splitting training samples")

    if Path(Path(data_dir) / "catalog.json").is_file:
        catalog = Client.open(Path(data_dir) / "catalog.json")
    else:
        raise NotImplementedError("Cannot find catalog.json file in data_dir")

    if sensor == "s1":
        scale_min, scale_max = 0, 100.0
    elif sensor == "s2":
        scale_min, scale_max = 0, 10000.0
    else:
        raise NotImplementedError(f"Sensor {str(sensor)} not supported ['s1', 's2']")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(out_dir) / "train/img").mkdir(parents=True, exist_ok=True)
    Path(Path(out_dir) / "train/msk").mkdir(parents=True, exist_ok=True)
    Path(Path(out_dir) / "test/img").mkdir(parents=True, exist_ok=True)
    Path(Path(out_dir) / "test/msk").mkdir(parents=True, exist_ok=True)
    Path(Path(out_dir) / "val/img").mkdir(parents=True, exist_ok=True)
    Path(Path(out_dir) / "val/msk").mkdir(parents=True, exist_ok=True)

    items = [item.to_dict() for item in catalog.get_all_items()]
    sys.stdout.flush()
    for i, item in tqdm.tqdm(enumerate(items), total=len(items)):
        split = item["properties"]["split"]
        subdir = Path(item["assets"][f"{sensor}_img"]["href"]).parent.name
        msk_file = Path(data_dir) / Path(subdir) / Path(item["assets"][f"{sensor}_msk"]["href"]).name
        valid_file = Path(data_dir) / Path(subdir) / Path(item["assets"][f"{sensor}_valid"]["href"]).name
        slope_file = Path(data_dir) / Path(subdir) / Path(item["assets"]["copdem30_slope"]["href"]).name
        img_file = Path(data_dir) / Path(subdir) / Path(item["assets"][f"{sensor}_img"]["href"]).name

        msk = Image(data=msk_file, dimorder="last")
        valid = Image(data=valid_file, dimorder="last")
        slope = Image(data=slope_file, dimorder="last") if slope else None
        img = Image(data=img_file, dimorder="last")
        img_scaled = scale_min_max(img.arr[:, :, img_bands_idx], min=scale_min, max=scale_max)

        if slope:
            slope.warp(resampling_method=2, dst_crs=img.dataset.crs, target_align=img)
            img_scaled = np.append(img_scaled, slope.arr, axis=2)

        img_tiles = tile_array(img_scaled, xsize=tile_shape[0], ysize=tile_shape[1], overlap=0.0, padding=False)
        msk_tiles = tile_array(msk.arr, xsize=tile_shape[0], ysize=tile_shape[1], overlap=0.0, padding=False)
        valid_tiles = (
            tile_array(valid.arr, xsize=tile_shape[0], ysize=tile_shape[1], overlap=0.0, padding=False)
            if exclude_nodata
            else None
        )

        for j in range(len(img_tiles)):
            if exclude_nodata:
                if 0 in valid_tiles[j, :, :, :]:
                    continue
            tiff.imsave(
                Path(out_dir) / f"{split}/img/{Path(img_file).stem}_{j}.tif",
                img_tiles[j, :, :, :],
                planarconfig="contig",
            )
            tiff.imsave(Path(out_dir) / f"{split}/msk/{Path(msk_file).stem}_{j}.tif", msk_tiles[j, :, :, :])
