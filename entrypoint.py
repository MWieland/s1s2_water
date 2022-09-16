import argparse
import prepare
import toml

from pathlib import Path


parser = argparse.ArgumentParser(
    description="prepare images and masks for train, validation and test of deep learning methods.",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument("--settings", help=f"path to JSON file with settings", required=True)
parser.add_argument(
    "--SPLIT", action="store_true", help=f"SPLIT images and masks into train, validation and test tiles", required=False
)
parser.add_argument("--AUGMENT", action="store_true", help=f"AUGMENT training tiles", required=False)
args = parser.parse_args()

if Path(args.settings).is_file():
    with open(args.settings) as f:
        settings = toml.load(f)
else:
    raise Exception("Cannot find settings file.")

if args.split:
    prepare.split.run(
        data_dir=settings["SPLIT"]["DATA_DIR"],
        sensor=settings["SPLIT"]["SENSOR"],
        tile_shape=settings["SPLIT"][""],
        img_band_idx=settings["SPLIT"]["IMG_BAND_IDX"],
        dem=settings["SPLIT"]["DEM"],
        out_dir=settings["SPLIT"]["OUT_DIR"],
        seed=settings["GENERAL"]["SEED"],
        num_threads=settings["GENERAL"]["NUM_THREADS"],
    )

if args.augment:
    prepare.augment.run(
        img_dir=settings["AUGMENT"]["IMG_DIR"],
        msk_dir=settings["AUGMENT"]["MSK_DIR"],
        naugs=settings["AUGMENT"]["NAUGS"],
        dem_in_bands=settings["AUGMENT"]["DEM_IN_BANDS"],
        seed=settings["GENERAL"]["SEED"],
    )
