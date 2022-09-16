import argparse
import toml

from pathlib import Path
from prepare.download import run as run_download
from prepare.split import run as run_split
from prepare.augment import run as run_augment


parser = argparse.ArgumentParser(
    description="Prepare s1s2_water images and masks for train, validation and test",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument("--settings", help=f"Path to TOML file with settings", required=True)
parser.add_argument("--download", action="store_true", help=f"Download s1s2_water dataset files", required=False)
parser.add_argument(
    "--split", action="store_true", help=f"Split images and masks into train, validation and test tiles", required=False
)
parser.add_argument("--augment", action="store_true", help=f"Augment training tiles", required=False)
args = parser.parse_args()

if Path(args.settings).is_file():
    with open(args.settings) as f:
        settings = toml.load(f)
else:
    raise Exception("Cannot find settings file.")

if args.download:
    run_download(data_dir=settings["GENERAL"]["DATA_DIR"])

if args.split:
    run_split(
        data_dir=settings["GENERAL"]["DATA_DIR"],
        out_dir=settings["SPLIT"]["OUT_DIR"],
        sensor=settings["SPLIT"]["SENSOR"],
        tile_shape=settings["SPLIT"]["TILE_SHAPE"],
        img_bands_idx=settings["SPLIT"]["IMG_BANDS_IDX"],
        slope=settings["SPLIT"]["SLOPE"],
        exclude_nodata=settings["SPLIT"]["EXCLUDE_NODATA"],
    )

if args.augment:
    run_augment(
        img_dir=settings["AUGMENT"]["IMG_DIR"],
        msk_dir=settings["AUGMENT"]["MSK_DIR"],
        naugs=settings["AUGMENT"]["NAUGS"],
        slope_in_bands=settings["AUGMENT"]["SLOPE_IN_BANDS"],
    )
