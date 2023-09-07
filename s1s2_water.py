import argparse
import toml

from pathlib import Path
from prepare.split import run as run_split


parser = argparse.ArgumentParser(
    description="Prepare s1s2_water images and masks for train, validation and test",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument("--settings", help=f"Path to TOML file with settings", required=True)
args = parser.parse_args()

if Path(args.settings).is_file():
    with open(args.settings) as f:
        settings = toml.load(f)
else:
    raise Exception("Cannot find settings file.")

run_split(
    data_dir=settings["DATA_DIR"],
    out_dir=settings["OUT_DIR"],
    sensor=settings["SENSOR"],
    tile_shape=settings["TILE_SHAPE"],
    img_bands_idx=settings["IMG_BANDS_IDX"],
    slope=settings["SLOPE"],
    exclude_nodata=settings["EXCLUDE_NODATA"],
)
