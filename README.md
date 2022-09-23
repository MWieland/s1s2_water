# s1s2_water
This repository refers to the [PUBLIC s1s2_water dataset stored on Zenodo](...), which contains less samples than our [internal s1s2_water dataset](...).

TODO: add here a general dataset description and link to our paper.

> If you use our dataset please cite the following publication ...

## TODOs
- implement and test download from Zenodo
- discuss with Florian were to publish this repo (part of ukis-data-tutorials -> make this one more general by removing reference to Geoservice)

## Data download and preparation
The following downloads the dataset (**--download**), splits images and masks into train, val and test tiles with predefined shape and band combination (**--split**) and applies basic augmentation on the training data split (**--augment**).

```python
$ python entrypoint.py --download --split --augment --settings settings.toml
```

Data preparation parameters are defined in a settings TOML file (**--settings**)

```toml
[GENERAL]
DATA_DIR = "/path/to/data_directory"

[SPLIT]
SENSOR = "s2"
TILE_SHAPE = [256, 256]
IMG_BANDS_IDX = [0, 1, 2, 3, 4, 5]
SLOPE = false
EXCLUDE_NODATA = false
OUT_DIR = "/path/to/output_directory"

[AUGMENT]
IMG_DIR = "/path/to/image_directory"
MSK_DIR = "/path/to/mask_directory"
NAUGS = 1
SLOPE_IN_BANDS = false
```

## Installation
#### Conda
```shell
$ conda create -n s1s2_water python=3.9
$ conda activate s1s2_water
$ poetry install
```

#### Docker
```shell
$ docker build -f entrypoint.Dockerfile --tag s1s2_water --network=host .
$ docker run --network=host --rm -v /media/datadrive/Datasets/referencedata/ikonos_ida/:/scratch/ s1s2_water --split --augment --settings /scratch/settings.toml
```