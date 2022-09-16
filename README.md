# prepare_refdata
Scripts to prepare reference datasets in a local directory. This includes splitting the images and masks into tiles, subsampling and augmentation.

> *TODO*: generalise the split.py module to work with any dataset (currently paths and scaler values are hardcoded for data_s1s2_water)

## Example Use
The following splits images and masks into train, val and test tiles with predefined shape and band combination (**--split**).
Sample splits are read from `data_s1s2_water_samples.geojson` and samples in each split are randomly shuffled.

It then samples the train tiles according to a predefined sample distribution using under- and oversampling with augmentation
(**--sample**) and apply additional augmentation of all training tiles (**--augment**).

Conda
```bash
$ python scripts/run_prepare.py --split --sample --augment --settings scripts/settings.json
```
Docker
```bash
$ docker run --rm -v /path/to/data_s1s2_water/data/:/data_s1s2_water/data/ -v /path/to/data_s1s2_water/sampling/:/data_s1s2_water/sampling data_s1s2_water_prepare --split --sample --augment --settings /data_s1s2_water/data/settings_docker.json
```

A random seed guarantees reproducibility of the results. Data preparation parameters are defined in settings.json

```json
{
    "general": {
        "logger": {
            "log_file": null,
            "log_level": "INFO"
        },
        "num_threads": 8,
        "seed": 4
    },

    "split": {
        "sensor": "s2",
        "tile_shape": [256, 256],
        "img_bands_idx": [0, 1, 2, 3, 4, 5],
        "dem": "copdem30",
        "exclude_substr": ["sentinel12_s2_1_", "sentinel12_s2_2_"],
        "out_dir": "data/tiles/"
    },

    "sample": {
        "img_dir": "data/tiles/train/img/",
        "msk_dir": "data/tiles/train/msk/",
        "nbins": 4,
        "dst_sample_dict": {"1": 0.5, "2": 0.20, "3": 0.15, "4": 0.15},
        "apply_to_data": false,
        "dem_in_bands": true
    },

    "augment": {
        "img_dir": "data/tiles/train/img/",
        "msk_dir": "data/tiles/train/msk/",
        "fda_ref_dir": null,
        "naugs": 1,
        "dem_in_bands": true
    }
}
```

### Installation
#### Conda
```shell
$ conda create -n data_s1s2_water python=3.8
$ conda activate data_s1s2_water
$ conda install rasterio
$ pip install ukis_pysat[raster] tqdm tifffile albumentations matplotlib imblearn
```

#### Docker
```shell
$ docker build -f run_prepare.Dockerfile --tag data_s1s2_water_prepare --network=host .
```

## Dataset Structure
The `data_s1s2_water` directory is split into subfolders containing data (data), sample locations and metadata (sampling), and helper scripts (scripts).
```
data_s1s2_water
├── data
│   ├── scenes
│   │   ├── aux_s1                          # auxiliary files Sentinel-1 (valid mask, scene metadata)
│   │   ├── aux_s2                          # auxiliary files Sentinel-2 (valid mask, scene metadata)
│   │   ├── dem_copdem30                    # CopDEM30 files (elevation, slope)
│   │   ├── dem_copdem90                    # CopDEM90 files (elevation, slope)
│   │   ├── dem_srtm1                       # SRTM1 files (elevation, slope)
│   │   ├── img_s1                          # image files Sentinel-1
│   │   ├── img_s2                          # image files Sentinel-2
│   │   ├── msk_s1                          # water masks Sentinel-1 (hand-labeled)
│   │   ├── msk_s1_water_processor          # water masks of the Sentinel-1 processor (rule-based)
│   │   ├── msk_s2                          # water masks Sentinel-2 (hand-labeled)
│   │   └── png                             # preview images
│   └── tiles
│       ├── train                           # training tiles (to be created by script)
│       ├── test                            # test tiles (to be created by script)
│       └── val                             # validation tiles (to be created by script)
├── sampling
│   └── data_s1s2_water_samples.geojson     # sample metadata
└── scripts                                 # helper scripts
```

## Sample Metadata
Locations and basic metadata for each sample are contained in `data_s1s2_water_samples.geojson`. More detailed metadata about the Sentinel scenes that compose each sample can be found in the respective scene metadata files.

| Field           | Description                                                                            |
| --------------- | -------------------------------------------------------------------------------------- |
| ID              | Unique ID for each sample                                                               |
| s1_srcids        | Source IDs of Sentinel-1 scenes for each sample                                                         |
| s1_img        | Sample ID of Sentinel-1 image
| s1_msk        | Sample ID of Sentinel-1 water mask
| s1_valid        | Sample ID of Sentinel-1 valid pixel mask
| s1_meta        | Sample ID of Sentinel-1 scene metadata
| s2_srcids        | Source IDs of Sentinel-2 scenes for each sample                                                         |
| s2_img        | Sample ID of Sentinel-2 image
| s2_msk        | Sample ID of Sentinel-2 water mask
| s2_valid        | Sample ID of Sentinel-2 valid pixel mask
| s2_meta        | Sample ID of Sentinel-2 scene metadata
| landcover        | Pre-dominant landcover type for each sample
| split        | Train, test or validation split that each sample belongs to

