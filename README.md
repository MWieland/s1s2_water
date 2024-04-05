# S1S2-Water: A global dataset for semantic segmentation of water bodies from Sentinel-1 and Sentinel-2 satellite images
This repository provides tools to work with the [S1S2-Water dataset](https://zenodo.org/records/8314175).

[S1S2-Water dataset](https://zenodo.org/records/8314175) is a global reference dataset for training, validation and testing of convolutional neural networks for semantic segmentation of surface water bodies in publicly available Sentinel-1 and Sentinel-2 satellite images. The dataset consists of 65 triplets of Sentinel-1 and Sentinel-2 images with quality checked binary water mask. Samples are drawn globally on the basis of the Sentinel-2 tile-grid (100 x 100 km) under consideration of pre-dominant landcover and availability of water bodies. Each sample is complemented with STAC-compliant metadata and Digital Elevation Model (DEM) raster from the Copernicus DEM.

The following pre-print article describes the dataset:

> Wieland, M., Fichtner, F., Martinis, S., Groth, S., Krullikowski, C., Plank, S., Motagh, M. (2023). S1S2-Water: A global dataset for semantic segmentation of water bodies from Sentinel-1 and Sentinel-2 satellite images. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, doi: [10.1109/JSTARS.2023.3333969](https://dx.doi.org/10.1109/JSTARS.2023.3333969).

## Dataset access
The dataset (~170 GB) is available for download at: https://zenodo.org/records/8314175

Download the dataset parts and extract them into a single data directory as follows.

```
.
└── data/
    ├── 1/
    │   ├── sentinel12_copdem30_1_elevation.tif
    │   ├── sentinel12_copdem30_1_slope.tif
    │   ├── sentinel12_s1_1_img.tif
    │   ├── sentinel12_s1_1_msk.tif
    │   ├── sentinel12_s1_1_valid.tif
    │   ├── sentinel12_s2_1_img.tif
    │   ├── sentinel12_s2_1_msk.tif
    │   ├── sentinel12_s2_1_valid.tif
    │   └── sentinel12_1_meta.json
    ├── 5/
    │   ├── sentinel12_copdem30_5_elevation.tif
    │   ├── sentinel12_copdem30_5_slope.tif
    │   ├── sentinel12_s1_5_img.tif
    │   ├── sentinel12_s1_5_msk.tif
    │   ├── sentinel12_s1_5_valid.tif
    │   ├── sentinel12_s2_5_img.tif
    │   ├── sentinel12_s2_5_msk.tif
    │   ├── sentinel12_s2_5_valid.tif
    │   └── sentinel12_5_meta.json
    ├── .../
    │   └── ...
    └── catalog.json
```

## Dataset information
Each file follows the naming scheme sentinel12_SENSOR_ID_LAYER.tif (e.g. `sentinel12_s1_5_img.tif`). Raster layers are stored as Cloud Optimized GeoTIFF (COG) and are projected to Universal Transverse Mercator (UTM).

| Sensor | Layer |Description | Values | Format | Bands |
| - | - | - | - | - | - |
| S1 | IMG | Sentinel-1 image <br> GRD product | Unit: dB (scaled by factor 100) | GeoTIFF <br> 10980 x 10980 px <br> 2 bands <br> Int16 | 0: VV <br> 1: VH
| S2 | IMG | Sentinel-2 image <br> L1C product | Unit: TOA reflectance (scaled by factor 10000) | GeoTIFF <br> 10980 x 10980 px <br> 6 bands <br> UInt16 | 0: Blue <br> 1: Green <br> 2: Red <br> 3: NIR <br> 4: SWIR1 <br> 5: SWIR2
| S1 / S2 | MSK | Annotation mask <br> Hand-labelled water mask | 0: No Water <br> 1: Water | GeoTIFF <br> 10980 x 10980 px <br> 1 band <br> UInt8 | 0: Water mask
| S1 / S2 | VALID | Valid pixel mask <br> Hand-labelled valid pixel mask | 0: Invalid (cloud, cloud-shadow, nodata) <br> 1: Valid | GeoTIFF <br> 10980 x 10980 px <br> 1 band <br> UInt8 | 0: Valid mask
| COPDEM30 | ELEVATION | Copernicus DEM elevation | Unit: Meters | GeoTIFF <br> 3660 x 3660 px <br> 1 band <br> Int16 | 0: Elevation
| COPDEM30 | SLOPE | Copernicus DEM slope | Unit: Degrees | GeoTIFF <br> 3660 x 3660 px <br> 1 band <br> Int16 | 0: Slope
| N.a. | META | METADATA | STAC metadata item | JSON | N.a.

## Data preparation
Make sure to download the dataset as described above. Clone this repository, adjust [settings.toml](settings.toml) and run [s1s2_water.py](s1s2_water.py) to prepare the dataset according to your desired settings. 

The following splits images and masks for a specific sensor (Sentinel-1 or Sentinel-2) into training, validation and testing tiles with predefined shape and band combination. Slope information can be appended to the image band stack if required.

```python
$ python s1s2_water.py --settings settings.toml
```

Data preparation parameters are defined in a [settings TOML file](settings.toml) (**--settings**)

```toml
SENSOR = "s2"                           # prepare Sentinel-1 or Sentinel-2 data ["s1", "s2"]
TILE_SHAPE = [256, 256]                 # desired tile shape in pixel
IMG_BANDS_IDX = [0, 1, 2, 3, 4, 5]      # desired image band combination
SLOPE = true                            # append slope band to image bands
EXCLUDE_NODATA = true                   # exclude tiles with nodata values
DATA_DIR = "/path/to/data_directory"    # data directory that holds the original images
OUT_DIR = "/path/to/output_directory"   # output directory that stores the prepared train, val and test tiles

# Sentinel-1 image bands
# {"VV": 0, "VH": 1}

# Sentinel-2 image bands
# {"Blue": 0, "Green": 1, "Red": 2, "NIR": 3, "SWIR1": 4, "SWIR2": 5}
```

Information on the deployed preprocessing steps for Sentinel-1 imagery can be found in the [SNAP GPT file](prepare/preproc_s1_snap.xml).

## Installation
```shell
$ conda env create --file environment.yaml
$ conda activate s1s2_water
```
