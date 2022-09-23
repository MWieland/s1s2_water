FROM eoc-gzs-db01-vm.eoc.dlr.de:4002/gzs-python-geo-base:latest

ENV PYTHONUNBUFFERED=1

RUN mkdir /scratch && \
    chmod ugo+rw- /scratch

RUN pip install --no-cache-dir toml numpy albumentations tqdm tifffile imagecodecs ukis-pysat[complete] pydantic rasterio shapely rio-toa

COPY prepare /prepare
COPY entrypoint.py /entrypoint.py

ENTRYPOINT ["python", "entrypoint.py"]
