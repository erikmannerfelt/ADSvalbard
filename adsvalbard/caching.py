import functools
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import projectfiles
import xarray as xr

from adsvalbard.constants import CONSTANTS


def precompile_cache(**kwargs):
    """Precompile projectfiles.cache with the given kwargs (can be overridden)"""

    if "cache_dir" not in kwargs:
        kwargs["cache_dir"] = CONSTANTS.cache_dir

    @functools.wraps(projectfiles.cache)
    def decorator(func = None, subdir: Path | None = None, **kwargs2):
        if subdir is not None:
            kwargs["cache_dir"] = Path(kwargs["cache_dir"]).joinpath(subdir)
        return projectfiles.cache(func=func, **(kwargs | kwargs2))

    return decorator

class NumpyArrayEncoder(json.JSONEncoder):
    """A JSON encoder that properly serializes numpy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_geojson(path: Path, obj: gpd.GeoDataFrame):
    obj.to_file(path, driver="GeoJSON")

def load_geojson(path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path) 


cache_geojson = precompile_cache(cache_dir=CONSTANTS.cache_dir, routine=(save_geojson, load_geojson, "geojson"))


def save_json(path: Path, obj: dict[Any, Any]):
    with open(path, "w") as outfile:
        json.dump(obj, outfile, cls=NumpyArrayEncoder)

def load_json(path: Path) -> dict[Any, Any]:
    with open(path) as infile:
        return json.load(infile)


cache_json = precompile_cache(routine=(save_json, load_json, "json"))

# cache_json_subdir = lambda subdir: functools.partial(projectfiles.cache, routine=(save_json, load_json, "json"), cache_dir=CONSTANTS.cache_dir.joinpath(subdir))
# cache_json_subdir.__doc__ = projectfiles.cache.__doc__

def save_feather(path: Path, obj: gpd.GeoDataFrame):
    obj.to_feather(path)

def load_feather(path: Path) -> gpd.GeoDataFrame:
    try:
        return gpd.read_feather(path)
    except ValueError:
        return pd.read_feather(path)

cache_feather = precompile_cache(routine=(save_feather, load_feather, "feather"))

def save_nc(path: Path, obj: xr.Dataset):
    obj.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 5} for v in obj.data_vars})

def load_nc(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, chunks=512)

cache_nc = precompile_cache(routine=(save_nc, load_nc, "nc"))

def test_precompile_cache():

    def func(arg1: int):
        return str(arg1)

    with tempfile.TemporaryDirectory() as temp_dir_str:
        cache_dir = Path(temp_dir_str)

        cached1 = projectfiles.cache(func, cache_dir=cache_dir)

        cached1(1)
        assert len(os.listdir(cache_dir)) == 1

        precompiled = precompile_cache(cache_dir=cache_dir, routine="text")

        cached2 = precompiled(func)

        cached2(2)

        assert len(os.listdir(cache_dir)) == 2

        cached_files = os.listdir(cache_dir)

        cache_name = list(filter(lambda p: ".txt" in p,  cached_files))[0]

        with open(cache_dir.joinpath(cache_name)) as infile:
            assert infile.read() == "2"

