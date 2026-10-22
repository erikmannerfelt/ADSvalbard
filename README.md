# Svalbard glacier elevation change 2013–2024 from ArcticDEM: processing

This repository turns ArcticDEM strips over Svalbard into co-registered annual DEM
mosaics, elevation-change rate rasters for 2013–2018, 2019–2024 and 2013–2024, and
their uncertainties. These are the products analysed in Mannerfelt et al. (in prep.).
The methods are described in the paper; this README covers the order in which the
code produces the products.

The pipeline is run stage by stage rather than through a single entry point. Each stage
caches its results and skips work that already exists, so a stage can be interrupted
and restarted.

## Related repositories

- [erikmannerfelt/ad_svalbard_post](https://github.com/erikmannerfelt/ad_svalbard_post):
  per-glacier statistics, comparisons with published estimates and the figures of the
  paper, starting from the products of this repository.

## Installation

The environment is a Nix flake:

```bash
nix develop
```

The Python dependencies are listed in `requirements.txt`. All commands below are run
from the repository root, since the code uses paths relative to it.

## Input data

Everything is downloaded on first use:

- the ArcticDEM strips (release s2s041, 2 m) from the Polar Geospatial Center;
- the Norwegian Polar Institute S0 terrain model (the reference DEM) and glacier
  outlines.

The URLs are in `adsvalbard/inputs.py` and `adsvalbard/arcticdem.py`. `shapes/` holds
the hand-drawn vector layers that are part of the method, among them the regions of
poor co-registration (`bad_coreg_regions.geojson`) and the patches used to train the
outlier classifier (`arcticdem_bad_patches.geojson`).

## Processing stages

Intermediate products are written to `temp/` (reference data) and `temp.svalbard/`
(everything else). The mosaics and trends end up in
`temp.svalbard/filt/svalbard/mosaics_3584/`.

| # | Stage | Command | Main output |
|---|---|---|---|
| 1 | Reference DEM mosaic and stable-terrain mask | `adsvalbard.rasters.build_npi_mosaic()`, `adsvalbard.rasters.build_stable_terrain_mask()` | `temp/npi_mosaic.vrt`, `temp/stable_terrain.tif` |
| 2 | Terrain slope and curvature of the reference DEM | `./generate_slopemaps.sh`, `adsvalbard.uncertainty.calculate_all_max_abs_curvature_chunks()` | `temp/npi_slope.tif`, `temp/npi_curvature.vrt` |
| 3 | Strip download and rigid co-registration to the reference DEM | `python main.py` | `temp.svalbard/arcticdem_coreg/` |
| 4 | Outlier probability of every strip | `adsvalbard.filtering.generate_all_masks()` | per-strip outlier-probability rasters |
| 5 | Vertical co-registration, annual median stacks, mosaics and trends | `python -m adsvalbard.vert_coreg` | `vertcoreg_results/`, `medians/`, `mosaics_3584/*.vrt` |
| 6 | Temporal bias correction of the 2019–2024 trend | `adsvalbard.vert_coreg.temporal_biascorr()` | `mosaics_3584/trend_2019-2024_slope_tcorr.vrt` |
| 7 | Uncertainty | `python -m adsvalbard.uncertainty`, `adsvalbard.uncertainty.make_fit_err_vrt()`, `python -m adsvalbard.variogram`, `adsvalbard.uncertainty.get_temporal_biascorr_uncertainty()` | `*_baseline_err.vrt`, `*_fit_err.vrt`, `temp.svalbard/uncertainty/` |
| 8 | Export to GeoTIFF | `./export_mosaics.sh`, then `python -m adsvalbard.doy` | `mosaics_3584/out/*.tif` |
| 9 | Uncertainty checks and point time series from the exported rasters | `adsvalbard.uncertainty.patch_method_uncertainty()`, `adsvalbard.mosaicking.sample_point_timeseries_trends()` | `temp.svalbard/uncertainty/patch_method_*.csv`, `temp.svalbard/point_timeseries_trends.csv` |
| 10 | Auxiliary tables | `python export_aux_files.py` | `temp.svalbard/aux_files.zip` |

Strips with implausible elevations at a set of check points
(`shapes/bad_vertcoreg_pts.geojson`) are excluded through the lists in `exclusions/`,
written by `adsvalbard.uncertainty.check_bad_pts()` with thresholds set by hand in that
function. The screening was iterated between stages 3 and 5, and each pass depends on
the results of the previous ones, so the lists are versioned here as the exclusions used
for the published products rather than regenerated:

| File | Excluded from |
|---|---|
| `exclusions/bad_dems.txt` | Everything (screened on the original strips) |
| `exclusions/bad_dems_dem.txt` | The rigidly co-registered stacks |
| `exclusions/bad_dems_dem_vertcoreg.txt` | Vertical co-registration and the vertically corrected stacks |

After a list changes, `adsvalbard.stacking.remove_bad_dem_chunks()` deletes the cached
stack chunks that contain newly excluded strips, so that stage 5 rebuilds them.

The outlier classifier of stage 4 is evaluated with
`adsvalbard.filtering.evaluate_gap_model_grouped()`.

## Outputs

`export_mosaics.sh` writes the released products to `mosaics_3584/out/`. They are
5 m rasters in EPSG:32633:

| File | Contents |
|---|---|
| `trend_<period>_slope.tif` | Elevation-change rate (m a⁻¹). For 2019–2024, the temporally corrected rate |
| `trend_2019-2024_slope_uncorr.tif` | The 2019–2024 rate without temporal bias correction |
| `trend_<period>_slope_fit_err.tif`, `trend_<period>_slope_baseline_err.tif` | Uncertainty components of the rates |
| `trend_2013-2024_intercept.tif` | Intercept of the 2013–2024 trend |
| `dem_<year>_{count,doy,quality,...}.tif` | Annual mosaic bands. `doy` is the day of year; `quality` is the processing level (1 heavily filtered, 2 lightly filtered, 3 vertically bias-corrected only, 4 globally deramped) |
| `dem_quality_mode.tif` | Most frequent quality level over all years |

The GDAL overviews are average-resampled.

## Processing times

All processing ran on one workstation (AMD Ryzen 9 3950X, 16 cores, 64 GB RAM).
Stacking ran 2–5 workers in parallel and mosaicking 8, so their totals are wall-clock
time. The strip stages include the strips from before 2013.

| Stage | Unit | Median | Total |
|---|---|---|---|
| Rigid co-registration | per strip (10,424) | 1.5 min | 455 h (19.0 days) |
| Outlier probability | per strip (10,418) | 14 s | 73 h (3.0 days) |
| Vertical co-registration | per region-year (72) | 3.1 min | 6 h |
| Stacking, main product (outlier probability < 0.75) | per year (12) | 11 h | 131 h (5.5 days) |
| Stacking, all four products |  | – | 414 h (17.2 days) |
| Mosaicking and trend fitting | 467 chunks | – | 4 h |

## Repository layout

| Path | Contents |
|---|---|
| `main.py` | Strip download and rigid co-registration (stage 3) |
| `adsvalbard/rasters.py` | Reference DEM mosaic and stable-terrain mask |
| `adsvalbard/arcticdem.py`, `adsvalbard/inputs.py` | Strip metadata and input downloads |
| `adsvalbard/filtering.py` | Outlier classifier and per-strip outlier probabilities |
| `adsvalbard/vert_coreg.py` | Vertical co-registration and temporal bias correction |
| `adsvalbard/stacking.py` | Annual median stacks |
| `adsvalbard/mosaicking.py` | Annual mosaics, quality levels and trend fitting |
| `adsvalbard/uncertainty.py`, `adsvalbard/variogram.py` | Uncertainty |
| `adsvalbard/doy.py` | Day-of-year band repair |
| `shapes/` | Hand-drawn vector layers |
| `exclusions/` | Strip exclusion lists |
| `extra/timings.py` | Processing times from file modification times |

## Licence

MIT; see `LICENSE`.
