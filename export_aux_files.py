import zipfile
from pathlib import Path
import io
import itertools

def main():
    files = [
        "temp.svalbard/uncertainty/binned_terrain_err_trend_2013-2018_slope.csv",
        "temp.svalbard/uncertainty/binned_terrain_err_trend_2013-2024_slope.csv",
        "temp.svalbard/uncertainty/binned_terrain_err_trend_2019-2024_slope.csv",
        "temp.svalbard/uncertainty/patch_method_trend_2013-2018_slope.csv",
        "temp.svalbard/uncertainty/patch_method_trend_2013-2024_slope.csv",
        "temp.svalbard/uncertainty/patch_method_trend_2019-2024_slope.csv",
        "temp.svalbard/uncertainty/temporal_biascorr_area_nmad.csv",
        "temp.svalbard/uncertainty/temporal_biascorr_distance_nmad.csv",
        "temp.svalbard/uncertainty/vgm_neff_cirq.csv",
        "temp.svalbard/uncertainty/variogram_model.csv",
        "temp.svalbard/uncertainty/empirical_variogram.csv",
        "temp.svalbard/chunk_outlines.geojson",
        "temp.svalbard/gap_model_importance_table.csv",
        "shapes/arcticdem_bad_patches.geojson",
        "shapes/bad_coreg_regions.geojson",
    ]

    inner_zips = {
        "vertcoreg_results.zip": io.BytesIO(),
        "coreg_results.zip": io.BytesIO(),
    }

    with zipfile.ZipFile(inner_zips["vertcoreg_results.zip"], "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for filepath in itertools.chain(
            Path("temp.svalbard/vertcoreg_results/").glob("vertcoreg*.csv"),
            Path("temp.svalbard/vertcoreg_results/").glob("vertcoreg*meta.json"),
        ):

            zip_file.write(filepath, filepath.name)

            

    with zipfile.ZipFile(inner_zips["coreg_results.zip"], "w", compression=zipfile.ZIP_DEFLATED) as zip_file:

        for filepath in Path("temp.svalbard/arcticdem_coreg/").glob("*.json"):
            zip_file.write(filepath, filepath.name)


    with zipfile.ZipFile("temp.svalbard/aux_files.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as zip_file:
        for filepath in map(Path, files):
            zip_file.write(filepath, filepath.name)

        for name in inner_zips:
            zip_file.writestr(name, inner_zips[name].getvalue())
    

          
if __name__ == "__main__":
    main()
