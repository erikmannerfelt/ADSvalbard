#!/usr/bin/env bash

filenames=("temp/npi_mosaic.vrt temp/npi_slope.tif" "temp/npi_slope.tif temp/npi_slope_of_slope.tif") 

for pair in "${filenames[@]}"; do
  pair=($pair)
  in_fp=${pair[0]};
  out_fp=${pair[1]};

  echo "$in_fp -> $out_fp"
  if ! [[ -f "$out_fp" ]]; then
    gdal raster pipeline ! read "$in_fp" ! unscale ! slope ! scale --ot "Uint16" --src-min 0 --src-max 90 --dst-min 0 --dst-max 65535 ! write --co "COMPRESS=DEFLATE" --co "ZLEVEL=9" --co "TILED=YES" --co "BIGTIFF=YES" -f "GTiff" "${out_fp}.tmp"

    mv "${out_fp}.tmp" "$out_fp"

    gdal_edit -scale 0.001373311970702678 "$out_fp"
  fi
done

