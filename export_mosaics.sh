#!/usr/bin/env bash

set -u
shopt -s nullglob

base_dir="temp.svalbard/filt/svalbard/mosaics_3584"
out_dir="$base_dir/out"

skip_vrt_patterns=(
  "*_se.vrt"
  "*_spatial_err.vrt"
  "*_temporal_err.vrt"
  "*accel*"
)

declare -a failed_files=()
declare -a corruption_hits=()

pythonso_from_active_python() {
    python3 - <<'PY'
import os
import sysconfig

libdir = sysconfig.get_config_var("LIBDIR")
ldlibrary = sysconfig.get_config_var("LDLIBRARY")

if not libdir or not ldlibrary:
    raise SystemExit(1)

print(os.path.join(libdir, ldlibrary))
PY
}

move_with_sidecars() {
    local src="$1"
    local dst="$2"

    mv -- "$src" "$dst"

    # Rename common GDAL sidecars too, if they were created.
    for suffix in .ovr .aux.xml .msk; do
        if [[ -e "${src}${suffix}" ]]; then
            mv -- "${src}${suffix}" "${dst}${suffix}"
        fi
    done
}

should_skip_vrt() {
    local vrt_path="$1"
    local vrt_name
    vrt_name="$(basename -- "$vrt_path")"

    local pattern
    for pattern in "${skip_vrt_patterns[@]}"; do
        if [[ "$vrt_name" == $pattern ]]; then
            return 0
        fi
    done

    return 1
}

vrt_files=("$base_dir"/*.vrt)

if (( ${#vrt_files[@]} == 0 )); then
    echo "No VRT files found in $base_dir" >&2
    exit 0
fi

for vrt_file in "${vrt_files[@]}"; do
    if should_skip_vrt "$vrt_file"; then
        echo "Skipping VRT: $(basename -- "$vrt_file")"
        continue
    fi

    tif_file="$out_dir/$(basename -- "${vrt_file%.vrt}").tif"

    # 1) Skip files that already exist
    if [[ -e "$tif_file" ]]; then
        echo "Skipping existing file: $tif_file"
        continue
    fi

    out_dir_for_file="$(dirname -- "$tif_file")"
    base_name="$(basename -- "${tif_file%.tif}")"

    # 2) Work on a temporary file, then rename at the very end
    tmp_tif="$(mktemp --tmpdir="$out_dir_for_file" --suffix=".tif" ".${base_name}.tmp.XXXXXX")"

    echo "Processing: $vrt_file -> $tif_file"
    echo "  Temporary file: $tmp_tif"

    pythonso="$(pythonso_from_active_python)"

    if [[ -z "$pythonso" ]]; then
        echo "ERROR: could not determine PYTHONSO from active python3" >&2
        failed_files+=("$vrt_file (pythonso)")
        continue
    fi

    if ! PYTHONSO="$pythonso" GDAL_VRT_ENABLE_PYTHON=YES gdal raster convert \
        --co "COMPRESS=DEFLATE" \
        --co "ZLEVEL=9" \
        --co "NUM_THREADS=ALL_CPUS" \
        --co "TILED=YES" \
        --co "BIGTIFF=YES" \
        --co "PREDICTOR=2" \
        --overwrite \
        "$vrt_file" \
        "$tmp_tif"
    then
        echo "ERROR: gdal raster convert failed for '$vrt_file'" >&2
        echo "       Temporary file kept: $tmp_tif" >&2
        failed_files+=("$vrt_file (convert)")
        continue
    fi

    if ! gdaladdo \
        -r average \
        --config COMPRESS_OVERVIEW DEFLATE \
        --config ZLEVEL_OVERVIEW 9 \
        --config PREDICTOR_OVERVIEW 2 \
        --config GDAL_NUM_THREADS ALL_CPUS \
        --config BIGTIFF_OVERVIEW YES \
        "$tmp_tif"
    then
        echo "ERROR: gdaladdo failed for '$tmp_tif'" >&2
        echo "       Temporary file kept for inspection." >&2
        failed_files+=("$vrt_file (gdaladdo)")
        continue
    fi

    if ! gdal_edit -stats "$tmp_tif"; then
        echo "ERROR: gdal_edit -stats failed for '$tmp_tif'" >&2
        echo "       Temporary file kept for inspection." >&2
        failed_files+=("$vrt_file (gdal_edit)")
        continue
    fi

    # 3) Run gdalinfo and detect the TIFF corruption signature you mentioned
    gdalinfo_output="$(gdalinfo "$tmp_tif" 2>&1)"
    gdalinfo_status=$?

    if (( gdalinfo_status != 0 )); then
        echo "ERROR: gdalinfo failed for '$tmp_tif' (not renaming to final name)" >&2
        printf '%s\n' "$gdalinfo_output" >&2

        if grep -qE \
            'TIFFFetchDirectory:.*Can not read TIFF directory count|TIFFReadDirectory:Failed to read directory at offset|gdalinfo failed - unable to open' \
            <<<"$gdalinfo_output"
        then
            echo "!!! CORRUPTION SIGNATURE DETECTED for '$tmp_tif'" >&2
            corruption_hits+=("$vrt_file -> $tmp_tif")
        fi

        failed_files+=("$vrt_file (gdalinfo)")
        continue
    fi

    # Extra safety: catch the signature even if gdalinfo unexpectedly exits 0
    if grep -qE \
        'TIFFFetchDirectory:.*Can not read TIFF directory count|TIFFReadDirectory:Failed to read directory at offset|gdalinfo failed - unable to open' \
        <<<"$gdalinfo_output"
    then
        echo "!!! CORRUPTION SIGNATURE DETECTED in gdalinfo output for '$tmp_tif'" >&2
        printf '%s\n' "$gdalinfo_output" >&2
        failed_files+=("$vrt_file (gdalinfo corruption signature)")
        corruption_hits+=("$vrt_file -> $tmp_tif")
        continue
    fi

    if ! move_with_sidecars "$tmp_tif" "$tif_file"; then
        echo "ERROR: failed to rename '$tmp_tif' to '$tif_file'" >&2
        failed_files+=("$vrt_file (rename)")
        continue
    fi

    echo "Finished: $tif_file"
done

echo
echo "==== Summary ===="

if (( ${#failed_files[@]} > 0 )); then
    echo "Failures:"
    printf '  - %s\n' "${failed_files[@]}"
else
    echo "No failures."
fi

if (( ${#corruption_hits[@]} > 0 )); then
    echo
    echo "Detected TIFF directory corruption signature in:"
    printf '  - %s\n' "${corruption_hits[@]}"
    exit 2
fi

(( ${#failed_files[@]} == 0 )) || exit 1
exit 0
