import pickle
import time
import typing
import warnings
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import scipy.ndimage
import scipy.spatial
import sklearn.ensemble
from tqdm import tqdm

import adsvalbard.utilities
from adsvalbard.constants import CONSTANTS


def get_eastings_northings(
    bounds: rasterio.coords.BoundingBox, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    res_hori = (bounds.right - bounds.left) / width
    res_vert = (bounds.top - bounds.bottom) / height
    eastings, northings = np.meshgrid(
        np.linspace(bounds.left + res_hori / 2, bounds.right - res_hori / 2, num=width),
        np.linspace(
            bounds.bottom + res_vert / 2, bounds.top - res_vert / 2, num=height
        )[::-1],
    )

    return eastings, northings


def get_patch_polygs() -> tuple[gpd.GeoDataFrame, str]:
    patch_polygs = gpd.read_file("shapes/arcticdem_bad_patches.geojson")

    checksum = adsvalbard.utilities.get_checksum([patch_polygs])

    return patch_polygs, checksum


def get_bad_patch_stats(
    force_redo: bool = False, add_extra: bool = False
) -> pd.DataFrame:
    """Sample data for training a gap detection algorithm.

    Parameters
    ----------
    force_redo
        Reprocess the data even if there is a cached version.
    add_extra
        Add extra good points around bad patches. Assumes everything around the bad patch is good.

    Returns
    -------
    A very big dataframe of labelled data to use for training/testing.

    """

    patch_polygs, checksum = get_patch_polygs()
    cache_filepath = (
        CONSTANTS.cache_dir
        / f"get_bad_patch_stats/get_bad_patch_stats-{checksum}.feather"
    )
    if cache_filepath.is_file() and not force_redo:
        return gpd.read_feather(cache_filepath)
    cache_filepath.parent.mkdir(exist_ok=True, parents=True)

    rng: np.random.Generator = np.random.default_rng(0)

    duplicated_patch_ids = pd.Series(
        *(np.unique(patch_polygs["patch_id"], return_counts=True))[::-1]
    )
    duplicated_patch_ids = duplicated_patch_ids[duplicated_patch_ids > 1]

    if duplicated_patch_ids.shape[0] > 0:
        raise ValueError(f"Found duplicated `patch_id`s\n{duplicated_patch_ids}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*'nopython' keyword.*")
        import xdem

    all_data = []
    with tqdm(
        total=patch_polygs.shape[0], desc="Getting patch stats", smoothing=0.0
    ) as progress_bar:
        for filename, file_data in patch_polygs.groupby("filename"):
            filepath = Path("temp.svalbard/arcticdem_coreg/") / f"{filename}.tif"

            year = int(filename.split("_")[3][:4])

            median_path = filepath.parent.parent / f"median_dem_{year}.tif"
            dh_path = filepath.parent.parent / f"dh/{year}/{filepath.stem}_dh.tif"
            dt_path = filepath.parent.parent / f"dt/{year}/{filepath.stem}_dt.tif"

            extra_paths = {
                "median_dem": median_path,
                "npi_dt": dt_path,
                "npi_dh": dh_path,
            }
            rasters = {
                key: xdem.DEM(str(fp), load_data=False)
                for key, fp in extra_paths.items()
            }
            dem = xdem.DEM(str(filepath))
            progress_bar.set_description("Loading DEM")

            progress_bar.set_description("Making gap distance map")
            rasters["gap_distance"] = xdem.DEM.from_array(
                scipy.ndimage.distance_transform_edt(~dem.data.mask) * dem.res[0],
                transform=dem.transform,
                crs=dem.crs,
            )

            per_filepath = []

            progress_bar.set_description("Getting patch stats")
            for _, patch in file_data.iterrows():

                # Crop the DEM to the patch, plus a buffer
                try:
                    buffer = 100 if add_extra else 10
                    new_bounds = adsvalbard.utilities.align_bounds(
                        rasterio.coords.BoundingBox(*patch.geometry.bounds),
                        buffer=buffer,
                    )
                    dem_sub = dem.crop(new_bounds, inplace=False)

                    rst_sub = {
                        key: raster.crop(dem_sub, inplace=False)
                        for key, raster in rasters.items()
                    }
                    # TODO: Simply refer this to the dict instead of declaring a variable
                    dist_sub = rst_sub["gap_distance"]
                except rasterio.windows.WindowError as exception:
                    raise ValueError(
                        f"Patch does not overlap with DEM;\n{patch}"
                    ) from exception

                # Convert the patch to a mask
                patch_mask = (
                    rasterio.features.rasterize(
                        [patch.geometry],
                        out_shape=dem_sub.shape,
                        transform=dem_sub.transform,
                    )
                    == 1
                )

                # Generate quadric coefficients for the terrain. These contain information such as slope, curvature, aspect, etc.
                coefs = xdem.terrain.get_quadric_coefficients(
                    dem_sub.data, resolution=dem_sub.res[0]
                )

                data_mask = np.isfinite(coefs[0])
                # Make a mask for data that exist and are within the patch
                in_patch_mask = data_mask & patch_mask
                # Make a mask for data that exist and are outside of the patch
                outside_patch_mask = data_mask & (~in_patch_mask)

                exact_mask = np.ones_like(in_patch_mask)

                # First generate easting/northing coordinates, then extract the coordinates of the finite values in the patch
                eastings, northings = get_eastings_northings(
                    dem_sub.bounds, *dem_sub.shape
                )

                flagged = np.ones(in_patch_mask.shape, dtype=int)
                # For "bad"-type patches, generate points outside of the patch and classify them as good
                if patch["type"] == "bad" and add_extra:

                    gap_distances_in_patch = dist_sub.data[in_patch_mask]
                    dist_bins = np.linspace(
                        gap_distances_in_patch.min(),
                        gap_distances_in_patch.max(),
                        num=11,
                    )

                    # Initialize an index array. Extra point (pixel) indices will be added here
                    extra_points = np.empty((0,), dtype=int)

                    # For each sampled bad point, try to find a "good" (unmarked) equivalent
                    # In these distance bins, try to match the amount of points in that bin and add as control points.
                    for i in range(1, dist_bins.shape[0]):
                        # First, find valid pixels with compatible distances
                        compat_distances = (
                            (dist_sub.data >= dist_bins[i - 1])
                            & (dist_sub.data <= dist_bins[i])
                            & outside_patch_mask
                        )
                        # Derive the indices of these compatible pixels
                        compat_indices = np.argwhere(compat_distances.ravel()).ravel()

                        # There might not be a single one, so then the bin has to be skipped.
                        if compat_indices.shape[0] == 0:
                            continue

                        # Count the number of points in this interval that were sampled within the patch
                        n_in_patch = np.count_nonzero(
                            (gap_distances_in_patch >= dist_bins[i - 1])
                            & (gap_distances_in_patch <= dist_bins[i])
                        )
                        # The number of points to sample should preferably be equal to inside the patch.
                        # This might not be possible though, as the number of compatible indices might be smaller
                        # Extract as many points as possible up to the number that was sampled in the patch
                        n_to_sample = min(n_in_patch, compat_indices.shape[0])

                        # Save a random choice of compatible distance points
                        extra_points = np.append(
                            extra_points, rng.choice(compat_indices, size=n_to_sample)
                        )

                    flagged.ravel()[extra_points] = 0
                    in_patch_mask.ravel()[extra_points] = True
                    exact_mask.ravel()[extra_points] = False

                coords_sub = np.transpose(
                    [coord[in_patch_mask] for coord in [eastings, northings]]
                )

                data = gpd.GeoDataFrame(
                    {
                        "bad": (flagged[in_patch_mask] == 1)
                        if (patch["type"] == "bad")
                        else np.zeros_like(flagged, dtype=bool)[in_patch_mask],
                        "exact": exact_mask[in_patch_mask],
                    },
                    geometry=gpd.points_from_xy(
                        coords_sub[:, 0], coords_sub[:, 1], crs=CONSTANTS.crs_epsg
                    ),
                )

                for key, arr in [
                    ("dem", dem_sub),
                    *list(rst_sub.items()),
                ]:  # , ("slope", slope), ("curvature", curv)]:
                    try:
                        data[key] = arr.data[in_patch_mask].filled(np.nan)
                    except IndexError as exception:
                        raise ValueError(
                            f"Failed to sample {key} on patch;\n{patch}"
                        ) from exception

                for j in range(coefs.shape[0]):
                    data[f"quadric_{str(j).zfill(2)}"] = coefs[j][in_patch_mask]

                data["patch_id"] = patch["patch_id"]

                per_filepath.append(data)
                progress_bar.update()

            # Concatenate all patch data for this filepath, and then sample the final values.
            per_filepath = pd.concat(per_filepath)

            all_data.append(per_filepath)

    pd.concat(all_data).to_feather(cache_filepath)

    return gpd.read_feather(cache_filepath)


class GapModel:
    def __init__(
        self,
        training_checksum: str,
        model: sklearn.ensemble.HistGradientBoostingClassifier,
        importance: pd.DataFrame,
        score: float,
    ):
        self.training_checksum = training_checksum
        self.model = model
        self.importance = importance
        self.score = score

    def to_pickle(self, filepath: Path) -> None:
        filepath.parent.mkdir(exist_ok=True)
        with open(filepath, "wb") as outfile:
            pickle.dump(self, outfile, protocol=5)

    @classmethod
    def from_pickle(cls, filepath: Path) -> typing.Self:
        with open(filepath, "rb") as infile:
            return pickle.load(infile)

    @classmethod
    def train(cls, force_redo: bool = False) -> typing.Self:
        _, checksum = get_patch_polygs()
        all_patches = get_bad_patch_stats(force_redo=force_redo)
        all_patches = all_patches[all_patches["exact"]]
        bad_vs_good_frac = (
            all_patches[all_patches["bad"]].shape[0] / all_patches.shape[0]
        )

        print(
            f"{bad_vs_good_frac * 100:.2f}% of values are flagged as bad. The rest are good."
        )

        all_patches = all_patches.dropna(how="all", axis="columns")


        import sklearn.inspection
        import sklearn.pipeline
        import sklearn.preprocessing

        x_cols = ["gap_distance", "npi_dh", "npi_dt", "median_dem"] + [
            str(col) for col in all_patches if "quadric_" in col
        ]
        y_col = "bad"

        xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(
            all_patches[x_cols], all_patches[y_col], random_state=0
        )

        model = sklearn.ensemble.HistGradientBoostingClassifier(random_state=0)
        model.fit(xtrain, ytrain)

        score = model.score(xtest, ytest)

        print(f"Model score: {score:.3f}")

        print("Asessing feature importance. This takes time..")
        importance = sklearn.inspection.permutation_importance(
            model, xtest, ytest, n_repeats=20, random_state=0, n_jobs=-1
        )

        importance_df = pd.DataFrame(
            {
                "name": model.feature_names_in_,
                "mean": importance.importances_mean,
                "std": importance.importances_std,
            }
        ).set_index("name")

        return cls(
            training_checksum=checksum,
            model=model,
            importance=importance_df,
            score=score,
        )

    @classmethod
    def default(cls, force_redo: bool = False) -> typing.Self:
        cache_path = CONSTANTS.cache_dir / "gap_model.pkl"

        _, checksum = get_patch_polygs()

        if cache_path.is_file() and not force_redo:
            model_result = cls.from_pickle(cache_path)

            if model_result.training_checksum != checksum:
                print(
                    "Model training data checksum is not the same as the stored checksum. Re-training recommended."
                )

            return model_result

        results = cls.train(force_redo=force_redo)
        results.to_pickle(cache_path)

        return results

    def __repr__(self) -> str:
        return (
            f"Model: {self.model}, score: {self.score:.3f}\n"
            f"Training checksum: {self.training_checksum[:8]}..."
        )


@dataclass(frozen=True)
class GroupedCVResult:
    """Results of patch-grouped cross-validation for the gap-detection model.

    This evaluates the *modeling procedure* used by `GapModel.train()`, not the exact
    fitted model that produced the outlier rasters currently on disk. `GapModel.default()`
    and its cached predictions are untouched by this evaluation; see
    `evaluate_gap_model_grouped()` for why the two need to be kept separate.

    Attributes
    ----------
    fold_scores
        One row per fold: classification metrics plus train/test group- and
        class-balance diagnostics.
    summary_scores
        Mean and std of each metric in `fold_scores`, aggregated across folds.
    fold_importance
        One row per (fold, feature): permutation importance mean/std within that
        fold's held-out subsample.
    summary_importance
        Mean and std of the per-fold importance mean, aggregated across folds.
        Indexed by feature name, matching the shape of `GapModel.importance`.
    """

    fold_scores: pd.DataFrame
    summary_scores: pd.DataFrame
    fold_importance: pd.DataFrame
    summary_importance: pd.DataFrame


def evaluate_gap_model_grouped(
    all_patches: pd.DataFrame,
    n_splits: int = 5,
    importance_sample_size: int = 200_000,
    importance_n_repeats: int = 10,
    random_state: int = 0,
    save_csv: bool = True,
    output_dir: Path = Path("temp.svalbard"),
) -> GroupedCVResult:
    """Evaluate the gap-detection modeling procedure with patch-grouped cross-validation.

    `GapModel.train()` evaluates a single point-level `train_test_split`, which lets
    pixels from the same labelled polygon (`patch_id`) land in both the training and
    testing sets. Because pixels within a polygon are spatially correlated, this leaks
    information and produces overly optimistic scores.

    This function instead uses `StratifiedGroupKFold`, keeping every pixel of a given
    `patch_id` in the same fold, and reports mean +/- std across folds rather than a
    single score from one arbitrary polygon partition.

    In the labelled data, almost every patch is homogeneously "bad" or "good" (the
    per-pixel label is effectively a per-patch label), and patch sizes are highly skewed
    (tens to hundreds of thousands of pixels for the largest patches). Plain `GroupKFold`
    can therefore leave some folds with a very different bad/good ratio than others;
    `StratifiedGroupKFold` balances the class ratio across folds while still keeping
    patches intact.

    This is a standalone evaluation, not a method on `GapModel`: it has a different output
    shape (per-fold scores/importances vs. a single fitted model) and a different purpose
    (estimating how well the procedure generalizes to unseen patches, vs. producing the
    deployed model artifact). It does not read or write the `GapModel` pickle cache, and
    does not affect `create_gap_mask()` / `generate_all_masks()`.

    Parameters
    ----------
    all_patches
        Labelled per-pixel training data, as returned by `get_bad_patch_stats()`. Must
        contain `patch_id`, `bad`, `exact`, and the feature columns used by `GapModel`.
    n_splits
        Number of grouped CV folds.
    importance_sample_size
        Maximum number of held-out rows per fold used for permutation importance, to
        bound runtime regardless of fold size (folds range from ~1k to several million
        rows). The full fold is used if it is smaller than this.
    importance_n_repeats
        Number of shuffle repeats per feature in
        `sklearn.inspection.permutation_importance`.
    random_state
        Seed for the fold splitter and the importance subsampling. The classifier itself
        is always seeded with `random_state=0`, matching `GapModel.train()`.
    save_csv
        Whether to write the four result tables as CSVs under `output_dir`.
    output_dir
        Directory the CSVs are written to when `save_csv` is True.

    Returns
    -------
    A `GroupedCVResult` with per-fold and aggregated scores and permutation importances.
    """
    import sklearn.inspection
    import sklearn.metrics
    import sklearn.model_selection

    all_patches = all_patches[all_patches["exact"]]
    all_patches = all_patches.dropna(how="all", axis="columns")

    bad_vs_good_frac = all_patches["bad"].sum() / all_patches.shape[0]
    print(
        f"{bad_vs_good_frac * 100:.2f}% of values are flagged as bad. The rest are good."
    )

    x_cols = ["gap_distance", "npi_dh", "npi_dt", "median_dem"] + [
        str(col) for col in all_patches if "quadric_" in col
    ]
    y_col = "bad"

    features = all_patches[x_cols]
    labels = all_patches[y_col]
    groups = all_patches["patch_id"]

    splitter = sklearn.model_selection.StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    rng = np.random.default_rng(random_state)

    score_rows = []
    importance_frames = []

    for fold, (train_idx, test_idx) in tqdm(
        enumerate(splitter.split(features, labels, groups=groups)),
        total=n_splits,
        desc="Grouped CV",
        smoothing=0.0,
    ):
        x_train, x_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

        model = sklearn.ensemble.HistGradientBoostingClassifier(random_state=0)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
            y_test, y_pred, labels=[False, True]
        ).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else np.nan
        )

        score_rows.append(
            {
                "fold": fold,
                "n_train": x_train.shape[0],
                "n_test": x_test.shape[0],
                "n_patches_train": groups.iloc[train_idx].nunique(),
                "n_patches_test": groups.iloc[test_idx].nunique(),
                "frac_bad_train": float(y_train.mean()),
                "frac_bad_test": float(y_test.mean()),
                "accuracy": sklearn.metrics.accuracy_score(y_test, y_pred),
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "fnr": fnr,
                "fpr": fpr,
            }
        )

        sample_size = min(importance_sample_size, x_test.shape[0])
        sample_idx = rng.choice(x_test.shape[0], size=sample_size, replace=False)

        importance = sklearn.inspection.permutation_importance(
            model,
            x_test.iloc[sample_idx],
            y_test.iloc[sample_idx],
            n_repeats=importance_n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )
        importance_frames.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "name": model.feature_names_in_,
                    "importance_mean": importance.importances_mean,
                    "importance_std": importance.importances_std,
                }
            )
        )

    fold_scores = pd.DataFrame(score_rows).set_index("fold")

    metric_cols = ["accuracy", "precision", "recall", "specificity", "f1", "fnr", "fpr"]
    summary_scores = fold_scores[metric_cols].agg(["mean", "std"]).T
    summary_scores.columns = ["mean", "std"]

    fold_importance = pd.concat(importance_frames, ignore_index=True)
    summary_importance = fold_importance.groupby("name")["importance_mean"].agg(
        ["mean", "std"]
    )

    result = GroupedCVResult(
        fold_scores=fold_scores,
        summary_scores=summary_scores,
        fold_importance=fold_importance,
        summary_importance=summary_importance,
    )

    if save_csv:
        output_dir.mkdir(parents=True, exist_ok=True)
        result.fold_scores.to_csv(
            output_dir / "gap_model_grouped_cv_scores_by_fold.csv"
        )
        result.summary_scores.to_csv(
            output_dir / "gap_model_grouped_cv_scores_summary.csv"
        )
        result.fold_importance.to_csv(
            output_dir / "gap_model_grouped_cv_importance_by_fold.csv", index=False
        )
        result.summary_importance.to_csv(
            output_dir / "gap_model_grouped_cv_importance_summary.csv"
        )

    return result


def make_gap_mask_name(filename: str, parent: Path | None = None) -> Path:
    year = int(filename.split("_")[3][:4])

    if parent is None:
        parent = Path("temp.svalbard/outlier_proba/")

    return parent / f"{year}/{filename.replace('.tif', '')}_outlier_proba.tif"


def create_gap_mask(
    filename: str, model: GapModel, verbose: bool = True, force_redo: bool = False
) -> Path:
    out_path = make_gap_mask_name(filename)
    if out_path.is_file() and not force_redo:
        return out_path

    start_time = time.time()

    def update(text: str):
        if not verbose:
            return
        duration = time.time() - start_time

        print(f"+{duration:.2f}s\t{text}")

    year = int(filename.split("_")[3][:4])
    filepaths = {
        "dem": Path(f"temp.svalbard/arcticdem_coreg/{filename}.tif"),
    }
    filepaths.update(
        {
            "median_dem": filepaths["dem"].parent.parent / f"median_dem_{year}.tif",
            "npi_dh": filepaths["dem"].parent.parent / f"dh/{year}/{filename}_dh.tif",
            "npi_dt": filepaths["dem"].parent.parent / f"dt/{year}/{filename}_dt.tif",
        }
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*'nopython' keyword.*")
        import xdem
    import scipy.ndimage

    update("Loading DEM")
    dem = xdem.DEM(filepaths["dem"])

    data_mask = ~scipy.ndimage.binary_dilation(
        dem.data.mask, structure=scipy.ndimage.generate_binary_structure(2, 2)
    )

    vals = pd.DataFrame({"dem": dem.data.filled(np.nan).ravel()})
    x_cols = [str(col) for col in model.importance.index]

    if "gap_distance" in x_cols:
        update("Making gap distances")

        dist = scipy.ndimage.distance_transform_edt(~dem.data.mask) * dem.res[0]
        vals["gap_distance"] = dist.ravel()
        del dist

        # del tree
        # del eastings
        # del northings

    if any(f"quadric_{str(i).zfill(2)}" in x_cols for i in range(11)):
        update("Making quadrics")

        quadric = xdem.terrain.get_quadric_coefficients(dem.data, resolution=dem.res[0])
        for i in range(quadric.shape[0]):
            vals[f"quadric_{str(i).zfill(2)}"] = quadric[i].ravel()

        del quadric

    for key in ["median_dem", "npi_dh", "npi_dt"]:
        if key not in x_cols:
            continue
        update(f"Loading {key}")

        data = xdem.DEM(filepaths[key], load_data=False)
        data.crop(dem.bounds)

        vals[key] = data.data.filled(np.nan).ravel()

    update("Predicting gap error probability")

    outliers = np.ones(vals.shape[0], dtype="uint8").reshape(dem.shape) * 255

    outliers[data_mask] = (
        model.model.predict_proba(vals[x_cols].loc[data_mask.ravel()])[:, 1] * 255
    ).astype("uint8")

    outliers = xdem.DEM.from_array(outliers, crs=dem.crs, transform=dem.transform)
    del dem

    update(f"Saving {out_path}")

    out_path.parent.mkdir(exist_ok=True, parents=True)
    outliers.save(out_path, tiled=True, co_opts={"ZLEVEL": "12"})

    return out_path


def generate_all_masks(
    subdir: str = "arcticdem_coreg", verbose_progress: bool = False
):
    filestems = list(
        map(
            lambda fp: fp.stem,
            sorted(
                Path(f"temp.svalbard/{subdir}/").rglob("*.tif"),
                key=lambda fp: fp.stem.split("_")[3],
                reverse=True,
            )
        )
    )

    model_result = GapModel.default()

    finished = []
    to_process = []

    for i, stem in enumerate(filestems):
        gap_path = make_gap_mask_name(filename=stem)

        if gap_path.is_file():
            finished.append(gap_path)
        else:
            to_process.append(i)

    with tqdm(total=len(to_process), disable=verbose_progress) as progress_bar:
        for i in to_process:
            stem = filestems[i]
            progress_bar.set_description(stem)
            try:
                create_gap_mask(filename=stem, model=model_result, verbose=verbose_progress)
            except ValueError as exception:
                print(f"Exception: {exception}")

            progress_bar.update()

    return


