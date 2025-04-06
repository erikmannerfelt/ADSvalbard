import pickle
import time
import typing
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
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


def get_eastings_northings(bounds: rasterio.coords.BoundingBox, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:

    res_hori = (bounds.right - bounds.left) / width
    res_vert = (bounds.top - bounds.bottom) / height
    eastings, northings = np.meshgrid(
        np.linspace(bounds.left + res_hori / 2, bounds.right - res_hori / 2, num=width),
        np.linspace(bounds.bottom + res_vert / 2, bounds.top - res_vert / 2, num=height)[::-1],
    )

    return eastings, northings
    

def get_gap_distance_tree(eastings: np.ndarray, northings: np.ndarray, nodata_mask: np.ndarray):
    import scipy.ndimage
    import sklearn.neighbors

    # eastings, northings = get_eastings_northings(bounds, *shape)
    # grad_x, grad_y = np.gradient(dem.data.mask.astype(int))

    grad = scipy.ndimage.convolve(nodata_mask.astype(int), [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], mode="constant") > 1

    # grad = xdem.terrain.slope(dem.data.mask.astype(int), resolution=dem.res[0])

    # plt.imshow(grad)
    # plt.show()

    # # grad = np.max([np.abs(grad_x), np.abs(grad_y)], axis=0)

    # # print(grad.shape, dem.data.shape)

    # return
    return sklearn.neighbors.KDTree(np.transpose([coord[grad] for coord in [eastings, northings]]))
    

def get_patch_polygs() -> tuple[gpd.GeoDataFrame, str]:
    patch_polygs = gpd.read_file("shapes/arcticdem_bad_patches.geojson")


    checksum = adsvalbard.utilities.get_checksum([patch_polygs])

    return patch_polygs, checksum

def get_bad_patch_stats(force_redo: bool = False, add_extra: bool = False) -> pd.DataFrame:
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
    # patch_polygs = gpd.read_file("shapes/arcticdem_bad_patches.geojson")


    # checksum = adsvalbard.utilities.get_checksum([patch_polygs])

    patch_polygs, checksum = get_patch_polygs()
    cache_filepath = CONSTANTS.cache_dir / f"get_bad_patch_stats/get_bad_patch_stats-{checksum}.feather"
    if cache_filepath.is_file() and not force_redo:
        return gpd.read_feather(cache_filepath)
    cache_filepath.parent.mkdir(exist_ok=True, parents=True)

    rng: np.random.Generator = np.random.default_rng(0)

    duplicated_patch_ids = pd.Series(*(np.unique(patch_polygs["patch_id"], return_counts=True))[::-1])
    duplicated_patch_ids = duplicated_patch_ids[duplicated_patch_ids > 1]

    if duplicated_patch_ids.shape[0] > 0:
        raise ValueError(f"Found duplicated `patch_id`s\n{duplicated_patch_ids}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*'nopython' keyword.*")
        import xdem

    # patch_id = 0

    all_data = []
    with tqdm(total=patch_polygs.shape[0], desc="Getting patch stats", smoothing=0.) as progress_bar:
        for filename, file_data in patch_polygs.groupby("filename"):
            filepath =  Path("temp.svalbard/arcticdem_coreg/") / f"{filename}.tif"

            year = int(filename.split("_")[3][:4])

            median_path = filepath.parent.parent / f"median_dem_{year}.tif"
            dh_path = filepath.parent.parent / f"dh/{year}/{filepath.stem}_dh.tif"
            dt_path = filepath.parent.parent / f"dt/{year}/{filepath.stem}_dt.tif"

            extra_paths = {
                "median_dem": median_path,
                "npi_dt": dt_path,
                "npi_dh": dh_path,
            }
            rasters = {key: xdem.DEM(str(fp), load_data=False) for key, fp in extra_paths.items()}
            dem = xdem.DEM(str(filepath))
            progress_bar.set_description("Loading DEM")

            progress_bar.set_description("Making gap distance map")
            rasters["gap_distance"] = xdem.DEM.from_array(
                scipy.ndimage.distance_transform_edt(~dem.data.mask) * dem.res[0],
                transform=dem.transform,
                crs=dem.crs
            )
            
            # tree = scipy.spatial.KDTree(np.transpose([coord[dem.data.mask] for coord in get_eastings_northings(dem.bounds, *dem.shape)]))

            # distance_field = tree.query(np.transpose([eastings.ravel(), northings.ravel()]), workers=-1)[0].reshape(eastings.shape)

            per_filepath = []

            progress_bar.set_description("Getting patch stats")
            for _, patch in file_data.iterrows():

                # print(patch)

                # Crop the DEM to the patch, plus a buffer
                try:
                    buffer = 100 if add_extra else 10
                    new_bounds = adsvalbard.utilities.align_bounds(rasterio.coords.BoundingBox(*patch.geometry.bounds), buffer=buffer)
                    dem_sub = dem.crop(new_bounds, inplace=False)

                    rst_sub = {key: raster.crop(dem_sub, inplace=False) for key, raster in rasters.items()}
                    # TODO: Simply refer this to the dict instead of declaring a variable
                    dist_sub = rst_sub["gap_distance"]
                except rasterio.windows.WindowError as exception:
                    raise ValueError(f"Patch does not overlap with DEM;\n{patch}") from exception
                    

                # Convert the patch to a mask
                patch_mask = rasterio.features.rasterize([patch.geometry], out_shape=dem_sub.shape, transform=dem_sub.transform) == 1

                # Generate quadric coefficients for the terrain. These contain information such as slope, curvature, aspect, etc.
                coefs = xdem.terrain.get_quadric_coefficients(dem_sub.data, resolution=dem_sub.res[0])

                data_mask = np.isfinite(coefs[0])
                # Make a mask for data that exist and are within the patch
                in_patch_mask = data_mask & patch_mask
                # Make a mask for data that exist and are outside of the patch 
                outside_patch_mask = data_mask & (~in_patch_mask)

                exact_mask = np.ones_like(in_patch_mask)

                # First generate easting/northing coordinates, then extract the coordinates of the finite values in the patch 
                eastings, northings = get_eastings_northings(dem_sub.bounds, *dem_sub.shape)

                flagged = np.ones(in_patch_mask.shape, dtype=int)
                # For "bad"-type patches, generate points outside of the patch and classify them as good
                if patch["type"] == "bad" and add_extra:
                    # coords_in_patch = np.transpose([coord[in_patch_mask] for coord in [eastings, northings]])

                    # Get the distances to a gap of the values in the patch
                    # gap_distances_in_patch = tree.query(coords_in_patch)[0]
                    gap_distances_in_patch = dist_sub.data[in_patch_mask]
                    dist_bins = np.linspace(gap_distances_in_patch.min(), gap_distances_in_patch.max(), num=11)
                    
                    # Calculate distances for every pixel in order to find matching training points outside of the patch
                    # distance_field = tree.query(np.transpose([eastings.ravel(), northings.ravel()]), workers=-1)[0].reshape(eastings.shape)

                    # Initialize an index array. Extra point (pixel) indices will be added here
                    extra_points = np.empty((0,), dtype=int)

                    # For each sampled bad point, try to find a "good" (unmarked) equivalent
                    # In these distance bins, try to match the amount of points in that bin and add as control points.
                    for i in range(1, dist_bins.shape[0]):
                        # First, find valid pixels with compatible distances
                        compat_distances = (dist_sub.data >= dist_bins[i -1]) & (dist_sub.data <= dist_bins[i]) & outside_patch_mask
                        # Derive the indices of these compatible pixels
                        compat_indices = np.argwhere(compat_distances.ravel()).ravel()

                        # There might not be a single one, so then the bin has to be skipped.
                        if compat_indices.shape[0] == 0:
                            continue

                        # Count the number of points in this interval that were sampled within the patch
                        n_in_patch = np.count_nonzero((gap_distances_in_patch >= dist_bins[i - 1]) & (gap_distances_in_patch <= dist_bins[i]))
                        # The number of points to sample should preferably be equal to inside the patch.
                        # This might not be possible though, as the number of compatible indices might be smaller
                        # Extract as many points as possible up to the number that was sampled in the patch
                        n_to_sample = min(n_in_patch, compat_indices.shape[0])

                        # Save a random choice of compatible distance points
                        extra_points = np.append(extra_points, rng.choice(compat_indices, size=n_to_sample))



                    flagged.ravel()[extra_points] = 0
                    in_patch_mask.ravel()[extra_points] = True
                    exact_mask.ravel()[extra_points] = False


                coords_sub = np.transpose([coord[in_patch_mask] for coord in [eastings, northings]])


                data = gpd.GeoDataFrame(
                    {
                        # "gap_distance": #tree.query(coords_sub)[0],
                        "bad": (flagged[in_patch_mask] == 1) if (patch["type"] == "bad") else np.zeros_like(flagged, dtype=bool)[in_patch_mask],
                        "exact": exact_mask[in_patch_mask],
                    },
                    geometry=gpd.points_from_xy(coords_sub[:, 0], coords_sub[:, 1], crs=CONSTANTS.crs_epsg)
                )


                for key, arr in [("dem", dem_sub), *list(rst_sub.items())]:#, ("slope", slope), ("curvature", curv)]:
                    try:
                        data[key] = arr.data[in_patch_mask].filled(np.nan)
                    except IndexError as exception:
                        raise ValueError(f"Failed to sample {key} on patch;\n{patch}") from exception
                        

                for j in range(coefs.shape[0]):
                    data[f"quadric_{str(j).zfill(2)}"] = coefs[j][in_patch_mask]


                data["patch_id"] = patch["patch_id"]

                per_filepath.append(data)
                progress_bar.update()

            # Concatenate all patch data for this filepath, and then sample the final values.
            per_filepath = pd.concat(per_filepath)

            # progress_bar.set_description("Reading rasters")
            # for key, filepath in [("median_dem", median_path), ("npi_dt", dt_path), ("npi_dh", dh_path)]:

            #     with rasterio.open(filepath) as raster, warnings.catch_warnings():
            #         warnings.filterwarnings("ignore", message=".*converting a masked element to nan.*")
            #         per_filepath[key] = np.fromiter(raster.sample(np.transpose([per_filepath["geometry"].x, per_filepath["geometry"].y]), masked=True), count=per_filepath.shape[0], dtype=raster.dtypes[0])

            all_data.append(per_filepath)


    pd.concat(all_data).to_feather(cache_filepath)

    return gpd.read_feather(cache_filepath)


def run_prediction(all_patches):
    all_patches = all_patches.sample(5000, random_state=0)
    import sklearn.model_selection
    from matplotlib.colors import ListedColormap
    from sklearn.datasets import make_circles, make_classification, make_moons
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.inspection import DecisionBoundaryDisplay
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025, random_state=42),
        SVC(gamma=2, C=1, random_state=42),
        # GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
        sklearn.ensemble.HistGradientBoostingClassifier(),
        DecisionTreeClassifier(max_depth=5, random_state=42),
        RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1, random_state=42
        ),
        MLPClassifier(alpha=1, max_iter=1000, random_state=42),
        MLPClassifier(hidden_layer_sizes=(1000, 1000), activation='relu'),
        AdaBoostClassifier(algorithm="SAMME", random_state=42),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]
    x_cols = ["dem", "slope", "curvature", "gap_distance", "npi_dh"]
    y_col = "in_patch"
    all_patches = all_patches.dropna(subset=x_cols)

    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(all_patches[x_cols], all_patches[y_col], random_state=0)


    
    for model in classifiers:
        model.fit(xtrain, ytrain)
        score = model.score(xtest, ytest)

        print(model, score)
        


    

class GapModel:
    def __init__(
        self,
        training_checksum: str,
        model: sklearn.ensemble.HistGradientBoostingClassifier,
        importance: pd.DataFrame,
        score: float
    ):
        self.training_checksum = training_checksum
        self.model = model
        self.importance = importance
        self.score = score

    def __repr__(self) -> str:

        return (
            f"Model: {self.model}, score: {self.score:.3f}\n"
            f"Training checksum: {self.training_checksum[:8]}..."
        )
        

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
        # all_patches = pd.read_feather("cache/get_bad_patch_stats/get_bad_patch_stats-5e56d0a7b7ccbb9b96412219d2c4169d4efc026fd300b38c74e2a1dba9965faf.feather")
        all_patches = all_patches[all_patches["exact"]]
        bad_vs_good_frac = all_patches[all_patches["bad"]].shape[0] / all_patches.shape[0]

        print(f"{bad_vs_good_frac * 100:.2f}% of values are flagged as bad. The rest are good.")


        all_patches = all_patches.dropna(how="all", axis="columns")

        import itertools

        import sklearn.inspection
        import sklearn.pipeline
        import sklearn.preprocessing

        x_cols = ["gap_distance", "npi_dh", "npi_dt", "median_dem"] + [str(col) for col in all_patches if "quadric_" in col]
        # x_cols = ["dem", "slope", "curvature", "npi_dh"]
        y_col = "bad"
        # all_patches = all_patches.dropna(subset=x_cols)

        xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(all_patches[x_cols], all_patches[y_col], random_state=0)

        # def get_score(combo: list[str]) -> float:
        model = sklearn.ensemble.HistGradientBoostingClassifier(random_state=0)
        model.fit(xtrain, ytrain)

        # print(importance)
        # print(importance.importances_mean)
        # print(model.feature_importances_)
        score = model.score(xtest, ytest)

        print(f"Model score: {score:.3f}")
        # raise NotImplementedError()

        print("Asessing feature importance. This takes time..")
        importance = sklearn.inspection.permutation_importance(
            model,
            xtest,
            ytest,
            n_repeats=20,
            random_state=0,
            n_jobs=-1
        )

        # print("Relative feature importance:")
        # for i in importance.importances_mean.argsort()[::-1]:
        #     # if importance.importances_mean[i] - 2 * importance.importances_std[i] > 0:
        #     print(
        #         f"\t{model.feature_names_in_[i]:<14}"
        #         f"{importance.importances_mean[i]:.3f}"
        #         f" +/- {importance.importances_std[i]:.3f}"
        #     )

        importance_df = pd.DataFrame({"name": model.feature_names_in_, "mean": importance.importances_mean, "std": importance.importances_std}).set_index("name")


        return cls(training_checksum=checksum, model=model, importance=importance_df, score=score)
        

    @classmethod
    def default(cls, force_redo: bool = False) -> typing.Self:

        cache_path = CONSTANTS.cache_dir / "gap_model.pkl"

        _, checksum = get_patch_polygs()

        if cache_path.is_file() and not force_redo:
            model_result = cls.from_pickle(cache_path)

            if model_result.training_checksum != checksum:
                print("Model training data checksum is not the same as the stored checksum. Re-training recommended.")

            return model_result


        results = cls.train(force_redo=force_redo)
        results.to_pickle(cache_path)

        return results


def make_gap_mask_name(filename: str, parent: Path | None = None) -> Path:
    year = int(filename.split("_")[3][:4])

    if parent is None:
        parent = Path("temp.svalbard/outlier_proba/")

    return parent / f"{year}/{filename.replace('.tif', '')}_outlier_proba.tif"


def create_gap_mask(filename: str, model: GapModel, verbose: bool = True, force_redo: bool = False) -> Path:

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
    import geoutils as gu
    import scipy.ndimage

    update("Loading DEM")
    dem = xdem.DEM(filepaths["dem"])

    data_mask = ~scipy.ndimage.binary_dilation(dem.data.mask, structure=scipy.ndimage.generate_binary_structure(2, 2))

    vals = pd.DataFrame({"dem": dem.data.filled(np.nan).ravel()})
    x_cols = [str(col) for col in model.importance.index]

    if "gap_distance" in x_cols:
        update("Making gap distances")
        # eastings, northings = get_eastings_northings(dem.bounds, *dem.shape)

        # tree = get_gap_distance_tree(eastings=eastings, northings=northings, nodata_mask=dem.data.mask)

        dist = scipy.ndimage.distance_transform_edt(~dem.data.mask) * dem.res[0]
        vals["gap_distance"] = dist.ravel()
        del dist

        # vals["gap_distance"] = 0.
        # vals.loc[data_mask.ravel(), "gap_distance"] = tree.query(np.transpose([eastings[data_mask], northings[data_mask]]))[0]

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

    # raw_predictions = model.model._raw_predict(vals[x_cols].loc[data_mask.ravel()], n_threads=-1)
    # outliers[data_mask] = (model.model._loss.predict_proba(raw_predictions)[:, 1] * 255).astype("uint8")

    outliers[data_mask] = (model.model.predict_proba(vals[x_cols].loc[data_mask.ravel()])[:, 1] * 255).astype("uint8")
    # outliers = (model.model.predict_proba(vals[x_cols])[:, 1] * 255).astype("uint8").reshape(dem.shape)
    # outliers[dem.data.mask] = 255
    # plt.imshow(outliers)
    # plt.show()

    
    outliers = xdem.DEM.from_array(outliers, crs=dem.crs, transform=dem.transform)
    del dem

    update(f"Saving {out_path}")

    # raise NotImplementedError()

    out_path.parent.mkdir(exist_ok=True, parents=True)
    outliers.save(out_path, tiled=True, co_opts={"ZLEVEL": "12"})

    return out_path


def generate_all_masks(subdir: str = "heerland_dem_coreg", verbose_progress: bool = False):

    filestems = list(map(lambda fp: fp.stem, Path(f"temp.svalbard/{subdir}/").glob("*/*.tif")))
    model_result = GapModel.default()

    finished = []
    to_process = []

    for i, stem in enumerate(filestems):
        gap_path = make_gap_mask_name(filename=stem)

        if gap_path.is_file():
            finished.append(gap_path)
        else:
            to_process.append(i)

    for i in tqdm(to_process, desc="Generating masks", disable=verbose_progress):
        stem = filestems[i]
        create_gap_mask(filename=stem, model=model_result, verbose=verbose_progress)

    return


def plot_bad_patches(force_redo: bool = False):


    model_result = GapModel.default()
    print(model_result.importance.sort_values("mean", ascending=False))
    print(model_result)

    test_file = "SETSM_s2s041_WV02_20160610_1030010056A0AB00_1030010058A14500_2m_lsf_seg1_dem_coreg"


    gap_mask_path = create_gap_mask(test_file, model_result)
    print(gap_mask_path)
    


    return
    all_patches = get_bad_patch_stats(force_redo=force_redo)

    bad_vs_good_frac = all_patches[all_patches["bad"]].shape[0] / all_patches.shape[0]

    print(f"{bad_vs_good_frac * 100:.2f}% of values are flagged as bad. The rest are good.")

    all_patches = all_patches.dropna(how="all", axis="columns")


    # slope, curv = 

    


    # in_patch = all_patches[all_patches["in_patch"]]

    # col = "npi_dh"

    # col_sum = in_patch[col].sum()

    # vline = 0
    # for dist in np.linspace(in_patch["gap_distance"].min(), in_patch["gap_distance"].max(), 100)[1:]:

    #     subset = in_patch[in_patch["gap_distance"] < dist]

    #     if vline == 0 and (subset[col].sum() > (col_sum * 0.8)):
    #         vline = dist
        

    #     plt.scatter(dist, subset["npi_dh"].mean())

    # print(vline)
    # ylim = plt.gca().get_ylim()
    # plt.vlines(vline, *ylim)

    # plt.show()

    # return

    # run_prediction(all_patches)
    # return


    import itertools

    import sklearn.ensemble
    import sklearn.inspection
    import sklearn.model_selection
    import sklearn.pipeline
    import sklearn.preprocessing

    x_cols = ["gap_distance", "npi_dh", "npi_dt", "median_dem"] + [str(col) for col in all_patches if "quadric_" in col]
    # x_cols = ["dem", "slope", "curvature", "npi_dh"]
    y_col = "bad"
    # all_patches = all_patches.dropna(subset=x_cols)

    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(all_patches[x_cols], all_patches[y_col], random_state=0)

    # def get_score(combo: list[str]) -> float:
    model = sklearn.ensemble.HistGradientBoostingClassifier(random_state=0)
    model.fit(xtrain, ytrain)

    # print(importance)
    # print(importance.importances_mean)
    # print(model.feature_importances_)
    score = model.score(xtest, ytest)

    print(f"Model score: {score:.3f}")

    importance = sklearn.inspection.permutation_importance(
        model,
        xtest,
        ytest,
        n_repeats=1,
        random_state=42,
        n_jobs=-1
    )

    print("Relative feature importance:")
    for i in importance.importances_mean.argsort()[::-1]:
        # if importance.importances_mean[i] - 2 * importance.importances_std[i] > 0:
        print(
            f"\t{model.feature_names_in_[i]:<14}"
            f"{importance.importances_mean[i]:.3f}"
            f" +/- {importance.importances_std[i]:.3f}"
        )

    all_patches["pred_bad"] = model.predict_proba(all_patches[x_cols])[:, 1]

    grouped = all_patches.groupby("patch_id")

    prob = grouped[["pred_bad"]].mean()
    prob["bad"] = grouped["bad"].median().astype(float)
    prob["count"] = grouped["pred_bad"].count()
    prob["diff"] = (prob["bad"] - prob["pred_bad"]).abs()

    big_patches = prob[prob["count"] > 15000]
    print(big_patches.sort_values("diff", ascending=False).iloc[:20])

    # prob["diff"] = 

    return

    plt.violinplot([patch["pred_bad"] for _, patch in all_patches.groupby("bad")])
    plt.show()
    return

    print(f"Model score: {score}")


    # test_file = "SETSM_s2s041_WV02_20230511_10300100E50CB500_10300100E75D2900_2m_seg1_dem_coreg"
    test_file = "SETSM_s2s041_WV02_20160610_1030010056A0AB00_1030010058A14500_2m_lsf_seg1_dem_coreg"
    year = int(test_file.split("_")[3][:4])

    print(year)

    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", message=".*'nopython' keyword.*")
      import xdem
    import geoutils as gu
    dem_path = Path(f"temp.svalbard/arcticdem_coreg/{test_file}.tif")
    dh_path = dem_path.parent.parent / f"dh/{year}/{test_file}_dh.tif"

    # slope_path = CONSTANTS.cache_dir / f"{test_file}_slope.tif"
    # curvature_path = CONSTANTS.cache_dir / f"{test_file}_curvature.tif"
    quadric_path = CONSTANTS.cache_dir / f"{test_file}_quadric.tif"
    gap_distance_path = CONSTANTS.cache_dir / f"{test_file}_distance.tif"

    print("Loading DEM")
    dem = xdem.DEM(dem_path)

    if not quadric_path.is_file():
        print("Making quadrics")
        quadric = gu.Raster.from_array(
            xdem.terrain.get_quadric_coefficients(dem.data, resolution=dem.res[0]),
            transform=dem.transform,
            crs=dem.crs,
            nodata=-9999.,
        )

        quadric.save(quadric_path)
    else:
        print("Loading quadrics")
        quadric = gu.Raster(str(quadric_path))

    
    if not gap_distance_path.is_file():
        print("Making gap distances")
        eastings, northings = get_eastings_northings(dem.bounds, *dem.shape)
        tree = scipy.spatial.KDTree(np.transpose([coord[dem.data.mask] for coord in [eastings, northings]]))

        gap_distances = gu.Raster.from_array(
            tree.query(np.transpose([eastings.ravel(), northings.ravel()]))[0].reshape(eastings.shape),
            transform=dem.transform,
            crs=dem.crs,
            nodata=-9999.
        )

        gap_distances.save(gap_distance_path)

        
    else:
        print("Loading gap distances")
        gap_distances = gu.Raster(gap_distance_path)
 
    # print("Loading slope")
    # if not slope_path.is_file():
    #     slope = xdem.terrain.slope(dem)
    #     slope.save(slope_path)
    # else:
    #     slope = xdem.DEM(Path(slope_path))

    # print("Loading curvature")
    # if not curvature_path.is_file():
    #     curvature = xdem.terrain.curvature(dem)
    #     curvature.save(curvature_path)
    # else:
    #     curvature = xdem.DEM(Path(curvature_path))

    print("Loading dh")
    dh = xdem.DEM(dh_path)

    vals = pd.DataFrame({key: vals.data.filled(np.nan).ravel() for key, vals in [("dem", dem), ("npi_dh", dh), ("gap_distance", gap_distances)]})#, ("slope", slope), ("curvature", curvature)]})

    for i in range(quadric.data.shape[0]):
        vals[f"quadric_{str(i).zfill(2)}"] = quadric.data[i].filled(np.nan).ravel()

    print("Predicting and saving gap error probability")
    outliers = xdem.DEM.from_array(model.predict_proba(vals[x_cols])[:, 1].reshape(dem.shape), crs=dem.crs, transform=dem.transform)

    outliers.save(f"{test_file}_outlier_proba.tif")


    print(outliers)
    print(outliers.shape)
        

    # score_nothing = get_score(x_cols)

    


    

    return

    for i in [len(x_cols) + 1, *list(range(len(x_cols)))]:

        combo = [x_cols[j] for j in range(len(x_cols)) if i != j]
        
        model = sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.Normalizer(),
            sklearn.ensemble.HistGradientBoostingClassifier()
        )
        model.fit(xtrain[combo], ytrain)
        score = model.score(xtest[combo], ytest)

        skipped = x_cols[i] if i < len(x_cols) else "nothing"
        if skipped == "nothing":
            score_nothing = score
            print(f"Skipping {skipped}\t{score}")
        else:
            print(f"Skipping {skipped}. Score deterioration: {score_nothing - score}")

    # model = sklearn.ensemble.HistGradientBoostingClassifier()

            
    return

    all_patches = all_patches[(all_patches["gap_distance"] > 0) & (all_patches["gap_distance"] < 600)]

    

    

    bins = np.linspace(0, all_patches["gap_distance"].max() + 1, 10)
    all_patches["digitized"] = np.digitize(all_patches["gap_distance"], bins=bins)
    all_patches.sort_values("digitized", inplace=True)

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    for i in range(2):
        axis = axes[i]

        patches = all_patches[all_patches["in_patch"] == (i == 1)]

        axis.set_title("Classified outlier" if i == 1 else "No class.")

        col = "npi_dh"

        masked = patches.dropna(subset=[col])

        axis.violinplot([vals[col].abs() for _, vals in masked.groupby("digitized")], bins[masked["digitized"].unique()], widths=np.mean(np.diff(bins)))


    plt.tight_layout()
    
    # plt.hist(patches["gap_distance"], bins=200)
    # plt.yscale("log")

    # bins = np.digitize(patches["gap_distance"], bins=10)
    # nan_mask = ~(patches["gap_distance"].isna() | patches["npi_dh"].isna())

    # plt.hist2d(patches["gap_distance"][nan_mask], np.log10(np.abs(patches["npi_dh"][nan_mask]) + 1e-3), density=True)
    # plt.scatter(patches["gap_distance"], np.log10(patches["npi_dh"].abs()), alpha=0.03)
    plt.show()


