import xdem
import matplotlib.pyplot as plt
import rasterio as rio
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
import time

import numpy as np
from shapely.geometry import Point
from shapely.prepared import prep

import numpy as np
from shapely.geometry import Point
from shapely.prepared import prep
import concurrent.futures
import tqdm.contrib.concurrent

try:
    from shapely import contains_xy  # Shapely >= 2
except ImportError:
    contains_xy = None


def sample_variogram_points(
    land,
    n_points,
    bins=None,
    n_init=8,
    pool=256,
    local_frac=0.75,
    min_sep=0.0,
    seed=0,
):
    """
    Generate points inside a shapely Polygon/MultiPolygon so that pairwise
    distances are distributed as evenly as possible across `bins`.

    Returns
    -------
    pts : (n_points, 2) ndarray
        Sampled coordinates.
    """
    if bins is None:
        bins = 10 ** np.linspace(np.log10(30), np.log10(100000), 50)
    bins = np.asarray(bins, dtype=float)

    rng = np.random.default_rng(seed)
    n_bins = len(bins) - 1
    min_sep2 = float(min_sep) ** 2

    # --- geometry helpers -------------------------------------------------
    if hasattr(land, "geoms"):
        parts = list(land.geoms)
    else:
        parts = [land]

    part_areas = np.array([p.area for p in parts], dtype=float)
    part_probs = part_areas / part_areas.sum()
    part_bounds = np.array([p.bounds for p in parts], dtype=float)

    if contains_xy is None:
        prepared_land = prep(land)
        prepared_parts = [prep(p) for p in parts]
    else:
        prepared_land = None
        prepared_parts = None

    def in_land(xy):
        if contains_xy is not None:
            return contains_xy(land, xy[:, 0], xy[:, 1])
        return np.array(
            [prepared_land.contains(Point(x, y)) for x, y in xy],
            dtype=bool,
        )

    def random_points_on_land(n):
        """
        Area-weighted polygon sampling. Much more efficient than sampling
        from the full MultiPolygon bounding box when land is fragmented.
        """
        out = np.empty((n, 2), dtype=float)
        filled = 0

        while filled < n:
            m = max(256, 2 * (n - filled))
            part_idx = rng.choice(len(parts), size=m, p=part_probs)
            b = part_bounds[part_idx]

            x = rng.uniform(b[:, 0], b[:, 2])
            y = rng.uniform(b[:, 1], b[:, 3])

            keep = np.zeros(m, dtype=bool)
            for i in np.unique(part_idx):
                j = np.where(part_idx == i)[0]
                if contains_xy is not None:
                    keep[j] = contains_xy(parts[i], x[j], y[j])
                else:
                    keep[j] = np.array(
                        [prepared_parts[i].contains(Point(xx, yy)) for xx, yy in zip(x[j], y[j])],
                        dtype=bool,
                    )

            xy = np.column_stack([x[keep], y[keep]])
            take = min(n - filled, len(xy))
            if take:
                out[filled:filled + take] = xy[:take]
                filled += take

        return out

    # --- numeric helpers --------------------------------------------------
    def farthest_subset(candidates, k):
        if len(candidates) <= k:
            return candidates.copy()
        sel = np.empty(k, dtype=int)
        sel[0] = rng.integers(len(candidates))
        d2 = ((candidates - candidates[sel[0]]) ** 2).sum(axis=1)
        for t in range(1, k):
            sel[t] = np.argmax(d2)
            d2 = np.minimum(d2, ((candidates - candidates[sel[t]]) ** 2).sum(axis=1))
        return candidates[sel]

    def pair_hist(points):
        if len(points) < 2:
            return np.zeros(n_bins, dtype=int)
        d = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
        iu = np.triu_indices(len(points), 1)
        return np.histogram(d[iu], bins=bins)[0]

    def hist_rows(dists):
        idx = np.searchsorted(bins, dists, side="right") - 1
        valid = (idx >= 0) & (idx < n_bins)

        row_ids = np.repeat(np.arange(dists.shape[0]), dists.shape[1])[valid.ravel()]
        bin_ids = idx.ravel()[valid.ravel()]

        counts = np.bincount(
            row_ids * n_bins + bin_ids,
            minlength=dists.shape[0] * n_bins,
        )
        return counts.reshape(dists.shape[0], n_bins)

    # --- initial anchors --------------------------------------------------
    init_candidates = random_points_on_land(max(10 * n_init, 200))
    pts = farthest_subset(init_candidates, min(n_init, n_points))

    if min_sep > 0 and len(pts) > 1:
        keep = [0]
        for i in range(1, len(pts)):
            if np.all(((pts[keep] - pts[i]) ** 2).sum(axis=1) >= min_sep2):
                keep.append(i)
        pts = pts[keep]

    while len(pts) < min(n_init, n_points):
        c = random_points_on_land(1)
        if len(pts) == 0 or np.all(((pts - c[0]) ** 2).sum(axis=1) >= min_sep2):
            pts = c if len(pts) == 0 else np.vstack([pts, c])

    hist = pair_hist(pts)

    # --- greedy additions -------------------------------------------------
    while len(pts) < n_points:
        n_existing = len(pts)
        target = (n_existing * (n_existing + 1) / 2) / n_bins

        deficits = np.clip(target - hist, 0, None) + 1e-12
        probs = deficits / deficits.sum()

        candidates = np.empty((pool, 2), dtype=float)
        n_cand = 0
        attempts = 0

        while n_cand < pool and attempts < 20 * pool:
            m = max(256, 2 * (pool - n_cand))
            attempts += m

            use_local = rng.random(m) < local_frac
            xy = np.empty((m, 2), dtype=float)

            # local ring proposals
            if np.any(use_local):
                j = np.where(use_local)[0]
                b = rng.choice(n_bins, size=len(j), p=probs)
                anchors = pts[rng.integers(len(pts), size=len(j))]
                r = np.exp(rng.uniform(np.log(bins[b]), np.log(bins[b + 1])))
                a = rng.uniform(0.0, 2 * np.pi, len(j))
                xy[j, 0] = anchors[:, 0] + r * np.cos(a)
                xy[j, 1] = anchors[:, 1] + r * np.sin(a)

            # global proposals
            if np.any(~use_local):
                j = np.where(~use_local)[0]
                xy[j] = random_points_on_land(len(j))

            # keep only candidates inside land
            xy = xy[in_land(xy)]

            if min_sep > 0 and len(xy):
                d2 = ((xy[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
                xy = xy[(d2 >= min_sep2).all(axis=1)]

            take = min(pool - n_cand, len(xy))
            if take:
                candidates[n_cand:n_cand + take] = xy[:take]
                n_cand += take

        if n_cand == 0:
            raise RuntimeError(
                "Could not generate valid candidate points. "
                "Try reducing min_sep or increasing pool."
            )

        candidates = candidates[:n_cand]

        dists = np.sqrt(((candidates[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        cand_hists = hist_rows(dists)
        scores = ((hist[None, :] + cand_hists - target) ** 2).sum(axis=1)

        best = np.argmin(scores)
        pts = np.vstack([pts, candidates[best]])
        hist += cand_hists[best]

    return pts

def sample_variogram_points_old(
    land,
    n_points,
    bins=None,
    n_init=8,
    pool=256,
    local_frac=0.75,
    min_sep=0.0,
    seed=0,
):
    """
    Generate point locations inside a shapely Polygon or MultiPolygon so that
    pairwise distances are distributed as evenly as possible across the given
    distance bins.

    Parameters
    ----------
    land : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        Sampling domain. Must be in a projected CRS with units in metres.

    n_points : int
        Number of points to generate.

    bins : array-like, optional
        Distance bin edges for the variogram. If None, defaults to:
        10 ** np.linspace(np.log10(30), np.log10(100000), 50)

    n_init : int, default 8
        Number of initial anchor points. These are chosen to be well spread
        across the domain and mainly help populate longer lag bins.

    pool : int, default 256
        Number of candidate points evaluated at each iteration.
        Larger values usually improve the result, but make the function slower.

    local_frac : float, default 0.75
        Fraction of proposed candidates that are generated locally on rings
        around existing points, rather than sampled globally across land.
        Higher values favour filling short and intermediate lag bins.

    min_sep : float, default 0.0
        Minimum allowed distance between any two sampled points.
        Use this to avoid near-duplicate points.

    seed : int, default 0
        Random seed for reproducibility.

    Returns
    -------
    pts : ndarray of shape (n_points, 2)
        Array of sampled point coordinates as [[x1, y1], [x2, y2], ...].

    hist : ndarray of shape (len(bins) - 1,)
        Final histogram of pairwise distances across the specified bins.

    Notes
    -----
    The algorithm is greedy, not globally optimal. It tries to flatten the
    pair-distance histogram by repeatedly adding the point that best improves
    the current distribution.

    This function assumes Euclidean planar distances in metres. Reproject your
    data first if `land` is in geographic coordinates (lon/lat).
    """
    if bins is None:
        bins = 10 ** np.linspace(np.log10(30), np.log10(100000), 50)
    bins = np.asarray(bins, dtype=float)

    rng = np.random.default_rng(seed)
    prepared_land = prep(land)
    n_bins = len(bins) - 1
    minx, miny, maxx, maxy = land.bounds

    def random_points_on_land(n):
        """
        Draw n random points uniformly from the bounding box and keep only
        those that fall inside land.
        """
        points = []
        while len(points) < n:
            m = max(256, 4 * (n - len(points)))
            xs = rng.uniform(minx, maxx, m)
            ys = rng.uniform(miny, maxy, m)
            for x, y in zip(xs, ys):
                if prepared_land.contains(Point(x, y)):
                    points.append((x, y))
                    if len(points) == n:
                        break
        return np.asarray(points, dtype=float)

    def farthest_subset(candidates, k):
        """
        Select k well-spread points from a candidate set using greedy
        farthest-point sampling.
        """
        if len(candidates) <= k:
            return candidates.copy()

        selected = [rng.integers(len(candidates))]
        d2 = np.sum((candidates - candidates[selected[0]]) ** 2, axis=1)

        for _ in range(1, k):
            i = np.argmax(d2)
            selected.append(i)
            d2 = np.minimum(d2, np.sum((candidates - candidates[i]) ** 2, axis=1))

        return candidates[selected]

    def pair_hist(points):
        """
        Compute the histogram of pairwise distances for a set of points.
        """
        if len(points) < 2:
            return np.zeros(n_bins, dtype=int)

        diffs = points[:, None, :] - points[None, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(len(points), 1)
        return np.histogram(dists[iu], bins=bins)[0]

    # ------------------------------------------------------------------
    # 1. Initial anchor points: well spread across land
    # ------------------------------------------------------------------
    init_candidates = random_points_on_land(max(10 * n_init, 200))
    pts = farthest_subset(init_candidates, min(n_init, n_points))

    # Enforce min_sep among initial points if requested
    if min_sep > 0 and len(pts) > 1:
        kept = [pts[0]]
        for p in pts[1:]:
            d = np.sqrt(np.sum((np.asarray(kept) - p) ** 2, axis=1))
            if np.all(d >= min_sep):
                kept.append(p)
        pts = np.asarray(kept, dtype=float)

    # If we somehow ended up with too few initial points, top up
    while len(pts) < min(n_init, n_points):
        candidate = random_points_on_land(1)[0]
        if len(pts) == 0:
            pts = np.asarray([candidate], dtype=float)
        else:
            d = np.sqrt(np.sum((pts - candidate) ** 2, axis=1))
            if min_sep <= 0 or np.all(d >= min_sep):
                pts = np.vstack([pts, candidate])

    hist = pair_hist(pts)

    # ------------------------------------------------------------------
    # 2. Greedily add points to flatten the pair-distance histogram
    # ------------------------------------------------------------------
    while len(pts) < n_points:
        n_existing = len(pts)

        # Approximate target: equal number of pairs in each bin
        total_pairs_after_addition = (n_existing + 1) * n_existing / 2
        target = total_pairs_after_addition / n_bins

        deficits = np.clip(target - hist, 0, None) + 1e-12
        probs = deficits / deficits.sum()

        candidates = []
        attempts = 0
        max_attempts = 20 * pool

        while len(candidates) < pool and attempts < max_attempts:
            attempts += 1

            use_local = (rng.random() < local_frac) and (len(pts) > 0)

            if use_local:
                # Choose an underfilled lag bin
                b = rng.choice(n_bins, p=probs)

                # Pick an anchor point
                anchor = pts[rng.integers(len(pts))]

                # Sample distance log-uniformly within the chosen bin
                r = np.exp(rng.uniform(np.log(bins[b]), np.log(bins[b + 1])))
                theta = rng.uniform(0, 2 * np.pi)

                x = anchor[0] + r * np.cos(theta)
                y = anchor[1] + r * np.sin(theta)
                candidate = np.array([x, y], dtype=float)
            else:
                candidate = random_points_on_land(1)[0]

            if not prepared_land.contains(Point(candidate)):
                continue

            if len(pts) > 0 and min_sep > 0:
                d = np.sqrt(np.sum((pts - candidate) ** 2, axis=1))
                if np.any(d < min_sep):
                    continue

            candidates.append(candidate)

        if not candidates:
            raise RuntimeError(
                "Could not generate valid candidate points. "
                "Try reducing min_sep or increasing pool."
            )

        candidates = np.asarray(candidates, dtype=float)

        # Distances from each candidate to all existing points
        dists = np.sqrt(
            np.sum((candidates[:, None, :] - pts[None, :, :]) ** 2, axis=2)
        )

        # Histogram contribution each candidate would add
        cand_hists = np.array(
            [np.histogram(row, bins=bins)[0] for row in dists],
            dtype=float,
        )

        # Choose candidate that makes histogram closest to flat
        scores = np.sum((hist[None, :] + cand_hists - target) ** 2, axis=1)
        best_idx = np.argmin(scores)

        pts = np.vstack([pts, candidates[best_idx]])
        hist = hist + cand_hists[best_idx]

    return pts, hist



def sample_variogram_points_bbox(
    bounds,
    n_points,
    bins=None,
    n_init=8,
    pool=256,
    local_frac=0.75,
    min_sep=0.0,
    seed=0,
):
    """
    Generate points inside a bounding box so that pairwise distances are
    distributed as evenly as possible across `bins`.

    Parameters
    ----------
    bounds : tuple[float, float, float, float]
        (minx, miny, maxx, maxy)
    n_points : int
        Number of points to generate
    bins : array-like, optional
        Distance bin edges. Defaults to log-spaced bins from 30 m to 100 km.
    n_init : int, default 8
        Number of initial well-spread anchor points
    pool : int, default 256
        Number of candidates evaluated per iteration
    local_frac : float, default 0.75
        Fraction of candidates proposed on rings around existing points
    min_sep : float, default 0.0
        Minimum allowed separation between sampled points
    seed : int, default 0
        Random seed

    Returns
    -------
    pts : (n_points, 2) ndarray
        Sampled coordinates
    """
    if bins is None:
        bins = 10 ** np.linspace(np.log10(30), np.log10(100000), 50)
    bins = np.asarray(bins, dtype=float)

    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = bounds
    n_bins = len(bins) - 1
    min_sep2 = float(min_sep) ** 2

    def random_points(n):
        return np.column_stack([
            rng.uniform(minx, maxx, n),
            rng.uniform(miny, maxy, n),
        ])

    def farthest_subset(candidates, k):
        if len(candidates) <= k:
            return candidates.copy()
        sel = np.empty(k, dtype=int)
        sel[0] = rng.integers(len(candidates))
        d2 = ((candidates - candidates[sel[0]]) ** 2).sum(axis=1)
        for i in range(1, k):
            sel[i] = np.argmax(d2)
            d2 = np.minimum(d2, ((candidates - candidates[sel[i]]) ** 2).sum(axis=1))
        return candidates[sel]

    def pair_hist(points):
        if len(points) < 2:
            return np.zeros(n_bins, dtype=int)
        d = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
        iu = np.triu_indices(len(points), 1)
        return np.histogram(d[iu], bins=bins)[0]

    def hist_rows(dists):
        idx = np.searchsorted(bins, dists, side="right") - 1
        valid = (idx >= 0) & (idx < n_bins)

        row_ids = np.repeat(np.arange(dists.shape[0]), dists.shape[1])[valid.ravel()]
        bin_ids = idx.ravel()[valid.ravel()]

        counts = np.bincount(
            row_ids * n_bins + bin_ids,
            minlength=dists.shape[0] * n_bins,
        )
        return counts.reshape(dists.shape[0], n_bins)

    # --- initial anchors
    init_candidates = random_points(max(10 * n_init, 200))
    pts = farthest_subset(init_candidates, min(n_init, n_points))

    if min_sep > 0 and len(pts) > 1:
        keep = [0]
        for i in range(1, len(pts)):
            if np.all(((pts[keep] - pts[i]) ** 2).sum(axis=1) >= min_sep2):
                keep.append(i)
        pts = pts[keep]

    while len(pts) < min(n_init, n_points):
        c = random_points(1)
        if len(pts) == 0 or np.all(((pts - c[0]) ** 2).sum(axis=1) >= min_sep2):
            pts = c if len(pts) == 0 else np.vstack([pts, c])

    hist = pair_hist(pts)

    # --- greedy additions
    while len(pts) < n_points:
        n_existing = len(pts)
        target = (n_existing * (n_existing + 1) / 2) / n_bins

        deficits = np.clip(target - hist, 0, None) + 1e-12
        probs = deficits / deficits.sum()

        m = max(pool, 256)
        use_local = rng.random(m) < local_frac
        candidates = np.empty((m, 2), dtype=float)

        # local ring proposals
        if np.any(use_local):
            j = np.where(use_local)[0]
            b = rng.choice(n_bins, size=len(j), p=probs)
            anchors = pts[rng.integers(len(pts), size=len(j))]
            r = np.exp(rng.uniform(np.log(bins[b]), np.log(bins[b + 1])))
            a = rng.uniform(0.0, 2 * np.pi, len(j))
            candidates[j, 0] = anchors[:, 0] + r * np.cos(a)
            candidates[j, 1] = anchors[:, 1] + r * np.sin(a)

        # global proposals
        if np.any(~use_local):
            j = np.where(~use_local)[0]
            candidates[j] = random_points(len(j))

        # clip to bbox
        candidates[:, 0] = np.clip(candidates[:, 0], minx, maxx)
        candidates[:, 1] = np.clip(candidates[:, 1], miny, maxy)

        if min_sep > 0:
            d2 = ((candidates[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
            candidates = candidates[(d2 >= min_sep2).all(axis=1)]

        if len(candidates) == 0:
            raise RuntimeError("No valid candidates; reduce min_sep or increase pool.")

        dists = np.sqrt(((candidates[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        cand_hists = hist_rows(dists)
        scores = ((hist[None, :] + cand_hists - target) ** 2).sum(axis=1)

        best = np.argmin(scores)
        pts = np.vstack([pts, candidates[best]])
        hist += cand_hists[best]

    return pts


import numpy as np


def sample_variogram_points_bbox_fast(
    bounds,
    n_points,
    bins=None,
    n_init=8,
    pool=256,
    local_frac=0.75,
    min_sep=0.0,
    score_subset=256,
    seed=0,
):
    """
    Generate points inside a bounding box such that pairwise distances are
    approximately evenly distributed across the supplied distance bins.

    This is a faster approximate alternative to the fully greedy sampler:
    each new candidate is scored only against a fixed-size random subset of
    existing points rather than against all existing points.

    Parameters
    ----------
    bounds : tuple of float
        Bounding box as (minx, miny, maxx, maxy).

    n_points : int
        Number of points to generate.

    bins : array-like, optional
        Distance bin edges. If None, defaults to:
        10 ** np.linspace(np.log10(30), np.log10(100000), 50)

    n_init : int, default 8
        Number of initial anchor points chosen to be well spread out.

    pool : int, default 256
        Number of candidate points evaluated for each added point.

    local_frac : float, default 0.75
        Fraction of candidates generated as local ring proposals around
        existing points. The remainder are sampled globally in the bbox.

    min_sep : float, default 0.0
        Minimum allowed separation between points.

    score_subset : int, default 256
        Number of existing points used to score each candidate.
        Smaller values are faster; larger values are more faithful to the
        full greedy objective.

    seed : int, default 0
        Random seed for reproducibility.

    Returns
    -------
    pts : ndarray of shape (n_points, 2)
        Sampled point coordinates.

    Notes
    -----
    This function samples only inside the bounding box. If you later clip the
    result to a land mask or polygon, the final pair-distance distribution will
    no longer exactly match the optimized bbox design.
    """
    if bins is None:
        bins = 10 ** np.linspace(np.log10(30), np.log10(100000), 50)
    bins = np.asarray(bins, dtype=float)

    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = bounds
    n_bins = len(bins) - 1
    min_sep2 = float(min_sep) ** 2

    def random_points(n):
        """Sample n uniform random points inside the bounding box."""
        return np.column_stack([
            rng.uniform(minx, maxx, n),
            rng.uniform(miny, maxy, n),
        ])

    def farthest_subset(candidates, k):
        """Greedy farthest-point subset selection."""
        if len(candidates) <= k:
            return candidates.copy()

        selected = np.empty(k, dtype=int)
        selected[0] = rng.integers(len(candidates))
        d2 = ((candidates - candidates[selected[0]]) ** 2).sum(axis=1)

        for i in range(1, k):
            selected[i] = np.argmax(d2)
            d2 = np.minimum(d2, ((candidates - candidates[selected[i]]) ** 2).sum(axis=1))

        return candidates[selected]

    def pair_hist(points):
        """Histogram of all pairwise distances among points."""
        if len(points) < 2:
            return np.zeros(n_bins, dtype=int)

        d = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
        iu = np.triu_indices(len(points), 1)
        return np.histogram(d[iu], bins=bins)[0]

    def hist_rows(dists):
        """
        Compute one histogram per row of `dists`, using the common bin edges in `bins`.

        Parameters
        ----------
        dists : ndarray of shape (n_rows, n_cols)

        Returns
        -------
        counts : ndarray of shape (n_rows, n_bins)
        """
        idx = np.searchsorted(bins, dists, side="right") - 1
        valid = (idx >= 0) & (idx < n_bins)

        flat_idx = idx.ravel()
        flat_valid = valid.ravel()

        row_ids = np.repeat(np.arange(dists.shape[0]), dists.shape[1])[flat_valid]
        bin_ids = flat_idx[flat_valid]

        counts = np.bincount(
            row_ids * n_bins + bin_ids,
            minlength=dists.shape[0] * n_bins,
        )
        return counts.reshape(dists.shape[0], n_bins)

    # ------------------------------------------------------------------
    # Optional spatial hash for fast min-separation checking
    # ------------------------------------------------------------------
    use_hash = min_sep > 0
    if use_hash:
        cell_size = float(min_sep)
        grid = {}

        def cell_coords(xy):
            return np.floor(xy / cell_size).astype(np.int64)

        def add_to_grid(idx):
            cell = tuple(cell_coords(pts[idx]))
            grid.setdefault(cell, []).append(idx)

        def valid_minsep(cands):
            keep = np.ones(len(cands), dtype=bool)
            cells = cell_coords(cands)

            for i, (cx, cy) in enumerate(cells):
                neighbours = []
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        neighbours.extend(grid.get((cx + dx, cy + dy), []))

                if neighbours:
                    d2 = ((pts[neighbours] - cands[i]) ** 2).sum(axis=1)
                    if np.any(d2 < min_sep2):
                        keep[i] = False

            return keep

    # ------------------------------------------------------------------
    # Initial anchor points
    # ------------------------------------------------------------------
    init_candidates = random_points(max(10 * n_init, 200))
    pts = farthest_subset(init_candidates, min(n_init, n_points))

    if min_sep > 0 and len(pts) > 1:
        keep = [0]
        for i in range(1, len(pts)):
            d2 = ((pts[keep] - pts[i]) ** 2).sum(axis=1)
            if np.all(d2 >= min_sep2):
                keep.append(i)
        pts = pts[keep]

    while len(pts) < min(n_init, n_points):
        c = random_points(1)[0]
        if len(pts) == 0:
            pts = np.array([c], dtype=float)
        else:
            d2 = ((pts - c) ** 2).sum(axis=1)
            if min_sep <= 0 or np.all(d2 >= min_sep2):
                pts = np.vstack([pts, c])

    if use_hash:
        for i in range(len(pts)):
            add_to_grid(i)

    # ------------------------------------------------------------------
    # Greedy additions using only a reference subset for scoring
    # ------------------------------------------------------------------
    while len(pts) < n_points:
        m_ref = min(score_subset, len(pts))
        ref_idx = rng.choice(len(pts), size=m_ref, replace=False)
        ref = pts[ref_idx]

        ref_hist = pair_hist(ref)
        target = (m_ref * (m_ref + 1) / 2) / n_bins

        deficits = np.clip(target - ref_hist, 0, None) + 1e-12
        probs = deficits / deficits.sum()

        m = max(pool, 256)
        use_local = rng.random(m) < local_frac
        candidates = np.empty((m, 2), dtype=float)

        # Local ring proposals
        if np.any(use_local):
            j = np.where(use_local)[0]
            b = rng.choice(n_bins, size=len(j), p=probs)
            anchors = pts[rng.integers(len(pts), size=len(j))]
            r = np.exp(rng.uniform(np.log(bins[b]), np.log(bins[b + 1])))
            a = rng.uniform(0.0, 2 * np.pi, len(j))
            candidates[j, 0] = anchors[:, 0] + r * np.cos(a)
            candidates[j, 1] = anchors[:, 1] + r * np.sin(a)

        # Global proposals
        if np.any(~use_local):
            j = np.where(~use_local)[0]
            candidates[j] = random_points(len(j))

        # Clip local proposals back into the bbox
        candidates[:, 0] = np.clip(candidates[:, 0], minx, maxx)
        candidates[:, 1] = np.clip(candidates[:, 1], miny, maxy)

        if use_hash:
            candidates = candidates[valid_minsep(candidates)]

        if len(candidates) == 0:
            raise RuntimeError(
                "No valid candidates found. Try reducing min_sep or increasing pool."
            )

        # Score candidates only against the reference subset
        dists = np.sqrt(((candidates[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2))
        cand_hists = hist_rows(dists)
        scores = ((ref_hist[None, :] + cand_hists - target) ** 2).sum(axis=1)

        best = np.argmin(scores)
        pts = np.vstack([pts, candidates[best]])

        if use_hash:
            add_to_grid(len(pts) - 1)

    return pts

import numpy as np


import numpy as np


def sample_variogram_points_bbox_mixture(
    bounds,
    n_points,
    bins=None,
    global_frac=0.2,
    min_sep=0.0,
    seed=0,
):
    """
    Generate points inside a bounding box using a simple multiscale mixture
    design that scales well to thousands of points.

    The method is intentionally approximate:
 sampled globally in the bounding box
    - the rest are sampled as offsets from previously accepted points
    - offset distances are drawn from the supplied lag bins so that all bins
      are used roughly evenly over time

    This does not greedily optimize the full pair-distance histogram, but it
    usually gives a much better spread of short, medium, and long distances
    than uniform random sampling, while scaling far better for large n.

    Parameters
    ----------
    bounds : tuple of float
        Bounding box as (minx, miny, maxx, maxy).

    n_points : int
        Number of points to generate.

    bins : array-like, optional
        Distance bin edges. If None, defaults to:
        10 ** np.linspace(np.log10(30), np.log10(100000), 50)

    global_frac : float, default 0.2
        Fraction of points sampled globally in the bbox rather than as local
        offsets from existing points. Increase this if you want stronger
        long-range coverage.

    min_sep : float, default 0.0
        Minimum allowed separation between points. If 0, no separation
        constraint is used and the function is fastest.

    seed : int, default 0
        Random seed for reproducibility.

    Returns
    -------
    pts : ndarray of shape (n_points, 2)
        Sampled point coordinates.

    Notes
    -----
    This function samples only inside the bounding box. If you later clip the
    points to a land mask or polygon, the final distance distribution will be
    altered by that masking step.

    The function uses reflected boundaries for local offsets: when a proposed
    point falls outside the bbox, it is mirrored back inside rather than
    clipped. That avoids piling up points exactly on the boundary.
    """
    if bins is None:
        bins = 10 ** np.linspace(np.log10(30), np.log10(100000), 50)
    bins = np.asarray(bins, dtype=float)

    if n_points <= 0:
        return np.empty((0, 2), dtype=float)

    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = map(float, bounds)
    widths = np.array([maxx - minx, maxy - miny], dtype=float)

    if np.any(widths <= 0):
        raise ValueError("Invalid bounds: max must be greater than min in both dimensions.")

    n_bins = len(bins) - 1
    if n_bins < 1:
        raise ValueError("`bins` must contain at least two edges.")

    def random_points(n):
        """Sample n uniform random points inside the bounding box."""
        return np.column_stack([
            rng.uniform(minx, maxx, n),
            rng.uniform(miny, maxy, n),
        ])

    def reflect_to_bbox(xy):
        """
        Reflect coordinates back into the bounding box instead of clipping.
        This preserves spread better near edges.
        """
        out = xy.copy()
        for dim, (lo, hi) in enumerate(((minx, maxx), (miny, maxy))):
            w = hi - lo
            z = (out[:, dim] - lo) % (2.0 * w)
            out[:, dim] = lo + np.where(z <= w, z, 2.0 * w - z)
        return out

    # ------------------------------------------------------------------
    # Optional spatial hash for fast minimum-separation checking
    # ------------------------------------------------------------------
    use_hash = min_sep > 0
    if use_hash:
        min_sep2 = float(min_sep) ** 2
        cell_size = float(min_sep)
        origin = np.array([minx, miny], dtype=float)
        grid = {}

        def cell_coords(xy):
            return np.floor((xy - origin) / cell_size).astype(np.int64)

        def point_is_valid(p):
            c = cell_coords(p.reshape(1, 2))[0]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cell = (c[0] + dx, c[1] + dy)
                    if cell in grid:
                        q = np.asarray(grid[cell], dtype=float)
                        d2 = np.sum((q - p) ** 2, axis=1)
                        if np.any(d2 < min_sep2):
                            return False
            return True

        def add_point_to_grid(p):
            c = tuple(cell_coords(p.reshape(1, 2))[0])
            grid.setdefault(c, []).append(p.copy())

    # ------------------------------------------------------------------
    # Build points
    # ------------------------------------------------------------------
    pts = np.empty((n_points, 2), dtype=float)
    n_accepted = 0

    # Start with a few global points
    n_seed = max(1, int(round(global_frac * n_points)))
    n_seed = min(n_seed, n_points)

    while n_accepted < n_seed:
        cand = random_points(1)[0]
        if not use_hash or point_is_valid(cand):
            pts[n_accepted] = cand
            if use_hash:
                add_point_to_grid(cand)
            n_accepted += 1

    # Cycle through bins in shuffled order so all lag bins are used roughly evenly
    bin_order = rng.permutation(n_bins)
    bin_ptr = 0

    def next_bin_index():
        nonlocal bin_order, bin_ptr
        if bin_ptr >= len(bin_order):
            bin_order = rng.permutation(n_bins)
            bin_ptr = 0
        b = bin_order[bin_ptr]
        bin_ptr += 1
        return b

    while n_accepted < n_points:
        use_global = rng.random() < global_frac

        if use_global or n_accepted == 0:
            cand = random_points(1)[0]
        else:
            parent = pts[rng.integers(n_accepted)]
            b = next_bin_index()

            r = np.exp(rng.uniform(np.log(bins[b]), np.log(bins[b + 1])))
            theta = rng.uniform(0.0, 2.0 * np.pi)

            cand = parent + np.array([r * np.cos(theta), r * np.sin(theta)])
            cand = reflect_to_bbox(cand.reshape(1, 2))[0]

        if not use_hash or point_is_valid(cand):
            pts[n_accepted] = cand
            if use_hash:
                add_point_to_grid(cand)
            n_accepted += 1

    return pts

def sample_variogram_points_bbox_wrapper(kwargs):
    return sample_variogram_points(**kwargs)

import numpy as np

def variogram_subset_indices_int32(
    pts,
    bins,
    target_points=12000,
    tile_size=2000,
    extra_small_bins=6,
    seed=0,
):
    """
    Lag-balanced subset for variogram estimation.

    Parameters
    ----------
    pts : (N, 2) ndarray
        Coordinates in meters, ideally int32.
    bins : 1D ndarray
        Lag bin edges.
    target_points : int
        Approximate target subset size.
    tile_size : int
        Coarse tile size in meters.
    extra_small_bins : int
        First this many bins get 2 samples per anchor, rest get 1.
    seed : int
        RNG seed.

    Returns
    -------
    idx : 1D ndarray
        Indices into pts.
    """
    rng = np.random.default_rng(seed)

    # keep storage as int32
    x = pts[:, 0].astype(np.int32, copy=False)
    y = pts[:, 1].astype(np.int32, copy=False)

    # ----- tile index in int32 -----
    tx = (x // tile_size).astype(np.int32)
    ty = (y // tile_size).astype(np.int32)

    tx0 = tx.min()
    ty0 = ty.min()
    txr = (tx - tx0).astype(np.int32)
    tyr = (ty - ty0).astype(np.int32)

    ny = int(tyr.max()) + 1
    tid = (txr.astype(np.int64) * ny + tyr.astype(np.int64)).astype(np.int32)

    order = np.argsort(tid)
    tid_sorted = tid[order]
    uniq, starts, counts = np.unique(tid_sorted, return_index=True, return_counts=True)

    # ----- choose anchors -----
    n_bins = len(bins) - 1
    mean_per_anchor = 1 + n_bins + extra_small_bins
    n_anchors = max(1, int(round(target_points / mean_per_anchor)))

    if len(uniq) <= n_anchors:
        chosen_tiles = np.arange(len(uniq))
    else:
        chosen_tiles = rng.choice(len(uniq), size=n_anchors, replace=False)

    selected = set()
    anchors = []

    for k in chosen_tiles:
        s = starts[k]
        c = counts[k]
        idx = order[s + rng.integers(c)]
        anchors.append(idx)
        selected.add(int(idx))

    anchors = np.asarray(anchors, dtype=np.int64)

    # ----- helper -----
    def sample_one(anchor_idx, r0, r1):
        ax = int(x[anchor_idx])
        ay = int(y[anchor_idx])

        itx0 = max(0, (ax - r1) // tile_size - int(tx0))
        itx1 = (ax + r1) // tile_size - int(tx0)
        ity0 = max(0, (ay - r1) // tile_size - int(ty0))
        ity1 = (ay + r1) // tile_size - int(ty0)

        best = -1
        seen = 0
        r0_2 = r0 * r0
        r1_2 = r1 * r1

        for i in range(itx0, itx1 + 1):
            base = i * ny
            for j in range(ity0, ity1 + 1):
                key = np.int32(base + j)
                pos = np.searchsorted(uniq, key)
                if pos >= len(uniq) or uniq[pos] != key:
                    continue

                s = starts[pos]
                c = counts[pos]
                ii = order[s:s+c]

                # only cast here, where int32 would overflow
                dx = x[ii].astype(np.int64) - ax
                dy = y[ii].astype(np.int64) - ay
                d2 = dx * dx + dy * dy

                mask = (d2 >= r0_2) & (d2 < r1_2) & (ii != anchor_idx)
                cand = ii[mask]
                if cand.size == 0:
                    continue

                for cidx in cand:
                    seen += 1
                    if seen == 1 or rng.integers(seen) == 0:
                        best = int(cidx)

        return best if best >= 0 else None

    # ----- fill bins -----
    for a in tqdm.tqdm(anchors):
        for b in range(n_bins):
            r0 = int(bins[b])
            r1 = int(bins[b + 1])
            reps = 2 if b < extra_small_bins else 1

            for _ in range(reps):
                j = sample_one(a, r0, r1)
                if j is not None:
                    selected.add(j)

    return np.fromiter(selected, dtype=np.int64)

import os
import numpy as np
from tqdm.contrib.concurrent import process_map

# ---- globals visible to worker processes (shared via fork on Linux) ----
_VG = {}

def _vg_worker(anchor_idx):
    x = _VG["x"]
    y = _VG["y"]
    uniq = _VG["uniq"]
    starts = _VG["starts"]
    counts = _VG["counts"]
    order = _VG["order"]
    bins = _VG["bins"]
    tile_size = _VG["tile_size"]
    extra_small_bins = _VG["extra_small_bins"]
    tx0 = _VG["tx0"]
    ty0 = _VG["ty0"]
    ny = _VG["ny"]
    base_seed = _VG["seed"]

    rng = np.random.default_rng(base_seed + int(anchor_idx))

    ax = int(x[anchor_idx])
    ay = int(y[anchor_idx])

    picked = [int(anchor_idx)]  # always keep anchor itself
    n_bins = len(bins) - 1

    for b in range(n_bins):
        r0 = int(bins[b])
        r1 = int(bins[b + 1])
        reps = 2 if b < extra_small_bins else 1

        itx0 = max(0, (ax - r1) // tile_size - tx0)
        itx1 = (ax + r1) // tile_size - tx0
        ity0 = max(0, (ay - r1) // tile_size - ty0)
        ity1 = (ay + r1) // tile_size - ty0

        r0_2 = np.int64(r0) * np.int64(r0)
        r1_2 = np.int64(r1) * np.int64(r1)

        for _ in range(reps):
            best = -1
            seen = 0

            for i in range(itx0, itx1 + 1):
                base = i * ny
                for j in range(ity0, ity1 + 1):
                    key = np.int32(base + j)
                    pos = np.searchsorted(uniq, key)
                    if pos >= len(uniq) or uniq[pos] != key:
                        continue

                    s = starts[pos]
                    c = counts[pos]
                    ii = order[s:s+c]

                    dx = x[ii].astype(np.int64) - ax
                    dy = y[ii].astype(np.int64) - ay
                    d2 = dx * dx + dy * dy

                    mask = (d2 >= r0_2) & (d2 < r1_2) & (ii != anchor_idx)
                    cand = ii[mask]
                    if cand.size == 0:
                        continue

                    # reservoir sample one candidate across all matching points
                    for cidx in cand:
                        seen += 1
                        if seen == 1 or rng.integers(seen) == 0:
                            best = int(cidx)

            if best >= 0:
                picked.append(best)

    return np.asarray(picked, dtype=np.int64)


def variogram_subset_indices_parallel(
    pts,
    bins,
    target_points=12000,
    tile_size=2000,
    extra_small_bins=6,
    seed=0,
    max_workers=None,
    chunksize=4,
):
    """
    Parallel lag-balanced subset for variogram estimation.

    Assumes pts[:,0], pts[:,1] are int32 coordinates in meters.
    Best on Linux/WSL where fork shares memory.
    """
    x = pts[:, 0].astype(np.int32, copy=False)
    y = pts[:, 1].astype(np.int32, copy=False)

    # --- tile index ---
    tx = (x // tile_size).astype(np.int32)
    ty = (y // tile_size).astype(np.int32)

    tx0 = int(tx.min())
    ty0 = int(ty.min())
    txr = (tx - tx0).astype(np.int32)
    tyr = (ty - ty0).astype(np.int32)

    ny = int(tyr.max()) + 1
    tid = (txr.astype(np.int64) * ny + tyr.astype(np.int64)).astype(np.int32)

    order = np.argsort(tid)
    tid_sorted = tid[order]
    uniq, starts, counts = np.unique(tid_sorted, return_index=True, return_counts=True)

    # --- choose anchors: one random point from each chosen occupied tile ---
    rng = np.random.default_rng(seed)
    n_bins = len(bins) - 1
    mean_per_anchor = 1 + n_bins + extra_small_bins
    n_anchors = max(1, int(round(target_points / mean_per_anchor)))

    if len(uniq) <= n_anchors:
        chosen_tiles = np.arange(len(uniq))
    else:
        chosen_tiles = rng.choice(len(uniq), size=n_anchors, replace=False)

    anchors = np.empty(len(chosen_tiles), dtype=np.int64)
    for m, k in enumerate(chosen_tiles):
        s = starts[k]
        c = counts[k]
        anchors[m] = order[s + rng.integers(c)]

    # --- publish read-only globals for workers ---
    _VG.clear()
    _VG.update({
        "x": x,
        "y": y,
        "uniq": uniq,
        "starts": starts,
        "counts": counts,
        "order": order,
        "bins": np.asarray(bins, dtype=np.int32),
        "tile_size": int(tile_size),
        "extra_small_bins": int(extra_small_bins),
        "tx0": int(tx0),
        "ty0": int(ty0),
        "ny": int(ny),
        "seed": int(seed),
    })

    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)

    # --- parallel over anchors ---
    parts = process_map(
        _vg_worker,
        anchors,
        max_workers=max_workers,
        chunksize=chunksize,
    )

    # flatten + unique
    idx = np.unique(np.concatenate(parts))
    return idx

def main():
    import adsvalbard.rasters
    bins = 10 ** np.linspace(np.log10(80), np.log10(100000), 50) 
    print("Reading land")

    xoff = 500000
    yoff = 8600000
    import scipy.spatial
    with rio.open("temp/stable_terrain.tif", overview_level=3) as raster:
        raw_bounds = rio.coords.BoundingBox(469128.7748,8554897.3123,583645.2916,8835786.8817)
        window = rio.windows.from_bounds(*raw_bounds, raster.transform).round_lengths().round_offsets()
        bounds = rio.coords.BoundingBox(*rio.windows.bounds(window, raster.transform))

        stable = np.argwhere(raster.read(1, window=window) == 1)

        x_coords = np.linspace(bounds.left + raster.res[0] / 2, bounds.right - raster.res[0] / 2, window.width, dtype="int32")[stable[:, 1]]
        y_coords = np.linspace(bounds.bottom + raster.res[1] / 2, bounds.top - raster.res[1] / 2, window.height, dtype="int32")[::-1][stable[:, 0]]
        del stable

        pts = np.vstack((x_coords, y_coords)).T

        # anchors = pts[np.random.choice(np.arange(pts.shape[0]), 100), :]

        # for anchor in anchors:
        #     for i in range(10):
        #         subset = pts[np.random.choice(np.arange(pts.shape[0]), 1000), :]

        #         distances = np.hypot(*(subset - anchor[None, :]).T)

        #         digitized = np.digitize(distances, bins)

                
        #         print(digitized)

                
        # print(anchors.shape)
        # return

        pts = pts[variogram_subset_indices_parallel(pts, bins=bins, target_points=24000)]
        # print(sub.shape)
        # return

    #     print(pts[:5, :])
    #     plt.scatter(pts[:5, 0], pts[:5, 1])
    #     plt.show()
    #     return

    print("Making sampling points")
    # pts = sample_variogram_points(land=land, n_points=5000, bins=bins)
    # start_time = time.time()
    # # for n in [500, 1000, 2000, 4000]:
    # n_pts = 0
    # all_pts = []

    # call_args = []
    # for i in range(50):
    #     call_args.append(
    #         {
    #             "land": land,
    #             "seed": i,
    #             "n_points": 100,
    #         }
    #     )

    # all_pts = tqdm.contrib.concurrent.process_map(sample_variogram_points_bbox_wrapper, call_args)

    # pts = np.concatenate(all_pts)

    # print(pts.shape)
    # return
    # for i in range(5):
    #     print(i)
    #     if n_pts > 20000:
    #         break
    #     pts = sample_variogram_points_bbox_mixture(land.bounds, n_points=50000, bins=bins, seed=i)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pts[:, 0], pts[:, 1], crs=32633))
    # pts = pts[pts.intersects(land)]
    #     n_pts += pts.shape[0]

    #     all_pts.append(pts)
    # pts = pd.concat(all_pts, ignore_index=True)
    print(pts.shape)
    print("Sampling stable terrain")
    # pts["stable"] = adsvalbard.rasters.sample_raster("temp/stable_terrain.tif", pts["geometry"]) == 1

    # pts = pts.query("stable").copy()

    var_col = "trend_2013-2024_slope"

    print("Sampling elevation change")
    pts[var_col] = adsvalbard.rasters.sample_raster(f"temp.svalbard/filt/svalbard/mosaics_3584/{var_col}.vrt", geometry=pts["geometry"])
    
    # centers = np.sqrt(bins[:-1] * bins[1:])  # geometric bin centers
    # plt.semilogx(centers, hist, marker="o")
    # plt.xlabel("Distance")
    # plt.ylabel("Pair count")
    # plt.show()

    # print(pts)
    # print(hist)

    # return
    # print(xdem.version.version)
    # return

    # bounds = rio.coords.BoundingBox(left=504602, bottom=8645983, right=544085, top=8673150)
    # bounds = rio.coords.BoundingBox(left=499302, bottom=8649321, right=552749, top=8686096)

    # dhdt = xdem.DEM("temp.svalbard/medians/svalbard/dem/dhdt_2024.tif", load_data=False)
    # dhdt.crop(bounds)

    # dhdt *= (2024 - 2009)
    # stable = xdem.DEM("temp/stable_terrain.tif", load_data=False)
    # stable.crop(dhdt)

    # dhdt.data.mask[stable.data == 0] = True

    # dhdt_pts = dhdt.to_points(subset=40000).ds.dropna(subset=["b1"])

    # dhdt_pts = gpd.read_feather("temp.svalbard/uncertainty/sampled_pts_1000000.arrow").query("stable")
    dhdt_pts = pts.copy()

    dhdt_pts["x"] = dhdt_pts.geometry.x - dhdt_pts.geometry.x.min()
    dhdt_pts["y"] = dhdt_pts.geometry.y - dhdt_pts.geometry.y.min()

    # bin_func = np.linspace(50, 10000, 10)
    # bin_func = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 2000, 3000, 5000, 7000, 9000, 10000]
    # np.random.seed(3)
    variogram = xdem.spatialstats.sample_empirical_variogram(
        values=dhdt_pts["trend_2013-2024_slope"].values,
        coords=dhdt_pts[["x", "y"]].values,
        subsample=2000,
        subsample_method="cdist_point",
        n_variograms=30,
        # runs=None,
        verbose=True,
        bin_func=bins,
        n_jobs=1,
        # random_seed=3,
    )
    print(variogram)

    vgm_model, params = xdem.spatialstats.fit_sum_model_variogram(["Sph", "Sph", "Sph"], variogram)

    areas = 10 ** np.linspace(0, np.log10(30000e6), 1000)
    neff = []
    for area in areas:
        neff.append(xdem.spatialstats.neff_circular_approx_numerical(area, params))

    pd.Series(neff, pd.Index(areas, name="neff")).to_csv("vgm_neff_cirq.csv")

    variogram["bin_width"] = np.r_[[0], np.diff(variogram.index)]
    fig = plt.figure(figsize=(8.3 * 0.81, 5 * 0.81))
    axes = fig.subplots(2, 1, sharex=True, height_ratios=[0.3, 0.7])
    axes[0].bar(variogram.index, variogram["count"], width=variogram["bin_width"] * 1.1, align="edge", color="gray")
    axes[1].plot(variogram.index, vgm_model(variogram.index), label="Variogram model", color="black")
    axes[1].errorbar(variogram.index, variogram["exp"], yerr=variogram["err_exp"], fmt="x", label="Empirical variogram", color="royalblue")

    axes[1].set_ylim(0, variogram["exp"].max() * 1.3)
    for r in params["range"].values:
        axes[1].vlines(r, *axes[1].get_ylim(), color="gray", linestyles="--")
        
    
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    axes[0].set_ylabel("Bin count")
    axes[1].set_ylabel("Variance (m²)")
    axes[1].legend(loc="lower right")
    axes[1].set_xlabel("Spatial lag (m)")


    print(params)

    plt.tight_layout()
    # xdem.spatialstats.plot_variogram(
    #     variogram,
    #     xscale_range_split=[100, 1000],
    #     list_fit_fun=[vgm_model],
    #     list_fit_fun_label=["Standardized double-range variogram"],
    # )
    # xdem.spatialstats.
    # dhdt.show()
    plt.savefig("figures/ad_variogram.pdf")
    plt.show()


if __name__ == "__main__":
    main()
