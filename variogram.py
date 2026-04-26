import xdem
import matplotlib.pyplot as plt
import rasterio as rio
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
import time
import tqdm
from pathlib import Path


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
    pool=32,               # <- much smaller default
    local_frac=0.75,
    min_sep=0.0,
    seed=0,
    dtype=np.float32,
):
    """
    Faster greedy sampler with the same basic idea as the original:
    choose candidate points that flatten the pair-distance histogram.

    Main speedups:
      - preallocated point array
      - no np.vstack in loop
      - squared distances instead of sqrt
      - precomputed log-bin edges
      - lower temporary array churn
    """

    if bins is None:
        bins = 10 ** np.linspace(np.log10(30), np.log10(100000), 50)

    bins = np.asarray(bins, dtype=np.float64)
    bins2 = bins * bins
    log_lo = np.log(bins[:-1])
    log_hi = np.log(bins[1:])

    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = map(float, bounds)

    n_bins = len(bins) - 1
    min_sep2 = float(min_sep) ** 2

    def random_points(n):
        pts = np.empty((n, 2), dtype=dtype)
        pts[:, 0] = rng.uniform(minx, maxx, n)
        pts[:, 1] = rng.uniform(miny, maxy, n)
        return pts

    def farthest_subset(candidates, k):
        if len(candidates) <= k:
            return candidates.copy()

        sel = np.empty(k, dtype=np.int32)
        sel[0] = rng.integers(len(candidates))
        d2 = np.sum((candidates - candidates[sel[0]]) ** 2, axis=1, dtype=np.float64)

        for i in range(1, k):
            sel[i] = np.argmax(d2)
            new_d2 = np.sum((candidates - candidates[sel[i]]) ** 2, axis=1, dtype=np.float64)
            np.minimum(d2, new_d2, out=d2)

        return candidates[sel]

    def pair_hist_sq(points):
        """Histogram of pairwise squared distances."""
        k = len(points)
        hist = np.zeros(n_bins, dtype=np.int64)
        if k < 2:
            return hist

        dx = points[:, 0][:, None] - points[:, 0][None, :]
        dy = points[:, 1][:, None] - points[:, 1][None, :]
        d2 = dx * dx + dy * dy

        iu = np.triu_indices(k, 1)
        idx = np.searchsorted(bins2, d2[iu], side="right") - 1
        valid = (idx >= 0) & (idx < n_bins)
        hist += np.bincount(idx[valid], minlength=n_bins)

        return hist

    def hist_rows_sq(d2):
        """
        d2: shape (m_candidates, k_existing)
        returns shape (m_candidates, n_bins)
        """
        idx = np.searchsorted(bins2, d2, side="right") - 1
        valid = (idx >= 0) & (idx < n_bins)

        row_ids = np.broadcast_to(
            np.arange(d2.shape[0], dtype=np.int64)[:, None], idx.shape
        )[valid]
        bin_ids = idx[valid]

        counts = np.bincount(
            row_ids * n_bins + bin_ids,
            minlength=d2.shape[0] * n_bins,
        )
        return counts.reshape(d2.shape[0], n_bins)

    # ----------------------------
    # Initial anchors
    # ----------------------------
    pts = np.empty((n_points, 2), dtype=dtype)

    init_candidates = random_points(max(10 * n_init, 200))
    init_pts = farthest_subset(init_candidates, min(n_init, n_points))

    k = len(init_pts)
    pts[:k] = init_pts

    if min_sep > 0 and k > 1:
        keep = [0]
        for i in range(1, k):
            d2 = np.sum((pts[keep] - pts[i]) ** 2, axis=1)
            if np.all(d2 >= min_sep2):
                keep.append(i)

        pts[:len(keep)] = pts[keep]
        k = len(keep)

    while k < min(n_init, n_points):
        c = random_points(1)[0]
        if k == 0:
            pts[k] = c
            k += 1
        else:
            d2 = np.sum((pts[:k] - c) ** 2, axis=1)
            if np.all(d2 >= min_sep2):
                pts[k] = c
                k += 1

    hist = pair_hist_sq(pts[:k])

    # ----------------------------
    # Greedy additions
    # ----------------------------
    while k < n_points:
        # adaptive pool helps a lot
        if k < 200:
            m = max(pool, 32)
        elif k < 2000:
            m = min(pool, 16)
        else:
            m = min(pool, 8)

        # total number of pairs after adding one point
        target = (k * (k + 1) / 2) / n_bins

        deficits = np.clip(target - hist, 0, None) + 1e-12
        probs = deficits / deficits.sum()

        use_local = rng.random(m) < local_frac
        candidates = np.empty((m, 2), dtype=dtype)

        # Local ring proposals
        if np.any(use_local):
            j = np.flatnonzero(use_local)
            b = rng.choice(n_bins, size=len(j), p=probs)
            anchors = pts[rng.integers(k, size=len(j))]

            r = np.exp(rng.uniform(log_lo[b], log_hi[b])).astype(dtype, copy=False)
            a = rng.uniform(0.0, 2 * np.pi, len(j))

            candidates[j, 0] = anchors[:, 0] + r * np.cos(a)
            candidates[j, 1] = anchors[:, 1] + r * np.sin(a)

        # Global proposals
        if np.any(~use_local):
            j = np.flatnonzero(~use_local)
            candidates[j] = random_points(len(j))

        # clip to bbox
        np.clip(candidates[:, 0], minx, maxx, out=candidates[:, 0])
        np.clip(candidates[:, 1], miny, maxy, out=candidates[:, 1])

        # squared distances candidate->existing
        dx = candidates[:, 0][:, None].astype(np.float64) - pts[:k, 0][None, :].astype(np.float64)
        dy = candidates[:, 1][:, None].astype(np.float64) - pts[:k, 1][None, :].astype(np.float64)
        d2 = dx * dx + dy * dy

        if min_sep > 0:
            valid_cand = np.all(d2 >= min_sep2, axis=1)
            if not np.any(valid_cand):
                raise RuntimeError("No valid candidates; reduce min_sep or increase pool.")
            candidates = candidates[valid_cand]
            d2 = d2[valid_cand]

        cand_hists = hist_rows_sq(d2)
        scores = np.sum((hist[None, :] + cand_hists - target) ** 2, axis=1)

        best = int(np.argmin(scores))
        pts[k] = candidates[best]
        hist += cand_hists[best]
        k += 1

    return pts

import math
from collections import OrderedDict

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds


class StableBlockSampler:
    """
    Multiresolution sampler for a huge binary stable-terrain raster.

    Coarse level:
        An 80 m (factor 16) support mask saying whether each coarse cell contains
        any stable 5 m pixels.

    Fine level:
        Each coarse cell can lazily load its corresponding 16x16 full-resolution
        pixels and cache the coordinates of stable 5 m cell centers.
    """

    def __init__(
        self,
        stable_path,
        bbox=None,
        coarse_factor=16,     # 5 m -> 80 m
        cache_size=50000,     # number of coarse cells to cache
    ):
        self.ds = rio.open(stable_path)
        self.coarse_factor = int(coarse_factor)
        self.cache_size = int(cache_size)

        if bbox is None:
            self.src_win = Window(0, 0, self.ds.width, self.ds.height)
        else:
            win = from_bounds(*bbox, transform=self.ds.transform)
            self.src_win = win.round_offsets().round_lengths()

        self.src_row0 = int(self.src_win.row_off)
        self.src_col0 = int(self.src_win.col_off)
        self.src_h = int(self.src_win.height)
        self.src_w = int(self.src_win.width)

        # Build coarse support mask with Resampling.max:
        # coarse cell = 1 if any fine pixel in block is stable.
        coarse_h = math.ceil(self.src_h / self.coarse_factor)
        coarse_w = math.ceil(self.src_w / self.coarse_factor)

        coarse = self.ds.read(
            1,
            window=self.src_win,
            out_shape=(coarse_h, coarse_w),
            # resampling=Resampling.max,
            masked=False,
        )

        self.coarse_mask = (coarse == 1)
        self.coarse_rows, self.coarse_cols = np.nonzero(self.coarse_mask)

        if self.coarse_rows.size == 0:
            raise ValueError("No stable terrain found in requested extent.")

        # Affine pieces for xy <-> row/col conversion
        self.x0 = self.ds.transform.c
        self.y0 = self.ds.transform.f
        self.dx = self.ds.transform.a
        self.dy = self.ds.transform.e  # usually negative for north-up rasters

        self.pixel_size = abs(self.dx)
        self.coarse_size = self.pixel_size * self.coarse_factor

        # LRU cache: (coarse_r, coarse_c) -> (N, 2) stable 5 m xy points
        self._fine_cache = OrderedDict()

    def close(self):
        self.ds.close()

    # ---------- coordinate helpers ----------

    def xy_to_src_rc(self, x, y):
        """Map XY to full-resolution source row/col."""
        col = int((x - self.x0) / self.dx)
        row = int((y - self.y0) / self.dy)
        return row, col

    def xy_to_coarse_rc(self, x, y):
        """Map XY to coarse row/col within bbox/source window."""
        src_r, src_c = self.xy_to_src_rc(x, y)
        rel_r = src_r - self.src_row0
        rel_c = src_c - self.src_col0
        coarse_r = rel_r // self.coarse_factor
        coarse_c = rel_c // self.coarse_factor
        return int(coarse_r), int(coarse_c)

    def coarse_rc_to_src_window(self, coarse_r, coarse_c):
        """Return full-resolution window for one coarse cell."""
        r0 = self.src_row0 + coarse_r * self.coarse_factor
        c0 = self.src_col0 + coarse_c * self.coarse_factor
        r1 = min(r0 + self.coarse_factor, self.src_row0 + self.src_h)
        c1 = min(c0 + self.coarse_factor, self.src_col0 + self.src_w)
        return Window(c0, r0, c1 - c0, r1 - r0)

    # ---------- fine cache ----------

    def _get_fine_points_in_coarse_cell(self, coarse_r, coarse_c):
        """
        Return stable 5 m cell centers inside one coarse cell.
        Cached after first read.
        """
        key = (int(coarse_r), int(coarse_c))
        if key in self._fine_cache:
            self._fine_cache.move_to_end(key)
            return self._fine_cache[key]

        # Outside coarse grid?
        if (
            coarse_r < 0
            or coarse_c < 0
            or coarse_r >= self.coarse_mask.shape[0]
            or coarse_c >= self.coarse_mask.shape[1]
        ):
            pts = np.empty((0, 2), dtype=np.float32)
            self._fine_cache[key] = pts
            return pts

        # No stable support in this coarse cell
        if not self.coarse_mask[coarse_r, coarse_c]:
            pts = np.empty((0, 2), dtype=np.float32)
            self._fine_cache[key] = pts
            return pts

        win = self.coarse_rc_to_src_window(coarse_r, coarse_c)
        arr = self.ds.read(1, window=win, masked=True)
        mask = (~np.ma.getmaskarray(arr)) & (arr == 1)

        rr, cc = np.nonzero(mask)
        if rr.size == 0:
            pts = np.empty((0, 2), dtype=np.float32)
        else:
            rr = rr + int(win.row_off)
            cc = cc + int(win.col_off)
            xs, ys = rio.transform.xy(self.ds.transform, rr, cc, offset="center")
            pts = np.column_stack([xs, ys]).astype(np.float32)

        self._fine_cache[key] = pts
        self._fine_cache.move_to_end(key)

        # LRU eviction
        if len(self._fine_cache) > self.cache_size:
            self._fine_cache.popitem(last=False)

        return pts

    # ---------- candidate generation ----------

    def random_global_point(self, rng):
        """Random stable 5 m point sampled via a stable coarse cell."""
        while True:
            i = rng.integers(0, self.coarse_rows.size)
            cr = self.coarse_rows[i]
            cc = self.coarse_cols[i]
            pts = self._get_fine_points_in_coarse_cell(cr, cc)
            if pts.shape[0] > 0:
                return pts[rng.integers(0, pts.shape[0])]

    def snap_local_point(self, x, y, rng, neighborhood=1):
        """
        Snap target XY to nearest available stable 5 m point inside a local
        coarse-cell neighborhood.

        neighborhood=1 -> search 3x3 coarse cells
        """
        cr, cc = self.xy_to_coarse_rc(x, y)

        all_pts = []
        for rr in range(cr - neighborhood, cr + neighborhood + 1):
            for cc2 in range(cc - neighborhood, cc + neighborhood + 1):
                pts = self._get_fine_points_in_coarse_cell(rr, cc2)
                if pts.shape[0] > 0:
                    all_pts.append(pts)

        if not all_pts:
            return None

        all_pts = np.vstack(all_pts)
        d2 = np.sum((all_pts.astype(np.float64) - np.array([x, y])) ** 2, axis=1)
        best = np.argmin(d2)
        return all_pts[best]


def _hist_rows_sq(d2, bins2, n_bins):
    """
    Candidate histogram contributions for squared distances.
    d2 has shape (m_candidates, k_reference_points)
    """
    idx = np.searchsorted(bins2, d2, side="right") - 1
    valid = (idx >= 0) & (idx < n_bins)

    row_ids = np.broadcast_to(
        np.arange(d2.shape[0], dtype=np.int64)[:, None],
        idx.shape
    )[valid]
    bin_ids = idx[valid]

    counts = np.bincount(
        row_ids * n_bins + bin_ids,
        minlength=d2.shape[0] * n_bins,
    )
    return counts.reshape(d2.shape[0], n_bins)


def sample_variogram_points_stable_multires(
    stable_path,
    n_points,
    bins,
    bbox=None,
    coarse_factor=16,
    n_init=16,
    pool=32,
    score_subset=256,
    local_frac=0.8,
    seed=0,
):
    """
    Sample ~balanced variogram points directly from a huge stable-terrain raster
    using a multiresolution approach.

    Final point coordinates are always exact 5 m stable cells.
    Only a coarse support mask is held in memory.
    """
    rng = np.random.default_rng(seed)
    bins = np.asarray(bins, dtype=np.float64)
    bins2 = bins * bins
    n_bins = len(bins) - 1
    log_lo = np.log(bins[:-1])
    log_hi = np.log(bins[1:])

    sampler = StableBlockSampler(
        stable_path=stable_path,
        bbox=bbox,
        coarse_factor=coarse_factor,
    )

    pts = np.empty((n_points, 2), dtype=np.float32)
    hist = np.zeros(n_bins, dtype=np.int64)

    # Optional de-duplication
    chosen = set()

    def point_key(p):
        # fine raster is 5 m; rounding to mm is plenty stable
        return (float(p[0]), float(p[1]))

    def unique_global():
        for _ in range(1000):
            p = sampler.random_global_point(rng)
            k = point_key(p)
            if k not in chosen:
                return p
        raise RuntimeError("Failed to generate a unique stable point.")

    # -------------------------
    # Initial spread-out points
    # -------------------------
    init_candidates = np.vstack([unique_global() for _ in range(max(10 * n_init, 300))])

    # Greedy farthest-point initialization
    sel = np.empty(min(n_init, n_points), dtype=np.int64)
    sel[0] = rng.integers(0, init_candidates.shape[0])
    d2 = np.sum(
        (init_candidates - init_candidates[sel[0]]) ** 2,
        axis=1,
        dtype=np.float64,
    )

    for i in range(1, len(sel)):
        sel[i] = np.argmax(d2)
        new_d2 = np.sum(
            (init_candidates - init_candidates[sel[i]]) ** 2,
            axis=1,
            dtype=np.float64,
        )
        np.minimum(d2, new_d2, out=d2)

    pts[:len(sel)] = init_candidates[sel]
    for i in range(len(sel)):
        chosen.add(point_key(pts[i]))

    k = len(sel)

    # Exact histogram for initial anchors
    for i in range(1, k):
        dx = pts[i, 0].astype(np.float64) - pts[:i, 0].astype(np.float64)
        dy = pts[i, 1].astype(np.float64) - pts[:i, 1].astype(np.float64)
        d2 = dx * dx + dy * dy
        idx = np.searchsorted(bins2, d2, side="right") - 1
        idx = idx[(idx >= 0) & (idx < n_bins)]
        hist += np.bincount(idx, minlength=n_bins)

    # -------------------------
    # Greedy additions
    # -------------------------
    with tqdm.tqdm(total=n_points, desc="Sampling points for vgm.") as progress_bar:
        while k < n_points:
            total_pairs_if_added = k * (k + 1) / 2
            target = total_pairs_if_added / n_bins

            weights = target - hist.astype(np.float64)
            probs = np.clip(weights, 0, None) + 1e-12
            probs /= probs.sum()

            cand_list = []

            # Build candidate pool
            n_local = int(round(pool * local_frac))
            n_global = pool - n_local

            # Local ring proposals, refined to exact 5 m stable cells
            for _ in range(n_local):
                b = rng.choice(n_bins, p=probs)
                anchor = pts[rng.integers(0, k)]

                r = np.exp(rng.uniform(log_lo[b], log_hi[b]))
                a = rng.uniform(0.0, 2.0 * np.pi)

                tx = float(anchor[0] + r * np.cos(a))
                ty = float(anchor[1] + r * np.sin(a))

                p = sampler.snap_local_point(tx, ty, rng, neighborhood=1)
                if p is not None and point_key(p) not in chosen:
                    cand_list.append(p)

            # Global proposals
            for _ in range(n_global):
                p = unique_global()
                cand_list.append(p)

            if len(cand_list) == 0:
                raise RuntimeError("No valid candidates produced.")

            cand_xy = np.asarray(cand_list, dtype=np.float32)

            # Score candidates against a subset of existing points
            ref_n = min(score_subset, k)
            ref_ids = rng.choice(k, size=ref_n, replace=False)
            refs = pts[ref_ids].astype(np.float64)

            dx = cand_xy[:, 0][:, None].astype(np.float64) - refs[:, 0][None, :]
            dy = cand_xy[:, 1][:, None].astype(np.float64) - refs[:, 1][None, :]
            d2 = dx * dx + dy * dy

            cand_hist = _hist_rows_sq(d2, bins2, n_bins)

            # Reward counts in deficit bins, penalize overfull bins
            scores = cand_hist @ weights
            best = int(np.argmax(scores))
            best_pt = cand_xy[best]

            pts[k] = best_pt
            chosen.add(point_key(best_pt))

            # Exact histogram update against all accepted points
            dx = pts[k, 0].astype(np.float64) - pts[:k, 0].astype(np.float64)
            dy = pts[k, 1].astype(np.float64) - pts[:k, 1].astype(np.float64)
            d2 = dx * dx + dy * dy

            idx = np.searchsorted(bins2, d2, side="right") - 1
            idx = idx[(idx >= 0) & (idx < n_bins)]
            hist += np.bincount(idx, minlength=n_bins)

            k += 1
            progress_bar.update()

    sampler.close()
    return pts, hist


# def sample_points(min_bin: float = np.log10(30), max_bin: np.log10(200000), n_points=3000, redo=False):

    

def main(verbose=True, n_points=10000, redo=False, random_seed=1):
    import adsvalbard.rasters
    # bbox = (469120,8554890,583640,8835780)
    # bbox = (408989.5978,8531849.9630,666111.5882,8867476.9870)
    bbox = (396000, 8479000, 727000, 8954000) # Spitsbergen and Nordaustlandet
    bins = 10 ** np.linspace(np.log10(30), np.log10(2.5e5), 50) 
    
    cache_path = Path("temp.svalbard/uncertainty/vgm_sample_pts.arrow")
    var_col = "trend_2013-2024_slope"
    err_col = var_col.replace("_slope", "_slope_err")
    # err_col = var_col + "_se"

    if cache_path.is_file() and not redo:
        pts = gpd.read_feather(cache_path)
    else:
        start_time = time.time()
        all_pts = []
        # Running multiple times increases mid- to long-range sample counts
        for i in range(3):
            pts_xy, hist = sample_variogram_points_stable_multires(
                stable_path="temp/stable_terrain.tif",
                n_points=n_points,
                bins=bins,
                bbox=bbox,          # important: only work inside the analysis region
                coarse_factor=16,   # 5 m -> 80 m support grid
                n_init=64,
                pool=32,
                score_subset=256,
                local_frac=0.8,
                seed=random_seed + i * 1000,
            )
            all_pts.append(pts_xy)
        pts_xy = np.concatenate(all_pts)
        print(f"{pts_xy.shape[0]}: {time.time() - start_time:.2f} s")
        print(pts_xy.shape)

        pts = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(pts_xy[:, 0], pts_xy[:, 1], crs=32633)
        )
        print(pts.shape)
        print("Sampling stable terrain")
        pts["stable"] = adsvalbard.rasters.sample_raster("temp/stable_terrain.tif", pts["geometry"]) == 1

        pts = pts.query("stable").copy()


        print("Sampling elevation change")
        pts[var_col] = adsvalbard.rasters.sample_raster(f"temp.svalbard/filt/svalbard/mosaics_3584/{var_col}.vrt", geometry=pts["geometry"])
        pts[err_col] = adsvalbard.rasters.sample_raster(f"temp.svalbard/filt/svalbard/mosaics_3584/{err_col}.vrt", geometry=pts["geometry"])
        pts.to_feather(cache_path)

    original_var = pts[var_col].var()
    pts = pts[pts[err_col] < (3 * pts[err_col].median())]
    pts = pts[np.abs(pts[var_col]) < (3 * np.abs(pts[var_col]).median())]
    scaled_col = var_col + "_scaled"
    pts[scaled_col] = (1e-3 + pts[err_col].mean()) * pts[var_col] / (pts[err_col] + 1e-3)

    
    # print(pts[scaled_col].describe())
    # print(pts[var_col].describe())

    
    # plt.hist(pts[err_col])
    # plt.show()
    # return
    # print(pts.shape)
    # return

    # print("Reading land")
    # if verbose:
    #     print("Reading land outlines")
    # land = gpd.read_file("zip://data/outlines/NP_S100_SHP.zip/NP_S100_SHP/S100_Land_f.shp", bbox=bbox).simplify(100).to_frame().dissolve().to_crs(32633)[0].item()

    # if verbose:
    #     print("Reading glacier outlines")
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore")
    #     glaciers = gpd.read_file("zip://data/outlines/NP_S100_SHP.zip/NP_S100_SHP/S100_Isbreer_f.shp", bbox=bbox).simplify(100).to_frame().dissolve().to_crs(32633)[0].item()


    # if verbose:
    #     print("Making stable terrain")
    # land = land.difference(glaciers)
    

    # print("Making sampling points")
    # start_time = time.time()
    # pts = sample_variogram_points_bbox_fast(bbox, 15000, bins=bins)
    # pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pts[:, 0], pts[:, 1], crs=32633))
    # print(pts.shape)
    # pts = pts[pts.intersects(land)]
    # print(f"{pts.shape[0]}: {time.time() - start_time:.2f} s")

    # print(pts.shape)
    # return
    # for i in range(5):
    #     print(i)
    #     if n_pts > 20000:
    #         break
    #     pts = sample_variogram_points_bbox_mixture(land.bounds, n_points=50000, bins=bins, seed=i)
    # pts = pts[pts.intersects(land)]
    #     n_pts += pts.shape[0]

    #     all_pts.append(pts)
    # pts = pd.concat(all_pts, ignore_index=True)
    
    # dhdt_pts = gpd.read_feather("temp.svalbard/uncertainty/sampled_pts_1000000.arrow").query("stable")

    pts["x"] = pts.geometry.x - pts.geometry.x.min()
    pts["y"] = pts.geometry.y - pts.geometry.y.min()

    # bin_func = np.linspace(50, 10000, 10)
    # bin_func = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 2000, 3000, 5000, 7000, 9000, 10000]
    # np.random.seed(3)
    print(pts[scaled_col].var())
    variogram = xdem.spatialstats.sample_empirical_variogram(
        values=pts[scaled_col].values,
        coords=pts[["x", "y"]].values,
        subsample=500,
        subsample_method="cdist_point",
        n_variograms=50,
        # runs=None,
        verbose=True,
        bin_func=bins,
        n_jobs=1,
        random_state=random_seed,
    )

    exp_scale = original_var / pts[scaled_col].var()
    variogram[["exp", "err_exp"]] *= exp_scale
    # Remove erroneous points that are far beyond the full variance of the data
    # variogram = variogram[variogram["exp"] < pts[scaled_col].var() * 1.15]
    print(variogram)

    vgm_model, params = xdem.spatialstats.fit_sum_model_variogram(["Sph", "Gaussian"], variogram, p0=[1e2, 1e-4, 1e4, 1e-4])

    areas = 10 ** np.linspace(0, np.log10(35000e6), 1000)
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

    axes[1].set_ylim(0, original_var * 1.32)
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
