"""Processing times of the pipeline stages, reconstructed from file modification times.

No stage logs its runtime, so the times are inferred from when each stage wrote its
outputs. The input is a listing of the processing directory, made on the processing
machine from the repository root with::

    find temp.svalbard -type f \\( -name '*.tif' -o -name '*.vrt' -o -name '*.json' -o -name '*.csv' \\) \\
      -printf '%T@\\t%s\\t%p\\n' | gzip > adsvalbard_mtimes.tsv.gz

and the table is printed with ``python extra/timings.py adsvalbard_mtimes.tsv.gz``.

The strip loops (download and rigid co-registration, outlier probabilities) and the
vertical co-registration ran one item at a time, so the time between consecutive outputs
is the time per item. Stacking and mosaicking ran several workers at once, so only their
total working time is meaningful. Pauses longer than ``MAX_GAP`` (interruptions) are left
out of all totals, as is any single item that took longer, so the totals are lower bounds. Outputs that were rewritten later (e.g. stack chunks rebuilt after an
exclusion list changed) carry the time of the last write, which makes those totals
approximate.
"""

import sys
from pathlib import Path
from typing import Final

import pandas as pd

MAX_GAP: Final = pd.Timedelta(hours=1)
STRIP_RE: Final = r"(SETSM_s2s041_[^/]*?_seg\d+)"
# The years of the published products; the stacks of other years are left over from tests
STACK_YEARS: Final = range(2013, 2025)
STACK_RE: Final = (
    r"temp\.svalbard/medians/svalbard/(?P<product>dem|dem_noncoreg|dem_vertcoreg)/"
    r"median_(?P<filt>filt_\d{3}_)?dem_(?P<year>\d{4})_chunks_3584/chunk_\d{3}_\d{3}\.tif$"
)
MOSAIC_RE: Final = (
    r"temp\.svalbard/filt/svalbard/chunks_3584/chunk_\d{3}_\d{3}/"
    r"chunk_\d{3}_\d{3}_(?:20\d\d|trend_2013-2024_slope)\.tif$"
)


def read_listing(path: Path) -> pd.DataFrame:
    """Read a ``find -printf '%T@\\t%s\\t%p\\n'`` listing.

    Returns
    -------
    A frame with the columns ``time`` (UTC timestamps) and ``path`` (relative to the
    repository root).
    """
    listing = pd.read_csv(path, sep="\t", names=["mtime", "size", "path"])
    listing["time"] = pd.to_datetime(listing["mtime"], unit="s")
    return listing[["time", "path"]]


def strip_times(listing: pd.DataFrame, pattern: str) -> pd.Series:
    """Modification time of one output per strip, indexed by the strip title."""
    files = listing[listing["path"].str.match(pattern)]
    titles = files["path"].str.extract(STRIP_RE)[0]
    return files.set_index(titles)["time"].groupby(level=0).first()


def busy_time(times: pd.Series) -> pd.Timedelta:
    """Working time of a sequence of outputs: the sum of the gaps between them, without pauses."""
    gaps = times.sort_values().diff().dropna()
    return gaps[gaps < MAX_GAP].sum()


def sequential_gaps(times: pd.Series) -> pd.Series:
    """Time per item of a loop that wrote one output per item, without pauses."""
    gaps = times.sort_values().diff().dropna()
    return gaps[gaps < MAX_GAP]


def stage_table(listing: pd.DataFrame) -> pd.DataFrame:
    """Median time per unit and total working time of every stage.

    Returns
    -------
    One row per stage with the columns ``stage``, ``unit`` (with the number of items
    where it is meaningful), ``median`` and ``total`` (``pd.Timedelta``; ``NaT`` where
    not meaningful).
    """
    vrt = strip_times(listing, r"temp\.svalbard/arcticdem_vrts/.*_dem_epsg32633\.vrt$")
    coreg = strip_times(listing, r"temp\.svalbard/arcticdem_coreg/.*_dem_coreg\.tif$")
    proba = strip_times(listing, r"temp\.svalbard/outlier_proba/.*\.tif$")

    # A strip's VRTs are written right after its download, at the start of its
    # co-registration
    strips = pd.concat({"vrt": vrt, "coreg": coreg}, axis=1).dropna()
    rigid = strips["coreg"] - strips["vrt"]

    vertcoreg = listing.loc[
        listing["path"].str.match(
            r"temp\.svalbard/vertcoreg_results/vertcoreg_results_\d{4}_[a-z]+\.csv$"
        ),
        "time",
    ]

    stacks = listing.join(listing["path"].str.extract(STACK_RE)).dropna(
        subset=["product"]
    )
    stacks = stacks[stacks["year"].astype(int).isin(STACK_YEARS)]
    stacks["product"] = stacks["product"] + "_" + stacks["filt"].fillna("")
    stack_busy = stacks.groupby(["product", "year"])["time"].apply(busy_time)
    main_stack = stack_busy.loc["dem_filt_075_"]

    mosaics = listing[listing["path"].str.match(MOSAIC_RE)]
    n_mosaic_chunks = (
        mosaics["path"].str.extract(r"chunks_3584/(chunk_\d{3}_\d{3})/")[0].nunique()
    )

    rows = [
        (
            "Rigid co-registration",
            f"per strip ({len(rigid):,})",
            rigid.median(),
            rigid.sum(),
        ),
        (
            "Outlier probability",
            f"per strip ({len(proba):,})",
            sequential_gaps(proba).median(),
            busy_time(proba),
        ),
        (
            "Vertical co-registration",
            f"per region-year ({len(vertcoreg)})",
            sequential_gaps(vertcoreg).median(),
            busy_time(vertcoreg),
        ),
        (
            "Stacking, main product (outlier probability < 0.75)",
            f"per year ({len(main_stack)})",
            main_stack.median(),
            main_stack.sum(),
        ),
        ("Stacking, all four products", "", pd.NaT, stack_busy.sum()),
        (
            "Mosaicking and trend fitting",
            f"{n_mosaic_chunks} chunks",
            pd.NaT,
            busy_time(mosaics["time"]),
        ),
    ]
    return pd.DataFrame(rows, columns=["stage", "unit", "median", "total"])


def format_duration(duration: pd.Timedelta) -> str:
    """Format a duration in the most readable unit, with days added above a day.

    Examples
    --------
    >>> format_duration(pd.Timedelta(seconds=14))
    '14 s'
    >>> format_duration(pd.Timedelta(minutes=1.52))
    '1.5 min'
    >>> format_duration(pd.Timedelta(hours=3.6))
    '4 h'
    >>> format_duration(pd.Timedelta(hours=455))
    '455 h (19.0 days)'
    >>> format_duration(pd.NaT)
    '–'
    """
    if pd.isna(duration):
        return "–"
    seconds = duration.total_seconds()
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    hours = seconds / 3600
    if hours < 24:
        return f"{hours:.0f} h"
    return f"{hours:.0f} h ({hours / 24:.1f} days)"


def main() -> None:
    table = stage_table(read_listing(Path(sys.argv[1])))
    print("| Stage | Unit | Median | Total |")
    print("|---|---|---|---|")
    for row in table.itertuples():
        print(
            f"| {row.stage} | {row.unit} | {format_duration(row.median)} | {format_duration(row.total)} |"
        )


if __name__ == "__main__":
    main()
