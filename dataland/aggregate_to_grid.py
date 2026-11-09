"""Streaming area-weighted aggregation onto regular projected regional cells."""

import argparse
import logging

import numpy as np
import xarray as xr
from tqdm import tqdm

if __package__:
    from .aggregation_defaults import EARTH_RADIUS_M, OUTPUT_DOMAINS, TRANSTABLE
    from .aggregation_grid import RegionalGrid, preset_crs
    from .aggregation_source import SourceField, source_tiles, tile_count, _source_axis, _slope_tile
    from .aggregation_statistics import Statistics
    from .aggregation_output import build_output
    from .neighbourhood import Neighbourhood
else:
    from aggregation_defaults import EARTH_RADIUS_M, OUTPUT_DOMAINS, TRANSTABLE
    from aggregation_grid import RegionalGrid, preset_crs
    from aggregation_source import SourceField, source_tiles, tile_count, _source_axis, _slope_tile
    from aggregation_statistics import Statistics
    from aggregation_output import build_output
    from neighbourhood import Neighbourhood

LOG = logging.getLogger(__name__)


def aggregate(input_file, grid, source_var="COVER", mode="fractions",
              latitude=None, longitude=None, memory_mb=128, categories=None,
              fill_missing_mean=None, fill_uncovered_mean=None, fill_uncovered=None,
              radius_m=None):
    """Aggregate valid source pixel areas, assigning whole pixels by centre.

    memory_mb bounds tile working space approximately, not output accumulators
    or backend I/O caches. Verified projection-specific bounds prune source
    windows; data tiles with no centres inside the domain are never loaded.
    Fills apply only after aggregation. fill_uncovered applies to both modes;
    fill_uncovered_mean is its compatibility alias.
    """
    if mode not in ("fractions", "mean", "std", "slope"):
        raise ValueError("mode must be fractions, mean, std or slope")
    if radius_m is not None:
        if mode not in ("std", "slope"):
            raise ValueError("radius_m is supported only for std and slope")
        if not np.isfinite(radius_m) or not 0 < radius_m <= np.pi * EARTH_RADIUS_M:
            raise ValueError("radius_m must be positive, finite and no greater than half the circumference")
    if not np.isfinite(memory_mb) or memory_mb < 1:
        raise ValueError("memory_mb must be finite and at least 1")
    if fill_uncovered_mean is not None:
        if fill_uncovered is not None and fill_uncovered != fill_uncovered_mean:
            raise ValueError("Conflicting fill_uncovered and fill_uncovered_mean values")
        fill_uncovered = fill_uncovered_mean
    for name, value in (("fill_missing_mean", fill_missing_mean),
                        ("fill_uncovered", fill_uncovered)):
        if value is not None:
            if name == "fill_missing_mean" and mode != "mean":
                raise ValueError(f"{name} is only supported in mean mode")
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
    if mode == "fractions" and fill_uncovered is not None and not 0 <= fill_uncovered <= 1:
        raise ValueError("Fraction fill_uncovered must be between 0 and 1")
    categories = TRANSTABLE if categories is None else categories
    statistics = Statistics(len(grid.x) * len(grid.y), mode, categories)
    max_pixels = max(1, int(memory_mb * 1024**2 / 256))
    neighbourhood = Neighbourhood(grid, radius_m, EARTH_RADIUS_M) if radius_m is not None else None
    with xr.open_dataset(input_file) as ds:
        source = SourceField(ds, source_var, latitude, longitude)
        lats, lons = source.lats, source.lons
        tile_rows, tile_cols = source.tile_shape(max_pixels)
        LOG.info("Source %s x %s; tile at most %s x %s; target %s x %s",
                 len(lats), len(lons), tile_rows, tile_cols, len(grid.y), len(grid.x))
        windows = (neighbourhood.windows(lats, lons) if neighbourhood is not None
                   else grid.source_windows(lats, lons))
        selected = sum((rs.stop - rs.start) * (cs.stop - cs.start) for rs, cs in windows)
        LOG.info("Source windows: %s; scanning %s / %s pixels (%.2f%%)",
                 len(windows), f"{selected:,}", f"{len(lats) * len(lons):,}",
                 100 * selected / (len(lats) * len(lons)))
        tiles = source_tiles(windows, tile_rows, tile_cols)
        with tqdm(total=tile_count(windows, tile_rows, tile_cols),
                  desc="Aggregating source", unit="tile") as progress:
            for rows, columns in tiles:
                lon, lat = np.meshgrid(lons[columns], lats[rows])
                matches = (neighbourhood.matches(lon.ravel(), lat.ravel(), max_pixels)
                           if neighbourhood is not None else grid.matches(lon.ravel(), lat.ravel()))
                tile_fields = None
                for source_indices, target_indices in matches:
                    if tile_fields is None:
                        tile_fields, tile_weights = source.read(rows, columns, mode)
                    fields = {name: values[source_indices] for name, values in tile_fields.items()}
                    statistics.add(target_indices, fields, tile_weights[source_indices])
                progress.update()
    return build_output(
        grid, statistics, input_file, source_var, source.units, tile_rows, tile_cols,
        fill_missing_mean, fill_uncovered, radius_m,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--source-var", default="COVER")
    parser.add_argument("--mode", choices=("fractions", "mean", "std", "slope"), default="fractions")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--grid-file", help="CF NetCDF with projected 1-D x/y axes and CRS")
    target.add_argument("--domain", choices=tuple(OUTPUT_DOMAINS))
    target.add_argument("--point-crs", help="Explicit CRS for fitting a native lattice to --point-file")
    parser.add_argument("--point-spacing", type=float, help="Native spacing in metres; required with --point-crs")
    parser.add_argument("--grid-var", help="CRS variable, or target data variable carrying grid_mapping")
    parser.add_argument("--point-file", help="Optional NetCDF paired output points selecting native reference cells")
    parser.add_argument("--point-latitude", default="latitude", help="Point-file latitude variable")
    parser.add_argument("--point-longitude", default="longitude", help="Point-file longitude variable")
    parser.add_argument("--latitude", help="Source latitude coordinate name")
    parser.add_argument("--longitude", help="Source longitude coordinate name")
    parser.add_argument("--memory-mb", type=float, default=128)
    parser.add_argument("--radius-m", type=float,
                        help="Std/slope: overlapping great-circle radius in metres; default native cells")
    parser.add_argument("--fill-missing-mean", type=float,
                        help="Mean mode: fill cells having source samples but no valid values")
    parser.add_argument("--fill-uncovered", "--fill-uncovered-mean", type=float,
                        help="Fill every output field where no source samples contribute (both modes)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.grid_var and not args.grid_file:
        parser.error("--grid-var requires --grid-file")
    if args.point_crs and (not args.point_file or args.point_spacing is None):
        parser.error("--point-crs requires --point-file and --point-spacing")
    if args.point_spacing is not None and not args.point_crs:
        parser.error("--point-spacing requires --point-crs")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.point_crs:
        grid = RegionalGrid.from_points(args.point_file, args.point_crs, args.point_spacing,
                                        args.point_latitude, args.point_longitude)
    elif args.domain and args.point_file:
        domain = OUTPUT_DOMAINS[args.domain]
        grid = RegionalGrid.from_points(
            args.point_file, preset_crs(args.domain), domain["resolution"],
            args.point_latitude, args.point_longitude,
        )
    else:
        grid = (RegionalGrid.from_netcdf(args.grid_file, args.grid_var)
                if args.grid_file else RegionalGrid.from_preset(args.domain))
        if args.point_file:
            grid = grid.select_points(args.point_file, args.point_latitude, args.point_longitude)
    output = aggregate(args.source_file, grid, args.source_var, args.mode,
                       args.latitude, args.longitude, args.memory_mb,
                       fill_missing_mean=args.fill_missing_mean,
                       fill_uncovered=args.fill_uncovered, radius_m=args.radius_m)
    output.to_netcdf(args.output)
    LOG.info("Wrote %s", args.output)



if __name__ == "__main__":
    main()
