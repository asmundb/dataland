"""Finalize statistics, report missing support and build NetCDF-ready output."""

import warnings

import numpy as np
from pyproj import Transformer
import xarray as xr

if __package__:
    from .aggregation_defaults import EARTH_RADIUS_M
else:
    from aggregation_defaults import EARTH_RADIUS_M


def build_output(grid, statistics, input_file, source_var, units,
                 tile_rows, tile_cols, fill_missing_mean, fill_uncovered, radius_m):
    mode, categories = statistics.mode, statistics.categories
    source_area, valid_area = statistics.source_area, statistics.valid_area
    source_count, valid_count = statistics.source_count, statistics.valid_count
    nout = len(source_count)
    if not source_count.any():
        raise ValueError("No source pixel centres fall inside the target grid")
    active = grid.points.mask if grid.points is not None else np.ones(nout, dtype=bool)
    missing = active & (source_count > 0) & (valid_area == 0)
    uncovered = active & (source_count == 0)
    missing_cells = int(np.count_nonzero(missing))
    uncovered_cells = int(np.count_nonzero(uncovered))
    empty = missing_cells + uncovered_cells
    if empty:
        if mode == "mean" or fill_uncovered is not None:
            missing_action = ("left as NaN" if fill_missing_mean is None
                              else f"filled with {fill_missing_mean}")
            uncovered_action = ("left as NaN" if fill_uncovered is None
                                else f"filled with {fill_uncovered}")
            warnings.warn(
                f"{empty} target cells have no valid source area: "
                f"{missing_cells} with only missing/non-finite source values ({missing_action}); "
                f"{uncovered_cells} with no source samples ({uncovered_action})",
                RuntimeWarning,
            )
        else:
            warnings.warn(f"{empty} target cells have no valid source area; statistics are NaN", RuntimeWarning)
    points = grid.points
    if points is None:
        dims = ("y", "x")
        shape = (len(grid.y), len(grid.x))
        output = xr.Dataset(coords={"x": grid.x, "y": grid.y})
        to_geographic = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
        target_lon = np.empty(shape)
        target_lat = np.empty(shape)
        for r in range(0, len(grid.y), tile_rows):
            for c in range(0, len(grid.x), tile_cols):
                x, y = np.meshgrid(grid.x[c:c + tile_cols], grid.y[r:r + tile_rows])
                target_lon[r:r + tile_rows, c:c + tile_cols], target_lat[r:r + tile_rows, c:c + tile_cols] = (
                    to_geographic.transform(x, y)
                )
    else:
        dims = ("points",)
        output = xr.Dataset(coords={
            "x": ("points", grid.x[points.columns]),
            "y": ("points", grid.y[points.rows]),
        })
        target_lat, target_lon = points.latitude, points.longitude

    def layout(values):
        if points is None:
            return values.reshape(shape)
        return values[points.rows * len(grid.x) + points.columns]

    for axis in ("x", "y"):
        output[axis].attrs = {"standard_name": f"projection_{axis}_coordinate", "units": "m"}
    output["crs"] = xr.DataArray(0, attrs={**grid.crs.to_cf(), "crs_wkt": grid.crs.to_wkt()})
    output = output.assign_coords(latitude=(dims, target_lat), longitude=(dims, target_lon))
    output.latitude.attrs = {"standard_name": "latitude", "units": "degrees_north"}
    output.longitude.attrs = {"standard_name": "longitude", "units": "degrees_east"}
    for name, total in statistics.finalize().items():
        if mode == "mean":
            if fill_missing_mean is not None:
                total[missing] = fill_missing_mean
        if fill_uncovered is not None:
            total[uncovered] = fill_uncovered
        output[name] = (dims, layout(total))
        output[name].attrs = {"units": "1" if mode in ("fractions", "slope") else units,
                              "grid_mapping": "crs",
                              "cell_methods": "area: standard_deviation" if name == "std" and mode == "std" else "area: mean"}
    for name, values, unit in (
        ("source_area", source_area, "m2"), ("valid_area", valid_area, "m2"),
        ("source_count", source_count, "1"), ("valid_count", valid_count, "1"),
    ):
        output[name] = (dims, layout(values))
        output[name].attrs = {"units": unit, "grid_mapping": "crs"}
    if mode != "fractions" or fill_uncovered is not None:
        status = np.zeros(nout, dtype=np.int8)
        status[missing] = 1
        status[uncovered] = 2
        status_name = f"{mode}_data_status"
        output[status_name] = (dims, layout(status))
        output[status_name].attrs = {
            "long_name": f"source support for {mode}, independent of output filling",
            "flag_values": np.array([0, 1, 2], dtype=np.int8),
            "flag_meanings": f"valid_{mode} only_missing_source_values no_source_samples",
            "grid_mapping": "crs",
        }
    output.attrs = {
        "aggregation_method": "source_pixel_centre_assignment",
        "weighting": "spherical source pixel area",
        "earth_radius_m": EARTH_RADIUS_M,
        "source_file": str(input_file),
        "source_variable": source_var,
        "mode": mode,
        "denominator": "valid source area; missing values excluded",
        "boundary_policy": "half-open cells in supplied axis order",
        "accuracy": "whole-pixel centre assignment, not exact area-overlap remapping",
    }
    if radius_m is not None:
        output.attrs["aggregation_method"] = "overlapping_great_circle_neighbourhood"
        output.attrs["radius_m"] = float(radius_m)
        output.attrs["distance_definition"] = "great-circle on weighting sphere; inclusive radius"
        output.attrs["boundary_policy"] = "source centres within radius, including beyond native target cells"
        output.attrs["accuracy"] = "whole source pixels selected by centre distance; not fractional circle overlap"
    if mode == "std":
        output.attrs["standard_deviation_definition"] = "area-weighted population; denominator=sum(weights)"
    if mode == "slope":
        output.attrs["slope_definition"] = (
            "area-weighted eastward/northward source gradients on the weighting sphere; "
            "common validity mask requires both components; elevations in metres"
        )
    if mode == "mean":
        output.attrs["missing_mean_cells"] = missing_cells
        output.attrs["uncovered_mean_cells"] = uncovered_cells
        for name, value in (("fill_missing_mean", fill_missing_mean),
                            ("fill_uncovered_mean", fill_uncovered)):
            if value is not None:
                output.attrs[name] = float(value)
    if fill_uncovered is not None:
        output.attrs["fill_uncovered"] = float(fill_uncovered)
        output.attrs["uncovered_cells"] = uncovered_cells
    if mode == "fractions":
        output.attrs["category_codes"] = "; ".join(
            f"{name}: {','.join(map(str, group))}" for name, group in categories.items()
        )
    if points is not None:
        output.attrs["point_selection"] = "native reference cells; supplied point order preserved"
        output.attrs["point_alignment_tolerance_m"] = min(1.0, abs(grid.dx) * 0.01, abs(grid.dy) * 0.01)
    return output
