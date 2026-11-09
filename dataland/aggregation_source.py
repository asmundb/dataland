"""Regular geographic source coordinates, areas and terrain derivatives."""

import numpy as np

if __package__:
    from .aggregation_defaults import EARTH_RADIUS_M
else:
    from aggregation_defaults import EARTH_RADIUS_M

def _regular_axis(values, name, tolerance=None):
    raw = np.asarray(values)
    if tolerance is None:
        epsilon = np.finfo(raw.dtype).eps if np.issubdtype(raw.dtype, np.floating) else 0
        tolerance = max(1e-8, 4 * epsilon * np.max(np.abs(raw), initial=0))
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite 1-D axis with at least two centres")
    step = (values[-1] - values[0]) / (len(values) - 1)
    differences = np.diff(values)
    if (step == 0 or not np.all(differences * step > 0)
            or not np.allclose(differences, step, rtol=1e-6, atol=tolerance)):
        raise ValueError(f"{name} must have regular, strictly monotonic spacing")
    return values, step



def _source_axis(ds, field, axis, name):
    if name is None:
        candidates = [
            str(key) for key in ds.variables
            if ds[key].ndim == 1 and ds[key].dims[0] in field.dims
            and (ds[key].attrs.get("standard_name") == axis
                 or str(key).lower() in (("lat", "latitude", "lats") if axis == "latitude"
                                        else ("lon", "longitude", "lons")))
        ]
        if len(candidates) != 1:
            raise ValueError(f"Specify source {axis}; expected one unambiguous 1-D coordinate")
        name = candidates[0]
    coord = ds[name]
    if coord.ndim != 1 or coord.dims[0] not in field.dims:
        raise ValueError(f"Source {axis} must be a 1-D coordinate of the input variable")
    units = coord.attrs.get("units", "degrees")
    if units not in ("degrees", "degrees_north", "degrees_east", "degree_north", "degree_east"):
        raise ValueError(f"Source {axis} must use degrees, not {units!r}")
    raw = coord.values
    epsilon = np.finfo(raw.dtype).eps if np.issubdtype(raw.dtype, np.floating) else 0
    tolerance = max(1e-8, 4 * epsilon * np.max(np.abs(raw), initial=0))
    values = np.asarray(raw, dtype=np.float64)
    if axis == "longitude":
        values = np.rad2deg(np.unwrap(np.deg2rad(values)))
    values, step = _regular_axis(values, f"source {axis}", tolerance=tolerance)
    if axis == "longitude":
        closure_tolerance = max(1e-6, 2 * tolerance)
        coverage = abs(step) * len(values)
        if abs(step) >= 180 or coverage > 360 + closure_tolerance:
            raise ValueError("Source longitude must not repeat or cover more than one globe")
        if abs(coverage - 360) <= closure_tolerance:
            # Float32 endpoints can imply slightly more than 360 degrees.
            # Restore exact closure without accepting a repeated seam column.
            step = np.copysign(360 / len(values), step)
            values = values[0] + np.arange(len(values)) * step
    edges = np.column_stack((values - step / 2, values + step / 2))
    if "bounds" in coord.attrs:
        bounds = ds[coord.attrs["bounds"]].transpose(coord.dims[0], ...).values
        if bounds.shape != edges.shape or not np.isfinite(bounds).all():
            raise ValueError(f"Invalid source {axis} bounds")
        if axis == "longitude":
            bounds = values[:, None] + (bounds - values[:, None] + 180) % 360 - 180
        if not np.allclose(np.sort(bounds, axis=1), np.sort(edges, axis=1),
                           atol=max(1e-7, tolerance), rtol=0):
            raise ValueError(f"Source {axis} bounds must match regular cell spacing")
        edges = bounds
    if axis == "latitude":
        if np.any(np.abs(values) > 90):
            raise ValueError("Source latitude centres are outside [-90, 90]")
        edges = np.clip(edges, -90, 90)
        widths = np.abs(np.diff(np.sin(np.deg2rad(edges)), axis=1)).ravel()
    else:
        widths = np.abs(np.deg2rad(edges[:, 1] - edges[:, 0]))
    return coord.dims[0], values, widths



def _slope_tile(field, lat_dim, lon_dim, lats, lons, rs, cs):
    """Derive geographic gradients using real neighbours beyond tile boundaries."""
    rows = np.arange(max(0, rs.start - 1), min(len(lats), rs.stop + 1))
    columns = np.arange(cs.start - 1, cs.stop + 1)
    step = (lons[-1] - lons[0]) / (len(lons) - 1)
    periodic = np.isclose(abs(step) * len(lons), 360, atol=1e-5, rtol=0)
    if not periodic:
        columns = columns[(columns >= 0) & (columns < len(lons))]
    values = field.isel({lat_dim: rows, lon_dim: columns % len(lons)}).transpose(
        lat_dim, lon_dim
    ).values.astype(np.float64)
    values[~np.isfinite(values)] = np.nan
    north = np.gradient(values, np.deg2rad(lats[rows]) * EARTH_RADIUS_M, axis=0)
    east = np.gradient(values, np.deg2rad(lons[0] + columns * step) * EARTH_RADIUS_M, axis=1)
    cosine = np.cos(np.deg2rad(lats[rows]))[:, None]
    np.divide(east, cosine, out=east, where=np.abs(cosine) > 1e-12)
    east[np.broadcast_to(np.abs(cosine) <= 1e-12, east.shape)] = np.nan
    # Missing centre elevations cannot produce valid terrain derivatives.
    east[~np.isfinite(values)] = np.nan
    north[~np.isfinite(values)] = np.nan
    r = slice(rs.start - rows[0], rs.stop - rows[0])
    c = slice(cs.start - columns[0], cs.stop - columns[0])
    return east[r, c], north[r, c]


class SourceField:
    """Validated source geometry and tile access within an open xarray dataset."""

    def __init__(self, dataset, variable, latitude=None, longitude=None):
        self.field = dataset[variable]
        self.lat_dim, self.lats, self.lat_weights = _source_axis(
            dataset, self.field, "latitude", latitude)
        self.lon_dim, self.lons, self.lon_weights = _source_axis(
            dataset, self.field, "longitude", longitude)
        if (self.field.ndim != 2 or self.lat_dim == self.lon_dim
                or set(self.field.dims) != {self.lat_dim, self.lon_dim}):
            raise ValueError("Source variable must have exactly the two latitude/longitude dimensions")
        self.units = self.field.attrs.get("units", "")

    def tile_shape(self, max_pixels):
        columns = min(len(self.lons), max(1, int(np.sqrt(max_pixels))))
        return max(1, max_pixels // columns), columns

    def read(self, rows, columns, mode):
        if mode == "slope":
            east, north = _slope_tile(self.field, self.lat_dim, self.lon_dim,
                                      self.lats, self.lons, rows, columns)
            fields = {"slope_x": east.ravel(), "slope_y": north.ravel()}
        else:
            values = self.field.isel({self.lat_dim: rows, self.lon_dim: columns}).transpose(
                self.lat_dim, self.lon_dim).values
            fields = {"mean": values.ravel()}
        weights = (EARTH_RADIUS_M**2 * self.lat_weights[rows, None]
                   * self.lon_weights[None, columns]).ravel()
        return fields, weights


def source_tiles(windows, rows_per_tile, columns_per_tile):
    for rows, columns in windows:
        for r in range(rows.start, rows.stop, rows_per_tile):
            for c in range(columns.start, columns.stop, columns_per_tile):
                yield (slice(r, min(r + rows_per_tile, rows.stop)),
                       slice(c, min(c + columns_per_tile, columns.stop)))


def tile_count(windows, rows_per_tile, columns_per_tile):
    return sum(((rows.stop - rows.start + rows_per_tile - 1) // rows_per_tile)
               * ((columns.stop - columns.start + columns_per_tile - 1) // columns_per_tile)
               for rows, columns in windows)
