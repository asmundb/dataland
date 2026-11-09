"""Regional target geometry, CF loading and native-cell selection."""

from dataclasses import dataclass
import logging
from typing import Optional

import numpy as np
from pyproj import CRS, Transformer
import xarray as xr

if __package__:
    from .aggregation_defaults import OUTPUT_DOMAINS
    from .aggregation_source import _regular_axis
else:
    from aggregation_defaults import OUTPUT_DOMAINS
    from aggregation_source import _regular_axis

LOG = logging.getLogger(__name__)


def preset_crs(name):
    projection = OUTPUT_DOMAINS[name]["projection"]
    return (CRS.from_cf(projection) if "grid_mapping_name" in projection
            else CRS.from_dict(projection))


def _read_points(path, latitude, longitude):
    with xr.open_dataset(path) as ds:
        lat, lon = ds[latitude], ds[longitude]
        if lat.ndim != 1 or lon.dims != lat.dims or lat.size == 0:
            raise ValueError("Point latitude/longitude must be nonempty paired 1-D arrays")
        for coord, allowed in ((lat, ("degrees", "degrees_north", "degree_north")),
                               (lon, ("degrees", "degrees_east", "degree_east"))):
            if coord.attrs.get("units", "degrees") not in allowed:
                raise ValueError("Point coordinates must use degrees")
        lats = np.asarray(lat.values, dtype=np.float64)
        lons = np.asarray(lon.values, dtype=np.float64)
    if (not np.isfinite(lats).all() or not np.isfinite(lons).all()
            or np.any(np.abs(lats) > 90)):
        raise ValueError("Point coordinates must be finite longitude/latitude in degrees")
    return lats, lons

@dataclass
class PointSelection:
    rows: np.ndarray
    columns: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    mask: np.ndarray



@dataclass
class RegionalGrid:
    """Projected cell centres; output retains the supplied axis ordering."""

    crs: CRS
    x: np.ndarray
    y: np.ndarray
    points: Optional[PointSelection] = None

    def __post_init__(self):
        self.crs = CRS.from_user_input(self.crs)
        if not self.crs.is_projected:
            raise ValueError("The regional grid must have a projected CRS")
        if any(not np.isclose(axis.unit_conversion_factor, 1.0)
               for axis in self.crs.axis_info[:2]):
            raise ValueError("The target CRS must use metres")
        self.x, self.dx = _regular_axis(self.x, "target x")
        self.y, self.dy = _regular_axis(self.y, "target y")
        self._transformer = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)

    @classmethod
    def from_preset(cls, name):
        domain = OUTPUT_DOMAINS[name]
        crs = preset_crs(name)
        cx, cy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
            *domain["center"]
        )
        height, width = domain["shape"]
        step = domain["resolution"]
        return cls(crs, cx + (np.arange(width) - (width - 1) / 2) * step,
                   cy - (np.arange(height) - (height - 1) / 2) * step)

    @classmethod
    def from_netcdf(cls, path, variable=None):
        """Read a CF grid_mapping and 1-D projection_x/y_coordinate axes."""
        with xr.open_dataset(path) as ds:
            crs_names = [name for name in ds.variables
                         if "grid_mapping_name" in ds[name].attrs
                         or "crs_wkt" in ds[name].attrs
                         or "spatial_ref" in ds[name].attrs]
            axis_dims = None
            if variable is not None:
                if variable not in ds:
                    raise ValueError(f"Target variable {variable!r} is absent")
                if variable in crs_names:
                    names = [variable]
                else:
                    mapping = ds[variable].attrs.get("grid_mapping")
                    names = [mapping] if mapping else []
                    axis_dims = ds[variable].dims
            else:
                names = crs_names
            if len(names) != 1 or names[0] not in ds:
                raise ValueError(
                    "Expected one CF CRS; --grid-var must select a CRS variable "
                    "or a data variable with a grid_mapping attribute"
                )
            attrs = ds[names[0]].attrs
            wkt = attrs.get("crs_wkt", attrs.get("spatial_ref"))
            crs = CRS.from_wkt(wkt) if wkt else CRS.from_cf(attrs)
            axes = []
            for axis in ("x", "y"):
                candidates = [
                    name for name in ds.variables
                    if ds[name].ndim == 1
                    and ds[name].attrs.get("standard_name") == f"projection_{axis}_coordinate"
                    and (axis_dims is None or ds[name].dims[0] in axis_dims)
                ]
                if len(candidates) != 1:
                    raise ValueError(f"Expected one CF projection_{axis}_coordinate axis")
                coord = ds[candidates[0]]
                unit = coord.attrs.get("units")
                if unit not in ("m", "metre", "metres", "meter", "meters", "km"):
                    raise ValueError(f"Unsupported or missing units for target {axis}: {unit!r}")
                values = coord.values * (1000 if unit == "km" else 1)
                values, step = _regular_axis(values, f"target {axis}")
                if "bounds" in coord.attrs:
                    bounds = ds[coord.attrs["bounds"]].transpose(coord.dims[0], ...).values
                    expected = np.sort(np.column_stack((values - step / 2, values + step / 2)), axis=1)
                    bounds = np.sort(bounds * (1000 if unit == "km" else 1), axis=1)
                    if bounds.shape != expected.shape or not np.allclose(bounds, expected):
                        raise ValueError(f"Target {axis} bounds do not match regular centre spacing")
                axes.append(values)
        return cls(crs, *axes)

    def select_points(self, path, latitude="latitude", longitude="longitude"):
        """Return a masked grid using paired coordinates aligned within one metre."""
        lats, lons = _read_points(path, latitude, longitude)
        x, y = self._transformer.transform(lons, lats)
        return self._select_coordinates(lats, lons, x, y, path)

    def _select_coordinates(self, lats, lons, x, y, path):
        col = (np.asarray(x) - self.x[0]) / self.dx
        row = (np.asarray(y) - self.y[0]) / self.dy
        if not np.isfinite(col).all() or not np.isfinite(row).all():
            raise ValueError("Point coordinates cannot be transformed to the reference CRS")
        columns = np.rint(col).astype(np.int64)
        rows = np.rint(row).astype(np.int64)
        tolerance = min(1.0, abs(self.dx) * 0.01, abs(self.dy) * 0.01)
        if (np.any(np.abs(col - columns) * abs(self.dx) > tolerance)
                or np.any(np.abs(row - rows) * abs(self.dy) > tolerance)):
            raise ValueError(f"Points do not align with reference grid within {tolerance:g} m")
        if (np.any(rows < 0) or np.any(rows >= len(self.y))
                or np.any(columns < 0) or np.any(columns >= len(self.x))):
            raise ValueError("Points lie outside the reference grid")
        indices = rows * len(self.x) + columns
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Multiple points map to the same reference cell")
        mask = np.zeros(len(self.x) * len(self.y), dtype=bool)
        mask[indices] = True
        LOG.info("Selected %s / %s reference cells from %s", len(indices), len(mask), path)
        return RegionalGrid(self.crs, self.x, self.y,
                            PointSelection(rows, columns, lats, lons, mask))

    @classmethod
    def from_points(cls, path, crs, spacing, latitude="latitude", longitude="longitude"):
        """Fit a regular lattice phase, then validate every point against it."""
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("Point grid spacing must be finite and positive")
        # Validate CRS and metre units before interpreting projected distances.
        reference = cls(crs, [0, spacing], [0, spacing])
        lats, lons = _read_points(path, latitude, longitude)
        x, y = reference._transformer.transform(lons, lats)
        axes = []
        for values in (np.asarray(x), np.asarray(y)):
            if not np.isfinite(values).all():
                raise ValueError("Point coordinates cannot be transformed to the reference CRS")
            phase = np.angle(np.mean(np.exp(2j * np.pi * values / spacing))) * spacing / (2 * np.pi)
            indices = np.rint((values - phase) / spacing)
            residual = np.abs(values - (phase + indices * spacing))
            tolerance = min(1.0, spacing * 0.01)
            if np.any(residual > tolerance):
                raise ValueError(f"Points do not form the specified lattice within {tolerance:g} m")
            count = int(indices.max() - indices.min()) + 1
            if count < 2:
                raise ValueError("Point-derived grid requires at least two occupied positions per axis")
            axes.append(phase + (indices.min() + np.arange(count)) * spacing)
        LOG.info("Inferred point grid: %s x %s at %s m; x origin %.6f, y origin %.6f",
                 len(axes[1]), len(axes[0]), spacing, axes[0][0], axes[1][0])
        return cls(crs, *axes)._select_coordinates(lats, lons, x, y, path)

    def locate(self, lon, lat):
        x, y = self._transformer.transform(lon, lat)
        col = (np.asarray(x) - self.x[0]) / self.dx + 0.5
        row = (np.asarray(y) - self.y[0]) / self.dy + 0.5
        # Stabilize exact edges after round-trip projection arithmetic.
        for indices in (row, col):
            finite = np.isfinite(indices)
            values = indices[finite]
            nearest = np.rint(values)
            tolerance = 16 * np.finfo(float).eps * np.maximum(1, np.abs(values))
            indices[finite] = np.where(np.abs(values - nearest) <= tolerance, nearest, values)
        inside = (np.isfinite(row) & np.isfinite(col)
                  & (row >= 0) & (row < len(self.y))
                  & (col >= 0) & (col < len(self.x)))
        bucket = (np.floor(row[inside]).astype(np.int64) * len(self.x)
                  + np.floor(col[inside]).astype(np.int64))
        if self.points is not None:
            selected = self.points.mask[bucket]
            inside[np.flatnonzero(inside)[~selected]] = False
            bucket = bucket[selected]
        return inside, bucket

    def source_windows(self, lats, lons):
        """Conservative windows for polar stereographic and same-hemisphere LCC.

        Latitude is monotonic in distance from the projection's pole/apex.
        Rectangle radial extrema are analytic, not sampled along curved edges.
        Polar longitude extrema occur at corners unless the rectangle contains
        the pole. LCC keeps all longitudes to avoid conic branch ambiguity.
        """
        operation = self.crs.coordinate_operation
        method = operation.method_name if operation else ""
        parameters = {p.name: p.value for p in operation.params} if operation else {}
        polar = method in ("Polar Stereographic (variant A)", "Polar Stereographic (variant B)",
                           "Polar Stereographic (variant C)")
        conic = method in ("Lambert Conic Conformal (1SP)", "Lambert Conic Conformal (2SP)")
        parallels = [value for name, value in parameters.items()
                     if "standard parallel" in name.lower()]
        if polar:
            latitude = parameters.get("Latitude of natural origin",
                                      parameters.get("Latitude of standard parallel", 0))
        elif conic and parallels and (all(p > 0 for p in parallels)
                                      or all(p < 0 for p in parallels)):
            latitude = parallels[0]
        else:
            LOG.warning("No verified source bounds for %s; scanning full source", method or self.crs.name)
            return [(slice(0, len(lats)), slice(0, len(lons)))]

        pole = 90.0 if latitude > 0 else -90.0
        px, py = self._transformer.transform(0, pole)
        xmin, xmax = sorted((self.x[0] - self.dx / 2, self.x[-1] + self.dx / 2))
        ymin, ymax = sorted((self.y[0] - self.dy / 2, self.y[-1] + self.dy / 2))
        near_x, near_y = np.clip(px, xmin, xmax), np.clip(py, ymin, ymax)
        rmin = np.hypot(near_x - px, near_y - py)
        rmax = max(np.hypot(x - px, y - py)
                   for x in (xmin, xmax) for y in (ymin, ymax))
        inverse = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        _, lat_bounds = inverse.transform([px + rmin, px + rmax], [py, py])
        if not np.isfinite([px, py, *lat_bounds]).all():
            LOG.warning("Non-finite radial source bounds; scanning full source")
            return [(slice(0, len(lats)), slice(0, len(lons)))]
        # One source spacing guards numerical round-off at the analytic bounds.
        lat_pad = abs(lats[1] - lats[0]) + 1e-7
        lat_mask = ((lats >= min(lat_bounds) - lat_pad)
                    & (lats <= max(lat_bounds) + lat_pad))
        lon_mask = np.ones(len(lons), dtype=bool)
        if polar and rmin > 0:
            centre_lon, _ = inverse.transform(near_x, near_y)
            corner_lon, _ = inverse.transform(
                [xmin, xmin, xmax, xmax], [ymin, ymax, ymin, ymax]
            )
            offsets = (np.asarray(corner_lon) - centre_lon + 180) % 360 - 180
            lon_pad = abs(lons[1] - lons[0]) + 1e-7
            source_offsets = (lons - centre_lon + 180) % 360 - 180
            lon_mask = ((source_offsets >= offsets.min() - lon_pad)
                        & (source_offsets <= offsets.max() + lon_pad))
        rows = _mask_slices(lat_mask)
        columns = _mask_slices(lon_mask)
        return [(rs, cs) for rs in rows for cs in columns]

    def matches(self, lon, lat):
        """Yield integer source/target index pairs for native-cell assignment."""
        inside, targets = self.locate(lon, lat)
        if len(targets):
            yield np.flatnonzero(inside), targets



def _mask_slices(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [slice(int(start), int(stop))
            for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
