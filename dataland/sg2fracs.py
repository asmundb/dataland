import dask.array as da
from dask.distributed import Client, LocalCluster
import numpy as np
import pyproj
import xarray as xr
import pyresample
import pyresample.bucket
import time
import warnings
from tqdm import tqdm
import cartopy.crs as ccrs
from scipy.spatial import cKDTree



if __package__:
    from .aggregation_defaults import OUTPUT_DOMAINS, TRANSTABLE
else:
    from aggregation_defaults import OUTPUT_DOMAINS, TRANSTABLE


def get_area_def(domain_name):
    """
    Returns a pyresample AreaDefinition for the given domain name.

    The area_extent is computed from the center (lon, lat), resolution, and shape.
    'center' can be passed explicitly as (lon, lat) to override the value in OUTPUT_DOMAINS.
    """
    if domain_name not in OUTPUT_DOMAINS:
        raise ValueError(f"Domain '{domain_name}' not found in OUTPUT_DOMAINS.")
    domain = OUTPUT_DOMAINS[domain_name]
    center_lon, center_lat = domain["center"]
    resolution = domain["resolution"]
    height, width = domain["shape"]

    crs = pyproj.CRS.from_dict(domain["projection"])
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    cx, cy = transformer.transform(center_lon, center_lat)

    half_x = width / 2 * resolution
    half_y = height / 2 * resolution
    area_extent = (cx - half_x, cy - half_y, cx + half_x, cy + half_y)

    area_def = pyresample.geometry.AreaDefinition(
        domain["area_id"],
        domain["description"],
        domain["area_id"],
        domain["projection"],
        width,
        height,
        area_extent,
    )
    pyresample.plot.save_quicklook("output_area.png", area_def, np.zeros((height, width)),coast_res="10m")
    return area_def


#def get_bool_map(field, category):
#    """
#    Returns a boolean map of the given category.
#    """
#    if category not in TRANSTABLE:
#        raise ValueError(f"Category {category} not found in TRANSTABLE.")
#    return np.isin(field, TRANSTABLE[category])


#def getecoclimap_sg_global_area_def():
#    """
#    Returns the AreaDefinition for ECOCLIMAP-SG Global 300m Grid.
#    """
#    
#    area_id = 'ecoclimap_sg_global'
#    description = 'ECOCLIMAP-SG Global 300m Grid'
#    projection = {'proj': 'longlat', 'datum': 'WGS84'}
#    
#    # Global bounding box: [lon_min, lat_min, lon_max, lat_max]
#    area_extent = [-180.0, -90.0, 180.0, 90.0]
#    
#    # Resolution in degrees (1/360 degree ≈ 300 meters)
#    resolution = 1.0 / 360.0  # 0.002777777777777778
#    
#    # Instantiate AreaDefinition
#    ecoclimap_area = pyresample.create_area_def(
#        area_id=area_id,
#        projection=projection,
#        area_extent=area_extent,
#        resolution=resolution,
#        units='degrees',
#        description=description
#    )
#    return ecoclimap_area


def _build_latlon_bucket_lut(out_lat_flat, out_lon_flat, lut_resolution=0.25):
    """Cache nearby spherical-tree candidates for each lat/lon lookup cell."""
    lat_min, lat_max = out_lat_flat.min(), out_lat_flat.max()
    lon_min, lon_max = out_lon_flat.min(), out_lon_flat.max()

    n_lat = max(1, int(np.ceil((lat_max - lat_min) / lut_resolution)))
    n_lon = max(1, int(np.ceil((lon_max - lon_min) / lut_resolution)))

    lat_edges = np.linspace(lat_min, lat_max, n_lat + 1)
    lon_edges = np.linspace(lon_min, lon_max, n_lon + 1)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    print(f"Building {n_lat}x{n_lon} bucket LUT ({n_lat * n_lon:,} cells)…", flush=True)
    tic = time.time()
    target_vectors = _latlon_unit_vectors(out_lon_flat, out_lat_flat)
    tree = cKDTree(target_vectors)
    k = min(8, len(out_lat_flat))
    lut = np.empty((n_lat, n_lon, k), dtype=np.int32)
    limits = np.empty((n_lat, n_lon), dtype=np.float64)
    rows_per_batch = max(1, 200_000 // n_lon)
    for start in range(0, n_lat, rows_per_batch):
        stop = min(start + rows_per_batch, n_lat)
        lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers[start:stop])
        distances, indices = tree.query(
            _latlon_unit_vectors(lon_grid.ravel(), lat_grid.ravel()),
            k=k, workers=-1,
        )
        if k == 1:
            distances = distances[:, np.newaxis]
            indices = indices[:, np.newaxis]
        lut[start:stop] = indices.reshape(stop - start, n_lon, k)
        limits[start:stop] = distances[:, -1].reshape(stop - start, n_lon)
    print(f"LUT built in {time.time() - tic:.1f}s", flush=True)
    return lut, limits, lat_edges, lon_edges, tree, target_vectors


def _cached_latlon_candidates(source_vectors, lat_bins, lon_bins, lookup):
    lut, limits, lat_edges, lon_edges, tree, target_vectors = lookup
    k = lut.shape[2]
    best_distance = np.full(len(source_vectors), np.inf)
    bucket = np.empty(len(source_vectors), dtype=np.intp)
    for candidate in range(k):
        indices = lut[lat_bins, lon_bins, candidate]
        difference = source_vectors - target_vectors[indices]
        distance = np.sqrt(np.einsum("ij,ij->i", difference, difference))
        better = distance < best_distance
        bucket[better] = indices[better]
        best_distance[better] = distance[better]

    uncertain = np.zeros(len(source_vectors), dtype=bool)
    if k < tree.n:
        center_lats = (lat_edges[lat_bins] + lat_edges[lat_bins + 1]) * 0.5
        center_lons = (lon_edges[lon_bins] + lon_edges[lon_bins + 1]) * 0.5
        centers = _latlon_unit_vectors(center_lons, center_lats)
        difference = source_vectors - centers
        center_distance = np.sqrt(np.einsum("ij,ij->i", difference, difference))
        # By the triangle inequality, no uncached target can be closer when
        # its center distance exceeds the best source distance plus this offset.
        uncertain = best_distance + center_distance >= limits[lat_bins, lon_bins] - 1e-12
    return bucket, uncertain


def _query_latlon_bucket_lut(lats, lons, lookup):
    """Find exact spherical nearest targets, using cached candidates when safe."""
    lut, _, lat_edges, lon_edges, tree, _ = lookup
    n_lat, n_lon, _ = lut.shape
    lat_bin = np.clip(np.searchsorted(lat_edges[1:-1], lats), 0, n_lat - 1)
    lon_bin = np.clip(np.searchsorted(lon_edges[1:-1], lons), 0, n_lon - 1)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    source_vectors = _latlon_unit_vectors(lon_grid.ravel(), lat_grid.ravel())
    lat_bins = np.repeat(lat_bin, len(lons))
    lon_bins = np.tile(lon_bin, len(lats))
    sample = np.linspace(0, len(source_vectors) - 1, min(1024, len(source_vectors)), dtype=np.intp)
    _, sample_uncertain = _cached_latlon_candidates(
        source_vectors[sample], lat_bins[sample], lon_bins[sample], lookup,
    )
    if sample_uncertain.mean() > 0.1:
        return tree.query(source_vectors, workers=-1)[1]
    bucket, uncertain = _cached_latlon_candidates(source_vectors, lat_bins, lon_bins, lookup)
    if uncertain.any():
        _, bucket[uncertain] = tree.query(source_vectors[uncertain], workers=-1)
    return bucket


def _match_lon_convention(lons, reference_lons):
    """Shift *lons* to the same [-180,180] or [0,360] convention as *reference_lons*."""
    if reference_lons.max() > 180:
        # reference uses [0, 360]
        return lons % 360
    else:
        # reference uses [-180, 180]
        return ((lons + 180) % 360) - 180


def _accumulate_fraction_stats(bucket, values_flat, counts, cat_counts, cat_lookup, max_cover_val, nan_to_num=None):
    """Accumulate categorical counts for COVER-like variables."""
    if np.issubdtype(values_flat.dtype, np.floating):
        fill = 0.0 if nan_to_num is None else float(nan_to_num)
        values_flat = np.nan_to_num(values_flat, nan=fill, posinf=fill, neginf=fill)
    cover_flat = np.clip(values_flat.astype(np.int32), 0, max_cover_val)

    counts += np.bincount(bucket, minlength=counts.size)
    for cat, cat_lut in cat_lookup.items():
        cat_counts[cat] += np.bincount(bucket[cat_lut[cover_flat]], minlength=counts.size)


def _accumulate_mean_stats(bucket, values_flat, sums, valid_counts, fill_value=None, nan_to_num=None,
                           sum_squares=None):
    """Accumulate moments for continuous variables (sum/count and optional sum of squares)."""
    raw_values = values_flat.astype(np.float64, copy=False)
    values = raw_values
    if nan_to_num is not None:
        fill = float(nan_to_num)
        values = np.nan_to_num(values, nan=fill, posinf=fill, neginf=fill)
        valid = np.ones(values.shape, dtype=bool)
    else:
        valid = np.isfinite(values)
    if fill_value is not None and np.isfinite(fill_value):
        valid &= raw_values != fill_value

    if valid.any():
        bucket_valid = bucket[valid]
        values_valid = values[valid]
        sums += np.bincount(bucket_valid, weights=values_valid, minlength=sums.size)
        valid_counts += np.bincount(bucket_valid, minlength=valid_counts.size)
        if sum_squares is not None:
            sum_squares += np.bincount(
                bucket_valid,
                weights=values_valid * values_valid,
                minlength=sum_squares.size,
            )


def _radius_projected_buckets(src_x, src_y, x_min, y_max, dx, dy, height, width, max_radius_m):
    """Return every output cell within max_radius_m of each source point."""
    row = np.floor((y_max - src_y) / dy).astype(np.int32)
    col = np.floor((src_x - x_min) / dx).astype(np.int32)
    row_offsets = range(-int(np.ceil(max_radius_m / dy)), int(np.ceil(max_radius_m / dy)) + 1)
    col_offsets = range(-int(np.ceil(max_radius_m / dx)), int(np.ceil(max_radius_m / dx)) + 1)
    buckets = []
    source_indices = []
    for row_offset in row_offsets:
        target_row = row + row_offset
        for col_offset in col_offsets:
            target_col = col + col_offset
            valid = (
                (target_col >= 0) & (target_col < width) &
                (target_row >= 0) & (target_row < height)
            )
            target_x = x_min + (target_col[valid] + 0.5) * dx
            target_y = y_max - (target_row[valid] + 0.5) * dy
            within_radius = (
                (src_x[valid] - target_x) ** 2 + (src_y[valid] - target_y) ** 2
                <= max_radius_m ** 2
            )
            source_index = np.flatnonzero(valid)[within_radius]
            buckets.append((target_row[valid][within_radius] * width + target_col[valid][within_radius]))
            source_indices.append(source_index)
    return np.concatenate(buckets), np.concatenate(source_indices)


def _nearest_projected_buckets(src_x, src_y, x_min, y_max, dx, dy, height, width, max_radius_m=None):
    """Assign each source point to its single nearest output cell.

    When *max_radius_m* is given, source points farther than that from their
    nearest cell centre are dropped, so every source point contributes to at
    most one output cell (non-overlapping footprints).
    """
    row = np.floor((y_max - src_y) / dy).astype(np.int32)
    col = np.floor((src_x - x_min) / dx).astype(np.int32)
    valid = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    if max_radius_m is not None:
        target_x = x_min + (col + 0.5) * dx
        target_y = y_max - (row + 0.5) * dy
        valid &= (src_x - target_x) ** 2 + (src_y - target_y) ** 2 <= max_radius_m ** 2
    source_index = np.flatnonzero(valid)
    return row[valid] * width + col[valid], source_index


def _latlon_unit_vectors(lons, lats):
    lon_radians = np.deg2rad(lons)
    lat_radians = np.deg2rad(lats)
    return np.column_stack((
        np.cos(lat_radians) * np.cos(lon_radians),
        np.cos(lat_radians) * np.sin(lon_radians),
        np.sin(lat_radians),
    ))


def _radius_latlon_buckets(source_lons, source_lats, target_tree, target_lons, target_lats, geod, max_radius_m):
    """Return every target point within max_radius_m of each source point."""
    earth_radius_m = 6_371_000.0
    search_radius = 2.0 * np.sin(max_radius_m / (2.0 * earth_radius_m)) * 1.01
    candidates = target_tree.query_ball_point(
        _latlon_unit_vectors(source_lons, source_lats), search_radius
    )
    source_indices = np.repeat(
        np.arange(len(candidates), dtype=np.intp),
        [len(candidate) for candidate in candidates],
    )
    buckets = np.concatenate([np.asarray(candidate, dtype=np.intp) for candidate in candidates])
    _, _, distance = geod.inv(
        source_lons[source_indices], source_lats[source_indices],
        target_lons[buckets], target_lats[buckets],
    )
    within_radius = distance <= max_radius_m
    return buckets[within_radius], source_indices[within_radius]


def _clip_lut_buckets_to_radius(bucket, source_lons, source_lats, target_lons, target_lats,
                                geod, max_radius_m):
    """Drop LUT-assigned source points that lie farther than *max_radius_m* from their target.

    Each source point keeps its single nearest-neighbour bucket, so footprints
    never overlap; only the memory-light per-source distance check is added.
    """
    _, _, distance = geod.inv(
        source_lons, source_lats, target_lons[bucket], target_lats[bucket],
    )
    within_radius = distance <= max_radius_m
    return bucket[within_radius], np.flatnonzero(within_radius)


def _find_coord_var(ds, dim, axis):
    """Return the name of the 1-D variable holding coordinates for *dim*.

    Handles files where the coordinate variable name differs from the dimension
    name (e.g. ``lats(lat)``), and where coordinates are plain data variables
    rather than registered xarray coordinates.
    """
    keywords = ("lat",) if axis == "lat" else ("lon",)
    candidates = [
        name for name in list(ds.coords) + list(ds.variables)
        if ds[name].ndim == 1 and ds[name].dims == (dim,)
        and any(k in str(name).lower() for k in keywords)
    ]
    if not candidates:
        # fall back to a variable named exactly like the dimension
        if dim in ds.variables and ds[dim].ndim == 1:
            return dim
        raise ValueError(
            f"Could not find a 1-D {axis} coordinate variable for dimension '{dim}'"
        )
    # prefer an exact dimension-name match, then the shortest name
    for name in candidates:
        if str(name) == str(dim):
            return name
    return sorted(candidates, key=lambda n: len(str(n)))[0]


def resolve_latlon(ds, var=None):
    """Resolve the lat/lon dimensions and coordinate variables of *ds*.

    Returns ``(lat_dim, lon_dim, lat_name, lon_name)``.  The dimension names are
    what ``isel`` needs, while the variable names are what holds the actual
    coordinate values -- these differ in files such as ``lats(lat)``.

    When *var* is given its dimensions are used to identify which axis is which,
    which is more reliable than name matching alone.
    """
    if var is not None and var in ds:
        dims = [str(d) for d in ds[var].dims]
        lat_dims = [d for d in dims if "lat" in d.lower() or d.lower() in ("y", "nj")]
        lon_dims = [d for d in dims if "lon" in d.lower() or d.lower() in ("x", "ni")]
        if len(lat_dims) == 1 and len(lon_dims) == 1:
            lat_dim, lon_dim = lat_dims[0], lon_dims[0]
            return (lat_dim, lon_dim,
                    _find_coord_var(ds, lat_dim, "lat"),
                    _find_coord_var(ds, lon_dim, "lon"))

    names = [str(n) for n in list(ds.coords) + list(ds.variables)]
    lat_names = [n for n in names if "lat" in n.lower() and ds[n].ndim == 1]
    lon_names = [n for n in names if "lon" in n.lower() and ds[n].ndim == 1]
    if not lat_names or not lon_names:
        raise ValueError("Could not identify 1-D latitude/longitude variables in dataset")
    lat_name = sorted(lat_names, key=len)[0]
    lon_name = sorted(lon_names, key=len)[0]
    return str(ds[lat_name].dims[0]), str(ds[lon_name].dims[0]), lat_name, lon_name


def _grid_step(coords):
    """Robust *signed* grid spacing in degrees, immune to a wrap seam in the array.

    ``np.mean(np.diff(...))`` telescopes to ``(last - first) / (n - 1)``, so a
    single 360-degree wrap (introduced when longitudes are rotated to match the
    output convention) collapses it to nearly zero.  The median ignores the one
    discontinuous step.

    The sign is preserved (not just the magnitude) so that a coordinate array
    stored in descending order yields a negative step. This makes the caller's
    gradient-to-metres conversion correct regardless of storage order, without
    any separate "is it ascending?" special-casing.
    """
    if coords.size < 2:
        return None
    steps = np.diff(np.asarray(coords, dtype=np.float64))
    # unwrap a longitude seam so the discontinuous step is not counted
    steps = (steps + 180.0) % 360.0 - 180.0
    step = float(np.median(steps))
    return step if step != 0 else None


def _slope_components(values, lats, lons):
    """Return (dz/dx eastward, dz/dy northward) in m/m for a regular lat/lon block.

    ``np.gradient`` differentiates in index space, so the result is rescaled to
    metres.  The east-west spacing shrinks with latitude, hence the per-row
    ``cos(lat)`` factor.
    """
    earth_radius_m = 6_371_000.0
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    lon_step_deg = _grid_step(lons)
    lat_step_deg = _grid_step(lats)
    lon_step = None if lon_step_deg is None else np.deg2rad(lon_step_deg)
    lat_step = None if lat_step_deg is None else np.deg2rad(lat_step_deg)

    d_row, d_col = np.gradient(values, edge_order=1)

    if lon_step is None:
        dzdx = np.full(values.shape, np.nan, dtype=np.float64)
    else:
        # lon_step is signed (from _grid_step), so dx_m is negative when the
        # column index runs westward; dividing by it yields the correct
        # eastward gradient regardless of storage order, with no special-casing.
        cos_lat = np.cos(np.deg2rad(lats))[:, np.newaxis]
        dx_m = earth_radius_m * lon_step * cos_lat
        with np.errstate(divide="ignore", invalid="ignore"):
            dzdx = d_col / dx_m
        # near the poles dx_m collapses; the zonal slope is undefined there
        dzdx = np.where(np.abs(dx_m) < 1.0, np.nan, dzdx)
        # the wrap seam has no valid neighbour on one side
        seam = np.flatnonzero(np.abs(np.diff(lons)) > 180.0)
        for s in seam:
            dzdx[:, s:s + 2] = np.nan

    if lat_step is None:
        dzdy = np.full(values.shape, np.nan, dtype=np.float64)
    else:
        # lat_step is signed, so dzdy comes out correctly oriented (northward
        # positive) whether rows run north-up or south-up.
        dzdy = d_row / (earth_radius_m * lat_step)

    return dzdx, dzdy


def _continuous_field_names(mode, source_var):
    """Names of the per-point fields accumulated for a continuous *mode*."""
    if mode == "slope":
        return [f"{source_var.lower()}_slope_x", f"{source_var.lower()}_slope_y"]
    return [source_var]


def _read_chunk_fields(ds, source_var, mode, lat_var, lon_var, lat_idx, lon_sel,
                       src_lats_all, src_lons_crop, chunk_start, chunk_end,
                       fill_value=None, nan_to_num=None):
    """Read one source chunk and derive the per-point field(s) required by *mode*.

    For 'slope' a one-row halo is read on each side so the finite-difference
    stencil at a chunk seam uses real neighbouring rows instead of a one-sided
    approximation; the halo rows are trimmed before the fields are returned, which
    keeps the result independent of *chunk_size*.
    """
    lats_chunk = src_lats_all[chunk_start:chunk_end]
    if mode != "slope":
        sel = _to_slice(lat_idx[chunk_start:chunk_end])
        values = ds[source_var].isel({lat_var: sel, lon_var: lon_sel}).values
        return {source_var: values}, lats_chunk

    pad_start = max(0, chunk_start - 1)
    pad_end = min(len(lat_idx), chunk_end + 1)
    sel = _to_slice(lat_idx[pad_start:pad_end])
    padded = ds[source_var].isel({lat_var: sel, lon_var: lon_sel}).values.astype(np.float64)
    if fill_value is not None and np.isfinite(fill_value):
        padded = np.where(padded == fill_value, np.nan, padded)
    if nan_to_num is not None:
        fill = float(nan_to_num)
        padded = np.nan_to_num(padded, nan=fill, posinf=fill, neginf=fill)

    dzdx, dzdy = _slope_components(padded, src_lats_all[pad_start:pad_end], src_lons_crop)
    lo = chunk_start - pad_start
    hi = lo + (chunk_end - chunk_start)
    name_x, name_y = _continuous_field_names(mode, source_var)
    return {name_x: dzdx[lo:hi], name_y: dzdy[lo:hi]}, lats_chunk


def _take(values_2d, source_index):
    """Flatten a chunk field and select the entries kept by the bucketing step."""
    flat = values_2d.ravel()
    return flat if source_index is None else flat[source_index]


def _accumulate_projected(ds, source_var, mode, lat_idx, lat_sel, lon_sel, src_lats_all, src_lons_crop,
                          transformer, x_min, y_min, x_max, y_max, height, width,
                          chunk_size, n_out, cat_lookup=None, max_cover_val=None, fill_value=None,
                          nan_to_num=None, max_radius_m=None, overlap=False):
    """Bucket source points onto a projected regular grid via coordinate transform."""
    dx = (x_max - x_min) / width
    dy = (y_max - y_min) / height
    lat_var, lon_var, _, _ = resolve_latlon(ds, source_var)
    if mode == "fractions":
        assert cat_lookup is not None and max_cover_val is not None
        counts = np.zeros(n_out, dtype=np.int64)
        cat_counts = {cat: np.zeros(n_out, dtype=np.int64) for cat in cat_lookup}

        for chunk_start in tqdm(range(0, len(lat_idx), chunk_size), desc="Processing chunks"):
            chunk_end = min(chunk_start + chunk_size, len(lat_idx))
            lat_chunk_sel = _to_slice(lat_idx[chunk_start:chunk_end])
            values_chunk = ds[source_var].isel({lat_var: lat_chunk_sel, lon_var: lon_sel}).values
            lats_chunk = src_lats_all[chunk_start:chunk_end]

            src_lon_grid, src_lat_grid = np.meshgrid(src_lons_crop, lats_chunk)
            src_x, src_y = transformer.transform(src_lon_grid.ravel(), src_lat_grid.ravel())
            if max_radius_m is not None and overlap:
                bucket, source_index = _radius_projected_buckets(
                    src_x, src_y, x_min, y_max, dx, dy, height, width, max_radius_m
                )
            else:
                bucket, source_index = _nearest_projected_buckets(
                    src_x, src_y, x_min, y_max, dx, dy, height, width, max_radius_m
                )
            values_flat = values_chunk.ravel()[source_index]
            _accumulate_fraction_stats(
                bucket, values_flat, counts, cat_counts, cat_lookup, max_cover_val, nan_to_num=nan_to_num
            )

        return counts, cat_counts, None

    field_names = _continuous_field_names(mode, source_var)
    sums = {name: np.zeros(n_out, dtype=np.float64) for name in field_names}
    valid_counts = {name: np.zeros(n_out, dtype=np.int64) for name in field_names}
    sum_squares = ({name: np.zeros(n_out, dtype=np.float64) for name in field_names}
                   if mode == "std" else None)
    # fill_value refers to the raw source variable, not to derived slope fields;
    # nan_to_num is applied to the elevations before differencing, so remaining
    # NaNs mark undefined slopes (poles, wrap seam) and must stay excluded
    stats_fill = None if mode == "slope" else fill_value
    stats_nan_to_num = None if mode == "slope" else nan_to_num
    for chunk_start in tqdm(range(0, len(lat_idx), chunk_size), desc="Processing chunks"):
        chunk_end = min(chunk_start + chunk_size, len(lat_idx))
        chunk_fields, lats_chunk = _read_chunk_fields(
            ds, source_var, mode, lat_var, lon_var, lat_idx, lon_sel,
            src_lats_all, src_lons_crop, chunk_start, chunk_end,
            fill_value=fill_value, nan_to_num=nan_to_num,
        )

        src_lon_grid, src_lat_grid = np.meshgrid(src_lons_crop, lats_chunk)
        src_x, src_y = transformer.transform(src_lon_grid.ravel(), src_lat_grid.ravel())
        if max_radius_m is not None and overlap:
            bucket, source_index = _radius_projected_buckets(
                src_x, src_y, x_min, y_max, dx, dy, height, width, max_radius_m
            )
        else:
            bucket, source_index = _nearest_projected_buckets(
                src_x, src_y, x_min, y_max, dx, dy, height, width, max_radius_m
            )
        for name in field_names:
            _accumulate_mean_stats(
                bucket,
                _take(chunk_fields[name], source_index),
                sums[name],
                valid_counts[name],
                fill_value=stats_fill,
                nan_to_num=stats_nan_to_num,
                sum_squares=None if sum_squares is None else sum_squares[name],
            )

    return sums, valid_counts, sum_squares


def _accumulate_latlon_lut(ds, source_var, mode, lat_idx, lat_sel, lon_sel, src_lats_all, src_lons_crop,
                           lookup,
                           chunk_size, n_out, cat_lookup=None, max_cover_val=None, fill_value=None,
                           nan_to_num=None, max_radius_m=None, out_lat_flat=None, out_lon_flat=None,
                           overlap=False):
    """Bucket source points onto an unstructured output grid via a lat/lon LUT."""
    lat_var, lon_var, _, _ = resolve_latlon(ds, source_var)
    geod = pyproj.Geod(ellps="WGS84") if max_radius_m is not None else None
    use_overlap = max_radius_m is not None and overlap
    target_tree = None
    if use_overlap:
        target_tree = cKDTree(_latlon_unit_vectors(out_lon_flat, out_lat_flat))
    # Bound temporary arrays independently of the user-specified read size.
    chunk_size = min(chunk_size, max(1, 500_000 // max(1, len(src_lons_crop))))
    if mode == "fractions":
        assert cat_lookup is not None and max_cover_val is not None
        counts = np.zeros(n_out, dtype=np.int64)
        cat_counts = {cat: np.zeros(n_out, dtype=np.int64) for cat in cat_lookup}

        for chunk_start in tqdm(range(0, len(lat_idx), chunk_size), desc="Processing chunks"):
            chunk_end = min(chunk_start + chunk_size, len(lat_idx))
            lat_chunk_sel = _to_slice(lat_idx[chunk_start:chunk_end])
            values_chunk = ds[source_var].isel({lat_var: lat_chunk_sel, lon_var: lon_sel}).values
            lats_chunk = src_lats_all[chunk_start:chunk_end]

            if use_overlap:
                src_lon_grid, src_lat_grid = np.meshgrid(src_lons_crop, lats_chunk)
                bucket, source_index = _radius_latlon_buckets(
                    src_lon_grid.ravel(), src_lat_grid.ravel(),
                    target_tree, out_lon_flat, out_lat_flat, geod, max_radius_m,
                )
                values_flat = values_chunk.ravel()[source_index]
            else:
                bucket = _query_latlon_bucket_lut(lats_chunk, src_lons_crop, lookup)
                values_flat = values_chunk.ravel()
                if geod is not None:
                    src_lon_grid, src_lat_grid = np.meshgrid(src_lons_crop, lats_chunk)
                    bucket, source_index = _clip_lut_buckets_to_radius(
                        bucket, src_lon_grid.ravel(), src_lat_grid.ravel(),
                        out_lon_flat, out_lat_flat, geod, max_radius_m,
                    )
                    values_flat = values_flat[source_index]
            _accumulate_fraction_stats(
                bucket, values_flat, counts, cat_counts, cat_lookup, max_cover_val, nan_to_num=nan_to_num
            )

        return counts, cat_counts, None

    field_names = _continuous_field_names(mode, source_var)
    sums = {name: np.zeros(n_out, dtype=np.float64) for name in field_names}
    valid_counts = {name: np.zeros(n_out, dtype=np.int64) for name in field_names}
    sum_squares = ({name: np.zeros(n_out, dtype=np.float64) for name in field_names}
                   if mode == "std" else None)
    # fill_value refers to the raw source variable, not to derived slope fields;
    # nan_to_num is applied to the elevations before differencing, so remaining
    # NaNs mark undefined slopes (poles, wrap seam) and must stay excluded
    stats_fill = None if mode == "slope" else fill_value
    stats_nan_to_num = None if mode == "slope" else nan_to_num
    for chunk_start in tqdm(range(0, len(lat_idx), chunk_size), desc="Processing chunks"):
        chunk_end = min(chunk_start + chunk_size, len(lat_idx))
        chunk_fields, lats_chunk = _read_chunk_fields(
            ds, source_var, mode, lat_var, lon_var, lat_idx, lon_sel,
            src_lats_all, src_lons_crop, chunk_start, chunk_end,
            fill_value=fill_value, nan_to_num=nan_to_num,
        )

        if use_overlap:
            src_lon_grid, src_lat_grid = np.meshgrid(src_lons_crop, lats_chunk)
            bucket, source_index = _radius_latlon_buckets(
                src_lon_grid.ravel(), src_lat_grid.ravel(),
                target_tree, out_lon_flat, out_lat_flat, geod, max_radius_m,
            )
        else:
            bucket = _query_latlon_bucket_lut(lats_chunk, src_lons_crop, lookup)
            source_index = None
            if geod is not None:
                src_lon_grid, src_lat_grid = np.meshgrid(src_lons_crop, lats_chunk)
                bucket, source_index = _clip_lut_buckets_to_radius(
                    bucket, src_lon_grid.ravel(), src_lat_grid.ravel(),
                    out_lon_flat, out_lat_flat, geod, max_radius_m,
                )
        for name in field_names:
            _accumulate_mean_stats(
                bucket,
                _take(chunk_fields[name], source_index),
                sums[name],
                valid_counts[name],
                fill_value=stats_fill,
                nan_to_num=stats_nan_to_num,
                sum_squares=None if sum_squares is None else sum_squares[name],
            )

    return sums, valid_counts, sum_squares

def plotshit():
    import matplotlib.pyplot as plt
    vmax = 1000
    fig, ax = plt.subplots(ncols=3, sharex="all", sharey="all")
    ax[0].set_title("Source values")
    ax[1].set_title("Buckets")
    ax[2].set_title("Averaged values")
    ax[0].pcolormesh(src_lons_crop, lats_chunk, values_chunk, vmin=0, vmax=vmax)
    ax[0].scatter(out_lon_flat, out_lat_flat, c='red', s=20,vmin=0, vmax=vmax)
    if max_radius_m is not None:
        ax[1].scatter(src_lon_grid.ravel()[source_index], src_lat_grid.ravel()[source_index], c=bucket, vmin=0, vmax=None)
    else:
        ax[1].pcolormesh(src_lons_crop, lats_chunk, bucket.reshape(lats_chunk.shape[0], src_lons_crop.shape[0]),vmin=0, vmax=None)
    ax[1].scatter(out_lon_flat, out_lat_flat, c='red', s=20,vmin=0, vmax=vmax)
    ax[2].scatter(out_lon_flat, out_lat_flat, c=(sums/valid_counts).reshape(2,2),vmin=0, vmax=vmax)
    #ax[2].pcolormesh(out_lon_flat, out_lat_flat, sums.reshape(2,2)
    plt.show()


def _to_slice(idx):
    """Convert a contiguous index array to a slice so xarray uses fast slice reads."""
    if len(idx) > 1 and idx[-1] - idx[0] == len(idx) - 1:
        return slice(int(idx[0]), int(idx[-1]) + 1)
    return idx


def compute_statistics(input_file, out_lat, out_lon, area_def=None, chunk_size=1000,
                       source_var="COVER", mode="fractions", output_name=None, nan_to_num=None,
                       max_radius_m=None, overlap=False, include_mean=False):
    """Compute gridded statistics for each output grid point.

    Args:
        input_file: Path to ECOCLIMAP-SG NetCDF file.
        out_lat:    1-D or 2-D array of output latitudes.
        out_lon:    1-D or 2-D array of output longitudes (same shape as out_lat).
        area_def:   Optional pyresample AreaDefinition.  When provided the source
                    points are projected directly onto the output grid using a
                    coordinate transform (fast path for regular projected grids).
                    Without it, an unstructured lat/lon LUT is used instead.
        chunk_size: Number of source latitude rows processed at a time.
        max_radius_m: Optional maximum source-to-target distance in metres.
        overlap:    When True and max_radius_m is set, every output point collects
                    all source points within max_radius_m, so footprints of
                    neighbouring output points overlap.  When False (default) each
                    source point is assigned to its single nearest output point and
                    dropped if it lies farther away than max_radius_m, giving
                    non-overlapping footprints at a fraction of the memory cost.
        mode:       'fractions', 'mean', 'std', or 'slope'.  In 'slope' mode the
                    eastward and northward slope components (m/m) are derived from
                    the source field with a one-row halo across chunk seams and
                    averaged per output point, yielding two output fields.
        include_mean: In 'std' mode, also return the mean field alongside the
                    standard deviation (both come from the same accumulators, so
                    this costs no extra passes over the source data).

    Returns:
        dict mapping output variable name(s) to array(s) with shape matching out_lat.
    """
    out_lat = np.squeeze(np.asarray(out_lat))
    out_lon = np.squeeze(np.asarray(out_lon))
    out_shape = out_lat.shape
    out_lat_flat = out_lat.ravel()
    out_lon_flat = out_lon.ravel()
    n_out = out_lat_flat.size

    radius_degrees = 0.0 if max_radius_m is None else np.degrees(max_radius_m / 6_371_000.0)
    lat_margin = max(1.0, radius_degrees)
    lon_margin = max(1.0, radius_degrees / max(np.cos(np.deg2rad(np.abs(out_lat_flat).max())), 1e-6))
    ds = xr.open_dataset(input_file)
    if source_var not in ds:
        raise ValueError(f"Variable '{source_var}' not found in {input_file}")

    # figure out lat/lon dimensions and coordinate variables
    lat_dim, lon_dim, s_lat, s_lon = resolve_latlon(ds, source_var)
    print(f"Source coords: {s_lat}({lat_dim}), {s_lon}({lon_dim})")

    fill_value = ds[source_var].attrs.get(
        "_FillValue",
        ds[source_var].attrs.get("missing_value", ds[source_var].encoding.get("_FillValue", None)),
    )
    
    src_lons_all = _match_lon_convention(ds[s_lon].values, out_lon_flat)
    lat_idx = np.where(
        (ds[s_lat].values >= out_lat_flat.min() - lat_margin) &
        (ds[s_lat].values <= out_lat_flat.max() + lat_margin)
    )[0]
    lon_idx = np.where(
        (src_lons_all >= out_lon_flat.min() - lon_margin) &
        (src_lons_all <= out_lon_flat.max() + lon_margin)
    )[0]
    if not lat_idx.size or not lon_idx.size:
        raise ValueError("The source coordinate subset is empty; check source and output domain overlap")
    src_lats_all = ds[s_lat].values[lat_idx]
    src_lons_crop = src_lons_all[lon_idx]
    lat_sel = _to_slice(lat_idx)
    lon_sel = _to_slice(lon_idx)
    print(f"Source subset: {len(lat_idx)} lats x {len(lon_idx)} lons "
          f"= {len(lat_idx) * len(lon_idx):,} points")

    if mode == "fractions":
        max_cover_val = max(v for vals in TRANSTABLE.values() for v in vals)
        cat_lookup = {}
        for cat, vals in TRANSTABLE.items():
            lut = np.zeros(max_cover_val + 2, dtype=bool)
            lut[vals] = True
            cat_lookup[cat] = lut
    elif mode in ["mean", "std", "slope"]:
        cat_lookup = None
        max_cover_val = None
    else:
        raise ValueError(
            f"Unsupported mode '{mode}'. Use 'fractions', 'mean', 'std', or 'slope'."
        )

    tic = time.time()
    counts = None
    cat_counts = None
    sums = None
    valid_counts = None
    sum_squares = None

    if area_def is not None:
        crs = pyproj.CRS.from_dict(area_def.crs.to_dict())
        transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x_min, y_min, x_max, y_max = area_def.area_extent
        height, width = area_def.shape
        if mode == "fractions":
            counts, cat_counts, _ = _accumulate_projected(
                ds, source_var, mode, lat_idx, lat_sel, lon_sel, src_lats_all, src_lons_crop,
                transformer, x_min, y_min, x_max, y_max, height, width,
                chunk_size, n_out,
                cat_lookup=cat_lookup, max_cover_val=max_cover_val, fill_value=fill_value,
                nan_to_num=nan_to_num, max_radius_m=max_radius_m, overlap=overlap,
            )
        else:
            sums, valid_counts, sum_squares = _accumulate_projected(
                ds, source_var, mode, lat_idx, lat_sel, lon_sel, src_lats_all, src_lons_crop,
                transformer, x_min, y_min, x_max, y_max, height, width,
                chunk_size, n_out,
                cat_lookup=cat_lookup, max_cover_val=max_cover_val, fill_value=fill_value,
                nan_to_num=nan_to_num, max_radius_m=max_radius_m, overlap=overlap,
            )
    else:
        # LUT resolution controls performance only; ambiguous lookups are
        # resolved against the spherical target tree.
        lat_range = out_lat_flat.max() - out_lat_flat.min()
        lon_range = out_lon_flat.max() - out_lon_flat.min()
        approx_spacing = np.sqrt(lat_range * lon_range / n_out)
        lut_resolution = max(0.5, np.sqrt(lat_range * lon_range / 50_000))
        print(f"Output grid spacing ~{approx_spacing:.4f}°, using LUT resolution {lut_resolution:.4f}°")
        lookup = None
        if not (overlap and max_radius_m is not None):
            lookup = _build_latlon_bucket_lut(
                out_lat_flat, out_lon_flat, lut_resolution=lut_resolution
            )
        if mode == "fractions":
            counts, cat_counts, _ = _accumulate_latlon_lut(
                ds, source_var, mode, lat_idx, lat_sel, lon_sel, src_lats_all, src_lons_crop,
                lookup,
                chunk_size, n_out,
                cat_lookup=cat_lookup, max_cover_val=max_cover_val, fill_value=fill_value,
                nan_to_num=nan_to_num, max_radius_m=max_radius_m, overlap=overlap,
                out_lat_flat=out_lat_flat, out_lon_flat=out_lon_flat,
            )
        else:
            sums, valid_counts, sum_squares = _accumulate_latlon_lut(
                ds, source_var, mode, lat_idx, lat_sel, lon_sel, src_lats_all, src_lons_crop,
                lookup,
                chunk_size, n_out,
                cat_lookup=cat_lookup, max_cover_val=max_cover_val, fill_value=fill_value,
                nan_to_num=nan_to_num, max_radius_m=max_radius_m, overlap=overlap,
                out_lat_flat=out_lat_flat, out_lon_flat=out_lon_flat,
            )

    ds.close()
    print(f"Total processing time: {time.time() - tic:.1f}s")

    results = {}
    if mode == "fractions":
        if counts is None or cat_counts is None:
            raise RuntimeError("Fraction accumulation did not return counts")
        counts = np.asarray(counts, dtype=np.int64)
        zero_cells = (counts == 0).sum()
        if zero_cells:
            warnings.warn(
                f"{zero_cells} output cells received no source points; their fractions are NaN",
                RuntimeWarning,
            )
        for cat in TRANSTABLE:
            fraction = np.full(n_out, np.nan, dtype=np.float64)
            np.divide(cat_counts[cat], counts, out=fraction, where=counts > 0)
            results[cat] = fraction.reshape(out_shape)
    elif mode in ("mean", "slope"):
        assert sums is not None and valid_counts is not None
        for name in sums:
            field_sums = np.asarray(sums[name], dtype=np.float64)
            field_counts = np.asarray(valid_counts[name], dtype=np.int64)
            mean = np.full(n_out, np.nan, dtype=np.float64)
            valid_out = field_counts > 0
            mean[valid_out] = field_sums[valid_out] / field_counts[valid_out]
            if mode == "mean":
                out_key = output_name or f"{source_var.lower()}_mean"
            else:
                out_key = name
            results[out_key] = mean.reshape(out_shape)
    elif mode == "std":
        assert sums is not None and valid_counts is not None and sum_squares is not None
        for name in sums:
            field_sums = np.asarray(sums[name], dtype=np.float64)
            field_counts = np.asarray(valid_counts[name], dtype=np.int64)
            field_squares = np.asarray(sum_squares[name], dtype=np.float64)
            std = np.full(n_out, np.nan, dtype=np.float64)
            var = np.full(n_out, np.nan, dtype=np.float64)
            valid_out = field_counts > 1
            n = field_counts[valid_out].astype(np.float64)
            # Unbiased sample variance: (sum(x^2) - sum(x)^2 / n) / (n - 1)
            var_num = field_squares[valid_out] - (field_sums[valid_out])**2 / n
            var[valid_out] = var_num / (n - 1.0)
            var[valid_out] = np.maximum(var[valid_out], 0.0)
            std[valid_out] = np.sqrt(var[valid_out])
            results[output_name or f"{source_var.lower()}_std"] = std.reshape(out_shape)
            if include_mean:
                mean = np.full(n_out, np.nan, dtype=np.float64)
                has_any = field_counts > 0
                mean[has_any] = field_sums[has_any] / field_counts[has_any]
                results[f"{source_var.lower()}_mean"] = mean.reshape(out_shape)

    else:
        raise ValueError(
            f"Unsupported mode '{mode}'. Use 'fractions', 'mean', 'std', or 'slope'."
        )

    ## _accumulate_projected produces north-up order (row 0 = northernmost).
    ## If out_lat is south-up (row 0 has lower latitude than row -1), flip to match.
    #if area_def is not None and out_lat.ndim == 2:
    #    if out_lat[0].mean() < out_lat[-1].mean():  # south-up
    #        results = {name: v[::-1, :] for name, v in results.items()}

    return results


def compute_fractions(input_file, out_lat, out_lon, area_def=None, chunk_size=2048, max_radius_m=None,
                      overlap=False):
    """Backward-compatible wrapper for COVER fractions."""
    return compute_statistics(
        input_file=input_file,
        out_lat=out_lat,
        out_lon=out_lon,
        area_def=area_def,
        chunk_size=chunk_size,
        source_var="COVER",
        mode="fractions",
        max_radius_m=max_radius_m,
        overlap=overlap,
    )


def save_fractions_to_netcdf(out_lat, out_lon, fractions, output_nc):
    out_lat_arr = np.squeeze(np.asarray(out_lat))
    out_lon_arr = np.squeeze(np.asarray(out_lon))
    is_2d = out_lat_arr.ndim == 2

    coords = (
        {"latitude": (["y", "x"], out_lat_arr), "longitude": (["y", "x"], out_lon_arr)}
        if is_2d
        else {"latitude": ("points", out_lat_arr), "longitude": ("points", out_lon_arr)}
    )
    dims = ["y", "x"] if is_2d else ["points"]

    ds_out = xr.Dataset(
        {cat: xr.DataArray(frac, dims=dims) for cat, frac in fractions.items()},
        coords=coords,
    )
    ds_out["latitude"].attrs = {"units": "degrees_north", "long_name": "latitude"}
    ds_out["longitude"].attrs = {"units": "degrees_east", "long_name": "longitude"}
    for cat in fractions:
        ds_out[cat].attrs = {"long_name": f"{cat} fraction", "units": "1"}

    ds_out.to_netcdf(output_nc)
    print(f"Fractions written to {output_nc}")

    print({k: v.shape for k, v in fractions.items()})


def save_statistics_to_netcdf(out_lat, out_lon, fields, output_nc, units_by_var=None, nan_to_num=None):
    """Save one or more computed fields on the output grid."""
    out_lat_arr = np.squeeze(np.asarray(out_lat))
    out_lon_arr = np.squeeze(np.asarray(out_lon))
    is_2d = out_lat_arr.ndim == 2

    coords = (
        {"latitude": (["y", "x"], out_lat_arr), "longitude": (["y", "x"], out_lon_arr)}
        if is_2d
        else {"latitude": ("points", out_lat_arr), "longitude": ("points", out_lon_arr)}
    )
    dims = ["y", "x"] if is_2d else ["points"]

    ds_out = xr.Dataset(
        {name: xr.DataArray(values, dims=dims) for name, values in fields.items()},
        coords=coords,
    )
    if nan_to_num is not None:
        for name in fields:
            ds_out[name] = ds_out[name].where(np.isfinite(ds_out[name]), other=nan_to_num)
    ds_out["latitude"].attrs = {"units": "degrees_north", "long_name": "latitude"}
    ds_out["longitude"].attrs = {"units": "degrees_east", "long_name": "longitude"}

    units_by_var = units_by_var or {}
    for name in fields:
        units = units_by_var.get(name, "1")
        ds_out[name].attrs = {"long_name": name.replace("_", " "), "units": units}

    ds_out.to_netcdf(output_nc)
    print(f"Fields written to {output_nc}")
    print({k: v.shape for k, v in fields.items()})



if __name__ == "__main__":  
    import argparse
    import anemoi.datasets

    input_file = "/home/asmundb/tmp/ecosg_final_map.nc"
    
    zarr_file = "/lustre/storeB/project/nwp/bris/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6.zarr"

    parser = argparse.ArgumentParser(description="Compute fractions, means, or standard deviations from ECOCLIMAP-SG-like data.")
    parser.add_argument("--source_file", type=str, default=input_file, help="Path to the input ECOCLIMAP-SG NetCDF file.")
    parser.add_argument("--output_domain", type=str, default=None, help="name of the output domain")
    parser.add_argument("--domain_file", type=str, default=None, help="Path to zarr with latlon (optional).")
    parser.add_argument("--out_nc", type=str, default="fractions_output.nc", help="Path to the output NetCDF file.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of source latitude rows processed at a time.")
    parser.add_argument("--mode", type=str, default="fractions", choices=["fractions", "mean", "std", "slope"], help="Aggregation mode. 'slope' writes the mean eastward and northward surface slope components (m/m).")
    parser.add_argument("--source_var", type=str, default="COVER", help="Source variable name to aggregate.")
    parser.add_argument("--output_var", type=str, default=None, help="Output variable name (used for mean/std mode).")
    parser.add_argument("--nan_to_num", type=float, default=None, help="Replace NaN source values before aggregation; for continuous modes also replace non-finite output values.")
    parser.add_argument("--max_radius_m", type=float, default=None, help="Maximum source-to-target distance in metres (optional).")
    parser.add_argument("--overlap", action="store_true", help="Let each output point collect all source points within --max_radius_m (overlapping footprints). Without this flag each source point contributes only to its nearest output point.")
    parser.add_argument("--include_mean", action="store_true", help="In --mode std, also write the mean field to the output file (no extra computation).")
    args = parser.parse_args()

    if args.domain_file:
        if args.domain_file == "test":
            out_lat = np.array([60, 61, 60, 61])
            out_lon = np.array([10, 10, 11, 11])
        elif args.domain_file.endswith(".zarr"):
            domain_ds = anemoi.datasets.open_dataset(args.domain_file)
            out_lat = domain_ds.latitudes
            out_lon = domain_ds.longitudes
        else:
            domain_ds = xr.open_dataset(args.domain_file)
            # domain lat/lon may be 1-D or 2-D, and stored as coords or data vars
            names = [str(v) for v in list(domain_ds.coords) + list(domain_ds.variables)]
            lat_vars = [v for v in names if "lat" in v.lower() and domain_ds[v].ndim in (1, 2)]
            lon_vars = [v for v in names if "lon" in v.lower() and domain_ds[v].ndim in (1, 2)]
            if len(lat_vars) == 0 or len(lon_vars) == 0:
                exit(f"Could not find lat/lon variables in {args.domain_file}.")
            # prefer 2-D (curvilinear) fields, then the shortest name
            lat_name = sorted(lat_vars, key=lambda v: (-domain_ds[v].ndim, len(v)))[0]
            lon_name = sorted(lon_vars, key=lambda v: (-domain_ds[v].ndim, len(v)))[0]
            print(f"Domain coords: {lat_name}, {lon_name}")
            out_lat = domain_ds[lat_name]
            out_lon = domain_ds[lon_name]
        area_def = None
    elif args.output_domain:
        area_def = get_area_def(args.output_domain)
        out_lon, out_lat = area_def.get_lonlats()  # returns (lons, lats)
    else:
        exit("Either --output_domain or --domain_file must be specified.")

    source_file = args.source_file
    fields = compute_statistics(
        source_file,
        out_lat,
        out_lon,
        area_def=area_def,
        chunk_size=args.chunk_size,
        source_var=args.source_var,
        mode=args.mode,
        output_name=args.output_var,
        nan_to_num=args.nan_to_num,
        max_radius_m=args.max_radius_m,
        overlap=args.overlap,
        include_mean=args.include_mean,
    )
    first_key = next(iter(fields))
    first_values = fields[first_key]
    assert first_values is not None
    print(np.asarray(out_lat).shape, np.asarray(out_lon).shape, np.asarray(first_values).shape)

    units = "1" if args.mode in ("fractions", "slope") else ""
    units_by_var = {name: units for name in fields}
    save_statistics_to_netcdf(
        out_lat, out_lon, 
        fields, args.out_nc, 
        units_by_var=units_by_var, 
        nan_to_num=None if args.mode == "fractions" else args.nan_to_num)

if False:
    import argparse 
    # setup test variables
    args = argparse.Namespace()
    args.domain_file = "treeheights_o96_3.nc"
    args.output_domain = "test"
    args.out_nc = "orography_o96_3.nc"
    args.source_file = "GMTED2010_30.EHdr.nc"
    args.source_var = "orography"
    args.output_var = "zs"
    args.chunk_size = 1024
    args.mode = "std"
    args.nan_to_num = 0
    args.max_radius_m = None
    args.overlap = False
    overlap = args.overlap
    max_radius_m = None
    source_file = args.source_file
    input_file = source_file
    chunk_size=args.chunk_size
    source_var=args.source_var
    mode=args.mode
    output_name=args.output_var
    nan_to_num=args.nan_to_num
    out_lat = np.array([60, 61, 60, 61])
    out_lon = np.array([10, 10, 11, 11])
