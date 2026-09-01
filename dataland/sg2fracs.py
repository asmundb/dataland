import dask.array as da
from dask.distributed import Client, LocalCluster
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import pyproj
import xarray as xr
import pyresample
import pyresample.bucket
import time
from tqdm import tqdm
import cartopy.crs as ccrs


TRANSTABLE = {  # CAUTION: fortran indexing
    "sea": [1],
    "lake": [2,3],
    "nature": [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
    "urban": [24,25,26,27,28,29,30,31,32,33],
    "forest": [7,8,9,10,11,12,13,14],
    "glacier": [6]
}

OUTPUT_DOMAINS = {
    "metcoop": {
        "area_id": "metcoop",
        "description": "MetCoOp domain",
        "shape": (1069, 949),
        "center": (17.5, 63.3),  # (lon, lat) of grid center
        "resolution": 2500,      # meters
        "projection": {
            "proj": "lcc",
            "lat_1": 63.3,
            "lat_2": 63.3,
            "lat_0": 63.3,
            "lon_0": 15.,
            "r": 6371229.0,
            "false_easting": 1060122.15845653,
            "false_northing": 1332565.78014384,
            "units": "m",
        },
    },
    "carra2": {
        "area_id": "carra2",
        "description": "CARRA2 domain",
        "shape": (2869, 2869),
        "center": (-45.0, 84.0),  # (lon, lat) of grid center
        "resolution": 2500,       # meters
        "projection": {
            "proj": "stere",
            "lat_0": 90,
            "lon_0": -30.,
            "datum": "WGS84",
            "units": "m",
        },
    },
}


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


def get_bool_map(field, category):
    """
    Returns a boolean map of the given category.
    """
    if category not in TRANSTABLE:
        raise ValueError(f"Category {category} not found in TRANSTABLE.")
    return np.isin(field, TRANSTABLE[category])


def getecoclimap_sg_global_area_def():
    """
    Returns the AreaDefinition for ECOCLIMAP-SG Global 300m Grid.
    """
    
    area_id = 'ecoclimap_sg_global'
    description = 'ECOCLIMAP-SG Global 300m Grid'
    projection = {'proj': 'longlat', 'datum': 'WGS84'}
    
    # Global bounding box: [lon_min, lat_min, lon_max, lat_max]
    area_extent = [-180.0, -90.0, 180.0, 90.0]
    
    # Resolution in degrees (1/360 degree ≈ 300 meters)
    resolution = 1.0 / 360.0  # 0.002777777777777778
    
    # Instantiate AreaDefinition
    ecoclimap_area = pyresample.create_area_def(
        area_id=area_id,
        projection=projection,
        area_extent=area_extent,
        resolution=resolution,
        units='degrees',
        description=description
    )
    return ecoclimap_area


def _build_latlon_bucket_lut(out_lat_flat, out_lon_flat, lut_resolution=0.25):
    """Build a 2D lat/lon → nearest-output-index lookup table.

    The KDTree query runs once on `n_lat x n_lon` LUT cells (typically ~1M)
    rather than on every source point.  Per-source lookup is then two
    searchsorted calls — O(n log k) with k ≈ 1000 instead of O(n log m).
    """
    from scipy.spatial import cKDTree

    lat_min, lat_max = out_lat_flat.min(), out_lat_flat.max()
    lon_min, lon_max = out_lon_flat.min(), out_lon_flat.max()

    n_lat = max(1, int(np.ceil((lat_max - lat_min) / lut_resolution)))
    n_lon = max(1, int(np.ceil((lon_max - lon_min) / lut_resolution)))

    lat_edges = np.linspace(lat_min, lat_max, n_lat + 1)
    lon_edges = np.linspace(lon_min, lon_max, n_lon + 1)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    print(f"Building {n_lat}x{n_lon} bucket LUT ({n_lat * n_lon:,} cells)…", flush=True)
    tic = time.time()
    tree = cKDTree(np.column_stack([out_lon_flat, out_lat_flat]))
    _, lut = tree.query(
        np.column_stack([lon_grid.ravel(), lat_grid.ravel()]), workers=-1
    )
    print(f"LUT built in {time.time() - tic:.1f}s", flush=True)
    return lut.reshape(n_lat, n_lon).astype(np.int32), lat_edges, lon_edges


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


def _accumulate_mean_stats(bucket, values_flat, sums, valid_counts, fill_value=None, nan_to_num=None):
    """Accumulate sum/count for mean of continuous variables."""
    values = values_flat.astype(np.float64, copy=False)
    if nan_to_num is not None:
        fill = float(nan_to_num)
        values = np.nan_to_num(values, nan=fill, posinf=fill, neginf=fill)
        valid = np.ones(values.shape, dtype=bool)
    else:
        valid = np.isfinite(values)
    if fill_value is not None and np.isfinite(fill_value) and nan_to_num is None:
        valid &= values != fill_value

    if valid.any():
        bucket_valid = bucket[valid]
        values_valid = values[valid]
        sums += np.bincount(bucket_valid, weights=values_valid, minlength=sums.size)
        valid_counts += np.bincount(bucket_valid, minlength=valid_counts.size)


def _accumulate_projected(ds, source_var, mode, lat_idx, lat_sel, lon_sel, src_lats_all, src_lons_crop,
                          transformer, x_min, y_min, x_max, y_max, height, width,
                          chunk_size, n_out, cat_lookup=None, max_cover_val=None, fill_value=None,
                          nan_to_num=None):
    """Bucket source points onto a projected regular grid via coordinate transform."""
    dx = (x_max - x_min) / width
    dy = (y_max - y_min) / height
    if mode == "fractions":
        assert cat_lookup is not None and max_cover_val is not None
        counts = np.zeros(n_out, dtype=np.int64)
        cat_counts = {cat: np.zeros(n_out, dtype=np.int64) for cat in cat_lookup}

        for chunk_start in tqdm(range(0, len(lat_idx), chunk_size), desc="Processing chunks"):
            chunk_end = min(chunk_start + chunk_size, len(lat_idx))
            lat_chunk_sel = _to_slice(lat_idx[chunk_start:chunk_end])
            values_chunk = ds[source_var].isel(lat=lat_chunk_sel, lon=lon_sel).values
            lats_chunk = src_lats_all[chunk_start:chunk_end]

            src_lon_grid, src_lat_grid = np.meshgrid(src_lons_crop, lats_chunk)
            src_x, src_y = transformer.transform(src_lon_grid.ravel(), src_lat_grid.ravel())
            col = np.floor((src_x - x_min) / dx).astype(np.int32)
            row = np.floor((y_max - src_y) / dy).astype(np.int32)
            valid = (col >= 0) & (col < width) & (row >= 0) & (row < height)
            bucket = row[valid] * width + col[valid]
            values_flat = values_chunk.ravel()[valid]
            _accumulate_fraction_stats(
                bucket, values_flat, counts, cat_counts, cat_lookup, max_cover_val, nan_to_num=nan_to_num
            )

        return counts, cat_counts

    sums = np.zeros(n_out, dtype=np.float64)
    valid_counts = np.zeros(n_out, dtype=np.int64)
    for chunk_start in tqdm(range(0, len(lat_idx), chunk_size), desc="Processing chunks"):
        chunk_end = min(chunk_start + chunk_size, len(lat_idx))
        lat_chunk_sel = _to_slice(lat_idx[chunk_start:chunk_end])
        values_chunk = ds[source_var].isel(lat=lat_chunk_sel, lon=lon_sel).values
        lats_chunk = src_lats_all[chunk_start:chunk_end]

        src_lon_grid, src_lat_grid = np.meshgrid(src_lons_crop, lats_chunk)
        src_x, src_y = transformer.transform(src_lon_grid.ravel(), src_lat_grid.ravel())
        col = np.floor((src_x - x_min) / dx).astype(np.int32)
        row = np.floor((y_max - src_y) / dy).astype(np.int32)
        valid = (col >= 0) & (col < width) & (row >= 0) & (row < height)
        bucket = row[valid] * width + col[valid]
        values_flat = values_chunk.ravel()[valid]
        _accumulate_mean_stats(
            bucket, values_flat, sums, valid_counts, fill_value=fill_value, nan_to_num=nan_to_num
        )

    return sums, valid_counts


def _accumulate_latlon_lut(ds, source_var, mode, lat_idx, lat_sel, lon_sel, src_lats_all, src_lons_crop,
                           bucket_lut, lat_edges, lon_edges,
                           chunk_size, n_out, cat_lookup=None, max_cover_val=None, fill_value=None,
                           nan_to_num=None):
    """Bucket source points onto an unstructured output grid via a lat/lon LUT."""
    n_lat_lut, n_lon_lut = bucket_lut.shape
    if mode == "fractions":
        assert cat_lookup is not None and max_cover_val is not None
        counts = np.zeros(n_out, dtype=np.int64)
        cat_counts = {cat: np.zeros(n_out, dtype=np.int64) for cat in cat_lookup}

        for chunk_start in tqdm(range(0, len(lat_idx), chunk_size), desc="Processing chunks"):
            chunk_end = min(chunk_start + chunk_size, len(lat_idx))
            lat_chunk_sel = _to_slice(lat_idx[chunk_start:chunk_end])
            values_chunk = ds[source_var].isel(lat=lat_chunk_sel, lon=lon_sel).values
            lats_chunk = src_lats_all[chunk_start:chunk_end]

            lat_bin = np.clip(np.searchsorted(lat_edges[1:-1], lats_chunk), 0, n_lat_lut - 1).astype(np.int32)
            lon_bin = np.clip(np.searchsorted(lon_edges[1:-1], src_lons_crop), 0, n_lon_lut - 1).astype(np.int32)
            bucket = bucket_lut[lat_bin[:, np.newaxis], lon_bin[np.newaxis, :]].ravel()
            values_flat = values_chunk.ravel()
            _accumulate_fraction_stats(
                bucket, values_flat, counts, cat_counts, cat_lookup, max_cover_val, nan_to_num=nan_to_num
            )

        return counts, cat_counts

    sums = np.zeros(n_out, dtype=np.float64)
    valid_counts = np.zeros(n_out, dtype=np.int64)
    for chunk_start in tqdm(range(0, len(lat_idx), chunk_size), desc="Processing chunks"):
        chunk_end = min(chunk_start + chunk_size, len(lat_idx))
        lat_chunk_sel = _to_slice(lat_idx[chunk_start:chunk_end])
        values_chunk = ds[source_var].isel(lat=lat_chunk_sel, lon=lon_sel).values
        lats_chunk = src_lats_all[chunk_start:chunk_end]

        lat_bin = np.clip(np.searchsorted(lat_edges[1:-1], lats_chunk), 0, n_lat_lut - 1).astype(np.int32)
        lon_bin = np.clip(np.searchsorted(lon_edges[1:-1], src_lons_crop), 0, n_lon_lut - 1).astype(np.int32)
        bucket = bucket_lut[lat_bin[:, np.newaxis], lon_bin[np.newaxis, :]].ravel()
        values_flat = values_chunk.ravel()
        _accumulate_mean_stats(
            bucket, values_flat, sums, valid_counts, fill_value=fill_value, nan_to_num=nan_to_num
        )

    return sums, valid_counts


def _to_slice(idx):
    """Convert a contiguous index array to a slice so xarray uses fast slice reads."""
    if len(idx) > 1 and idx[-1] - idx[0] == len(idx) - 1:
        return slice(int(idx[0]), int(idx[-1]) + 1)
    return idx


def compute_statistics(input_file, out_lat, out_lon, area_def=None, chunk_size=1000,
                       source_var="COVER", mode="fractions", output_name=None, nan_to_num=None):
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

    Returns:
        dict mapping output variable name(s) to array(s) with shape matching out_lat.
    """
    out_lat = np.squeeze(np.asarray(out_lat))
    out_lon = np.squeeze(np.asarray(out_lon))
    out_shape = out_lat.shape
    out_lat_flat = out_lat.ravel()
    out_lon_flat = out_lon.ravel()
    n_out = out_lat_flat.size

    margin = 1.0
    ds = xr.open_dataset(input_file)
    if source_var not in ds:
        raise ValueError(f"Variable '{source_var}' not found in {input_file}")

    fill_value = ds[source_var].attrs.get(
        "_FillValue",
        ds[source_var].attrs.get("missing_value", ds[source_var].encoding.get("_FillValue", None)),
    )
    src_lons_all = _match_lon_convention(ds.lons.values, out_lon_flat)
    lat_idx = np.where(
        (ds.lats.values >= out_lat_flat.min() - margin) &
        (ds.lats.values <= out_lat_flat.max() + margin)
    )[0]
    lon_idx = np.where(
        (src_lons_all >= out_lon_flat.min() - margin) &
        (src_lons_all <= out_lon_flat.max() + margin)
    )[0]
    src_lats_all = ds.lats.values[lat_idx]
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
    elif mode == "mean":
        cat_lookup = None
        max_cover_val = None
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'fractions' or 'mean'.")

    tic = time.time()
    if area_def is not None:
        crs = pyproj.CRS.from_dict(area_def.crs.to_dict())
        transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x_min, y_min, x_max, y_max = area_def.area_extent
        height, width = area_def.shape
        accum_a, accum_b = _accumulate_projected(
            ds, source_var, mode, lat_idx, lat_sel, lon_sel, src_lats_all, src_lons_crop,
            transformer, x_min, y_min, x_max, y_max, height, width,
            chunk_size, n_out,
            cat_lookup=cat_lookup, max_cover_val=max_cover_val, fill_value=fill_value,
            nan_to_num=nan_to_num,
        )
    else:
        # LUT resolution must be finer than the output grid spacing so that
        # every output point is the nearest neighbour of at least one LUT cell.
        # Estimate the average output point spacing from the domain area and count.
        lat_range = out_lat_flat.max() - out_lat_flat.min()
        lon_range = out_lon_flat.max() - out_lon_flat.min()
        approx_spacing = np.sqrt(lat_range * lon_range / n_out)
        lut_resolution = min(approx_spacing * 0.5, 0.25)
        print(f"Output grid spacing ~{approx_spacing:.4f}°, using LUT resolution {lut_resolution:.4f}°")
        bucket_lut, lat_edges, lon_edges = _build_latlon_bucket_lut(
            out_lat_flat, out_lon_flat, lut_resolution=lut_resolution
        )
        accum_a, accum_b = _accumulate_latlon_lut(
            ds, source_var, mode, lat_idx, lat_sel, lon_sel, src_lats_all, src_lons_crop,
            bucket_lut, lat_edges, lon_edges,
            chunk_size, n_out,
            cat_lookup=cat_lookup, max_cover_val=max_cover_val, fill_value=fill_value,
            nan_to_num=nan_to_num,
        )

    ds.close()
    print(f"Total processing time: {time.time() - tic:.1f}s")

    results = {}
    if mode == "fractions":
        counts, cat_counts = accum_a, accum_b
        zero_cells = (counts == 0).sum()
        assert zero_cells == 0, f"{zero_cells} output cells received no source points — check domain/source overlap"
        for cat in TRANSTABLE:
            results[cat] = (cat_counts[cat] / counts.astype(np.float64)).reshape(out_shape)
    elif mode == "mean":
        sums = np.asarray(accum_a, dtype=np.float64)
        valid_counts = np.asarray(accum_b, dtype=np.int64)
        mean = np.full(n_out, np.nan, dtype=np.float64)
        valid_out = valid_counts > 0
        mean[valid_out] = sums[valid_out] / valid_counts[valid_out]
        results[output_name or f"{source_var.lower()}_mean"] = mean.reshape(out_shape)
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'fractions' or 'mean'.")

    ## _accumulate_projected produces north-up order (row 0 = northernmost).
    ## If out_lat is south-up (row 0 has lower latitude than row -1), flip to match.
    #if area_def is not None and out_lat.ndim == 2:
    #    if out_lat[0].mean() < out_lat[-1].mean():  # south-up
    #        results = {name: v[::-1, :] for name, v in results.items()}

    return results


def compute_fractions(input_file, out_lat, out_lon, area_def=None, chunk_size=2048):
    """Backward-compatible wrapper for COVER fractions."""
    return compute_statistics(
        input_file=input_file,
        out_lat=out_lat,
        out_lon=out_lon,
        area_def=area_def,
        chunk_size=chunk_size,
        source_var="COVER",
        mode="fractions",
    )


def save_fractions_to_netcdf(out_lat, out_lon, fractions, output_nc):
    out_lat_arr = np.squeeze(np.asarray(out_lat))
    out_lon_arr = np.squeeze(np.asarray(out_lon))
    is_2d = out_lat_arr.ndim == 2

    coords = (
        {"lat": (["y", "x"], out_lat_arr), "lon": (["y", "x"], out_lon_arr)}
        if is_2d
        else {"lat": ("points", out_lat_arr), "lon": ("points", out_lon_arr)}
    )
    dims = ["y", "x"] if is_2d else ["points"]

    ds_out = xr.Dataset(
        {cat: xr.DataArray(frac, dims=dims) for cat, frac in fractions.items()},
        coords=coords,
    )
    ds_out["lat"].attrs = {"units": "degrees_north", "long_name": "latitude"}
    ds_out["lon"].attrs = {"units": "degrees_east", "long_name": "longitude"}
    for cat in fractions:
        ds_out[cat].attrs = {"long_name": f"{cat} fraction", "units": "1"}

    ds_out.to_netcdf(output_nc)
    print(f"Fractions written to {output_nc}")

    print({k: v.shape for k, v in fractions.items()})


def save_statistics_to_netcdf(out_lat, out_lon, fields, output_nc, units_by_var=None):
    """Save one or more computed fields on the output grid."""
    out_lat_arr = np.squeeze(np.asarray(out_lat))
    out_lon_arr = np.squeeze(np.asarray(out_lon))
    is_2d = out_lat_arr.ndim == 2

    coords = (
        {"lat": (["y", "x"], out_lat_arr), "lon": (["y", "x"], out_lon_arr)}
        if is_2d
        else {"lat": ("points", out_lat_arr), "lon": ("points", out_lon_arr)}
    )
    dims = ["y", "x"] if is_2d else ["points"]

    ds_out = xr.Dataset(
        {name: xr.DataArray(values, dims=dims) for name, values in fields.items()},
        coords=coords,
    )
    ds_out["lat"].attrs = {"units": "degrees_north", "long_name": "latitude"}
    ds_out["lon"].attrs = {"units": "degrees_east", "long_name": "longitude"}

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

    parser = argparse.ArgumentParser(description="Compute fractions or means from ECOCLIMAP-SG-like data.")
    parser.add_argument("--source_file", type=str, default=input_file, help="Path to the input ECOCLIMAP-SG NetCDF file.")
    parser.add_argument("--output_domain", type=str, default=None, help="name of the output domain")
    parser.add_argument("--domain_file", type=str, default=None, help="Path to zarr with latlon (optional).")
    parser.add_argument("--out_nc", type=str, default="fractions_output.nc", help="Path to the output NetCDF file.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of source latitude rows processed at a time.")
    parser.add_argument("--mode", type=str, default="fractions", choices=["fractions", "mean"], help="Aggregation mode.")
    parser.add_argument("--source_var", type=str, default="COVER", help="Source variable name to aggregate.")
    parser.add_argument("--output_var", type=str, default=None, help="Output variable name (used for mean mode).")
    parser.add_argument("--nan_to_num", type=float, default=None, help="Replace NaN values with this number before aggregation (optional).")
    args = parser.parse_args()

    if args.domain_file:
        if args.domain_file == "test":
            out_lat = np.array([60, 61, 62])
            out_lon = np.array([10, 11, 12])
        elif args.domain_file.endswith(".zarr"):
            domain_ds = anemoi.datasets.open_dataset(args.domain_file)
            out_lat = domain_ds.latitudes
            out_lon = domain_ds.longitudes
        else:
            domain_ds = xr.open_dataset(args.domain_file)
            out_lat = domain_ds.latitude
            out_lon = domain_ds.longitude
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
    )
    first_key = next(iter(fields))
    first_values = fields[first_key]
    assert first_values is not None
    print(np.asarray(out_lat).shape, np.asarray(out_lon).shape, np.asarray(first_values).shape)

    units = "1" if args.mode == "fractions" else ""
    units_by_var = {name: units for name in fields}
    save_statistics_to_netcdf(out_lat, out_lon, fields, args.out_nc, units_by_var=units_by_var)