import numpy as np
import matplotlib
import matplotlib.tri as mtri   
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import xarray as xr
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature
from pathlib import Path


def meps_crs():
    cartopy_crs = ccrs.LambertConformal(
        central_longitude=15,
        central_latitude=63.3,
        standard_parallels=(
            63.3, 63.3
        ),
        globe=ccrs.Globe(semimajor_axis=6371000.0,
                         semiminor_axis=6371000.0)

    )
    return cartopy_crs, (-1060084.0, 1309916.0, -1332517.875, 1337482.125)


def map_all_vars(ds, figdir="figs/"):
    fig_dir = Path(figdir)
    fig_dir.mkdir(exist_ok=True)
    min_max = ds.quantile(q=[0.05, 0.95])
    for i, x in enumerate(ds.data_vars):
        fig, ax = plt.subplots()
        im = ax.imshow(ds[x][::-1], vmin=min_max[x][0].values, vmax=min_max[x][1].values)
        plt.colorbar(im, ax=ax)
        plt.savefig(f"{fig_dir}/{x}.png")
        plt.close()


def plot_detail_map(ds, extent=[10.2,11,59.8,60.05], figdir="fig_zoom/"):
    crs, extent = meps_crs()
    esri_sat = cimgt.GoogleTiles(style="satellite")
    
    for var in ds.data_vars:
        if "VALLEY_NORM" in var:
            data = np.minimum(np.round(ds[var].values[::-1]), 10)
        else:
            data = ds[var].values[::-1]
        hot = plt.get_cmap("hot")
        norm = Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
        rgba = hot(norm(data))
        rgba[..., -1] = norm(data)
        
        fig, ax = plt.subplots(ncols=1, subplot_kw=dict(projection=crs), sharex="all",sharey="all")
        ax.add_image(esri_sat, 13, zorder=0)
        im = ax.imshow(rgba, transform=crs, extent=extent, zorder=10, vmax=.2)
        #ax.set_extent([10.2,11,59.8,60.05]) # oslo
        ax.set_extent([9.31,9.73,58.81,58.94]) # jomfruland
        ax.coastlines(resolution="10m")
        plt.savefig(f"figs2/{var}.png")
        plt.close()
    

def _scatter_size(ax, lon, lat):
    """Bound circular marker diameter by the closest display-space pair."""
    from scipy.spatial import cKDTree

    projected = ax.projection.transform_points(ccrs.PlateCarree(), lon, lat)[:, :2]
    display = ax.transData.transform(projected)
    display = display[np.isfinite(display).all(axis=1)]
    if len(display) < 2:
        return 1.0
    tree = cKDTree(display)
    minimum = np.inf
    for start in range(0, len(display), 65536):
        spacing = tree.query(display[start:start + 65536], k=2)[0][:, 1]
        minimum = min(minimum, float(spacing.min()))
    if minimum == 0:
        raise ValueError("Coincident scatter points cannot be drawn without overlap")
    diameter_pixels = minimum * 0.95
    return (diameter_pixels * 72 / ax.figure.dpi) ** 2


def plot_unstructured(
    lat, lon, values, title="", vmin=0, vmax=1, cmap="viridis",
    output_file="fraction.png", method="scatter", satellite=False, zoom=None,
    point_size=None, background_color=None
):
    """Plot unstructured data, optionally over satellite tiles."""
    if method not in ("scatter", "trimesh"):
        raise ValueError("method must be scatter or trimesh")
    if point_size is not None and (not np.isfinite(point_size) or point_size <= 0):
        raise ValueError("point_size must be finite and positive (points squared)")
    if background_color is not None and not matplotlib.colors.is_color_like(background_color):
        raise ValueError(f"Invalid background color: {background_color!r}")
    lat, lon, values = (np.asarray(a).ravel() for a in (lat, lon, values))
    if not (lat.shape == lon.shape == values.shape):
        raise ValueError("Latitude, longitude and values must have matching shapes")
    lon = (lon + 180) % 360 - 180
    keep = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(values)
    if zoom is not None:
        west, south, east, north = zoom
        if not (np.isfinite(zoom).all() and -180 <= west < east <= 180
                and -90 <= south < north <= 90):
            raise ValueError("zoom must be lon_min, lat_min, lon_max, lat_max with increasing bounds")
        keep &= (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
    if not keep.any():
        raise ValueError(f"No finite data points to plot for {title!r} in the requested region")
    lat, lon, values = lat[keep], lon[keep], values[keep]
    if method == "trimesh" and len(values) < 3:
        raise ValueError("trimesh requires at least three finite points")
    fig, ax = plt.subplots(
        figsize=(14, 7),
        subplot_kw={"projection": ccrs.Robinson()}
    )
    if background_color is not None:
        fig.set_facecolor(background_color)
        ax.set_facecolor(background_color)
    if zoom is not None:
        ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
    elif method == "trimesh":
        ax.set_global()

    if satellite:
        ax.add_image(cimgt.GoogleTiles(style="satellite"), 2, zorder=0)

    ax.coastlines(resolution="10m", linewidth=0.5, zorder=3)
    ax.gridlines(linewidth=0.3, alpha=0.5, zorder=3)

    lon = np.where(lon > 180, lon - 360, lon)
    if method == "trimesh":
        triang = mtri.Triangulation(lon, lat)
        sc = ax.tripcolor(triang, values, cmap=cmap, vmin=vmin, vmax=vmax,
                          transform=ccrs.PlateCarree(), rasterized=True,
                          alpha=0.78, zorder=2)
    
    elif method == "scatter":
        sc = ax.scatter(
            lon, lat,
            c=values,
            s=1 if point_size is None else point_size,
            marker="o",
            linewidths=0,
            edgecolors="none",
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(),
            rasterized=True,
            alpha=1,
            zorder=2,
            
        )

    plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.04, fraction=0.03, label=title)
    ax.set_title(title)
    if method == "scatter":
        fig.canvas.draw()
        maximum_size = _scatter_size(ax, lon, lat)
        requested_size = point_size if point_size is not None else (maximum_size if zoom is not None else 1)
        sc.set_sizes([min(requested_size, maximum_size)])
    plt.savefig(output_file, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {output_file}")


def plot_file(input_file, output_dir, variables=None, zoom=None, indices=None,
              method="scatter", cmap="viridis", vmin=None, vmax=None, file_format="auto",
              point_size=None, background_color=None):
    """Save one spatial slice per variable; unspecified non-spatial indices are zero."""
    import logging
    import re

    indices = {} if indices is None else dict(indices)
    if any(index < 0 for index in indices.values()):
        raise ValueError("Slice indices must be nonnegative")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    is_zarr = str(input_file).rstrip("/").endswith(".zarr")
    anemoi = file_format == "anemoi"
    if file_format == "auto" and is_zarr:
        import zarr

        group = zarr.open_group(str(input_file), mode="r")
        anemoi = ("variables" in group.attrs
                  and all(name in group for name in ("data", "latitudes", "longitudes")))
    paths = []
    filenames = set()

    def save(name, lat, lon, values, units=""):
        filename = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name)) + ".png"
        if filename in filenames:
            raise ValueError(f"Variables map to the same output filename: {filename}")
        filenames.add(filename)
        path = output_dir / filename
        plot_unstructured(lat, lon, values,
                          title=f"{name} [{units}]" if units else str(name),
                          output_file=path, method=method, cmap=cmap,
                          vmin=vmin, vmax=vmax, zoom=zoom, point_size=point_size,
                          background_color=background_color)
        paths.append(path)

    if anemoi:
        import anemoi.datasets

        ds = anemoi.datasets.open_dataset(str(input_file))
        unknown = set(indices) - {"time", "member"}
        if unknown:
            raise ValueError(f"Anemoi indices must be time or member, not {sorted(unknown)}")
        selected = list(ds.variables) if variables is None else variables
        if not selected or any(name not in ds.name_to_index for name in selected):
            raise ValueError("Requested variables are absent or no variables were selected")
        time, member = indices.get("time", 0), indices.get("member", 0)
        if time >= ds.shape[0] or member >= ds.shape[2]:
            raise ValueError("Time or member index is outside the Anemoi dataset")
        logging.info("Plotting Anemoi time=%s, member=%s", time, member)
        for name in selected:
            save(name, ds.latitudes, ds.longitudes,
                 ds[time, ds.name_to_index[name], member, :])
    else:
        opener = xr.open_zarr if is_zarr else xr.open_dataset
        with opener(input_file) as ds:
            unknown = set(indices) - set(ds.dims)
            if unknown:
                raise ValueError(f"Unknown index dimensions: {sorted(unknown)}")
            for dim, index in indices.items():
                if index >= ds.sizes[dim]:
                    raise ValueError(f"Index {index} is outside dimension {dim}")
            coords = []
            for axis, names in (("latitude", ("latitude", "lat", "lats")),
                                ("longitude", ("longitude", "lon", "lons"))):
                matches = [name for name in ds.variables
                           if ds[name].attrs.get("standard_name") == axis]
                if not matches:
                    matches = [name for name in names if name in ds]
                if len(matches) != 1:
                    raise ValueError(f"Expected one unambiguous {axis} variable")
                coords.append(ds[matches[0]])
            lat, lon = coords
            spatial_dims = set(lat.dims) | set(lon.dims)
            if not spatial_dims:
                raise ValueError("Latitude/longitude must have spatial dimensions")
            if set(indices) & spatial_dims:
                raise ValueError("Index overrides must not select spatial dimensions")
            selected = (list(variables) if variables is not None else
                        [name for name in ds.data_vars
                         if name not in (lat.name, lon.name)
                         and spatial_dims.issubset(ds[name].dims)
                         and np.issubdtype(ds[name].dtype, np.number)])
            if not selected:
                raise ValueError("No spatial variables to plot")
            for name in selected:
                if name not in ds.data_vars:
                    raise ValueError(f"Data variable {name!r} is absent")
                field = ds[name]
                if not spatial_dims.issubset(field.dims):
                    raise ValueError(f"{name!r} does not contain all spatial dimensions")
                selection = {dim: indices.get(dim, 0) for dim in field.dims
                             if dim not in spatial_dims}
                logging.info("Plotting %s at indices %s", name, selection)
                field = field.isel(selection)
                field, lat_grid, lon_grid = xr.broadcast(field, lat, lon)
                save(name, lat_grid.values, lon_grid.values, field.values,
                     field.attrs.get("units", ""))
    return paths


def main():
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="Plot file variables as individual PNG maps.")
    parser.add_argument("input_file", help="Anemoi/xarray Zarr or NetCDF")
    parser.add_argument("--output-dir", default="figs")
    parser.add_argument("--variable", "--variables", nargs="+", help="Default: all spatial variables")
    parser.add_argument("--zoom", type=float, nargs=4,
                        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    parser.add_argument("--index", action="append", default=[], metavar="DIM=INDEX",
                        help="Repeat to override first-slice defaults; Anemoi uses time and member")
    parser.add_argument("--format", choices=("auto", "xarray", "anemoi"), default="auto")
    parser.add_argument("--method", choices=("scatter", "trimesh"), default="scatter")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument("--point-size", type=float,
                        help="Requested scatter area in points squared, capped to prevent overlap")
    parser.add_argument("--background-color", help="Map and figure background: color name or '#RRGGBB'")
    args = parser.parse_args()
    indices = {}
    for specification in args.index:
        dim, separator, value = specification.partition("=")
        if not separator or not dim or not value.isdigit() or dim in indices:
            parser.error("--index requires a unique DIM=INDEX with a nonnegative integer")
        indices[dim] = int(value)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    plot_file(args.input_file, args.output_dir, args.variable, args.zoom, indices,
              args.method, args.cmap, args.vmin, args.vmax, args.format, args.point_size,
              args.background_color)


def _examples():
    import anemoi.datasets
    ds = xr.open_dataset("data/output_data/fractions_N320.nc")
    ds = xr.open_dataset("data/output_data/fractions_carra2.nc")
    #ds = xr.open_dataset("treeheights_O96_2.nc")
    #ds2 = xr.open_dataset("treeheights_o96.nc")
    #ds2 = xr.open_dataset("/lustre/storeB/users/asmundb/dataset/decoder/forcings_with_topo_descriptors.nc")


    d = {"sea": "SFX.FRAC_SEA", "nature": "SFX.FRAC_NATURE", 
         "forest": "FOREST", "lake": "SFX.FRAC_WATER", "urban": "SFX.FRAC_TOWN",
         "glacier": "SFX.COVER006",
         "tree_height": "H_TREE"}

    d = {"H_TREE": "tree_height"}

    for cat in ds.data_vars:
        plot_unstructured(
            ds["latitude"].values,
            ds["longitude"].values,
            ds[cat].values,
            title=f"{cat} Fraction", 
            output_file=f"fraction_{cat}.png",
            method="scatter",
            cmap="plasma",
            vmin=None,
            vmax=None)

        if np.array_equal(ds[cat].values, ds2[cat].values):
            print(f"Skipping {cat} as it is identical in both datasets.")
            continue
        plot_unstructured(
            ds["lat"].values,
            np.where(ds["lon"].values > 180, ds["lon"].values - 360, ds["lon"].values),
            ds[cat].values - ds2[cat].values,
            title=f"{cat} Fraction", 
            output_file=f"diff_fraction_{cat}.png",
            method="trimesh",
            cmap="RdBu_r", vmin=-5, vmax=5)

    cat = "H_TREE"
    ds = xr.open_dataset("data/output_data/tree_height_carra2.nc")
    plot_unstructured(
        ds["latitude"].values,
        ds["longitude"].values,
        ds["h_tree_mean"].values,
        title=f"Tree Height", 
        output_file=f"tree_height_carra2.png",
        method="scatter",
        cmap="plasma",
        vmin=None,
        vmax=None
    )
    #ds2 = xr.open_dataset("/lustre/storeB/users/asmundb/dataset/decoder/forcings_with_topo_descriptors.nc")
    origin="lower"
    #ref = np.nan_to_num(ds2[d[cat]].values.squeeze(), nan=0.0) #[::-1]

    #ref = ds2["H_TREE"].fillna(0).values.squeeze()
    diff = ds[cat].values - ref
    vmax = ref.max()
    vmin = 0
    vm = np.nanstd(np.abs(diff))*3
    fig, ax = plt.subplots(ncols=3, sharey="all", sharex="all", figsize=(12, 4))
    im0 = ax[0].imshow(diff, vmin=-vm, vmax=vm, cmap="RdBu_r", origin=origin, interpolation="none")
    im1 = ax[1].imshow(ds[cat].values, vmin=vmin, vmax=vmax, cmap="viridis", origin=origin, interpolation="none")
    im2 = ax[2].imshow(ref, vmin=vmin, vmax=vmax, cmap="viridis", origin=origin, interpolation="none")
    plt.colorbar(im0, ax=ax[0], orientation="horizontal", pad=0.04, fraction=0.03, label=f"{cat} Difference")
    plt.colorbar(im1, ax=ax[1:], orientation="horizontal", pad=0.04, fraction=0.03, label=f"{cat} Value")
    plt.show()
    plt.savefig(f"sea_diff_{cat}.png", dpi=150, bbox_inches="tight")
    plt.close()

    
    ds = xr.open_dataset("data/output_data/tree_height_N320.nc")
    plot_unstructured(
        ds["latitude"].values,
        ds["longitude"].values,
        ds["h_tree_mean"].values,
        title="Tree Height", 
        output_file=f"tree_height_N320.png",
        method="trimesh",
        cmap="plasma",
        vmin=None,
        vmax=None
    )

    res = "n320"
    rad = "100"
    ds = xr.open_dataset(f"data/output_data/std_oro_{res}_{rad}.nc")
    plot_unstructured(
            ds["latitude"].values,
            ds["longitude"].values,
            ds[f"std_oro_{rad}km"].values,
            title="Orography std", 
            output_file=f"orography_std_{res}_{rad}.png",
            method="scatter",
            cmap="plasma",
            vmin=0,
            vmax=None
        )
    plot_unstructured(
                ds["latitude"].values,
                ds["longitude"].values,
                ds[f"orography_mean"].values,
                title="Orography mean", 
                output_file=f"orography_mean_{res}_{rad}.png",
                method="scatter",
                cmap="plasma",
                vmin=0,
                vmax=None
            )

    res = "carra2"
    rad = "100"
    
    ds = xr.open_dataset(f"data/output_data/slope_oro_{res}_{rad}.nc")
    vm = np.abs(ds["orography_slope_x"].values).max()
    plot_unstructured(
            ds["latitude"].values,
            ds["longitude"].values,
            ds["orography_slope_x"].values,
            title="Orography slope", 
            output_file=f"orography_slope_x_{res}_{rad}.png",
            method="scatter",
            cmap="RdBu_r",
            vmin=-vm,
            vmax=vm
        )
    plot_unstructured(
                ds["latitude"].values,
                ds["longitude"].values,
                ds["orography_slope_y"].values,
                title="Orography slope", 
                output_file=f"orography_slope_y_{res}_{rad}.png",
                method="scatter",
                cmap="RdBu_r",
                vmin=-vm,
                vmax=vm
            )


    
    ds = xr.open_dataset("std_oro_075.nc")
    plot_unstructured(
            ds["latitude"].values,
            ds["longitude"].values,
            ds["std_oro"].values,
            title="Orography std", 
            output_file=f"std_oro_metcoop_30.png",
            method="trimesh",
            cmap="plasma",
            vmin=0,
            vmax=1200
        )
    
    ds1 = anemoi.datasets.open_dataset("/lustre/storeB/project/nwp/bris/datasets/aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6.zarr")

    var = "sdor"
    plot_unstructured(
            ds1.latitudes,
            ds1.longitudes,
            ds1[0, ds1.name_to_index[var] , 0, :],
            title=var, 
            output_file=f"{var}.png",
            method="trimesh",
            cmap="plasma",
            vmin=0,
            vmax=1200
        )

    ds = xr.open_dataset("std_oro_metcoop_30.nc")
    plt.imshow(ds["std_oro"].values, vmin=0, vmax=1200, origin="lower", interpolation="none")
    plt.show()

    
    x = ds1[0, ds1.name_to_index[var] , 0, :]

    y = ds["std_oro"].values
    


def old():
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt
    meps_file = "/lustre/arkivB/projects/metproduction/MEPS/2026/10/06/meps_det_2_5km_20261006T00Z.nc"
    ds = xr.open_dataset(meps_file)
    var = "air_temperature_0m"
    field = ds[var].isel(time=0).values.squeeze() -273.15
    crs, extent = meps_crs()
    zoom = [-255000, -233000, -380000, -359500]
    
    fig, ax = plt.subplots(subplot_kw={"projection": crs})
    ax.set_extent(zoom, crs=crs)
    #ax.imshow(field, origin="lower", interpolation="none", transform=crs, extent=extent)
    #ax.coastlines(resolution='10m')
    ax.add_image(cimgt.GoogleTiles(style="satellite"), 13, zorder=0)
    pixel_width = (extent[1] - extent[0]) / field.shape[-1]
    pixel_height = (extent[3] - extent[2]) / field.shape[-2]

    x_edges = extent[0] + np.arange(field.shape[-1] + 1) * pixel_width
    y_edges = extent[2] + np.arange(field.shape[-2] + 1) * pixel_height
    x_edges = x_edges[(x_edges >= zoom[0]) & (x_edges <= zoom[1])]
    y_edges = y_edges[(y_edges >= zoom[2]) & (y_edges <= zoom[3])]
    ax.vlines(x_edges, zoom[2], zoom[3], transform=crs, color="white", linewidth=0.5, alpha=0.6)
    ax.hlines(y_edges, zoom[0], zoom[1], transform=crs, color="white", linewidth=0.5, alpha=0.6)
        
    plt.savefig("oslo_satellite.png")
    plt.close()

    # THIS FIGURE
    var = "SFX_LE"
    field = ds[var].isel(time=12).values.squeeze()
        
    fig, ax = plt.subplots(subplot_kw={"projection": crs})
    ax.imshow(field, origin="lower", interpolation="none", transform=crs, extent=extent,cmap="RdBu_r", vmin=-160, vmax=160)
    ax.set_extent(zoom, crs=crs)
    plt.savefig("oslo_LE.png")
    plt.close()

    
    plt.show()


    ds = xr.open_dataset("data/output_data/fractions_carra2_sub.nc")
    ds = xr.open_dataset("data/output_data/tree_height_carra2_sub.nc")
    var = "mean"
    plt.scatter(np.where(ds.longitude>180, ds.longitude-360, ds.longitude), 
                ds.latitude, c=ds[var])
    plt.show()

    ds2 = xr.open_dataset("data/output_data/std_oro_carra2_2p5.nc")



    # quick maths
    a = 2*(ds.x.size + ds.y.size)*65
    b = ds.x.size * ds.y.size
    b/a


    # icrements
    meps_file0 = "/lustre/arkivB/projects/metproduction/MEPS/2026/07/13/meps_det_2_5km_20260713T09Z.nc"
    meps_file1 = "/lustre/arkivB/projects/metproduction/MEPS/2026/07/13/meps_det_2_5km_20260713T12Z.nc"
    
    ds0 = xr.open_dataset(meps_file0)
    ds1 = xr.open_dataset(meps_file1)

    var = "SFX_X001WG1"

    field0 = ds0[var].isel(time=3).values.squeeze()
    field1 = ds1[var].isel(time=0).values.squeeze()
    
    inc0 = field1 - field0
            
    fig, ax = plt.subplots(subplot_kw={"projection": crs})
    ax.imshow(inc0, origin="lower", interpolation="none", transform=crs, extent=extent,cmap="RdBu_r", vmin=-.012, vmax=.012)
    ax.coastlines()
    plt.savefig("sm_inc.png")
    plt.close()
    plt.show()


    
    var = "precipitation_amount_acc"
    h = 6
    field0 = ds0[var].isel(time=h+3).values.squeeze() - ds0[var].isel(time=h+3-6).values.squeeze()
    field1 = ds1[var].isel(time=h).values.squeeze() - ds1[var].isel(time=h-6).values.squeeze()
    inc0 = field1 - field0
            
    fig, ax = plt.subplots(subplot_kw={"projection": crs})
    ax.imshow(inc0, origin="lower", interpolation="none", transform=crs, extent=extent,cmap="RdBu_r", vmin=-4, vmax=4)
    ax.coastlines()
    plt.savefig("tp_inc.png")
    plt.close()

    plt.show()


    meps_file = "https://thredds.met.no/thredds/dodsC/mepslatest/meps_det_2_5km_20261020T00Z.ncml"
    #meps_file = "/lustre/arkivB/projects/metproduction/MEPS/2026/10/06/meps_det_2_5km_20261006T00Z.nc"
    ds = xr.open_dataset(meps_file)
    var = "air_temperature_0m"
    field = ds[var].isel(time=0).values.squeeze() -273.15
    crs, extent = meps_crs()

    eco = xr.open_dataset("data/input_data/ecosg_final_map.nc")
    covers = {
        1: "sea and oceans",
        2: "lakes",
        3: "rivers",
        4: "bare land",
        5: "bare rock",
        6: "permanent snow",
        7: "boreal broadleaf deciduous",
        8: "temperate broadleaf deciduous",
        9: "tropical broadleaf deciduous",
        10: "temperate broadleaf evergreen",
        11: "tropical broadleaf evergreen",
        12: "boreal needleleaf evergreen",
        13: "temperate needleleaf evergreen",
        14: "boreal needleleaf deciduous",
        15: "shrubs",
        16: "boreal grassland",
        17: "temperate grassland",
        18: "tropical grassland",
        19: "winter C3 crops",
        20: "summer C3 crops",
        21: "C4 crops",
        22: "flooded trees",
        23: "flooded grassland",
        24: "LCZ1: compact high-rise",
        25: "LCZ2: compact midrise",
        26: "LCZ3: compact low-rise",
        27: "LCZ4: open high-rise",
        28: "LCZ5: open midrise",
        29: "LCZ6: open low-rise",
        30: "LCZ7: lightweight low-rise",
        31: "LCZ8: large low-rise",
        32: "LCZ9: sparsely built",
        33: "LCZ10: heavy industry"
    }
 
    x0 = -330000
    y0 = -410000
    w = 100000


def plot_ecosg_zoomed(x0, y0, w=10000, sat=False, histbarh=True, sat_res=13):

    h = w
    zoom = [x0, x0+w, y0, y0+h]

    lonlat = ccrs.PlateCarree().transform_points(crs, np.array([x0, x0+w, x0 , x0+w]), np.array([y0, y0+h, y0+h, y0]))
    lons = lonlat[..., 0]
    lats = lonlat[..., 1]
    lextent = [lons.min(), lons.max(), lats.min(), lats.max()]
    
    il = eco["lons"] > lons[0]
    il &= eco["lons"] < lons[1]
    jl = eco["lats"] > lats[0]
    jl &= eco["lats"] < lats[1]
    ecofield = eco["COVER"].isel(lon=il, lat=jl)
    
    fig, ax = plt.subplots(subplot_kw={"projection": crs})
    ax.set_extent(zoom, crs=crs)
    im = ax.imshow(ecofield, origin="upper", interpolation="none", transform=ccrs.PlateCarree(), extent=lextent,
                   vmin=0,vmax=len(covers)-1)
    if histbarh:
        fig.subplots_adjust(right=0.55)
        hist_ax = ax.inset_axes([1.04, 0, 0.24, 1])
        eco_values = im.get_array().compressed()
        eco_values = eco_values[np.isfinite(eco_values)]
        cover_values, counts = np.unique(eco_values, return_counts=True)
        hist_ax.barh(cover_values, np.log(counts), height=0.8, color=im.cmap(im.norm(cover_values)))
        hist_ax.set_ylim(im.norm.vmin - 0.5, im.norm.vmax + 0.5)
        hist_ax.set_yticks(cover_values)
        hist_ax.set_yticklabels([covers.get(val, str(val)) for val in cover_values], fontsize=8)
        hist_ax.yaxis.tick_right()
        hist_ax.yaxis.set_label_position("right")
        #hist_ax.set_ylabel("COVER")
        hist_ax.set_xlabel("log")
        hist_ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    if sat:
        ax.add_image(cimgt.GoogleTiles(style="satellite"), sat_res, zorder=0)
    pixel_width = (extent[1] - extent[0]) / field.shape[-1]
    pixel_height = (extent[3] - extent[2]) / field.shape[-2]
    x_edges = extent[0] + np.arange(field.shape[-1] + 1) * pixel_width
    y_edges = extent[2] + np.arange(field.shape[-2] + 1) * pixel_height
    x_edges = x_edges[(x_edges >= zoom[0]) & (x_edges <= zoom[1])]
    y_edges = y_edges[(y_edges >= zoom[2]) & (y_edges <= zoom[3])]
    #ax.vlines(x_edges, zoom[2], zoom[3], transform=crs, color="white", linewidth=0.2, alpha=0.6)
    #ax.hlines(y_edges, zoom[0], zoom[1], transform=crs, color="white", linewidth=0.2, alpha=0.6) 
    return fig, ax

 
    x0 = -330000
    y0 = -410000
    w = 100000

    fig, ax = plot_ecosg_zoomed(x0, y0, w)
    ax.set_title("100 km")
    plt.savefig("zoomed_100km.png")
    plt.close()

    fig, ax = plot_ecosg_zoomed(-265000, -380000, 30000)
    ax.set_title("30 km")
    plt.savefig("zoomed_30km.png")
    plt.close()
    
    fig, ax = plot_ecosg_zoomed(-258500, -375200, 2500)
    ax.set_title("2.5 km")
    plt.savefig("zoomed_2.5km.png")
    plt.close()

    fig, ax = plot_ecosg_zoomed(-258500, -375200, 2500, sat=True)
    ax.set_title("2.5 km Satellite")
    plt.savefig("zoomed_2.5km_sat.png")
    plt.close()

    fig, ax = plot_ecosg_zoomed(-257300, -374400, 1000, sat=True, sat_res=15)
    ax.set_title("1 km Satellite")
    plt.savefig("zoomed_1km_sat.png")
    plt.close()


if __name__ == "__main__":
    main()