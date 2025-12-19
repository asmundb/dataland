import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import xarray as xr
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
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
    

