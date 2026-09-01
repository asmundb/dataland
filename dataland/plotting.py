import numpy as np
import matplotlib
import matplotlib.tri as mtri   
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
    

def plot_unstructured(lat, lon, values, title="", vmin=0, vmax=1, cmap="viridis", output_file="fraction.png",method="scatter"):
    fig, ax = plt.subplots(
        figsize=(14, 7),
        subplot_kw={"projection": ccrs.Robinson()}
    )
    ax.set_global()
    ax.coastlines(resolution="110m", linewidth=0.5)
    ax.gridlines(linewidth=0.3, alpha=0.5)

    if method == "trimesh":
        triang = mtri.Triangulation(lon, lat)
        sc = ax.tripcolor(triang, values, cmap=cmap, vmin=vmin, vmax=vmax,
                          transform=ccrs.PlateCarree(), rasterized=True)
    
    elif method == "scatter":
        sc = ax.scatter(
            lon, lat,
            c=values,
            s=1,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(),
            rasterized=True,
        )

    plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.04, fraction=0.03, label=title)
    ax.set_title(title)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_file}")


if __name__ == "__main__":
    ds = xr.open_dataset("fractions_metcoop2.nc")
    ds = xr.open_dataset("treeheights_O96_2.nc")
    ds2 = xr.open_dataset("treeheights_o96.nc")
    #ds2 = xr.open_dataset("/lustre/storeB/users/asmundb/dataset/decoder/forcings_with_topo_descriptors.nc")


    d = {"sea": "SFX.FRAC_SEA", "nature": "SFX.FRAC_NATURE", 
         "forest": "FOREST", "lake": "SFX.FRAC_WATER", "urban": "SFX.FRAC_TOWN",
         "glacier": "SFX.COVER006",
         "tree_height": "H_TREE"}

    d = {"H_TREE": "tree_height"}

    for cat in ds.data_vars:
        plot_unstructured(
            ds["lat"].values,
            np.where(ds["lon"].values > 180, ds["lon"].values - 360, ds["lon"].values),
            ds[cat].values,
            title=f"{cat} Fraction", 
            output_file=f"fraction_{cat}.png",
            method="trimesh",
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
    ds = xr.open_dataset("treeheights_metcoop_3.nc")
    ds2 = xr.open_dataset("/lustre/storeB/users/asmundb/dataset/decoder/forcings_with_topo_descriptors.nc")
    origin="lower"
    #ref = np.nan_to_num(ds2[d[cat]].values.squeeze(), nan=0.0) #[::-1]

    ref = ds2["H_TREE"].fillna(0).values.squeeze()
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

    
    ds = xr.open_dataset("treeheights_metcoop_2.nc")
    plot_unstructured(
        ds["lat"].values,
        np.where(ds["lon"].values > 180, ds["lon"].values - 360, ds["lon"].values),
        ds["SFX.H_TREE"].values,
        title="Tree Height", 
        output_file=f"tree_height.png",
        method="trimesh",
        cmap="plasma",
        vmin=None,
        vmax=None
    )


#import xarray as xr
#ds = xr.open_dataset("/lustre/storeB/users/asmundb/dataset/ecoclimap_sg/new_ht_c.nc", chunks={})
#allowed_netcdf4 = {
#    "contiguous", "compression", "dtype", "_FillValue", "least_significant_digit",
#    "szip_pixels_per_block", "shuffle", "significant_digits", "quantize_mode",
#    "complevel", "fletcher32", "szip_coding", "endian", "zlib", "chunksizes",
#    "blosc_shuffle",
#}
#encoding = {
#    var: {k: v for k, v in ds[var].encoding.items() if k in allowed_netcdf4}
#    for var in ds.variables
#}
#ds["H_TREE"] = ds["H_TREE"].fillna(0.0) / 100.0
#ds.to_netcdf("new_ht_c_m.nc", encoding=encoding)