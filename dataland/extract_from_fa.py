import numpy as np
import xarray as xr
import yaml
import pyproj
from importlib import resources
import glob
import epygram


def read_state_fa(r, varnames=None):
    sle = []
    for j in range(len(varnames)):
        field = r.readfield(varnames[j])
        data = field.getdata(subzone='CI')
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)
        sle.append(data)
    return np.stack(sle, axis=-1)


def process(pgdfile, prepfile, output_filename, config_filename=None, domain="meps"):
    
    if config_filename is None:
        if domain in ["carra_ne", "carra_igb"]:  # one patch systems 
            config_filename = str(resources.files("dataland.config").joinpath("FA_variables_p1.yaml"))
        else:
            config_filename = str(resources.files("dataland.config").joinpath("FA_variables_p2.yaml"))

    # Define variable metadata 
    with open(config_filename, "r") as f:
        varmeta = yaml.safe_load(f)
    
    r_prep = epygram.formats.resource(prepfile, openmode='r', fmt="FA")
    r_pgd = epygram.formats.resource(pgdfile, openmode='r', fmt="FA")
    
    #fl = r_pgd.listfields()
    #for f in fl:
    #    if "ZS" in f:
    #        print(f)
    
    dims = r_pgd.geometry.get_datashape(subzone='CI')
    
    ds = xr.Dataset(
        coords={
            "x": np.arange(dims[1].item()),
            "y": np.arange(dims[0].item())
        },
    )

    for vnam in varmeta:
        if vnam in r_prep.listfields():
            x = read_state_fa(r_prep, varnames=[vnam]).astype(np.float32).squeeze()
        elif vnam in r_pgd.listfields():
            x = read_state_fa(r_pgd, varnames=[vnam]).astype(np.float32).squeeze()
        else:
            raise ValueError(f"Variable {vnam} not found in either file.")
        x = x.astype(np.float32).squeeze()
        x = np.where(x == 1e20, np.nan, x)
        if vnam == "SFX.ZS":  # workaround for hardcoded minimum topography
            x = xr.where(x < -100, -99.9, x)
        ds[vnam] = (("y", "x"), x)
        ds[vnam].attrs.update(varmeta.get(vnam, {}))
    
    if domain in ["meps", "carra2"]:  # aggregate for 2 patch setups
        # Hacks for some variables
        ds["FOREST"] = xr.where(np.logical_and(np.isnan(ds["X002PATCH"]), ds["SFX.FRAC_NATURE"]>0), 0, ds["X002PATCH"])
        ds["FOREST"].attrs.update({
            "long_name": "Forest fraction",
            "comment": "Fraction of nature tile covered by forest X002PATCH and set to 0 where no forest",
            "units": "1",
        })
        ds["H_TREE"] = xr.where(np.logical_and(np.isnan(ds["X002PATCH"]), ds["SFX.FRAC_NATURE"]>0), 0, ds["X002H_TREE"])
        ds["H_TREE"].attrs.update({
            "long_name": "Tree height",
            "comment": "Mean canopy/tree height over grid cell X002H_TREE and set to 0 where no forest",
            "units": "m",
        })

        ds = ds.drop_vars(["X002PATCH", "X002H_TREE"])
    
    else:
        ds = ds.rename({"X001H_TREE": "H_TREE"})


    ds.attrs.update({
        "title": "Forcings from FA/PGD",
        "source_files": f"{prepfile}, {pgdfile}"
    })

    geo = get_geo(domain=domain)

    ds["x"] = geo["x"]
    ds["y"] = geo["y"]
    ds = xr.merge((ds, geo))
    #ds = ds.assign(
    #    longitude=(("y","x"), geo["longitude"].values),
    #    latitude=(("y","x"), geo["latitude"].values)
    #)
    
    ds.to_netcdf(output_filename)


def get_geo(domain):
    if domain == "meps":
        # read x and y and lon and lat from meps file
        meps_filepath = "https://thredds.met.no/thredds/dodsC/meps25epsarchive/2025/12/15/meps_det_2_5km_20251215T21Z.nc"
        geo = xr.open_dataset(meps_filepath)
    
    elif domain == "carra2":
        geo = geo_from_harmonie(
            lat0=90,
            lon0=-30,
            latc=84,
            lonc=-45,
            nx=2869,        
            ny=2869,
            dx=2500)
    elif domain == "carra_ne":
        geo = geo_from_harmonie(
            lat0=80,
            lon0=-34,
            latc=74,
            lonc=26,
            nx=789,        
            ny=989,
            dx=2500)
    return geo

def geo_from_harmonie(lat0, lon0, latc, lonc, nx, ny, dx, ezone=0):
    earth = 6.37122e6
    proj_type = "lcc" if lat0 < 90 else "stere"
    ellps = pyproj.CRS.from_string("EPSG:4326")
    proj2 = {
        "proj": proj_type,
        "lat_0": lat0,
        "lon_0": lon0,
        "R": earth,
        "no_defs": True,
        "units": "m",
    }
    if lat0 < 90:
        proj2["lat_1"] = lat0
        proj2["lat_2"] = lat0
    
    crs = pyproj.CRS.from_json_dict(proj2)
    p2 = pyproj.Transformer.from_crs(ellps, crs, always_xy=True)
    center = p2.transform(lonc, latc)
    ll = (center[0] - dx*(nx-1)/2, center[1] - dx*(ny-1)/2)
    x = ll[0] + np.arange(nx)*dx
    y = ll[1] + np.arange(ny)*dx
    p_inv = pyproj.Transformer.from_crs(crs, ellps, always_xy=True)
    X, Y = np.meshgrid(x, y)
    lon, lat = p_inv.transform(X, Y)
    ds = xr.Dataset(
        {
            "longitude": (("y", "x"), lon),
            "latitude": (("y", "x"), lat),
        },
        coords={
            "x": x,
            "y": y,
        },
    )
    ds["projection"] = xr.DataArray(1, attrs=crs.to_cf())
    return ds


if __name__ == "__main__":
    import sys

    domain = sys.argv[1]
    if domain == "meps":
        pgdfile = "/nobackup/prod1/cooper/harmonie/MEPS_prod/climate/METCOOP25D/PGD.fa"
        prepfile = glob.glob("/nobackup/prod1/cooper/harmonie/MEPS_prod/archive/*/*/*/*/mbr001/ICMSHHARM+0003.sfx")[-1]
        output_filename = "MEPS_raw_forcings.nc"
    elif domain == "carra2":
        pgdfile = "/hpcperm/fac2/carraclim/CARRA2_2500/Const.Clim.sfx"
        prepfile = glob.glob("/scratch/fac2/hm_home/carra2_201909/archive/*/*/*/*/ICMSHHARM+0003.sfx")[-1]
        output_filename = "/scratch/fab0/CARRA2_raw_forcings.nc"
    elif domain == "carra_ne":
        pgdfile = "/hpcperm/nhx/carraclim/CARRA_NE/Const.Clim.sfx"
        prepfile = glob.glob("/scratch/nhx/hm_home/carra_NE_TU/archive/*/*/*/*/ICMSHFULL+0003.sfx")[-1]
        output_filename = "/scratch/fab0/CARRA_NE_raw_forcings.nc"


    
    process(pgdfile, prepfile, output_filename, domain=domain)