import numpy as np
import epygram
import xarray as xr
import yaml
from importlib import resources
import glob


def read_state_fa(r, varnames=None):
    sle = []
    for j in range(len(varnames)):
        field = r.readfield(varnames[j])
        data = field.getdata(subzone='CI')
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)
        sle.append(data)
    return np.stack(sle, axis=-1)


pgdfile = "/nobackup/prod1/cooper/harmonie/MEPS_prod/climate/METCOOP25D/PGD.fa"
prepfile = glob.glob("/nobackup/prod1/cooper/harmonie/MEPS_prod/archive/*/*/*/*/mbr001/ICMSHHARM+0003.sfx")[-1]

# read x and y and lon and lat from meps file
meps_filepath = "https://thredds.met.no/thredds/dodsC/meps25epsarchive/2025/12/15/meps_det_2_5km_20251215T21Z.nc",
meps = xr.open_dataset(meps_filepath)

output_filename = "MEPS_raw_forcings.nc"

config_filename = str(resources.files("dataland.config").joinpath("FA_variables.yml"))

# Define variable metadata 
with open(config_filename, "r") as f:
    varmeta = yaml.safe_load(f)

r_prep = epygram.formats.resource(prepfile, openmode='r', fmt="FA")
r_pgd = epygram.formats.resource(pgdfile, openmode='r', fmt="FA")

fl = r_pgd.listfields()
for f in fl:
    if "ZS" in f:
        print(f)

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
    ds[vnam] = (("y", "x"), x)
    ds[vnam].attrs.update(varmeta.get(vnam, {}))


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
ds.attrs.update({
    "title": "Forcings from FA/PGD",
    "source_files": f"{prepfile}, {pgdfile}"
})

ds["x"] = meps["x"]
ds["y"] = meps["y"]
ds = ds.assign(
    longitude=(("y","x"), meps["longitude"].values),
    latitude=(("y","x"), meps["latitude"].values)
)


ds.to_netcdf(output_filename)
