import logging
import numpy as np
from pathlib import Path
import yaml
import shutil
from importlib import resources

import xarray as xr
from topo_descriptors import topo
from topo_descriptors.helpers import get_dem_netcdf, scale_to_pixel, to_netcdf


logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

DESCRIPTOR_FUNCS = {
    "tpi": topo.compute_tpi,
    "gradient": topo.compute_gradient,
    "std": topo.compute_std,
    "sx": topo.compute_sx,
    "valley_ridge": topo.compute_valley_ridge,
}

def make_dem_file(filename, meps_path, varname="SFX.ZS"):
    ds = get_dem_netcdf(filename)
    dem_ds = ds[[varname]]
    dem_ds.attrs.update(crs="epsg:")
    dem2_ds = xr.open_dataset(meps_path)
    dem_ds = dem_ds.update({"x": dem2_ds["x"], "y": dem2_ds["y"]})
    mask = ds["SFX.FRAC_NATURE"] == 0
    zs_field = np.where(mask, 0, dem_ds[varname].values)
    to_netcdf(zs_field, dem_ds, varname)
    return f"topo_{varname}.nc"


def compute_topo_descriptors(dem_filename, opt, tmp_dir="tmp"):
    dem_ds = get_dem_netcdf(dem_filename)

    tmpdir = Path(tmp_dir+"/")
    shutil.rmtree(tmpdir, ignore_errors=True)
    tmpdir.mkdir(exist_ok=True)

    default_scales = opt["scales"]
    descriptors = opt["descriptors"]

    for name, cfg in descriptors.items():
        # Determine scales (override or default)
        scales = cfg.get("scales", default_scales)

        # Common kwargs for all descriptors
        # Remove keys that are handled explicitly
        kw = {k: v for k, v in cfg.items() if k not in ("scales", "azimuths")}

        func = DESCRIPTOR_FUNCS[name]

        # Special-case descriptor: sx
        if name == "sx":
            azimuths = cfg.get("azimuths", [])
            for radius in scales:
                for az in azimuths:
                    func(
                        dem_ds,
                        radius=radius,
                        azimuth=az,
                        outdir=tmpdir,
                        **kw
                    )
        else:
            # Normal descriptor
            func(
                dem_ds,
                scales=scales,
                outdir=tmpdir,
                **kw
            )


def main(args):

    cfgfile = args.recipe
    if cfgfile is None:
        cfgfile = str(resources.files("dataland.config").joinpath("topo_descriptors.yml"))
    meps_filename = args.meps_filepath
    raw_filename = args.raw_filepath
    output_filename = args.output_filepath

    with open(cfgfile) as f:
        cfg = yaml.safe_load(f)
    
    for var_name, var_cfg in cfg["variables"].items():
        cfg["variables"][var_name] = {**cfg["default"], **var_cfg}
    
    meps_ds = xr.open_dataset(meps_filename)
    ds = xr.open_dataset(raw_filename)
    ds["x"] = meps_ds["x"]
    ds["y"] = meps_ds["y"]
    ds = ds.assign(
        longitude=(("y","x"), meps_ds["longitude"].values),
        latitude=(("y","x"), meps_ds["latitude"].values)
    )
    for variable in cfg["variables"]:
        dem_filename = make_dem_file(raw_filename, variable)
        opt = cfg["variables"][variable]
        compute_topo_descriptors(dem_filename, opt, tmp_dir=variable)
        ds1 = xr.open_mfdataset(str(f"{variable}/topo_*.nc"))
        ds1 = ds1.rename({v: f"{variable}_{v}" for v in ds1.data_vars})
        ds = xr.merge([ds1, ds])
        shutil.rmtree(variable)
    ds.to_netcdf(output_filename)
    
def get_args():
    import argparse

    default_args = {
        "recipe": None,
        "meps_filepath": "https://thredds.met.no/thredds/dodsC/meps25epsarchive/2025/12/15/meps_det_2_5km_20251215T21Z.nc",
        "raw_filepath": "/lustre/storeB/users/asmundb/dataset/decoder/MEPS_raw_forcings.nc",
        "output_filepath": "forcings_with_topo_descriptors.nc",
    }
    
    parser = argparse.ArgumentParser(description="Compute topo descriptors and merge with MEPS forcings")

    parser.add_argument(
        "--recipe",
        default=default_args["recipe"],
        help="Path to topo descriptor recipe YAML file"
    )

    parser.add_argument(
        "--meps-filepath",
        default=default_args["meps_filepath"],
        help="Path or URL to MEPS dataset"
    )

    parser.add_argument(
        "--raw-filepath",
        default=default_args["raw_filepath"],
        help="Path to raw MEPS forcings file"
    )

    parser.add_argument(
        "-o",
        "--output-filepath",
        default=default_args["output_filepath"],
        help="Output NetCDF file with topo descriptors added"
    )

    return parser.parse_args()
    


if __name__ == "__main__":
    
    main(get_args())