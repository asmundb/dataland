import numpy as np
import xarray as xr
from netCDF4 import Dataset as NetCDFDataset
import os

def read_surfex_dir(header_path, dir_path, output_path=None, chunk_size=1024,
                    var_name="orography", compression_level=4):
    """
    Reads a SURFEX model .dir binary file and its corresponding header file
    and writes a chunked NetCDF file without loading the full field into memory.
    
    Parameters:
        header_path (str): Path to the header file.
        dir_path (str): Path to the corresponding .dir binary file.
        output_path (str | None): Output NetCDF path. If None, uses dir_path + ".nc".
        chunk_size (int): Number of rows to read/write per chunk.
        var_name (str): Name of output data variable.
        compression_level (int): zlib compression level (0-9).
    
    Returns:
        xarray.Dataset: Lazily opened output dataset.
    """
    # Parse the header file
    header_info = {}
    with open(header_path, "r") as header_file:
        for line in header_file:
            if ":" in line:
                key, value = line.split(":", 1)
                header_info[key.strip().lower()] = value.strip()
            elif "resolution" in line.lower():
                header_info["resolution"] = line.split("resolution")[1].strip()

    
    # Extract values from the header
    nodata = int(header_info.get("nodata", -32768))
    north = float(header_info["north"])
    south = float(header_info["south"])
    west = float(header_info["west"])
    east = float(header_info["east"])
    rows = int(header_info["rows"])
    cols = int(header_info["cols"])
    record_type = header_info["recordtype"].split()[0].lower()  # e.g., "integer"
    bit_depth = int(header_info["recordtype"].split()[1])  # e.g., 16
    shape = (rows, cols)
    
    # Determine the NumPy dtype based on the recordtype and bit depth
    if record_type == "integer" and bit_depth == 16:
        dtype = np.dtype(np.int16)
    elif record_type == "integer" and bit_depth == 32:
        dtype = np.dtype(np.int32)
    elif record_type in ["float", "real"]:
        dtype = np.dtype(np.float32 if bit_depth == 32 else np.float64)
    else:
        raise ValueError(f"Unsupported record type: {record_type} with bit depth {bit_depth}")

    if chunk_size is None or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    if output_path is None:
        output_path = f"{dir_path}.nc"

    # Validate file size against header metadata.
    expected_values = rows * cols
    expected_bytes = expected_values * dtype.itemsize
    actual_bytes = os.path.getsize(dir_path)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Binary size mismatch for {dir_path}: got {actual_bytes} bytes, "
            f"expected {expected_bytes} bytes ({rows}x{cols}x{dtype.itemsize})."
        )

    # Generate latitude and longitude arrays
    latitudes = np.linspace(north, south, rows)
    longitudes = np.linspace(west, east, cols)

    # Stream binary chunks directly to chunked NetCDF.
    with open(dir_path, "rb") as src, NetCDFDataset(output_path, "w") as nc:
        nc.createDimension("latitude", rows)
        nc.createDimension("longitude", cols)

        lat_var = nc.createVariable("latitude", "f4", ("latitude",))
        lon_var = nc.createVariable("longitude", "f4", ("longitude",))
        lat_var[:] = latitudes.astype(np.float32)
        lon_var[:] = longitudes.astype(np.float32)
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"

        chunk_rows = min(int(chunk_size), rows)
        chunk_cols = min(cols, 2048)
        comp = int(max(0, min(9, compression_level)))
        encoding_kwargs = {
            "zlib": True,
            "complevel": comp,
            "shuffle": True,
            "chunksizes": (chunk_rows, chunk_cols),
            "fill_value": np.float32(np.nan),
        }
        data_var = nc.createVariable(
            var_name,
            "f4",
            ("latitude", "longitude"),
            **encoding_kwargs,
        )

        nc.description = "Processed SURFEX DIR/HDR field"
        nc.resolution = header_info.get("resolution", "unknown")
        nc.nodata_value = nodata
        data_var.nodata_value = nodata

        for row_start in range(0, rows, chunk_rows):
            row_end = min(row_start + chunk_rows, rows)
            this_rows = row_end - row_start
            count = this_rows * cols

            raw = np.fromfile(src, dtype=dtype, count=count)
            if raw.size != count:
                raise ValueError(
                    f"Unexpected EOF while reading {dir_path}: "
                    f"needed {count} values for rows {row_start}:{row_end}, got {raw.size}."
                )

            chunk = raw.reshape(this_rows, cols).astype(np.float32, copy=False)
            chunk[chunk == nodata] = np.nan
            data_var[row_start:row_end, :] = chunk

    return xr.open_dataset(output_path, chunks={})


if __name__ == "__main__":
    filepath = "/lustre/storeA/project/nwp/H2O/climate/GMTED2010_075.EHdr"
    ds = read_surfex_dir(
        filepath + ".hdr",
        filepath + ".dir",
        output_path="GMTED2010_075.EHdr.nc",
        chunk_size=1024,
    )
