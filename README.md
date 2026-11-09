# dataland
prepare and process land datasets


# Install

python3 -m -venv .venv
. .venv/bin/activate

pip install -e .

or with extras [fa], [anemoi]

## Plot file variables

Save all spatial variables, or selected variables, as individual PNG maps:

```sh
python -m dataland.plotting dataset.zarr --output-dir figures \
    --zoom -25 55 35 80
python -m dataland.plotting fields.nc --variable sea nature \
    --output-dir figures --zoom -25 55 35 80
python -m dataland.plotting dataset.zarr --variable t2m \
    --index time=3 --index member=0 --output-dir figures
```

Zoom order is **lon_min, lat_min, lon_max, lat_max**; longitude bounds must be
increasing within [-180, 180] (dateline-crossing zoom is not supported).
Plots use `plot_unstructured`, default scatter rendering and automatic colour
limits. Optional `--method trimesh`, `--cmap`, `--vmin`, and `--vmax` control
rendering. Non-finite points are excluded; empty regions raise an error.
Scatter maps use opaque, outline-free circles. Marker diameter is limited to
95% of the closest pair's spacing in display coordinates, checking all plotted
points. Zoomed maps use this size automatically; unzoomed maps request size 1
and apply the same limit. `--point-size 9` requests a marker area in points
squared, also capped to avoid overlap. Coincident plotted points raise an error.
Gaps are intentional: scatter circles cannot tile rotated cells without gaps
or overlaps. This is not a rendering of exact native cell boundaries.
Use `--background-color lightgrey` or `--background-color '#202020'` to set
both the map background (visible between dots) and the saved figure background.
Omitting it preserves Matplotlib's default background.

Anemoi Zarr and ordinary xarray Zarr are detected automatically; `--format
anemoi` or `--format xarray` can select explicitly. Anemoi support requires the
`anemoi` extra; xarray Zarr requires Zarr support in the environment. NetCDF
and xarray Zarr need identifiable latitude/longitude coordinates (paired points,
2-D coordinates, or independent 1-D latitude/longitude axes).
All-variable selection skips non-spatial metadata. Each extra dimension defaults
to index zero, logged for each variable; override using repeated
`--index DIM=INDEX` options. Anemoi uses `time` and `member` index names.
PNG names are derived from variable names; filename collisions raise errors.

## Gridding statistics

For `sg2fracs.py --domain_file`, source points are assigned to their nearest
output point on a sphere. A lookup table caches candidates, with an exact
nearest-neighbor search when the cached candidates cannot be certified.
This avoids gaps caused by assigning an entire lookup cell to one output point.

In fractions mode, output points that receive no source samples are reported
and saved as NaN, including when `--nan_to_num` is set. That option still
replaces non-finite source values before aggregation; for continuous modes it
also replaces non-finite output values.

### Regional area-weighted aggregation (new)

`aggregate_to_grid.py` is a separate regional implementation. It assigns each
source pixel's whole spherical area to the projected target cell containing its
centre. Fractions use valid contributing area, not sample counts; means use the
same area weights. This is **not exact area-overlap remapping**, especially at
coastlines and target-cell edges. The existing script remains available.

Use the existing domain geometry:

```sh
python -m dataland.aggregate_to_grid \
    --source-file data/input_data/ecosg_final_map.nc \
    --domain metcoop --output fractions_metcoop.nc
```

Or read the native projected geometry from a CF NetCDF file:

```sh
python -m dataland.aggregate_to_grid \
    --source-file data/input_data/ecosg_final_map.nc \
    --grid-file regional_grid.nc --grid-var surface_field \
    --memory-mb 128 --output fractions_regional.nc
```

The target NetCDF must contain:

- A projected CF `grid_mapping` variable (CF projection attributes, `crs_wkt`,
  or `spatial_ref`). Select `--grid-var` if the CRS is ambiguous.
  This may name the CRS variable itself (for example `--grid-var projection`)
  or a data variable referencing it through its `grid_mapping` attribute.
- Regular one-dimensional axes with standard names
  `projection_x_coordinate` and `projection_y_coordinate`, in metres or km.
  At least two centres per axis are required; cells extend half a spacing
  beyond the outer centres. Explicit bounds, if present, must agree.

Only projected metre-based CRSs are supported. Output preserves each axis's
ordering, including south-up grids. Latitude/longitude centres alone are
deliberately insufficient: the tool does not guess the projection or footprint.
Presets `metcoop` and `carra2` reproduce the existing definitions, not geometry
inferred from a model file; use a CF file when authoritative geometry is available.

#### Masked point subsets of a regional grid

If output coordinates are a flattened, possibly irregular subset, add
`--point-file` alongside the native reference `--grid-file` (or `--domain`).
The reference supplies exact projection, lattice alignment and spacing. The point
file supplies paired 1-D latitude/longitude in degrees, default variable names
`latitude` and `longitude`; override with `--point-latitude`/`--point-longitude`.

```sh
python -m dataland.aggregate_to_grid \
    --source-file data/input_data/ecosg_final_map.nc \
    --grid-file data/input_data/pred_collective-lama_step50000_CARRA_2023-03-19T00.nc \
    --grid-var projection \
    --point-file data/output_data/std_oro_carra2_2p5.nc \
    --output fractions_carra2_points.nc
```

Coordinates must align with reference centres within 1 metre (or 1% of spacing,
if smaller). Duplicates, non-finite coordinates, misalignment and points outside
the reference are errors. No native cell is inferred from nearest-point
footprints: unselected reference cells are masked, and their source pixels are
discarded rather than reassigned to neighbouring selected points.

Output fields use a `points` dimension in exactly the input point order, retaining
the supplied latitude/longitude and aligned native x/y coordinates. Fill warnings
and counts refer only to selected cells. Source-window pruning and accumulator
storage still use the full reference rectangle; the mask does not yet tighten
source windows. The point file is not modified.
Point outputs are constructed directly in input point order without generating
latitude/longitude arrays for the full reference rectangle.

If no authoritative x/y reference is available, supply an explicit projection
and known spacing instead. The tool fits the lattice offset from the projected
points and validates every point; it does not guess the CRS or spacing:

```sh
python -m dataland.aggregate_to_grid \
    --source-file data/input_data/ecosg_final_map.nc \
    --point-file data/output_data/std_oro_carra2_2p5.nc \
    --point-crs '+proj=stere +lat_0=90 +lon_0=-30 +R=6371229 +units=m' \
    --point-spacing 2500 --mode fractions --fill-uncovered 0 \
    --output fractions_carra2_points.nc
```

This explicitly selected spherical CRS fits the supplied CARRA points within
one metre. The `carra2` preset now uses the same sphere (radius 6,371,229 m),
replacing its previous WGS84 ellipsoid. This changes preset-generated geometry,
including callers of the legacy script using the shared preset.
As a shortcut, use `--domain carra2 --point-file ...` instead of `--point-crs`
and `--point-spacing`: with a point file, a domain supplies only projection and
spacing, while the lattice offset and bounding rectangle come from the points.
Without a point file, the preset centre and shape still define the full grid.
Agreement with a lattice validates alignment, not the scientific provenance
of a projection. At least two occupied positions on each axis are required.
Bounds and axes are derived from the fitted lattice; unselected cells remain
masked and output point ordering is preserved as above.

For a continuous mean:

```sh
python -m dataland.aggregate_to_grid \
    --source-file tree_height.nc --source-var H_TREE --mode mean \
    --grid-file regional_grid.nc --grid-var surface_field \
    --output tree_height_regional.nc
```

The source must be a two-dimensional variable on regular one-dimensional
latitude/longitude axes in degrees. Use `--latitude` and `--longitude` to
specify nonstandard coordinate names. Ascending/descending axes, wrapped
longitude coordinates, and transposed variables are supported. Pixel edges
come from CF bounds or half-spacing; latitude edges are clipped at the poles.
Source areas use a sphere of radius 6,371,229 m, independently of target CRS.
Global longitude closure checks allow coordinate-precision round-off (including
float32 DEM axes) and normalize valid global spacing to exactly 360 degrees.
Repeated seam columns or genuine coverage beyond one globe remain errors.

Missing values decoded by xarray, NaNs, and infinities are excluded from the
denominator, not replaced with zero. Unknown finite cover codes inside the
target fail explicitly. The ECOCLIMAP defaults cover codes 1--33; `forest`
and `glacier` are subsets of `nature`, not additional exclusive classes.
Custom categories can be supplied through the Python `aggregate` API.

For mean mode, two optional **output-only** fills handle different causes of
undefined means:

- `--fill-missing-mean 0`: source samples exist, but all their values are missing
  or non-finite.
- `--fill-uncovered 0` (also accepted as `--fill-uncovered-mean 0`): no source pixel centres contribute (for example
  beyond a DEM's latitude extent). This also includes sampling gaps; it is not
  proof that a cell lies outside the source footprint.

Use both flags to replace all undefined output means with zero. Missing source
values remain excluded: a cell containing valid values of 8 and missing values
still has mean 8, not a zero-diluted mean. Defaults remain NaN. Fill values must
be finite. `--fill-missing-mean` is rejected in fractions mode. Complete source/target non-overlap,
invalid grid metadata and numerical overflow remain errors, even with filling.

Warnings report counts and actions separately for both cases. Mean outputs
include `mean_data_status` (0 = valid mean, 1 = only missing source values,
2 = no source samples), whether filled or not. Counts, areas, fill settings
and affected-cell counts are preserved in the NetCDF so artificial zeros are
distinguishable from measured zeros.

`--fill-uncovered` also works in **fractions** mode, setting each category to
the specified value only in cells receiving no source samples. Fraction fills
must lie in [0, 1]; cells with source samples but only missing values remain NaN.
Filled category values are placeholders, not an inferred cover distribution:
zero-filling all categories does not sum to one. Warnings, `fractions_data_status`
(same 0/1/2 meanings), unchanged sample/area diagnostics, and fill metadata
identify these cells. The Python API accepts `fill_uncovered` in either mode;
`fill_uncovered_mean` remains a compatibility alias.

For example:

```sh
python -m dataland.aggregate_to_grid \
    --source-file data/input_data/new_ht_c_m.nc \
    --grid-file data/input_data/pred_collective-lama_step50000_CARRA_2023-03-19T00.nc \
    --grid-var projection --source-var H_TREE --mode mean \
    --fill-missing-mean 0 --fill-uncovered-mean 0 \
    --memory-mb 128 --output data/output_data/tree_height_carra2.nc
```

Output includes fractions (or `mean`), projected axes and CRS, 2-D latitude and
longitude, `source_area`, `valid_area` (m2), `source_count`, `valid_count`, and
method provenance. The source diagnostics include missing pixels; valid
diagnostics exclude them. These areas are sums of whole assigned source
pixels, **not target-cell areas or exact coverage fractions**. Cells with no
valid data remain NaN unless explicitly filled in mean mode, and produce a
warning; no intersecting source centres
is an error. Edges are half-open in the supplied axis order, with projection
round-off at exact edges stabilized.

Processing first selects conservative source windows for polar stereographic
and same-hemisphere Lambert conformal conic grids. Latitude bounds use exact
minimum/maximum distances from the target rectangle's outer edges to the
projection pole/apex, rather than a sampled geographic boundary. Polar grids
also restrict longitude using the rectangle's angular extent, splitting source
windows at longitude seams; grids containing the pole retain all longitudes.
Lambert grids conservatively retain all longitudes within the latitude band.
One source spacing pads bounds for numerical round-off. Other projections
use an explicitly logged full-source scan.

Within those windows, processing uses bounded two-dimensional tiles and skips
reading source data for tiles entirely outside the domain. The projected
containment check remains authoritative; source-window bounds only reduce work.
`--memory-mb` is an approximate tile workspace budget. A tile-level progress bar
shows completion and ETA for selected tiles, including selected tiles skipped
outside the target. Logs report the selected source pixel count and percentage.
The memory budget is not a process RAM limit: output accumulators and backend I/O
caches are additional. Default fractions need approximately 96 bytes per target
cell for accumulators and output geographic coordinates, plus temporary arrays.
No global source mesh or full source-to-target LUT is allocated.

Window selection for the supplied 1638 x 1246 CARRA NetCDF geometry and global
64800 x 129600 ECOCLIMAP axes reduced the scan from 8.40 billion pixels to
469 million (5.59%). This is a pixel-count measurement, not a runtime benchmark;
full-size processing time and peak RAM have not been measured. Pruned fractions,
means, areas and counts are tested against full-source scans.

### Standard deviation and terrain slope

`--mode std` writes `std` and `mean` in source units. Standard deviation is the
**area-weighted population** value, not the legacy sample estimator. A cell
with one valid sample has standard deviation zero. Weighted centred moments
are merged across tiles to avoid cancellation from large elevation offsets.

`--mode slope` writes `slope_x` (eastward) and `slope_y` (northward), in m/m.
These are area-weighted means of source-resolution elevation derivatives, not
derivatives of the coarse mean, nor slope magnitude. Source elevations are
interpreted as metres; unit metadata is not currently enforced. Geographic distances use the same spherical radius as
area weighting. Central differences use neighbour halos in both dimensions,
including across cropped windows, and periodic longitude neighbours on global
grids. Physical source edges use one-sided differences. Missing centre/stencil
values invalidate derivatives; undefined polar eastward gradients are excluded.
Both components share a validity mask requiring both derivatives to be finite.

```sh
python -m dataland.aggregate_to_grid \
    --source-file dem.nc --source-var orography --mode std \
    --domain carra2 --point-file output_points.nc --output dem_std.nc
python -m dataland.aggregate_to_grid \
    --source-file dem.nc --source-var orography --mode slope \
    --domain carra2 --point-file output_points.nc --output dem_slope.nc
```

Both modes support `--fill-uncovered` and native point subsets. Cells with
samples but no valid statistic stay NaN; `--fill-missing-mean` remains exclusive
to mean mode. `std_data_status` and `slope_data_status` identify valid, missing,
and uncovered cells using the same flag convention as mean mode. Neither
mode implements global Gaussian grids.

### Overlapping neighbourhoods

For std and slope, `--radius-m 15000` selects an overlapping **15 km radius**
around each output centre. Omitting the option preserves native-cell
aggregation. Distances are great-circle metres on the 6,371,229 m weighting
sphere, not distances in the target projection. Source pixels contribute their
whole area if their centres lie within the radius; no fractional circle/pixel
overlaps are calculated.

```sh
python -m dataland.aggregate_to_grid \
    --source-file dem.nc --source-var orography --mode std \
    --domain carra2 --point-file output_points.nc \
    --radius-m 15000 --output dem_std_30km_diameter.nc
python -m dataland.aggregate_to_grid \
    --source-file dem.nc --source-var orography --mode slope \
    --domain carra2 --point-file output_points.nc \
    --radius-m 50000 --output dem_slope_100km_diameter.nc
```

Neighbourhoods include source pixels beyond the native target boundary and may
overlap masked-out native cells: the point mask selects **centres**, not the
source support for a circle. Std remains area-weighted population std plus mean;
slope remains the mean of signed source gradient components, so opposing slopes
can cancel. Missing data, status flags and output fills follow the native path.
Radius and distance semantics are saved in output metadata.

The matching engine lives in `neighbourhood.py`. It prunes source latitude to
cover all circles (keeping all longitudes for safe pole/dateline handling),
rejects tiles using a conservative 3-D chord-distance bound, and batches
source-target matches with at most one tile's pixel budget of pairs per batch.
Each intersecting tile is read/derived once and reused across match batches.
Output accumulators and spatial trees are additional to the tile memory budget.
Large circles can still be expensive because each source pixel contributes
to many targets; no full-resolution neighbourhood runtime benchmark is available.

### Aggregation code organization

The `metcoop` projection uses an explicit `R=6371000` metre sphere, matching
the older MetCoOp point coordinates. Its former lowercase `r` parameter was
not interpreted as the PROJ sphere radius and resolved to WGS84. This correction
also changes full-preset geometry in legacy callers using the shared defaults.
Source-pixel area weighting remains on the separate 6,371,229 metre sphere.

- `aggregate_to_grid.py`: CLI, option validation and tile-loop orchestration.
- `aggregation_grid.py`: target geometry, CF loading, point-lattice fitting,
  source windows and native-cell matching.
- `aggregation_source.py`: geographic axis normalization, source areas,
  tile reads and slope halos.
- `aggregation_statistics.py`: weighted fractions, means and centred variance
  accumulation; no file, projection or output-fill logic.
- `aggregation_output.py`: missing-data warnings, explicit fills, diagnostics,
  provenance and rectangular/point output construction.
- `neighbourhood.py`: spherical radius matching, without progress or source I/O.

Both matchers yield integer `(source_indices, target_indices)` pairs. The
orchestrator reads each intersecting source tile once, reuses it across match
batches, and owns progress updates. Existing CLI options and numerical definitions
are retained; compatibility names are normalized before processing.
Source-window fallback and neighbourhood batching behavior are unchanged by this
refactoring. Accumulators still use the reference rectangle, including for points.

Run its tests with:

```sh
python -m unittest discover -s tests -p 'test_aggregate_to_grid.py'
python -m unittest discover -s tests -p 'test_aggregation_components.py'
```
