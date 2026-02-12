import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster
import scipy.spatial.distance as ssd
from sklearn.metrics.pairwise import cosine_similarity


# find similarity of forcings

preselect = {
    "FOREST",
    "H_TREE",
    "SFX.AVG_ZS",
    "SFX.CLAY",
    "SFX.COVER006",
    "SFX.FRAC_NATURE",
    "SFX.FRAC_NATURE_TPI_12500M",
    "SFX.FRAC_SEA",
    "SFX.FRAC_TOWN",
    "SFX.FRAC_WATER",
    "SFX.MAX_ZS",
    "SFX.MIN_ZS",
    "SFX.SAND",
    "SFX.SIL_ZS",
    "SFX.SSO_ANIS",
    "SFX.SSO_STDEV",
    "SFX.ZS_SN_DERIVATIVE_12500M_SIGRATIO1",
    "SFX.ZS_STD_17500M",
    "SFX.ZS_SX_RADIUS12500_AZIMUTH180",
    "SFX.ZS_TPI_12500M",
    "SFX.ZS_VALLEY_NORM_12500M_x",
    "SFX.ZS_VALLEY_NORM_12500M_y",
    "SFX.ZS_WE_DERIVATIVE_12500M_SIGRATIO1",
    "subgrid_slope_x",
    "subgrid_slope_y",
}


ds = xr.open_dataset("forcings_with_topo_descriptors.nc").isel(time=0).drop_vars(("longitude","latitude","projection_lambert"))[preselect]

da = ds.stack(points=("x","y")).to_array().transpose("points","variable")
seamask = ds["SFX.FRAC_SEA"].stack(points=("x","y")) == 0
varnames = np.array(ds.data_vars)


corrmat = np.corrcoef(da, rowvar=False)
corrmat_m = np.corrcoef(da[seamask], rowvar=False)

cos_sim = cosine_similarity(da.values.T)
cos_sim_m = cosine_similarity(da[seamask].values.T)

fig, ax = plt.subplots(ncols=2,nrows=2, sharex="all", sharey="all")
ax[0,0].set_title("corr")
ax[0,1].set_title("cos_sim")
ax[1,0].set_title("corr mask")
ax[1,1].set_title("cos_sim mask")
ax[0,0].imshow(corrmat  , vmin=-1, vmax=1, cmap="RdBu_r")
ax[0,1].imshow(cos_sim  , vmin=-1, vmax=1, cmap="RdBu_r")
ax[1,0].imshow(corrmat_m, vmin=-1, vmax=1, cmap="RdBu_r")
ax[1,1].imshow(cos_sim_m, vmin=-1, vmax=1, cmap="RdBu_r")
plt.show()


cmap = "hot"
fig, ax = plt.subplots(ncols=3, sharex="all", sharey="all")
ax[0].set_title("corr")
ax[1].set_title("cos_sim")
ax[2].set_title("both")
ax[0].imshow(1 - np.abs(corrmat), vmin=0, vmax=1, cmap=cmap)
ax[1].imshow(1 - np.abs(cos_sim), vmin=0, vmax=1, cmap=cmap)
ax[2].imshow(2 - np.sqrt(corrmat**2 + cos_sim**2), vmin=0, vmax=2, cmap=cmap)
plt.show()


def cluster(distmat):
    # enforce symmetry explicitly
    distmat = (distmat + distmat.T) / 2
    # enforce zero diagonal
    np.fill_diagonal(distmat, 0)
    dist = ssd.squareform(distmat)  # distance from correlation
    Z = linkage(dist, method="average")
    clusters = fcluster(Z, t=0.3, criterion="distance")
    groups = np.unique(clusters)
    group_dict = {}
    for g in groups:
        group_dict[g.item()] = varnames[np.where(clusters==g)].tolist()
    return group_dict



distmat = 1 - np.abs(corrmat)

g1 = cluster(distmat)
ga_m = cluster(1 - np.abs(np.nan_to_num(corrmat_m, nan=0)))
g2 = cluster(1 - np.abs(cos_sim))
g3 = cluster(2 - np.sqrt(corrmat**2 + cos_sim**2))
plt.imshow(2 - np.sqrt(corrmat**2 + cos_sim**2))
plt.show()

for g in ga_m:
    if len(ga_m[g]) > 1:
        print(g)
        corrs = {}
        for v in ga_m[g]:
            r = []
            for v1 in ga_m[g]:
                if v == v1:
                    continue
                r.append(xr.corr(ds[v], ds[v1]).values.item())
            corrs[v] = r
        
        for v in corrs:
            print(f"{v}: {np.mean(np.abs(corrs[v]))}")

        fig, ax = plt.subplots(ncols=len(ga_m[g]))
        [ax[i].imshow(ds[v], origin="lower") for i, v in enumerate(ga_m[g])]
        plt.show()


lines = ["Group\tVariables"]
for g, vars_ in ga_m.items():
    #lines.append(f"{g}; {', '.join(vars_)}")
    lines.append(", ".join(vars_))

print("\n".join(lines))


