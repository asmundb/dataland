"""Bounded source-to-target matches for spherical overlapping footprints."""

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree


def unit_vectors(lon, lat):
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    return np.column_stack((np.cos(lat) * np.cos(lon),
                            np.cos(lat) * np.sin(lon), np.sin(lat)))


class Neighbourhood:
    def __init__(self, grid, radius_m, earth_radius):
        self.chord = 2 * np.sin(radius_m / (2 * earth_radius))
        if grid.points is not None:
            lon, lat = grid.points.longitude, grid.points.latitude
            self.targets = grid.points.rows * len(grid.x) + grid.points.columns
        else:
            x, y = np.meshgrid(grid.x, grid.y)
            lon, lat = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True).transform(
                x.ravel(), y.ravel())
            self.targets = np.arange(len(lon))
        if not np.isfinite(lon).all() or not np.isfinite(lat).all():
            raise ValueError("Non-finite neighbourhood target coordinates")
        self.vectors = unit_vectors(lon, lat)
        self.tree = cKDTree(self.vectors)
        self.south = max(-90, np.min(lat) - np.rad2deg(radius_m / earth_radius))
        self.north = min(90, np.max(lat) + np.rad2deg(radius_m / earth_radius))

    def windows(self, lats, lons):
        # Latitude bounds are conservative for every great-circle cap, including
        # caps crossing the dateline or containing poles. Keep all longitudes.
        pad = abs(lats[1] - lats[0]) + 1e-7
        rows = np.flatnonzero((lats >= self.south - pad) & (lats <= self.north + pad))
        return [] if not len(rows) else [(slice(int(rows[0]), int(rows[-1]) + 1),
                                          slice(0, len(lons)))]

    def matches(self, lon, lat, max_pairs):
        """Yield integer source/target index pairs for one flattened source tile."""
        vectors = unit_vectors(lon, lat)
        centre = vectors.mean(axis=0)
        bound = np.max(np.linalg.norm(vectors - centre, axis=1)) + self.chord
        candidates = np.asarray(self.tree.query_ball_point(centre, bound + 1e-14), dtype=np.intp)
        if not len(candidates):
            return
        source_tree = cKDTree(vectors)
        # At most max_pairs matches even in a completely overlapping batch.
        target_batch = max(1, max_pairs // len(vectors))
        for start in range(0, len(candidates), target_batch):
            chosen = candidates[start:start + target_batch]
            matches = source_tree.query_ball_point(self.vectors[chosen], self.chord)
            lengths = np.array([len(match) for match in matches])
            if lengths.sum() == 0:
                continue
            source = np.concatenate([np.asarray(match, dtype=np.intp) for match in matches])
            target = np.repeat(self.targets[chosen], lengths)
            yield source, target
