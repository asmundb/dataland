"""Area-weighted statistics, independent of geometry, file I/O and output fills."""

import numpy as np


class Statistics:
    def __init__(self, size, mode, categories):
        self.mode = mode
        self.categories = categories
        self.source_area = np.zeros(size)
        self.valid_area = np.zeros(size)
        self.source_count = np.zeros(size, dtype=np.int64)
        self.valid_count = np.zeros(size, dtype=np.int64)
        names = (list(categories) if mode == "fractions" else
                 ["slope_x", "slope_y"] if mode == "slope" else ["mean"])
        self.totals = {name: np.zeros(size) for name in names}
        self.moments = np.zeros(size) if mode == "std" else None
        self.codes = None
        if mode == "fractions":
            self._validate_categories()

    def _validate_categories(self):
        if not self.categories or any(not codes for codes in self.categories.values()):
            raise ValueError("Categories must contain nonempty code lists")
        self.codes = np.array(sorted({code for group in self.categories.values() for code in group}))
        if not np.isfinite(self.codes).all() or np.any(self.codes != np.floor(self.codes)):
            raise ValueError("Category codes must be finite integers")
        reserved = {"source_area", "valid_area", "source_count", "valid_count", "crs",
                    "latitude", "longitude", "x", "y", "fractions_data_status"}
        if reserved.intersection(self.categories):
            raise ValueError("Category names collide with output diagnostics or coordinates")

    def add(self, target_indices, fields, weights):
        values = fields["slope_x"] if self.mode == "slope" else fields["mean"]
        valid = np.isfinite(values)
        if self.mode == "slope":
            valid &= np.isfinite(fields["slope_y"])
        if self.mode == "fractions":
            unknown = valid & ~np.isin(values, self.codes)
            if unknown.any():
                raise ValueError(f"Unknown cover codes inside target: {np.unique(values[unknown])[:20]}")

        occupied, inverse = np.unique(target_indices, return_inverse=True)
        size = len(occupied)
        self.source_area[occupied] += np.bincount(inverse, weights=weights, minlength=size)
        self.source_count[occupied] += np.bincount(inverse, minlength=size)
        bins = inverse[valid]
        valid_weights = weights[valid]
        areas = np.bincount(bins, weights=valid_weights, minlength=size)

        if self.mode == "std":
            if valid.any():
                self._merge_moments(occupied, bins, values[valid], valid_weights, areas)
        else:
            for name, total in self.totals.items():
                contribution = (np.isin(values[valid], self.categories[name])
                                if self.mode == "fractions" else fields[name][valid])
                total[occupied] += np.bincount(bins, weights=valid_weights * contribution,
                                              minlength=size)
        self.valid_area[occupied] += areas
        self.valid_count[occupied] += np.bincount(bins, minlength=size)

    def _merge_moments(self, occupied, bins, values, weights, areas):
        # Centre each group before summation, then merge centred moments.
        groups, first = np.unique(bins, return_index=True)
        origin = np.zeros(len(occupied))
        origin[groups] = values[first]
        shifted_sum = np.bincount(bins, weights=weights * (values - origin[bins]),
                                  minlength=len(occupied))
        means = origin[groups] + shifted_sum[groups] / areas[groups]
        centres = np.zeros(len(occupied))
        centres[groups] = means
        moment = np.bincount(bins, weights=weights * (values - centres[bins]) ** 2,
                             minlength=len(occupied))
        targets = occupied[groups]
        old_area = self.valid_area[targets]
        combined = old_area + areas[groups]
        delta = means - self.totals["mean"][targets]
        self.moments[targets] += moment[groups] + delta**2 * old_area * (areas[groups] / combined)
        self.totals["mean"][targets] += delta * (areas[groups] / combined)

    def finalize(self):
        """Return statistics without mutating accumulators or applying fills."""
        if self.mode == "std":
            fields = {
                "mean": self.totals["mean"].copy(),
                "std": np.sqrt(np.divide(self.moments, self.valid_area,
                                         out=np.zeros_like(self.valid_area),
                                         where=self.valid_area > 0)),
            }
        else:
            fields = {
                name: np.divide(total, self.valid_area, out=np.zeros_like(total),
                                where=self.valid_area > 0)
                for name, total in self.totals.items()
            }
        for name, values in fields.items():
            if not np.isfinite(values[self.valid_area > 0]).all():
                raise FloatingPointError(f"Non-finite {name} despite valid source area; check numeric overflow")
            values[self.valid_area == 0] = np.nan
        return fields
