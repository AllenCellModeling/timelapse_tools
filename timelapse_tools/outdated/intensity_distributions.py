#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from .computation_manager import ComputationManager


class IntensityDistributions(ComputationManager):
    """Find intensity distributions over the data cubes"""

    def __init__(self):
        self._T = None
        self.median_intensity_across_dim = None
        self.intens_bins = None
        self.intens_dist = None

    def process_data(self, data, dimensions, file_pointer, read_dims):
        # Pre-calculate and allocate the properties we'll be writing to
        if self._T is None:
            self._T = file_pointer.dims()["T"]
        if self.median_intensity_across_dim is None:
            z, y, x = data.shape[-3:]
            t_n = self._T[1] - self._T[0]
            self.median_intensity_across_dim = [np.zeros((p, t_n)) for p in (x, y, z)]
        if self.intens_bins is None:
            # Split the space into bins
            dtypeinfo = np.iinfo(data.flat[0].dtype)
            self.intens_bins = np.arange(int(dtypeinfo.min), int(dtypeinfo.max) + 1)
        if self.intens_dist is None:
            t_n = self._T[1] - self._T[0]
            self.intens_dist = np.zeros((t_n, len(self.intens_bins) - 1))
        # Find index of current time, not assuming starting at 0
        t_ind = [v for k, v in dimensions if k == "T"][0] - self._T[0]
        # Write median projection
        for axis, dim in enumerate(dimensions):
            if dim[0] not in "XYZ":
                continue
            take_over = [i for i in range(len(data.shape)) if i != axis]
            median_proj = np.median(data, take_over)
            out_arr = self.median_intensity_across_dim["XYZ".find(dim[0])]
            out_arr[:, t_ind] = median_proj
        # Write intensity distribution
        intens_hist, _ = np.histogram(data.flat, self.intens_bins)
        self.intens_dist[t_ind, :] = intens_hist

    def crop_intensity_distribution(self, distribution, bins, percentile=None):
        """From a dist[time, bins], crop bins at top and bottom that equal 0

        Parameters
        ----------
        distribution: np.array
            distribution of shape [time, bins]
        bins: np.array
            list of bin edges, of length distribution.shape[1]+1
        percentile: float
            percentile to use as cut off, defaults to 0

        Returns
        -------
        cropped_distribution: np.array
            distribution cropped to be [time, bins-n]
        cropped_bins: np.array
            bins cropped to match new distribution
        """
        if percentile is None:
            limit = 0
        else:
            limit = np.percentile(distribution.flat, percentile)
        # Find edges
        has_entries = (distribution > limit).any(0)
        first = has_entries.searchsorted(True)
        last = -has_entries[::-1].searchsorted(True)
        # Crop
        cropped_distribution = distribution[:, first:last]
        cropped_bins = bins[first:last]
        return cropped_distribution, cropped_bins
