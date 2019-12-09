#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pytest
from aicspylibczi import CziFile
from imageio import imread

from timelapse_tools import projection

###############################################################################


@pytest.fixture
def small_timelapse_czi(data_dir) -> Path:
    return (data_dir / "s_1_t_5_c_1_z_1.czi").resolve(strict=True)


@pytest.fixture
def single_timepoint_png(data_dir) -> Path:
    return (data_dir / "s_1_t_1_c_1_z_1.png").resolve(strict=True)


@pytest.fixture
def single_timepoint_all_axes_png(data_dir) -> Path:
    return (data_dir / "s_1_t_1_c_1_z_1_all_axes.png").resolve(strict=True)


def test_im2proj_valid(small_timelapse_czi, single_timepoint_png):
    # Read CZI
    czi = CziFile(small_timelapse_czi)
    data, shape = czi.read_image(B=0, S=0, T=0)

    # Generate projection
    proj = projection.im2proj(data=data, shape=shape)

    # Read expected projection
    expected = imread(single_timepoint_png)

    # Compare
    assert np.array_equal(proj, expected)


@pytest.mark.parametrize(
    "inject_shape, project_axis",
    [
        # fails because dimension D not found in shape
        pytest.param(
            [("S", 1), ("T", 1), ("C", 1), ("Z", 1), ("Y", 624), ("X", 924)],
            "D",
            marks=pytest.mark.raises(exception=ValueError),
        ),
        # fails because data.shape is a different length than shape
        pytest.param(
            [
                ("A", 1),
                ("B", 1),
                ("S", 1),
                ("T", 1),
                ("C", 1),
                ("Z", 1),
                ("Y", 624),
                ("X", 924),
            ],
            "Z",
            marks=pytest.mark.raises(exception=IndexError),
        ),
    ],
)
def test_im2proj_exceptions(small_timelapse_czi, inject_shape, project_axis):
    # Read CZI
    czi = CziFile(small_timelapse_czi)
    data, _ = czi.read_image(B=0, S=0, T=0)

    # Generate projection
    projection.im2proj(data=data, shape=inject_shape, project_axis=project_axis)


def test_im2proj_all_axes_valid(small_timelapse_czi, single_timepoint_all_axes_png):
    # Read CZI
    czi = CziFile(small_timelapse_czi)
    data, shape = czi.read_image(B=0, S=0, T=0)

    # Generate projection
    proj = projection.im2proj_all_axes(data=data, shape=shape)

    # Read expected projection
    expected = imread(single_timepoint_all_axes_png)

    # Compare
    assert np.array_equal(proj, expected)
