#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest

from timelapse_tools import conversion

###############################################################################


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent / "data"

###############################################################################


@pytest.mark.parametrize("img, operating_dim", [
    ("s_1_t_5_c_1_z_1.czi", "T"),
    ("s_1_t_5_c_1_z_1.czi", "Z"),
    ("s_None_t_5_c_1_z_None.czi", "T"),
    pytest.param(
        "s_1_t_5_c_1_z_1.czi",
        "A",
        marks=pytest.mark.raises(exception=ValueError)
    ),
    pytest.param(
        "s_1_t_5_c_1_z_1.czi",
        "S",
        marks=pytest.mark.raises(exception=ValueError)
    ),
    pytest.param(
        "s_None_t_5_c_1_z_None.czi",
        "Z",
        marks=pytest.mark.raises(exception=ValueError)
    )
])
def test_img_prep(data_dir, img, operating_dim):
    img, dims = conversion._img_prep.run(data_dir / img, operating_dim)
