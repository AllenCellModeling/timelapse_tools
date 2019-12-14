#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import dask.array as da
import pytest

from timelapse_tools import conversion

###############################################################################


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent / "data"

###############################################################################


def test_img_prep(data_dir):
    img, dims = conversion._img_prep.run(data_dir / "s_1_t_5_c_1_z_1.czi", "T")
    print(img)
    print(img.shape)
    print(dims)
    assert isinstance(img, da.core.Array)
    assert isinstance(dims, str)
