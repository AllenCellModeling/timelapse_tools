#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest

from timelapse_tools import daread

###############################################################################


@pytest.mark.parametrize("img, expected_shape, expected_chunksize", [
    ("s_1_t_5_c_1_z_1.czi", (1, 5, 1, 1, 624, 924), (1, 1, 1, 1, 624, 924)),
    (Path("s_1_t_5_c_1_z_1.czi"), (1, 5, 1, 1, 624, 924), (1, 1, 1, 1, 624, 924)),
])
def test_daread(data_dir, img, expected_shape, expected_chunksize):
    pass
