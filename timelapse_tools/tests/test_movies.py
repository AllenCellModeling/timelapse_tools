#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest

###############################################################################


@pytest.fixture
def small_timelapse_czi(data_dir) -> Path:
    return (data_dir / "s_1_t_5_c_1_z_1.czi").resolve(strict=True)


@pytest.fixture
def small_timelapse_mp4(data_dir) -> Path:
    return (data_dir / "s_1_t_5_c_1_z_1.mp4").resolve(strict=True)


# @pytest.mark.parametrize()
# def test_generate_movie(tmpdir, small_timelapse_czi, small_timelapse_mp4)
