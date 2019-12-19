#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime

import pytest
from aicspylibczi import CziFile
from dateutil.tz import tzoffset

from timelapse_tools.utils import czi_metadata

###############################################################################


@pytest.mark.parametrize(
    "filename, expected_datetime",
    [
        (
            "s_1_t_5_c_1_z_1.czi",
            datetime(2019, 11, 6, 8, 10, 17, 757536, tzinfo=tzoffset(None, -28800)),
        ),
        (
            "s_None_t_5_c_1_z_None.czi",
            datetime(2019, 11, 21, 14, 5, 47, 684194, tzinfo=tzoffset(None, -28800)),
        ),
    ],
)
def test_datetime_created(data_dir, filename, expected_datetime):
    czi = CziFile(data_dir / filename)
    assert czi_metadata.datetime_created(czi) == expected_datetime


@pytest.mark.parametrize(
    "filename, expected_creator",
    [
        ("s_1_t_5_c_1_z_1.czi", "caroline.hookway"),
        ("s_None_t_5_c_1_z_None.czi", "jamieg"),
    ],
)
def test_collected_by(data_dir, filename, expected_creator):
    czi = CziFile(data_dir / filename)
    assert czi_metadata.created_by(czi) == expected_creator


@pytest.mark.parametrize(
    "filename, expected_channels",
    [("s_1_t_5_c_1_z_1.czi", ["EGFP"]), ("s_None_t_5_c_1_z_None.czi", ["Bright_2"])],
)
def test_channel_names(data_dir, filename, expected_channels):
    czi = CziFile(data_dir / filename)
    assert czi_metadata.channel_names(czi) == expected_channels
