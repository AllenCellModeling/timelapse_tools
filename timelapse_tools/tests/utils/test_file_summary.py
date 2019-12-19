#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from timelapse_tools.utils import file_summary

###############################################################################


@pytest.mark.parametrize(
    "filename, expected_size",
    [
        ("s_1_t_5_c_1_z_1.czi", "5.9 MB"),
        ("s_None_t_5_c_1_z_None.czi", "22.3 MB"),
        pytest.param(
            "does_not_exist.txt",
            None,
            marks=pytest.mark.raises(exception=FileNotFoundError),
        ),
    ],
)
def test_file_size(data_dir, filename, expected_size):
    assert file_summary.file_size(data_dir / filename) == expected_size
