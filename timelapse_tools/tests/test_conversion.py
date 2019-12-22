#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from imageio import mimread

from timelapse_tools import conversion
from timelapse_tools.constants import Dimensions

###############################################################################


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent / "data"


select_all = slice(None, None, None)

###############################################################################


@pytest.mark.parametrize(
    "save_path, overwrite, fname, expected",
    [
        (Path(__file__) / "test_dir", False, None, Path(__file__) / "test_dir"),
        (Path(__file__).parent, True, None, Path(__file__).parent),
        pytest.param(
            Path(__file__).parent,
            False,
            None,
            None,
            marks=pytest.mark.raises(exception=FileExistsError),
        ),
        pytest.param(
            Path(__file__),
            False,
            None,
            None,
            marks=pytest.mark.raises(exception=FileExistsError),
        ),
    ],
)
def test_get_save_path(save_path, overwrite, fname, expected):
    actual = conversion._get_save_path.run(save_path, overwrite, fname)
    assert actual == expected


@pytest.mark.parametrize(
    "img, operating_dim",
    [
        ("s_1_t_5_c_1_z_1.czi", "T"),
        ("s_1_t_5_c_1_z_1.czi", "Z"),
        ("s_None_t_5_c_1_z_None.czi", "T"),
        pytest.param(
            "s_1_t_5_c_1_z_1.czi", "A", marks=pytest.mark.raises(exception=ValueError)
        ),
        pytest.param(
            "s_1_t_5_c_1_z_1.czi", "S", marks=pytest.mark.raises(exception=ValueError)
        ),
        pytest.param(
            "s_None_t_5_c_1_z_None.czi",
            "Z",
            marks=pytest.mark.raises(exception=ValueError),
        ),
    ],
)
def test_img_prep(data_dir, img, operating_dim):
    img, dims = conversion._img_prep.run(data_dir / img, operating_dim)


@pytest.mark.parametrize(
    "img, dims, dim_name, dim_indicies_selected, expected_shape, expected_dims",
    [
        (da.ones((1, 2, 3, 4, 5)), "SCZYX", Dimensions.Scene, 0, (2, 3, 4, 5), "CZYX"),
        (
            da.ones((1, 2, 3, 4, 5)),
            "SCZYX",
            Dimensions.Scene,
            slice(0, 1, 1),
            (1, 2, 3, 4, 5),
            "SCZYX",
        ),
        (
            da.ones((1, 2, 3, 4, 5)),
            "SCZYX",
            Dimensions.Channel,
            0,
            (1, 3, 4, 5),
            "SZYX",
        ),
        (
            da.ones((1, 2, 3, 4, 5)),
            "SCZYX",
            Dimensions.Channel,
            slice(0, 1, 1),
            (1, 1, 3, 4, 5),
            "SCZYX",
        ),
        (da.ones((1, 1)), "YX", Dimensions.Channel, 0, (1, 1), "YX"),
        (da.ones((1, 1)), "YX", Dimensions.Channel, slice(0, 1, 1), (1, 1), "YX"),
        pytest.param(
            da.ones((1, 1, 1)),
            "CYX",
            Dimensions.Channel,
            "Wrong selector type",
            None,
            None,
            marks=pytest.mark.raises(exception=TypeError),
        ),
    ],
)
def test_select_dimension(
    img, dims, dim_name, dim_indicies_selected, expected_shape, expected_dims
):
    img, dims = conversion._select_dimension.run(
        img, dims, dim_name, dim_indicies_selected
    )
    assert img.shape == expected_shape
    assert dims == expected_dims


@pytest.mark.parametrize(
    "img, expected_shape",
    [(da.ones((1, 2, 3)), (1, 2, 3)), (da.ones((1, 2, 3, 4, 5)), (1, 2, 3, 4, 5))],
)
def test_get_image_shape(img, expected_shape):
    assert conversion._get_image_shape.run(img) == expected_shape


@pytest.mark.parametrize(
    "img_shape, dims, expected_getitem_indicies",
    [
        (
            (1, 2, 3, 4, 5, 6),
            "STCZYX",
            [
                (0, select_all, 0, select_all, select_all, select_all),
                (0, select_all, 1, select_all, select_all, select_all),
                (0, select_all, 2, select_all, select_all, select_all),
            ],
        ),
        (
            (3, 2, 1, 4, 5, 6),
            "STCZYX",
            [
                (0, select_all, 0, select_all, select_all, select_all),
                (1, select_all, 0, select_all, select_all, select_all),
                (2, select_all, 0, select_all, select_all, select_all),
            ],
        ),
        (
            (3, 2, 3, 4, 5),
            "STZYX",
            [
                (0, select_all, select_all, select_all, select_all),
                (1, select_all, select_all, select_all, select_all),
                (2, select_all, select_all, select_all, select_all),
            ],
        ),
        (
            (3, 3, 3, 4, 5),
            "TCZYX",
            [
                (select_all, 0, select_all, select_all, select_all),
                (select_all, 1, select_all, select_all, select_all),
                (select_all, 2, select_all, select_all, select_all),
            ],
        ),
        ((1, 2, 3, 4), "TZYX", [(select_all, select_all, select_all, select_all)]),
    ],
)
def test_generate_getitem_indicies(img_shape, dims, expected_getitem_indicies):
    actual = conversion._generate_getitem_indicies.run(img_shape, dims)
    assert actual == expected_getitem_indicies


@pytest.mark.parametrize(
    "img, getitem_indicies, expected_process_list",
    [
        (
            da.ones((1, 2, 3, 4)),
            [(select_all, select_all, select_all, select_all)],
            [da.ones((1, 2, 3, 4))],
        ),
        (
            da.ones((3, 2, 3, 4, 5)),
            [
                (0, select_all, select_all, select_all, select_all),
                (1, select_all, select_all, select_all, select_all),
            ],
            [da.ones((2, 3, 4, 5)), da.ones((2, 3, 4, 5)), da.ones((2, 3, 4, 5))],
        ),
        (
            da.ones((2, 4, 2, 4, 4, 4)),
            [
                (0, select_all, 0, select_all, select_all, select_all),
                (1, select_all, 0, select_all, select_all, select_all),
                (0, select_all, 1, select_all, select_all, select_all),
                (1, select_all, 1, select_all, select_all, select_all),
            ],
            [
                da.ones((4, 4, 4, 4)),
                da.ones((4, 4, 4, 4)),
                da.ones((4, 4, 4, 4)),
                da.ones((4, 4, 4, 4)),
            ],
        ),
    ],
)
def test_generate_process_list(img, getitem_indicies, expected_process_list):
    actual = conversion._generate_process_list.run(img, getitem_indicies)
    for to_process, expected in zip(actual, expected_process_list):
        assert np.array_equal(to_process.compute(), expected.compute())


@pytest.mark.parametrize(
    "dims, getitem_indicies, expected",
    [
        (
            "STCZYX",
            [
                (0, select_all, 0, select_all, select_all, select_all),
                (1, select_all, 0, select_all, select_all, select_all),
                (0, select_all, 1, select_all, select_all, select_all),
                (1, select_all, 1, select_all, select_all, select_all),
            ],
            [{"S": 0, "C": 0}, {"S": 1, "C": 0}, {"S": 0, "C": 1}, {"S": 1, "C": 1}],
        ),
        (
            "STZYX",
            [
                (0, select_all, select_all, select_all, select_all),
                (1, select_all, select_all, select_all, select_all),
            ],
            [{"S": 0}, {"S": 1}],
        ),
        (
            "TCZYX",
            [
                (select_all, 0, select_all, select_all, select_all),
                (select_all, 1, select_all, select_all, select_all),
            ],
            [{"C": 0}, {"C": 1}],
        ),
        ("TZYX", [(select_all, select_all, select_all, select_all)], [{}]),
    ],
)
def test_generate_selected_dims_list(dims, getitem_indicies, expected):
    actual = conversion._generate_selected_dims_list.run(dims, getitem_indicies)
    assert actual == expected


@pytest.mark.parametrize(
    "img, expected",
    [
        ("s_1_t_5_c_1_z_1.czi", "s_1_t_5_c_1_z_1.mp4"),
        ("s_None_t_5_c_1_z_None.czi", "s_None_t_5_c_1_z_None.mp4"),
    ],
)
def test_generate_movies(data_dir, tmpdir, img, expected):
    # Generate movies
    save_dir = conversion.generate_movies(
        data_dir / img, save_path=tmpdir, overwrite=True, fps=1, quality=10
    )

    # There _should_ only be one file produced from this, select it
    produced_files = [f for f in save_dir.iterdir()]
    assert len(produced_files) == 1

    # Read the only one and compare with expected
    actual = np.stack(mimread(produced_files[0]))
    expected = np.stack(mimread(data_dir / expected))

    # This entire test is basically just "does the workflow run."
    # All other functions are tested above so this is really just
    # testing imageio, ffmpeg, and whichever callables the user provides.
    assert actual.shape == expected.shape
