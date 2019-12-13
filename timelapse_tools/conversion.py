#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from itertools import product
from pathlib import Path
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Tuple,
                    Union)

import dask.array as da
from aicspylibczi import CziFile
from prefect import Flow, task, unmapped

from .constants import AVAILABLE_OPERATING_DIMENSIONS, Dimensions
from .normalization.single_channel_percentile_norm import percentile_norm
from .projection.single_channel_max_project import single_channel_max_project
from .utils.czi_reading import daread

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class ImageDetails(NamedTuple):
    data: da.core.Array
    dims: str

###############################################################################


@task
def _generate_movie(
    data: da.core.Array,
    dims: str,
    operating_dim: str,
    save_path: Path,
    normalization_func: Callable,
    normalization_kwargs: Dict[str, Any],
    projection_func: Callable,
    projection_kwargs: Dict[str, Any]
) -> da.core.Array:
    pass


@task
def _select_dimension(
    img: da.core.Array,
    dims: str,
    dim_name: str,
    dim_indicies_selected: Optional[Union[int, slice]] = None
) -> ImageDetails:
    # Select which dimensions to process
    if dim_name in dims:
        # If specific dimension indicies were provided select them
        if dim_indicies_selected is not None:
            if not isinstance(dim_indicies_selected, (int, slice)):
                raise TypeError(
                    f"{dim_name} selection may only be done by providing either an "
                    f"integer or slice. Received: {type(dim_indicies_selected)}."
                )

            # Generate operations required to select the data
            ops = []
            for dim in dims:
                if dim is dim_name:
                    ops.append(dim_indicies_selected)
                else:
                    ops.append(slice(None, None, None))

            # Select the specified data
            img = img[tuple(ops)]

            # Remove the dimension from the dims if a specific integer was requested
            if isinstance(dim_indicies_selected, int):
                dims = dims.replace(dim_name, "")

    # Dimension not present dims
    else:
        if dim_indicies_selected is not None:
            log.warn(
                f"Ignoring the specified {dim_name} dimension(s) "
                f"({dim_indicies_selected}) as it was not found in the file."
            )

    return ImageDetails(data=img, dims=dims)


@task
def _generate_getitem_indicies(
    img_shape: tuple,
    dims: str
) -> List[Tuple[Union[int, slice]]]:
    # Generate getitem ops to process for each scene x channel pair
    if Dimensions.Scene in dims and Dimensions.Channel in dims:
        sc_indicies = list(product(
            range(img_shape[dims.index(Dimensions.Scene)]),
            range(img_shape[dims.index(Dimensions.Channel)])
        ))
        getitem_indicies = []
        for sc_index_pair in sc_indicies:
            this_pair_getitem_indices = []
            for dim in dims:
                if dim is Dimensions.Scene:
                    this_pair_getitem_indices.append(sc_index_pair[0])
                elif dim is Dimensions.Channel:
                    this_pair_getitem_indices.append(sc_index_pair[1])
                else:
                    this_pair_getitem_indices.append(slice(None, None, None))

            getitem_indicies.append(tuple(this_pair_getitem_indices))

    # Generate getitem ops to process for each scene
    elif Dimensions.Scene in dims:
        s_indicies = list(range(img_shape[dims.index(Dimensions.Scene)]))
        getitem_indicies = []
        for s_index in s_indicies:
            this_index_getitem_indices = []
            for dim in dims:
                if dim is Dimensions.Scene:
                    this_index_getitem_indices.append(s_index)
                else:
                    this_index_getitem_indices.append(slice(None, None, None))

            getitem_indicies.append(tuple(this_index_getitem_indices))

    # Generate getitem ops to process for each channel
    elif Dimensions.Channel in dims:
        c_indicies = list(range(img_shape[dims.index(Dimensions.Channel)]))
        getitem_indicies = []
        for c_index in c_indicies:
            this_index_getitem_indices = []
            for dim in dims:
                if dim is Dimensions.Channel:
                    this_index_getitem_indices.append(c_index)
                else:
                    this_index_getitem_indices.append(slice(None, None, None))

            getitem_indicies.append(tuple(this_index_getitem_indices))

    # Just pass through a list of a single getitem all
    else:
        getitem_indicies = [tuple([slice(None, None, None) for dim in dims])]

    return getitem_indicies


@task
def _img_prep(img: Union[str, Path, CziFile], operating_dim: str) -> ImageDetails:
    # Convert to dask.array
    img, dims = daread(img)

    # Get valid operating dimensions for this image by using set intersection
    valid_op_dims = set([d for d in dims]) & AVAILABLE_OPERATING_DIMENSIONS

    # Check that the operating dim provided is valid for this image
    if operating_dim not in valid_op_dims:
        raise ValueError(
            f"Invalid operating dimension provided. "
            f"Provided operating dimension: '{operating_dim}'. "
            f"Valid operating dimensions for this image: {valid_op_dims}."
        )

    return ImageDetails(data=img, dims=dims)


def convert_to_mp4(
    img: Union[str, Path, CziFile],
    distributed: bool = False,
    distributed_executor_port: Union[str, int] = 8888,
    save_path: Optional[Union[str, Path]] = None,
    normalization_func: Callable = percentile_norm,
    normalization_kwargs: Dict[str, Any] = {},
    projection_func: Callable = single_channel_max_project,
    projection_kwargs: Dict[str, Any] = {},
    operating_dim: str = "T",
    S: Optional[Union[int, slice]] = None,
    C: Optional[Union[int, slice]] = None,
) -> Path:
    if distributed:
        from prefect.engine.executors import DaskExecutor
        executor = DaskExecutor(address=f"tcp://localhost:{distributed_executor_port}")
    else:
        from prefect.engine.executors import LocalExecutor
        executor = LocalExecutor()

    # Run all processing through prefect + dask for better
    # parallelization and task optimization
    with Flow("czi_to_mp4_conversion") as flow:
        # Setup and check image and operating dimension provided
        img_details = _img_prep(img=img, operating_dim=operating_dim)

        # TODO:
        # Determine save path

        # Select scene data
        img_details = _select_dimension(
            img=img_details.data,
            dims=img_details.dims,
            dim_name=Dimensions.Scene,
            dim_indicies_selected=S
        )
        # Select channel data
        img_details = _select_dimension(
            img=img_details.data,
            dims=img_details.dims,
            dim_name=Dimensions.Channel,
            dim_indicies_selected=C
        )

        # Generate all the indicie sets we will need to process
        getitem_indicies = _generate_getitem_indicies(
            img_shape=img_details.data.shape,
            dims=img_details.dims
        )

        # Generate all the movie selections
        movies = [
            img_details.data[getitem_selection]
            for getitem_selection in getitem_indicies
        ]

        # Generate movies for each
        _generate_movie.map(
            data=movies,
            dims=unmapped(img_details.dims),
            operating_dim=unmapped(operating_dim),
            save_path=unmapped(save_path),
            normalization_func=unmapped(normalization_func),
            normalization_kwargs=unmapped(normalization_kwargs),
            projection_func=unmapped(projection_func),
            projection_kwargs=unmapped(projection_kwargs)
        )

    # Run the flow
    state = flow.run(executor=executor)

    # Get resulting path(s)
    save_path = state.result[flow.get_tasks(name="_get_save_path")[0]].result

    return save_path
