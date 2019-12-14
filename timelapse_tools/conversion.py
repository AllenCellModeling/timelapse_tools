#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dask.array as da
from prefect import Flow, task, unmapped

from .constants import AVAILABLE_OPERATING_DIMENSIONS, Dimensions
from .normalization.single_channel_percentile_norm import percentile_norm
from .projection.single_channel_max_project import single_channel_max_project
from .utils.czi_reading import daread

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

ImageDetails = Tuple[da.core.Array, str]

###############################################################################


@task
def _get_save_path(
    save_path: Optional[Union[str, Path, None]],
    overwrite: bool,
    fname: str
) -> Path:
    # If save_path was provided just double check it is valid
    if save_path is not None:
        # Resolve path
        save_path = Path(save_path).expanduser().resolve()

        # Check that no directory or file exists at this location
        if save_path.exists() and not overwrite:
            raise FileExistsError(
                f"The save path provided already points to an existing resource and "
                f"overwrite wasn't specified. "
                f"Provided: {save_path}"
            )

        return save_path

    return Path(fname).expanduser().resolve()


@task
def _img_prep(img: Union[str, Path], operating_dim: str) -> ImageDetails:
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

    return img, dims


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

    return img, dims


@task
def _get_image_shape(img: da.core.Array) -> Tuple[int]:
    return img.shape


@task
def _generate_getitem_indicies(
    img_shape: tuple,
    dims: str
) -> List[Tuple[Union[int, slice]]]:
    getitem_indicies = []

    # Generate getitem ops to process for each scene x channel pair
    if Dimensions.Scene in dims and Dimensions.Channel in dims:
        sc_indicies = list(product(
            range(img_shape[dims.index(Dimensions.Scene)]),
            range(img_shape[dims.index(Dimensions.Channel)])
        ))
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
        getitem_indicies.append(tuple([slice(None, None, None) for dim in dims]))

    return getitem_indicies


@task
def _generate_process_list(
    img: da.core.Array,
    getitem_indicies: List[Tuple[Union[int, slice]]]
) -> List[da.core.Array]:
    return [img[getitem_selection] for getitem_selection in getitem_indicies]


@task
def _generate_movie(
    data: da.core.Array,
    dims: str,
    operating_dim: str,
    save_path: Path,
    # normalization_func: Callable,
    normalization_kwargs: Dict[str, Any],
    # projection_func: Callable,
    projection_kwargs: Dict[str, Any]
) -> da.core.Array:
    return


def convert_to_mp4(
    img: Union[str, Path],
    distributed: bool = False,
    distributed_executor_port: Union[str, int] = 8888,
    save_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    # normalization_func: Callable = percentile_norm,
    normalization_kwargs: Dict[str, Any] = {},
    # projection_func: Callable = single_channel_max_project,
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
        # Convert img to Path
        img = Path(img).expanduser().resolve(strict=True)

        # Determine save path
        save_path = _get_save_path(
            save_path=save_path,
            overwrite=overwrite,
            fname=img.with_suffix("").name
        )

        # Setup and check image and operating dimension provided
        img_details = _img_prep(img=img, operating_dim=operating_dim)

        # Select scene data
        img_details = _select_dimension(
            img=img_details[0],
            dims=img_details[1],
            dim_name=Dimensions.Scene,
            dim_indicies_selected=S
        )

        # Select channel data
        img_details = _select_dimension(
            img=img_details[0],
            dims=img_details[1],
            dim_name=Dimensions.Channel,
            dim_indicies_selected=C
        )
        # Unpack image details because they will be used all over the place now
        img = img_details[0]
        dims = img_details[1]

        # Generate all the indicie sets we will need to process
        getitem_indicies = _generate_getitem_indicies(
            img_shape=_get_image_shape(img),
            dims=dims
        )

        # Generate all the movie selections
        to_process = _generate_process_list(
            img=img,
            getitem_indicies=getitem_indicies
        )

        # Generate movies for each
        _generate_movie.map(
            data=to_process,
            dims=unmapped(dims),
            operating_dim=unmapped(operating_dim),
            save_path=unmapped(save_path),
            # normalization_func=unmapped(normalization_func),
            normalization_kwargs=unmapped(normalization_kwargs),
            # projection_func=unmapped(projection_func),
            projection_kwargs=unmapped(projection_kwargs)
        )

    # Run the flow
    state = flow.run(executor=executor)

    # Get resulting path
    save_path = state.result[flow.get_tasks(name="_get_save_path")[0]].result

    return save_path
