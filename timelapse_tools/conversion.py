#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dask.array as da
import imageio
import numpy as np
from prefect import Flow, task, unmapped

from .constants import AVAILABLE_OPERATING_DIMENSIONS, Dimensions
from .normalization.single_channel_percentile_norm import \
    single_channel_percentile_norm
from .projection.single_channel_max_project import single_channel_max_project
from .utils.czi_reading import daread

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

ImageDetails = Tuple[da.core.Array, str]

###############################################################################


@task
def _get_save_path(
    save_path: Optional[Union[str, Path, None]], overwrite: bool, fname: str
) -> Path:
    # If save_path was provided just double check it is valid
    if save_path is not None:
        save_path = Path(save_path).expanduser().resolve()

    # Else use filename to generate location
    else:
        save_path = Path(fname).expanduser().resolve()

    # Check that no directory or file exists at this location
    if save_path.exists() and not overwrite:
        raise FileExistsError(
            f"The save path provided already points to an existing resource and "
            f"overwrite wasn't specified. "
            f"Provided: {save_path}"
        )

    return save_path


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
    dim_indicies_selected: Optional[Union[int, slice]] = None,
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
    img_shape: tuple, dims: str
) -> List[Tuple[Union[int, slice]]]:
    getitem_indicies = []

    # Generate getitem ops to process for each scene x channel pair
    if Dimensions.Scene in dims and Dimensions.Channel in dims:
        sc_indicies = list(
            product(
                range(img_shape[dims.index(Dimensions.Scene)]),
                range(img_shape[dims.index(Dimensions.Channel)]),
            )
        )
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
    img: da.core.Array, getitem_indicies: List[Tuple[Union[int, slice]]]
) -> List[da.core.Array]:
    return [img[getitem_selection] for getitem_selection in getitem_indicies]


@task
def _generate_selected_dims_list(
    dims: str, getitem_indicies: List[Tuple[Union[int, slice]]]
) -> List[Dict[str, int]]:
    selected_dims = []
    for getitem_selection in getitem_indicies:
        this_set_dims = {}
        for i, dim in enumerate(dims):
            if isinstance(getitem_selection[i], int):
                this_set_dims[dim] = getitem_selection[i]

        selected_dims.append(this_set_dims)

    return selected_dims


@task
def _generate_movie(
    data: da.core.Array,
    selected_indices: Dict[str, int],
    dims: str,
    operating_dim: str,
    save_path: Path,
    fps: int,
    save_format: str,
    normalization_func: Callable,
    normalization_kwargs: Dict[str, Any],
    projection_func: Callable,
    projection_kwargs: Dict[str, Any],
) -> da.core.Array:
    # Normalize the data
    data = normalization_func(data=data, **normalization_kwargs)

    # Get the dims for this movie
    dims = "".join(dim for dim in dims if dim not in selected_indices)

    # Generate projections for each index of the operating dim
    frame_getitem_indicies = []
    for i in range(data.shape[dims.index(operating_dim)]):
        this_frame_set = []
        for dim in dims:
            if dim is operating_dim:
                this_frame_set.append(i)
            else:
                this_frame_set.append(slice(None, None, None))
        frame_getitem_indicies.append(tuple(this_frame_set))

    # Project all frames
    frames = []
    for frame_getitem_set in frame_getitem_indicies:
        frames.append(
            projection_func(
                data=data[frame_getitem_set],
                dims=dims.replace(operating_dim, ""),
                **projection_kwargs,
            )
        )

    # Generate output file name
    this_file = []
    for dim, selected in selected_indices.items():
        this_file.append(dim),
        this_file.append(str(selected))
    this_file = "_".join(this_file)

    # Remove any leading period from save format
    if save_format[0] == ".":
        save_format = save_format[1:]
    output_file = save_path / f"dims-{this_file}.{save_format}"

    # Make save dir if doesn't exist yet
    save_path.mkdir(parents=True, exist_ok=True)

    # Init writer
    writer = imageio.get_writer(output_file, fps=fps)

    # Iter over frames and append to writer
    for frame in frames:
        writer.append_data(frame.compute().astype(np.uint8))

    # Close writer
    writer.close()


def generate_movies(
    img: Union[str, Path],
    distributed_executor_port: Optional[Union[str, int]] = None,
    save_path: Optional[Union[str, Path]] = None,
    operating_dim: str = Dimensions.Time,
    overwrite: bool = False,
    fps: int = 12,
    save_format: str = "mp4",
    normalization_func: Callable = single_channel_percentile_norm,
    normalization_kwargs: Dict[str, Any] = {},
    projection_func: Callable = single_channel_max_project,
    projection_kwargs: Dict[str, Any] = {},
    S: Optional[Union[int, slice]] = None,
    C: Optional[Union[int, slice]] = None,
    B: Union[int, slice] = 0,
) -> Path:
    """
    Generate a movie for every scene and channel pair found in a file through an
    operating dimension.

    Parameters
    ----------
    img: Union[str, Path]
        Path to a CZI file to read and generate movies for.
    distributed_executor_port: Optional[Union[str, int]]
        If provided a port to use for connecting to the distributed scheduler. All image
        computation and workflow tasks will be distributed using Dask.
        Default: None
    save_path: Optional[Union[str, Path]]
        A specific path to save the generated movies to.
        Default: The a directory named after the provided file.
    operating_dim: str
        Which dimension to operating through for each frame of the movie.
        Default: Dimensions.Time ("T")
    overwrite: bool
        Should existing files found under the same directory name be overwritten.
        Default: False
    fps: int
        Frames per second of each produces movie.
        Default: 12
    save_format: str
        Which movie format should be used for each produced file.
        Default: mp4
        Available: mov, avi, mpg, mpeg, mp4, mkv, wmv
    normalization_func: Callable
        A function to normalize the entire movie data prior to projection.
        Default: timelapse_tools.normalization.single_channel_percentile_norm
    normalization_kwargs: Dict[str, Any]
        Any extra arguments to pass to the normalization function.
        Default: {}
    projection_func: Callable
        A function to project the data for at each frame of the movie.
        Default: timelapse_tools.projection.single_channel_max_project
    projection_kwargs: Dict[str, Any]
        Any extra arguments to pass to the projection function.
        Default: {}
    S: Optional[Union[int, slice]]
        A specific integer or slice to use for selecting down the scenes to process.
        Default: None (process all scenes)
    C: Optional[Union[int, slice]]
        A specific integer or slice to use for selecting down the channels to process.
        Default: None (process all channels)
    B: Union[int, slice]
        A specific integer or slice to use for selecting down the channels to process.
        Default: 0
    Returns
    -------
    save_path: Path
        The path to the produced scene-channel pairings of movies.
    """
    if distributed_executor_port:
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
            save_path=save_path, overwrite=overwrite, fname=img.with_suffix("").name
        )

        # Setup and check image and operating dimension provided
        img_details = _img_prep(
            img=img,
            operating_dim=operating_dim,
            # Don't run if save path checking failed
            upstream_tasks=[save_path],
        )

        # Select scene data
        img_details = _select_dimension(
            img=img_details[0],
            dims=img_details[1],
            dim_name=Dimensions.Scene,
            dim_indicies_selected=S,
        )

        # Select channel data
        img_details = _select_dimension(
            img=img_details[0],
            dims=img_details[1],
            dim_name=Dimensions.Channel,
            dim_indicies_selected=C,
        )

        # Select 'B' data
        img_details = _select_dimension(
            img=img_details[0],
            dims=img_details[1],
            dim_name=Dimensions.B,
            dim_indicies_selected=B,
        )

        # Generate all the indicie sets we will need to process
        getitem_indicies = _generate_getitem_indicies(
            img_shape=_get_image_shape(img_details[0]), dims=img_details[1]
        )

        # Generate all the movie selections
        to_process = _generate_process_list(
            img=img_details[0], getitem_indicies=getitem_indicies
        )

        # Generate a list of dictionaries that map dimension to selected data
        selected_indices = _generate_selected_dims_list(
            dims=img_details[1], getitem_indicies=getitem_indicies
        )

        # Generate movies for each
        _generate_movie.map(
            data=to_process,
            selected_indices=selected_indices,
            dims=unmapped(img_details[1]),
            operating_dim=unmapped(operating_dim),
            save_path=unmapped(save_path),
            fps=unmapped(fps),
            save_format=unmapped(save_format),
            normalization_func=unmapped(normalization_func),
            normalization_kwargs=unmapped(normalization_kwargs),
            projection_func=unmapped(projection_func),
            projection_kwargs=unmapped(projection_kwargs),
        )

    # Run the flow
    state = flow.run(executor=executor)

    # Get resulting path
    save_path = state.result[flow.get_tasks(name="_get_save_path")[0]].result

    return save_path
