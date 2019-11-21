#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import imageio
import numpy as np
from aicspylibczi import CziFile
from lxml.etree import _Element
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from . import projection

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def _process_timepoint(
    img: CziFile,
    metadata: _Element,
    T: int,
    S: int,
    C: Optional[int] = None,
    projection_func: Callable = projection.im2proj,
    projection_kwargs: Dict = {},
    label: Optional[Union[Callable, str]] = None,
    font: Optional[ImageFont.FreeTypeFont] = None,
    begin_t: int = 0,
) -> np.ndarray:
    """
    Process a single frame of the movie. This read the desired plane then step through the projection function and the
    label function.

    Parameters
    ----------
    img: CziFile
        The initialized CziFile to read from.
    metadata: _Element
        The already read file wide metadata.
    T: int
        Which timepoint to read.
    S: int
        Which scene to read.
    C: Optional[int]
        Optionally, which channel to read. If None, reads all channels for that timepoint and scene.
    projection_func: Callable
        Which function to be used to generate a projection for the plane.
    projection_kwargs: Dict
        Any additional kwargs to be passed to the projection function.
    label: Optional[Union[Callable, str]]
        Optionally, either: a function to generate a string or a string to placed on each frame.
    font: Optional[ImageFont.FreeTypeFont]
        Optionally, a preloaded font to use for the label.
    begin_t: int
        The first timepoint index to be processed by the generate_movie function.

    Returns
    -------
    projection: np.ndarray
        The 2D (YX) or 3D (CYX) ndarray to add as a frame to the movie.
    """
    # Read timepoint
    log.debug(f"Reading timepoint: {T}")
    if C:
        read_slice, shape = img.read_image(T=T, B=0, S=S, C=C)
    else:
        read_slice, shape = img.read_image(T=T, B=0, S=S)

    # Generate projection
    log.debug(f"Generating projection...")
    proj = projection_func(read_slice, shape, **projection_kwargs)

    # Always cast to uint8
    proj = proj.astype(np.uint8)

    # Enforce projection dimensions
    if len(proj.shape) not in (2, 3):
        raise ValueError(
            f"The array returned from provided projection function was {len(proj.shape)} dimensional. Expected 2 or 3."
        )

    # Add label if provided
    if label:
        # Call label function with parameters
        if isinstance(label, Callable):
            label = label(metadata=metadata, start_T=begin_t, current_T=T, shape=shape)

        # Always cast to a string
        if not isinstance(label, str):
            log.debug(
                f"Label for frame: {T} was provided as {label} with type: {type(label)}. Casting to string."
            )
            label = str(label)

        # Read projection as Image
        im = Image.fromarray(proj)

        # Get height and set intensity for text
        if len(proj.shape) == 3:
            height = proj.shape[1]
            intensity = (255, 255, 255)

        else:
            height = proj.shape[0]
            intensity = 255

        # Label position is lower left
        position = (10, height - 20)

        # Attach label in lower left of image
        drawer = ImageDraw.Draw(im)
        drawer.text(position, label, intensity, font=font)

        # Return to numpy array
        proj = np.asarray(im).astype(np.uint8)

    log.debug(f"Completed timepoint: {T}")
    return proj


def generate_movie(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    projection_func: Callable = projection.im2proj,
    projection_kwargs: Dict = {},
    series_range: slice = slice(None, None, None),
    fps: int = 1,
    label: Optional[Union[Callable, str]] = None,
    font: Path = (Path(__file__).parent / "fonts" / "DejaVuSansMono.ttf"),
    show_progress: bool = False,
    S: int = 0,
    C: Optional[int] = None,
) -> Path:
    """
    Convert a single large file into a much smaller movie. Image reading and writing is done a single frame at a time.

    Parameters
    ----------
    input_file: Union[str, Path]
        The path to the file to be processed.
    output_file: Optional[Union[str, Path]]
        Optionally, a path to where the output file should be stored. If None provided, stores it in the same directory
        as the input_file, with the same name but with the .mp4 suffix.
    overwrite: bool
        If a file already exists at the output_file location, allow it to be overwritten.
    projection_func: Callable
        Which function to be used to generate a projection for the plane.
    projection_kwargs: Dict
        Any additional kwargs to be passed to the projection function.
    series_range: slice
        A slice object to inform which timepoints to use.
    fps: int
        How many frames should be shown per second in the output file.
    label: Optional[Union[Callable, str]]
        Optionally, either: a function to generate a string or a string to placed on each frame.
    font: Optional[ImageFont.FreeTypeFont]
        Optionally, a preloaded font to use for the label.
    S: int
        Which scene to read.
    C: Optional[int]
        Which channel to read. Default to reading all channels.

    Returns
    -------
    saved_file: Path
        The resulting save path for the produced file.
    """
    # Resolve paths
    input_file = Path(input_file).expanduser().resolve(strict=True)

    # Create output filename if none passed
    if output_file is None:
        output_file = input_file.with_suffix(".mp4")

    # Check path
    output_file = Path(output_file).expanduser().resolve()

    # Check input path
    if input_file.is_dir():
        raise IsADirectoryError(
            f"Please provide a single file to the `input_file` parameter. Receieved directory: {input_file}"
        )

    # Check output path
    if output_file.is_file() and not overwrite:
        raise FileExistsError(
            f"A file already exists at the specified output location: {output_file}."
        )

    # Init czi
    img = CziFile(input_file)
    metadata = img.read_meta()

    # Check czi
    if len(img.dims()) == 0:
        raise IOError(
            f"The input file provided appears to be corrupted or is unreadable by `aicspylibczi`."
        )

    # Warn user about potential memory and IO
    if C is None:
        log.warn(
            f"All dimension `C` (Channel) data will be read on each iteration. This may increase time and memory."
        )

    # Read the font file if labels are desired
    if label:
        font = ImageFont.truetype(str(font), 12)
    else:
        font = None

    # Iterate over time dim
    len_T = img.dims()["T"][1]

    # Generate iterator
    iterator = range(len_T)[series_range]

    # Store beginning timepoint as it may be useful for labeling if desired
    begin_t = iterator[0]

    # Change iterator over to tqdm if desired
    if show_progress:
        iterator = tqdm(iterator)

    # Init writer
    writer = imageio.get_writer(output_file, fps=fps)

    # Iter through timepoints
    for i in iterator:
        # Process timepoint
        processed = _process_timepoint(
            img=img,
            metadata=metadata,
            T=i,
            S=S,
            C=C,
            projection_func=projection_func,
            projection_kwargs=projection_kwargs,
            label=label,
            font=font,
            begin_t=begin_t,
        )

        # Append projection
        writer.append_data(processed)

    # Close the writer
    writer.close()

    return output_file
