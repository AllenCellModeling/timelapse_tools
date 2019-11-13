#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import imageio
import numpy as np
from aicspylibczi import CziFile
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from . import normalization

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def _process_timepoint(
    img: CziFile,
    T: int,
    S: int,
    C: Optional[int] = None,
    norm_func: Callable = normalization.im2proj,
    norm_kwargs: Dict = {},
    label: Optional[Union[Callable, str]] = None,
    font: Optional[ImageFont.FreeTypeFont] = None
) -> np.ndarray:
    log.info(f"Reading timepoint: {T}...")
    if C:
        read_slice, shape = img.read_image(T=T, B=0, S=S, C=C)
    else:
        read_slice, shape = img.read_image(T=T, B=0, S=S)

    # Generate projection
    log.debug(f"Generating projection...")
    proj = norm_func(read_slice, shape, **norm_kwargs)

    # Always cast to uint8
    proj = proj.astype(np.uint8)

    # TODO: Add label generation

    log.info(f"Completed timepoint {T} processing")
    return proj


def generate_movie(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    overwrite: bool = False,
    norm_func: Callable = normalization.im2proj,
    norm_kwargs: Dict = {},
    series_range: slice = slice(None, None, None),
    fps: int = 1,
    label: Optional[Union[Callable, str]] = None,
    font: Path = (Path(__file__).parent / "fonts" / "DejaVuSansMono.ttf"),
    show_progress: bool = False,
    S: int = 0,
    C: Optional[int] = None
) -> Path:
    # Resolve paths
    input_file = Path(input_file).expanduser().resolve(strict=True)
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
    writer = imageio.get_writer(output_file, fps=fps)

    # Warn user about potential memory and IO
    if C is None:
        log.warn(f"All dimension `C` (Channel) data will be read on each iteration. This may increase time and memory.")

    # Read the font file if labels are desired
    if label:
        font = ImageFont.truetype(font, 12)
    else:
        font = None

    # Iterate over time dim
    len_T = img.dims()["T"][1]

    # Show or hide progress bar
    if show_progress:
        for i in tqdm(range(len_T)[series_range]):
            # Process timepoint
            processed = _process_timepoint(
                img=img,
                T=i,
                S=S,
                C=C,
                norm_func=norm_func,
                norm_kwargs=norm_kwargs,
                label=label,
                font=font
            )

            # Append projection
            writer.append_data(processed)
    else:
        for i in range(len_T)[series_range]:
            # Process timepoint
            processed = _process_timepoint(
                img=img,
                T=i,
                S=S,
                C=C,
                norm_func=norm_func,
                norm_kwargs=norm_kwargs,
                label=label,
                font=font
            )

            # Append projection
            writer.append_data(processed)

    # Close the writer
    writer.close()

    return output_file
