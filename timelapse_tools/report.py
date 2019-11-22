#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional, Union

from aicspylibczi import CziFile
from jinja2 import Template

from . import projection
from .computation import compute, file_summary, intensity_distributions
from .movie import generate_movie
from .render import plots

###############################################################################

RESOURCES = (Path(__file__).parent / "resources").resolve(strict=True)

###############################################################################


def _projection(channel_index, channel_names, filepath, save_dir_resources_dir, save_dir):
    curr_channel = channel_names[channel_index]
    saved_projection = generate_movie(
        input_file=filepath,
        output_file=save_dir_resources_dir / f"{curr_channel}.mp4",
        overwrite=True,
        projection_func=projection.im2proj_all_axes,
        C=channel_index,
        fps=12
    )

    # Append the data to media
    # Make the file relative to the save directory instead of a hard coded full path
    return {"name": channel_names[channel_index], "src": str(saved_projection.relative_to(save_dir))}


def _computation(channel_index, filepath, channel_names):
    intensity_distribution_calc = intensity_distributions.IntensityDistributions()
    compute(
        input_file=filepath,
        computation_manager=intensity_distribution_calc,
        C=channel_index
    )

    # Get projections from computed
    yz_proj = intensity_distribution_calc.median_intensity_across_dim[0]
    yz_proj_b64 = plots.fig_to_base64(plots.small_heatmap(yz_proj))
    return {"name": f"YZ Projection for Channel: {channel_names[channel_index]}", "src": yz_proj_b64}


def generate_report(
    filepath: Union[str, Path],
    save_dir: Optional[Union[str, Path]] = None
) -> Path:
    # Validate filepath
    filepath = Path(filepath).expanduser().resolve(strict=True)

    # Ensure it is a file
    if filepath.is_dir():
        raise IsADirectoryError(f"Reports can only be generated for a single file. Provided directory: {filepath}")

    # Check save dir
    if save_dir is None:
        save_dir = Path(f"file_report-{filepath.name}").resolve()
    else:
        save_dir = Path(save_dir).resolve()

    # Make save dir if needed
    save_dir.mkdir(parents=True, exist_ok=True)

    # Copy all report resources
    save_dir_resources_dir = save_dir / "resources"
    save_dir_resources_dir.mkdir(parents=True, exist_ok=True)

    # Copy css
    shutil.copyfile(RESOURCES / "template.css", save_dir_resources_dir / "main.css")

    # Open CZI
    czi = CziFile(filepath)

    # Map of display names and their values
    attributes = {
        "File Size": file_summary.file_size(filepath),
        "Created": file_summary.date_created(czi),
        "Created By": file_summary.collected_by(czi),
        "Dimensions": file_summary.dimensions(czi)
    }

    # Convert attributes dictionary to format for jinja
    attributes = [{"name": name, "value": value} for name, value in attributes.items()]

    # Generate media and suppliments
    dims = czi.dims()
    channel_names = file_summary.channel_names(czi)
    if "C" in dims:
        # Generate channel indicies list to process
        channel_indices = list(range(dims["C"][0], dims["C"][1]))

        # Generate max project movies for each channel
        projection_func = partial(
            _projection,
            channel_names=channel_names,
            filepath=filepath,
            save_dir_resources_dir=save_dir_resources_dir,
            save_dir=save_dir
        )

        # Generate computations for each channel
        computation_func = partial(
            _computation,
            channel_names=channel_names,
            filepath=filepath
        )

        with ProcessPoolExecutor() as exe:
            media = list(exe.map(projection_func, channel_indices))

        with ProcessPoolExecutor() as exe:
            suppliments = list(exe.map(computation_func, channel_indices))

    # Read and parse the template
    with open(RESOURCES / "template.html", "r") as read_template:
        template = Template(read_template.read())

    # Fill template
    filled = template.render(filename=filepath.name, attributes=attributes, suppliments=suppliments, media=media)

    # Save dir of outputs
    with open(save_dir / "index.html", "w") as write_index:
        write_index.write(filled)
