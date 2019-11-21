#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
from pathlib import Path
from typing import Optional, Union

from aicspylibczi import CziFile
from jinja2 import Template

from . import projection
from .computation import file_summary
from .movie import generate_movie

###############################################################################

RESOURCES = (Path(__file__).parent / "resources").resolve(strict=True)

###############################################################################


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

    # Generate media
    channel_names = file_summary.channel_names(czi)
    media = []
    if "C" in czi.dims():
        # Generate max project movies for each channel
        for channel_index in range(czi.dims()["C"][1]):
            curr_channel = channel_names[channel_index]
            saved_projection = generate_movie(
                input_file=filepath,
                output_file=save_dir_resources_dir / f"{curr_channel}.mp4",
                overwrite=True,
                projection_func=projection.im2proj_all_axes,
                C=channel_index
            )

            # Append the data to media
            # Make the file relative to the save directory instead of a hard coded full path
            media.append({"name": curr_channel, "src": str(saved_projection.relative_to(save_dir))})

    # Read and parse the template
    with open(RESOURCES / "template.html", "r") as read_template:
        template = Template(read_template.read())

    # Fill template
    filled = template.render(filename=filepath.name, attributes=attributes, suppliments=[], media=media)

    # Save dir of outputs
    with open(save_dir / "index.html", "w") as write_index:
        write_index.write(filled)
