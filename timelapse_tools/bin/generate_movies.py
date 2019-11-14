#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from timelapse_tools import generate_movie, label, projection

###############################################################################

log = logging.getLogger()
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(name)s - %(lineno)3d][%(levelname)s] %(message)s')

###############################################################################


# Write passthrough function that will be multithreaded
def passthrough(io):
    input_fp, output_fp = io

    log.info(f"Beginning processing for: {input_fp}")
    try:
        generate_movie(
            input_file=input_fp,
            output_file=output_fp,
            projection_func=projection.im2proj_all_axes,
            fps=24,
            label=label.t_index_labeler,
            C=0
        )
        log.info(f"Completed processing for: {input_fp}. Saved to: {output_fp}")
    except Exception as e:
        log.error(f"Recieved error from file: {input_fp}. Exception:")
        log.error(e)


# Read data csv
data = (Path(__file__).resolve().parent.parent.parent / "data" / "raw_image_list.csv").resolve(strict=True)
data = pd.read_csv(data)

# Generate list of input file paths and output save paths
input_file_paths = [
    (Path("/allen") / row["Isilon path"][1:] / row["File Name"]).resolve(strict=True)
    for i, row in data.iterrows()
]
output_save_paths = [
    (Path("/allen/aics/modeling/jacksonb/projects/timelapse_movies") / row["File Name"]).resolve().with_suffix(".mp4")
    for i, row in data.iterrows()
]

# Zip them together for passthrough function to unpack
io_paths = zip(input_file_paths, output_save_paths)

# Run threadpool
with ThreadPoolExecutor(max_workers=6) as exe:
    exe.map(passthrough, io_paths)
