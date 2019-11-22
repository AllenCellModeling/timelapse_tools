#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd

from timelapse_tools import generate_report

###############################################################################

log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(name)s - %(lineno)3d][%(levelname)s] %(message)s",
)

###############################################################################


# Write passthrough function that will be multithreaded
def passthrough(input_fp, save_dir):
    save_dir = save_dir / input_fp.name
    log.info(f"Beginning processing for: {save_dir}")
    try:
        generate_report(
            filepath=input_fp,
            save_dir=save_dir
        )
        log.info(f"Completed processing for: {input_fp}. Saved to: {save_dir}")
    except Exception as e:
        log.error(f"Recieved error from file: {input_fp}. Exception:")
        log.error(e)


# Read data csv
data = (
    Path(__file__).resolve().parent.parent.parent / "data" / "raw_image_list.csv"
).resolve(strict=True)
data = pd.read_csv(data)

# Generate list of input file paths and output save paths
input_file_paths = [
    (Path("/allen") / row["Isilon path"][1:] / row["File Name"]).resolve(strict=True)
    for i, row in data.iterrows()
]

# Create partially filled function
processing_func = partial(
    passthrough,
    save_dir=Path("/allen/aics/modeling/jacksonb/projects/timelapse_reports_fixed").resolve()
)

# Run threadpool
with ProcessPoolExecutor() as exe:
    exe.map(processing_func, input_file_paths)
