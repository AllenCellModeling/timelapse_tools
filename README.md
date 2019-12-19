# Timelapse Tools

[![Build Status](https://github.com/AllenCellModeling/timelapse_tools/workflows/Build%20Master/badge.svg)](https://github.com/AllenCellModeling/timelapse_tools/actions)
[![Documentation](https://github.com/AllenCellModeling/timelapse_tools/workflows/Documentation/badge.svg)](https://AllenCellModeling.github.io/timelapse_tools)
[![Code Coverage](https://codecov.io/gh/AllenCellModeling/timelapse_tools/branch/master/graph/badge.svg)](https://codecov.io/gh/AllenCellModeling/timelapse_tools)

Load and convert timelapse CZI files to movie formats

---

## Features
* Generate movies function that can operate on the `T` or `Z` axis
* General purpose CZI delayed reader
* Supported output formats:
    * `mov`
    * `avi`
    * `mpg`
    * `mpeg`
    * `mp4`
    * `mkv`
    * `wmv`

## Quick Start

_**Read and interact with a large file:**_
```python
from timelapse_tools import daread

# Dask array with delayed reads for every YX plane
img = daread("my_very_large_image.czi")
```

_**Generate all scene and channel movie pairs from a file:**_
```python
from timelapse_tools import generate_movies

# Generates a folder with every scene and channel pair of movies in the file
generate_movies("my_very_large_image.czi")
```

## Distributed
If you want to generate these movies in a distributed fashion, spin up a Dask scheduler.
The following settings generally work pretty well for our (AICS) SLURM cluster:
```python
from dask_jobqueue import SLURMCluster
import dask, dask.distributed

cluster = SLURMCluster(
    cores=2,
    memory="16GB",
    walltime="12:00:00",
    queue="aics_cpu_general"
)
cluster.adapt(minimum_jobs=2, maximum_jobs=40)
client = dask.distributed.Client(cluster)
```

From there you simply need to pass the distributed executor port to the
`generate_movies` function:
```python
from timelapse_tools import generate_movies

generate_movies(
    "my_very_large_image.czi",
    distributed_executor_port=cluster.scheduler_info["address"].split(":")[-1]
)
```

_It is also recommended that whichever machine you run the scheduler on, to also set the
following environment variable:_
```bash
export DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING="False"
```
_More details under the "work stealing" section
[here](https://docs.prefect.io/core/tutorials/dask-cluster.html)._

## Installation

`pip install git+https://github.com/AllenCellModeling/timelapse_tools.git`

## Documentation
For full package documentation please visit [AllenCellModeling.github.io/timelapse_tools](https://AllenCellModeling.github.io/timelapse_tools).

License: Allen Institute Software License
