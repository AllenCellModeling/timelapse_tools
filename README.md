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

# Generates a folder with every scene and channel pair of videos in the file
generate_movies("my_very_large_image.czi")
```

![Example Generated Timelapse Movie](data/example.mp4)

## Installation

`pip install git+https://github.com/AllenCellModeling/timelapse_tools.git`

## Documentation
For full package documentation please visit [AllenCellModeling.github.io/timelapse_tools](https://AllenCellModeling.github.io/timelapse_tools).

License: Allen Institute Software License
