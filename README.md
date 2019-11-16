# Timelapse Tools

[![Build Status](https://github.com/AllenCellModeling/timelapse_tools/workflows/Build%20Master/badge.svg)](https://github.com/AllenCellModeling/timelapse_tools/actions)
[![Documentation](https://github.com/AllenCellModeling/timelapse_tools/workflows/Documentation/badge.svg)](https://AllenCellModeling.github.io/timelapse_tools)

Load and convert timelapses

---

## Features
* Load and convert time lapse images

## Quick Start
```python
from timelapse_tools import generate_movie, projection, label

generate_movie(
    "my_very_large_image.czi",
    projection_func=projection.im2proj_all_axes,
    label=label.t_index_labeler,
    C=0
)
```

![Example Generated Timelapse Movie](data/example.mp4)

## Installation

`pip install git+https://github.com/AllenCellModeling/timelapse_tools.git`

## Documentation
For full package documentation please visit [AllenCellModeling.github.io/timelapse_tools](https://AllenCellModeling.github.io/timelapse_tools).

License: Allen Institute Software License
