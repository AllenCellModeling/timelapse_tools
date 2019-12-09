#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################


class Dimensions:
    Scene = "S"
    Time = "T"
    Channel = "C"
    SpatialZ = "Z"
    SpatialY = "Y"
    SpatialX = "X"


AVAILABLE_OPERATING_DIMENSIONS = [
    Dimensions.Scene,
    Dimensions.Time,
    Dimensions.Channel,
    Dimensions.SpatialZ,
]
