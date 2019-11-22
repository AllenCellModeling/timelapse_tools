#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from dateutil import parser


def file_size(fn):
    """File size in GB"""
    return os.path.getsize(fn) / 2 ** 20


def date_created(czi):
    """Date created as indicated by czi metadata"""
    creation_xml = czi.read_meta().xpath("//CreationDate")
    assert len(creation_xml) == 1, "Wrong number of creation dates"
    creation_date = parser.isoparse(creation_xml[0].text)
    creation_string = creation_date.date().isoformat()
    return creation_string


def collected_by(czi):
    """User of record"""
    user = czi.read_meta().xpath("//UserName")[0].text
    return user


def dimensions(czi):
    """The STCZ dimensions of the czi as indicated by czi.dims()"""
    dims = czi.dims()
    keys = [k for k in ["S", "T", "C", "Z"] if k in dims.keys()]
    dim_string = "_".join([f"{k}{dims[k][1]-dims[k][0]}" for k in keys])
    return dim_string


def channel_names(czi):
    """Metadata specified channel names"""
    path = "./Metadata/Information/Image/Dimensions/Channels/Channel"
    channels = czi.read_meta().findall(path)
    names = [c.get("Name") for c in channels]
    return names


def time_per_frame(czi):
    """Waiting on subblock metadata"""
    return "Unavailable"


def total_duration(czi):
    """Waiting on subblock metadata"""
    return "Unavailable"
