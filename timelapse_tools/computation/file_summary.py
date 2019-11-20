#!/usr/bin/env python
# -*- coding: utf-8 -*-

import imageio 
from aicspylibczi import CziFile
from path import Path
import os
import dateutil


def file_size(fn):
        return os.path.getsize(fn) / 2**20

def date_created(czi):
    """Date created as indicated by czi metadata"""
    creation_xml = czi.read_meta().xpath('//CreationDate')
    assert len(creationxml) == 1, "Wrong number of creation dates"
    creation_date = dateutil.parser.isoparse(creation_xml[0].text)
    creation_string = creation_date.date().isoformat()

def dimensions(czi):
    """The STCZ dimensions of the czi as indicated by czi.dims()"""
    dims = czi.dims()
    keys = [k for k in ['S', 'T', 'C', 'Z'] if k in dims.keys()]
    dim_string = "_".join([f"{k}{dims[k][1]-dims[k][0]}" for k in keys])
    return dim_string

def time_per_frame(czi):
    """Waiting on subblock metadata"""
    return "Unavailable" 

def total_duration(czi):
    """Waiting on subblock metadata"""
    return "Unavailable" 


