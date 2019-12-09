#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dateutil.parser import isoparse

###############################################################################


def datetime_created(czi):
    """Date created as indicated by czi metadata"""
    creation_xml = czi.read_meta().xpath("//CreationDate")
    assert len(creation_xml) == 1, "Wrong number of creation dates"
    return isoparse(creation_xml[0].text)


def created_by(czi):
    """User of record"""
    user = czi.read_meta().xpath("//UserName")[0].text
    return user


def channel_names(czi):
    """Metadata specified channel names"""
    path = "./Metadata/Information/Image/Dimensions/Channels/Channel"
    channels = czi.read_meta().findall(path)
    return [c.get("Name") for c in channels]


def time_per_frame(czi):
    """Waiting on subblock metadata"""
    return "Unavailable"


def total_duration(czi):
    """Waiting on subblock metadata"""
    return "Unavailable"
