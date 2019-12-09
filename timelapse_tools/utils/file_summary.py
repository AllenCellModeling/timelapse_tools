#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

###############################################################################


def _human_readable_file_size(value):
    """
    Pulled from:
    https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size#answer-1094933
    """
    for unit in ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"]:
        if abs(value) < 1024.0:
            return "%3.1f %s" % (value, unit)
        value /= 1024.0
    return "%.1f%s%s" % (value, "YB")


def file_size(fn):
    """Get human readable size of a file."""
    return _human_readable_file_size(os.path.getsize(fn))
