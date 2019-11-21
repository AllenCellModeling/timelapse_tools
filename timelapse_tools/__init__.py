# -*- coding: utf-8 -*-

"""Top-level package for Timelapse Tools."""

__author__ = "Jackson Brown, Dave Williams"
__email__ = "jacksonb@alleninstitute.org, cdavew@alleninstitute.org"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.1.0"


def get_module_version():
    return __version__


from .computation import compute  # noqa: F401
from .movie import generate_movie  # noqa: F401
from .report import generate_report  # noqa: F401
