# -*- coding: utf-8 -*-

"""Top-level package for Timelapse Tools."""

__author__ = "Jackson Brown, Dave Williams"
__email__ = "jacksonb@alleninstitute.org, cdavew@alleninstitute.org"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.1.0"


from .conversion import convert_to_mp4  # noqa: F401
from .utils import daread  # noqa: F401


def get_module_version():
    return __version__
