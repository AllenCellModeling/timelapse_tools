#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
from typing import List, Tuple

from lxml.etree import _Element

###############################################################################


def t_index_labeler(metadata: _Element, start_T: int, current_T: int, shape: List[Tuple[str, int]]) -> str:
    return f"T: {current_T}"


def t_plus_duration_labeler(metadata: _Element, start_T: int, current_T: int, shape: List[Tuple[str, int]]) -> str:
    # Get timeline elements from metadata
    timeline_elements = metadata.xpath(".//TimelineElement/Time")

    # Get the first and current time point
    begin = datetime.strptime(timeline_elements[start_T].text, "%Y-%m-%dT%H:%M:%S.%fZ")
    current = datetime.strptime(timeline_elements[current_T].text, "%Y-%m-%dT%H:%M:%S.%fZ")

    # Get duration
    duration = str(current - begin)

    # Return the duration from start
    return f"Duration: {duration}"
