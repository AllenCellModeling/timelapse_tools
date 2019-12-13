#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################


class ConflictingArgumentsError(Exception):
    pass


class InvalidShapeError(Exception):
    def __init__(self, actual: int, expected: int):
        self.actual = actual
        self.expected = expected

    def __str__(self):
        return (
            f"Invalid dimensions for this operation. "
            f"Recieved array with {self.actual} dimensions. "
            f"Expected array with {self.expected} dimensions. "
        )
