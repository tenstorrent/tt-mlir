# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import inspect
import subprocess
import torch
import pytest
from typing import Callable, List, Optional, Tuple, Union, Literal, Dict

from ttmlir.dialects import func
from ttmlir.ir import *
from ttmlir.passmanager import PassManager
from ttmlir.passes import (
    tt_populate_argument_types,
    ttir_to_ttnn_backend_pipeline,
    ttnn_to_flatbuffer_file,
    ttir_to_ttmetal_backend_pipeline,
    ttmetal_to_flatbuffer_file,
    translate_to_cpp,
    MLIRModuleLogger,
)


class Marks:
    """
    Convenience class for adding pytest marks.

    Example
    -------
    >>> skip_test = Marks(pytest.mark.skip)
    >>> test_case | skip_test  # Marks test_case to be skipped
    """

    def __init__(self, *marks):
        """
        Initialize with pytest marks.

        Parameters
        ----------
        *marks : tuple
            Variable number of pytest.mark objects
        """
        self.marks = marks

    def __ror__(self, lhs):
        """
        Apply marks to a test parameter.

        Parameters
        ----------
        lhs : Any
            Test parameter to mark

        Returns
        -------
        pytest.param
            Marked test parameter
        """
        return pytest.param(lhs, marks=self.marks)


def shape_str(shape):
    """
    Converts shape tuple to string.

    Parameters
    ----------
    shape : *Union[Tuple[int, ...], List[int]]*
        Shape to convert to string

    Returns
    -------
    str
        String representation of the shape (e.g., '32x32' for shape (32, 32))
    """
    return "x".join(map(str, shape))
