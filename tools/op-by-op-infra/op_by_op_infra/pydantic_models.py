# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Definition of the pydantic models used for data production.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class TensorDesc(BaseModel):
    """
    Contains descriptions of tensors used as inputs or outputs of the operation in a ML
    kernel operation test.
    """

    # All ops have the following two attributes.
    shape: List[int] = Field(description="Shape of the tensor.")
    data_type: str = Field(
        description="Data type of the tensor, e.g. Float32, " "BFloat16, etc."
    )
    # NOTE Only TTNN ops have the following attributes. Thus set to optional.
    buffer_type: Optional[str] = Field(
        default=None,
        description="Memory space of the tensor, e.g. Dram, L1, " "System.",
    )
    layout: Optional[str] = Field(
        default=None,
        description="Layout of the tensor, e.g. Interleaved, "
        "HeightSharded, WidthSharded.",
    )
    grid_shape: Optional[List[int]] = Field(
        default=None,
        description="The grid shape describes a 2D region of cores which are used to "
        "store the tensor in memory. E.g. You have a tensor with shape "
        "128x128, you might decide to put this on a 2x2 grid of cores, "
        "meaning each core has a 64x64 slice.",
    )


class OpTest(BaseModel):
    """
    Contains information about ML kernel operation tests, such as test execution,
    results, configuration.
    """

    # NOTE Set during job collection and running the CI pipeline, can't be set earlier.
    # Set as optional.
    github_job_id: Optional[int] = Field(
        default=None,
        description="Identifier for the Github Actions CI job, which ran the test.",
    )
    # NOTE This is pytest metadata, can't be set in op by op infra nor provided by
    # frontend. Must be set somewhere in pydantic model -> db table conversion.
    # Set as optional.
    full_test_name: Optional[str] = Field(
        default=None, description="Test name plus config."
    )
    test_start_ts: datetime = Field(
        description="Timestamp with timezone when the test execution started."
    )
    test_end_ts: datetime = Field(
        description="Timestamp with timezone when the test execution ended."
    )
    # NOTE This is pytest metadata, can't be set in op by op infra nor provided by
    # frontend. Must be set somewhere in pydantic model -> db table conversion.
    # Set as optional.
    test_case_name: Optional[str] = Field(
        default=None, description="Name of the pytest function."
    )
    # NOTE This is pytest metadata, can't be set in op by op infra nor provided by
    # frontend. Must be set somewhere in pydantic model -> db table conversion.
    # Set as optional.
    filepath: Optional[str] = Field(
        default=None, description="Test file path and name."
    )
    success: bool = Field(description="Test execution success.")
    # NOTE This is pytest metadata, can't be set in op by op infra nor provided by
    # frontend. Must be set somewhere in pydantic model -> db table conversion.
    # Set as optional.
    skipped: bool = Field(
        default=False, description="Some tests in a job can be skipped."
    )
    error_message: Optional[str] = Field(
        None, description="Succinct error string, such as exception type."
    )
    # NOTE Unused for now.
    config: Optional[dict] = Field(
        default=None, description="Test configuration, as key/value pairs."
    )
    # NOTE This field must be provided from frontend.
    frontend: Optional[str] = Field(
        default=None, description="ML frontend or framework used to run the test."
    )
    # NOTE This field must be provided from frontend.
    model_name: Optional[str] = Field(
        default=None,
        description="Name of the ML model in which this operation is used.",
    )
    # TODO is this important? How to determine it? Set as optional for now.
    op_kind: Optional[str] = Field(
        default=None, description="Kind of operation, e.g. Eltwise."
    )
    # Origin op name.
    op_name: str = Field(description="Name of the operation, e.g. ttnn.conv2d")
    # TODO what should be set here? Set as optional for now.
    framework_op_name: Optional[str] = Field(
        default=None,
        description="Name of the operation within the framework, e.g. torch.conv2d",
    )
    inputs: List[TensorDesc] = Field(description="List of input tensors.")
    outputs: List[TensorDesc] = Field(description="List of output tensors.")
    # NOTE Unused for now.
    op_params: Optional[dict] = Field(
        default=None,
        description="Parametrization criteria for the operation, based on its kind, "
        "as key/value pairs, e.g. stride, padding, etc.",
    )


def to_str(v) -> str:
    if isinstance(v, datetime):
        return v.isoformat()
    else:
        return str(v)


def model_to_dict(model: BaseModel) -> dict:
    return {k: to_str(v) for k, v in model}


def model_to_list(model: BaseModel) -> list:
    return [(k, to_str(v)) for k, v in model]
