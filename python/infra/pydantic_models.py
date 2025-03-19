# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Definition of the pydantic models used for data production.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Test(BaseModel):
    """
    Table containing information about the execution of CI/CD tests, each one associated
    with a specific CI/CD job execution.

    Only some CI/CD jobs execute tests, which are executed sequentially.
    """

    # capture here
    test_start_ts: datetime = Field(
        description="Timestamp with timezone when the test execution started."
    )
    # capture here
    test_end_ts: datetime = Field(
        description="Timestamp with timezone when the test execution ended."
    )
    # capture here
    test_case_name: str = Field(description="Name of the pytest function.")
    # capture in fe
    filepath: str = Field(description="Test file path and name.")
    # capture in fe
    category: str = Field(description="Name of the test category.")
    # capture in fe
    group: Optional[str] = Field(None, description="Name of the test group.")
    # capture in fe
    owner: Optional[str] = Field(None, description="Developer of the test.")

    error_message: Optional[str] = Field(
        None, description="Succinct error string, such as exception type."
    )
    success: bool = Field(description="Test execution success.")
    skipped: bool = Field(description="Some tests in a job can be skipped.")
    full_test_name: str = Field(description="Test name plus config.")
    config: Optional[dict] = Field(
        None, description="Test configuration key/value " "pairs."
    )
    tags: Optional[dict] = Field(
        None, description="Tags associated with the test, as key/value pairs."
    )


class TensorDesc(BaseModel):
    """
    Contains descriptions of tensors used as inputs or outputs of the operation in a ML
    kernel operation test.
    """

    shape: List[int] = Field(description="Shape of the tensor.")
    data_type: str = Field(
        description="Data type of the tensor, e.g. Float32, " "BFloat16, etc."
    )
    buffer_type: str = Field(
        description="Memory space of the tensor, e.g. Dram, L1, " "System."
    )
    layout: str = Field(
        description="Layout of the tensor, e.g. Interleaved, "
        "SingleBank, HeightSharded."
    )
    grid_shape: List[int] = Field(
        description="The grid shape describes a 2D region of cores which are used to "
        "store the tensor in memory. E.g. You have a tensor with shape "
        "128x128, you might decide to put this on a 2x2 grid of cores, "
        "meaning each core has a 64x64 slice."
    )


class OpTest(BaseModel):
    """
    Contains information about ML kernel operation tests, such as test execution,
    results, configuration.
    """

    github_job_id: int = Field(
        description="Identifier for the Github Actions CI job, which ran the test.",
    )
    full_test_name: str = Field(description="Test name plus config.")
    test_start_ts: datetime = Field(
        description="Timestamp with timezone when the test execution started."
    )
    test_end_ts: datetime = Field(
        description="Timestamp with timezone when the test execution ended."
    )
    test_case_name: str = Field(description="Name of the pytest function.")
    filepath: str = Field(description="Test file path and name.")
    success: bool = Field(description="Test execution success.")
    skipped: bool = Field(description="Some tests in a job can be skipped.")
    error_message: Optional[str] = Field(
        None, description="Succinct error string, such as exception type."
    )
    config: Optional[dict] = Field(
        default=None, description="Test configuration, as key/value pairs."
    )
    frontend: str = Field(description="ML frontend or framework used to run the test.")
    model_name: str = Field(
        description="Name of the ML model in which this operation is used."
    )
    op_kind: str = Field(description="Kind of operation, e.g. Eltwise.")
    op_name: str = Field(description="Name of the operation, e.g. ttnn.conv2d")
    framework_op_name: str = Field(
        description="Name of the operation within the framework, e.g. torch.conv2d"
    )
    inputs: List[TensorDesc] = Field(description="List of input tensors.")
    outputs: List[TensorDesc] = Field(description="List of output tensors.")
    op_params: Optional[dict] = Field(
        default=None,
        description="Parametrization criteria for the operation, based on its kind, "
        "as key/value pairs, e.g. stride, padding, etc.",
    )
