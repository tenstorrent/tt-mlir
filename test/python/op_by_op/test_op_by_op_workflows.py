# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from op_by_op_infra.execution_result import ExecutionPhase
from op_by_op_infra.workflow_internal import (
    compile_split_and_execute,
    split_and_execute,
    split_compile_split_and_execute,
)

from .fixtures import *


def test_split_and_execute_shlo_module(shlo_module_str: str):
    results = split_and_execute(shlo_module_str)

    assert all(
        result.execution_phase == ExecutionPhase.EXECUTED_FLATBUFFER
        for result in results
    ), f"Expected all results to be in EXECUTED_FLATBUFFER phase, got: {results}"


def test_compile_split_and_execute_shlo_module(shlo_module_str: str):
    results = compile_split_and_execute(shlo_module_str)

    expected_results = [
        ExecutionPhase.EXECUTED_FLATBUFFER,
        ExecutionPhase.EXECUTED_FLATBUFFER,
    ]
    for i, result in enumerate(results):
        assert result.execution_phase == expected_results[i]


def test_split_compile_split_and_execute_shlo_module(shlo_module_str: str):
    results = split_compile_split_and_execute(shlo_module_str)

    expected_results = [
        ExecutionPhase.EXECUTED_FLATBUFFER,
        ExecutionPhase.EXECUTED_FLATBUFFER,
    ]
    for i, result in enumerate(results):
        assert result.execution_phase == expected_results[i]


def test_split_and_execute_ttir_module(ttir_module_str: str):
    results = split_and_execute(ttir_module_str)

    assert all(
        result.execution_phase == ExecutionPhase.EXECUTED_FLATBUFFER
        for result in results
    ), f"Expected all results to be in EXECUTED_FLATBUFFER phase, got: {results}"


def test_compile_split_and_execute_ttir_module(ttir_module_str: str):
    results = compile_split_and_execute(ttir_module_str)

    expected_results = [
        ExecutionPhase.EXECUTED_FLATBUFFER,
        ExecutionPhase.EXECUTED_FLATBUFFER,
    ]
    for i, result in enumerate(results):
        assert result.execution_phase == expected_results[i]


def test_split_compile_split_and_execute_ttir_module(ttir_module_str: str):
    results = split_compile_split_and_execute(ttir_module_str)

    expected_results = [
        ExecutionPhase.EXECUTED_FLATBUFFER,
        ExecutionPhase.EXECUTED_FLATBUFFER,
    ]
    for i, result in enumerate(results):
        assert result.execution_phase == expected_results[i]


def test_split_and_execute_ttnn_module(ttnn_module_str: str):
    results = split_and_execute(ttnn_module_str)

    expected_results = [
        ExecutionPhase.EXECUTED_FLATBUFFER,
        ExecutionPhase.EXECUTED_FLATBUFFER,
    ]
    for i, result in enumerate(results):
        assert result.execution_phase == expected_results[i]


def test2_compile_split_and_execute_ttnn_module(ttnn_module_str: str):
    results = compile_split_and_execute(ttnn_module_str)

    expected_results = [
        ExecutionPhase.EXECUTED_FLATBUFFER,
        ExecutionPhase.EXECUTED_FLATBUFFER,
    ]
    for i, result in enumerate(results):
        assert result.execution_phase == expected_results[i]


def test_split_compile_split_and_execute_ttnn_module(ttnn_module_str: str):
    results = split_compile_split_and_execute(ttnn_module_str)

    expected_results = [
        ExecutionPhase.EXECUTED_FLATBUFFER,
        ExecutionPhase.EXECUTED_FLATBUFFER,
    ]
    for i, result in enumerate(results):
        assert result.execution_phase == expected_results[i]
