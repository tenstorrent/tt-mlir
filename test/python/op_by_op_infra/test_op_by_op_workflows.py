# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from ttmlir.execution_result import ExecutionPhase
from ttmlir.workflow_internal import (
    compile_split_and_execute,
    split_and_execute,
    split_compile_split_and_execute,
)


@pytest.fixture
def shlo_module_str() -> str:
    return """
        module {
            func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
                %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
                %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
                %2 = stablehlo.add %0, %1 : tensor<1x128xf32>
                return %2 : tensor<1x128xf32>
            }
        }
    """


@pytest.fixture
def ttir_module_str() -> str:
    return """
        module {
        func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
            %0 = ttir.empty() : tensor<1x128xf32>
            %1 = "ttir.broadcast"(%arg0, %0) <{broadcast_dimensions = array<i64: 1, 1>}> : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
            %2 = ttir.empty() : tensor<1x128xf32>
            %3 = "ttir.reshape"(%arg1, %2) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
            %4 = ttir.empty() : tensor<1x128xf32>
            %5 = "ttir.broadcast"(%3, %4) <{broadcast_dimensions = array<i64: 1, 1>}> : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
            %6 = ttir.empty() : tensor<1x128xf32>
            %7 = "ttir.add"(%1, %5, %6) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
            return %7 : tensor<1x128xf32>
        }
        }
    """


@pytest.fixture
def ttnn_module_str() -> str:
    return """
        #dram = #ttnn.buffer_type<dram>
        #system_desc = #tt.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 97248, erisc_l1_unreserved_base = 104992, dram_unreserved_base = 32, dram_unreserved_end = 1073158336, physical_helper_cores = {dram = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  0x8,  0x9,  0x10,  0x11] eth_inactive = [ 16x18,  16x19,  16x20,  16x21,  16x22,  16x23,  16x24,  16x25,  17x19,  17x20,  17x22,  17x23,  17x24]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [3 : i32], [ 0x0x0x0]>
        #ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
        #ttnn_layout1 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
        module {
            tt.device_module {
                builtin.module attributes {tt.system_desc = #system_desc} {
                    tt.device @default_device = <workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5] -> (0, 0, (((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv s4) mod 12, ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv (s4 * 12) + ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
                    func.func @main(%arg0: tensor<1x128xf32, #ttnn_layout>, %arg1: tensor<128xf32, #ttnn_layout1>) -> tensor<1x128xf32, #ttnn_layout> {
                        %0 = "ttnn.reshape"(%arg1) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32, #ttnn_layout1>) -> tensor<1x128xf32, #ttnn_layout>
                        "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<128xf32, #ttnn_layout1>) -> ()
                        %1 = "ttnn.add"(%arg0, %0) : (tensor<1x128xf32, #ttnn_layout>, tensor<1x128xf32, #ttnn_layout>) -> tensor<1x128xf32, #ttnn_layout>
                        "ttnn.deallocate"(%0) <{force = false}> : (tensor<1x128xf32, #ttnn_layout>) -> ()
                        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x128xf32, #ttnn_layout>) -> ()
                        return %1 : tensor<1x128xf32, #ttnn_layout>
                    }
                }
            }
        }
    """


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
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.EXECUTED_FLATBUFFER,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.GENERATED_TTNN,
    ]
    for i, result in enumerate(results):
        assert result.execution_phase == expected_results[i]


def test_split_compile_split_and_execute_shlo_module(shlo_module_str: str):
    results = split_compile_split_and_execute(shlo_module_str)

    expected_results = [
        ExecutionPhase.EXECUTED_FLATBUFFER,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.EXECUTED_FLATBUFFER,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.GENERATED_TTNN,
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
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.EXECUTED_FLATBUFFER,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.GENERATED_TTNN,
    ]
    for i, result in enumerate(results):
        assert result.execution_phase == expected_results[i]


def test_split_compile_split_and_execute_ttir_module(ttir_module_str: str):
    results = split_compile_split_and_execute(ttir_module_str)

    expected_results = [
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.EXECUTED_FLATBUFFER,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.EXECUTED_FLATBUFFER,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.GENERATED_TTNN,
    ]
    for i, result in enumerate(results):
        assert result.execution_phase == expected_results[i]


def test_split_and_execute_ttnn_module(ttnn_module_str: str):
    results = split_and_execute(ttnn_module_str)

    expected_results = [
        ExecutionPhase.EXECUTED_FLATBUFFER,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.EXECUTED_FLATBUFFER,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.GENERATED_TTNN,
    ]
    for i, result in enumerate(results):
        assert result.execution_phase == expected_results[i]


def test2_compile_split_and_execute_ttnn_module(ttnn_module_str: str):
    results = compile_split_and_execute(ttnn_module_str)

    expected_results = [
        ExecutionPhase.EXECUTED_FLATBUFFER,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.EXECUTED_FLATBUFFER,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.GENERATED_TTNN,
    ]
    for i, result in enumerate(results):
        assert result.execution_phase == expected_results[i]


def test_split_compile_split_and_execute_ttnn_module(ttnn_module_str: str):
    results = split_compile_split_and_execute(ttnn_module_str)

    expected_results = [
        ExecutionPhase.EXECUTED_FLATBUFFER,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.EXECUTED_FLATBUFFER,
        ExecutionPhase.GENERATED_TTNN,
        ExecutionPhase.GENERATED_TTNN,
    ]
    for i, result in enumerate(results):
        assert result.execution_phase == expected_results[i]
