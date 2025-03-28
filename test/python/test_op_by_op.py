# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pprint import pprint
from typing import List

from ttmlir.execution_result import ExecutionPhase, ExecutionResult
from ttmlir.workflows import (
    compile_split_and_execute,
    convert_results_to_pydantic_models,
    split_and_execute,
    split_compile_split_and_execute,
)


def _print_results_as_pydantic_models(results: List[ExecutionResult]):
    models = convert_results_to_pydantic_models(results)
    for m in models:
        print(f"Showing pydantic report for op {m.op_name}:")
        pprint(m.model_dump())


def test_shlo(print_results: bool = False):
    def test1(print_results: bool = False):
        results = split_and_execute(shlo_module_str)

        assert all(
            result.execution_phase == ExecutionPhase.EXECUTED_FLATBUFFER
            for result in results
        ), f"Expected all results to be in EXECUTED_FLATBUFFER phase, got: {results}"

        if print_results:
            _print_results_as_pydantic_models(results)

    def test2(print_results: bool = False):
        results = compile_split_and_execute(shlo_module_str)

        # TODO special case where module consists solely of dealloc op. See what should
        # be done with it.
        assert results[-1].execution_phase == ExecutionPhase.GENERATED_TTNN
        assert all(
            result.execution_phase == ExecutionPhase.EXECUTED_FLATBUFFER
            for result in results[:-1]
        ), f"Expected all results to be in EXECUTED_FLATBUFFER phase, got: {results}"

        if print_results:
            _print_results_as_pydantic_models(results)

    def test3(print_results: bool = False):
        results = split_compile_split_and_execute(shlo_module_str)

        assert all(
            result.execution_phase == ExecutionPhase.EXECUTED_FLATBUFFER
            for result in results
        ), f"Expected all results to be in EXECUTED_FLATBUFFER phase, got: {results}"

        if print_results:
            _print_results_as_pydantic_models(results)

    shlo_module_str = """
        module {
            func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
                %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
                %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
                %2 = stablehlo.add %0, %1 : tensor<1x128xf32>
                return %2 : tensor<1x128xf32>
            }
        }
    """

    test1(print_results)
    test2(print_results)
    test3(print_results)
    print("SHLO tests passed.")


def test_ttir(print_results: bool = False):
    def test1(print_results: bool = False):
        results = split_and_execute(ttir_module_str)

        assert all(
            result.execution_phase == ExecutionPhase.EXECUTED_FLATBUFFER
            for result in results
        ), f"Expected all results to be in EXECUTED_FLATBUFFER phase, got: {results}"

        if print_results:
            _print_results_as_pydantic_models(results)

    def test2(print_results: bool = False):
        results = compile_split_and_execute(ttir_module_str)

        # TODO special case where module consists solely of dealloc op. See what should
        # be done with it.
        assert results[-1].execution_phase == ExecutionPhase.GENERATED_TTNN
        assert all(
            result.execution_phase == ExecutionPhase.EXECUTED_FLATBUFFER
            for result in results[:-1]
        ), f"Expected all results to be in EXECUTED_FLATBUFFER phase, got: {results}"

        if print_results:
            _print_results_as_pydantic_models(results)

    def test3(print_results: bool = False):
        results = split_compile_split_and_execute(ttir_module_str)

        assert all(
            result.execution_phase == ExecutionPhase.EXECUTED_FLATBUFFER
            for result in results
        ), f"Expected all results to be in EXECUTED_FLATBUFFER phase, got: {results}"

        if print_results:
            _print_results_as_pydantic_models(results)

    ttir_module_str = """
        module {
            func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
                %0 = tensor.empty() : tensor<1x128xf32>
                %1 = "ttir.broadcast"(%arg0, %0) <{broadcast_dimensions = array<i64: 1, 1>}> : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
                %2 = tensor.empty() : tensor<1x128xf32>
                %3 = "ttir.reshape"(%arg1, %2) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
                %4 = tensor.empty() : tensor<1x128xf32>
                %5 = "ttir.broadcast"(%3, %4) <{broadcast_dimensions = array<i64: 1, 1>}> : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
                %6 = tensor.empty() : tensor<1x128xf32>
                %7 = "ttir.add"(%1, %5, %6) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
                return %7 : tensor<1x128xf32>
            }
        }
    """

    test1(print_results)
    test2(print_results)
    print("TTIR tests passed.")
    # test3(print_results)  # TODO skipped due to a TTNN-graph level bug.
    # `ttnn.get_device` and `ttnn.empty` cannot be executed on
    # their own. First one fails in runtime, second one when
    # trying to generate fb.


def test_ttnn(print_results: bool = False):
    def test1(print_results: bool = False):
        results = split_and_execute(ttnn_module_str)

        # TODO special case where module consists solely of dealloc op. See what should
        # be done with it.
        assert results[-1].execution_phase == ExecutionPhase.GENERATED_TTNN
        assert all(
            result.execution_phase == ExecutionPhase.EXECUTED_FLATBUFFER
            for result in results[:-1]
        ), f"Expected all results to be in EXECUTED_FLATBUFFER phase, got: {results}"

        if print_results:
            _print_results_as_pydantic_models(results)

    def test2(print_results: bool = False):
        results = compile_split_and_execute(ttnn_module_str)

        # TODO special case where module consists solely of dealloc op. See what should
        # be done with it.
        assert results[-1].execution_phase == ExecutionPhase.GENERATED_TTNN
        assert all(
            result.execution_phase == ExecutionPhase.EXECUTED_FLATBUFFER
            for result in results[:-1]
        ), f"Expected all results to be in EXECUTED_FLATBUFFER phase, got: {results}"

        if print_results:
            _print_results_as_pydantic_models(results)

    def test3(print_results: bool = False):
        results = split_compile_split_and_execute(ttnn_module_str)

        # TODO special case where module consists solely of dealloc op. See what should
        # be done with it.
        assert results[-1].execution_phase == ExecutionPhase.GENERATED_TTNN
        assert all(
            result.execution_phase == ExecutionPhase.EXECUTED_FLATBUFFER
            for result in results[:-1]
        ), f"Expected all results to be in EXECUTED_FLATBUFFER phase, got: {results}"

        if print_results:
            _print_results_as_pydantic_models(results)

    ttnn_module_str = """
        #device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
        #dram = #ttnn.buffer_type<dram>
        #system_desc = #tt.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 99488, erisc_l1_unreserved_base = 104864, dram_unreserved_base = 32, dram_unreserved_end = 1073144896, physical_cores = {worker = [ 18x18,  18x19,  18x20,  18x21,  18x22,  18x23,  18x24,  18x25,  19x18,  19x19,  19x20,  19x21,  19x22,  19x23,  19x24,  19x25,  20x18,  20x19,  20x20,  20x21,  20x22,  20x23,  20x24,  20x25,  21x18,  21x19,  21x20,  21x21,  21x22,  21x23,  21x24,  21x25,  22x18,  22x19,  22x20,  22x21,  22x22,  22x23,  22x24,  22x25,  23x18,  23x19,  23x20,  23x21,  23x22,  23x23,  23x24,  23x25,  24x18,  24x19,  24x20,  24x21,  24x22,  24x23,  24x24,  24x25,  25x18,  25x19,  25x20,  25x21,  25x22,  25x23,  25x24,  25x25] dram = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  0x8,  0x9,  0x10,  0x11] eth_inactive = [ 16x18,  16x19,  16x20,  16x21,  16x22,  16x23,  16x24,  16x25,  17x19,  17x20,  17x22,  17x23,  17x24]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32}], [0], [3 : i32], [ 0x0x0x0]>
        #ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
        #ttnn_layout1 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
        module attributes {tt.device = #device, tt.system_desc = #system_desc} {
            func.func @main(%arg0: tensor<1x128xf32, #ttnn_layout>, %arg1: tensor<128xf32, #ttnn_layout1>) -> tensor<1x128xf32, #ttnn_layout> {
                %0 = "ttnn.reshape"(%arg1) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32, #ttnn_layout1>) -> tensor<1x128xf32, #ttnn_layout>
                %1 = "ttnn.add"(%arg0, %0) : (tensor<1x128xf32, #ttnn_layout>, tensor<1x128xf32, #ttnn_layout>) -> tensor<1x128xf32, #ttnn_layout>
                "ttnn.deallocate"(%0) <{force = false}> : (tensor<1x128xf32, #ttnn_layout>) -> ()
                return %1 : tensor<1x128xf32, #ttnn_layout>
            }
        }
    """

    test1(print_results)
    test2(print_results)
    test3(print_results)
    print("TTNN tests passed.")


if __name__ == "__main__":
    test_shlo(True)
    test_ttir(True)
    test_ttnn(True)
