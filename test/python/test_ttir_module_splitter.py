# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttmlir.ir import *
from ttmlir.ttir_module_splitter import TTIRModuleSplitter


def test1():
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

    splitter: TTIRModuleSplitter = TTIRModuleSplitter.create_from_module_str(
        ttir_module_str
    )

    for op in splitter.get_sub_ops():
        print(op)

    for m in splitter.get_sub_modules():
        print(str(m))


def test2():
    ttir_module_str = """
    module {
        func.func public @main(%arg0: tensor<1x4xi32>, %arg1: tensor<512x768xf32>, %arg2: tensor<2x768xf32>, %arg3: tensor<30522x768xf32>) -> tensor<1x4x768xf32> {
            %0 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
            %1 = tensor.empty() : tensor<1x1xi32>
            %2 = "ttir.reshape"(%0, %1) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
            %3 = tensor.empty() : tensor<1x4xi32>
            %4 = "ttir.broadcast"(%2, %3) <{broadcast_dimensions = array<i64: 1, 4>}> : (tensor<1x1xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
            %5 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 4 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<4xi32>
            %6 = tensor.empty() : tensor<1x4xi32>
            %7 = "ttir.reshape"(%5, %6) <{shape = [1 : i32, 4 : i32]}> : (tensor<4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
            %8 = tensor.empty() : tensor<1x4xi32>
            %9 = "ttir.broadcast"(%7, %8) <{broadcast_dimensions = array<i64: 1, 1>}> : (tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
            %10 = call @_take(%arg3, %arg0) : (tensor<30522x768xf32>, tensor<1x4xi32>) -> tensor<1x4x768xf32>
            %11 = call @_take_0(%arg1, %9) : (tensor<512x768xf32>, tensor<1x4xi32>) -> tensor<1x4x768xf32>
            %12 = call @_take_1(%arg2, %4) : (tensor<2x768xf32>, tensor<1x4xi32>) -> tensor<1x4x768xf32>
            %13 = tensor.empty() : tensor<1x4x768xf32>
            %14 = "ttir.add"(%10, %12, %13) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x4x768xf32>, tensor<1x4x768xf32>, tensor<1x4x768xf32>) -> tensor<1x4x768xf32>
            %15 = tensor.empty() : tensor<1x4x768xf32>
            %16 = "ttir.add"(%14, %11, %15) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x4x768xf32>, tensor<1x4x768xf32>, tensor<1x4x768xf32>) -> tensor<1x4x768xf32>
            return %16 : tensor<1x4x768xf32>
        }
        func.func private @_take(%arg0: tensor<30522x768xf32>, %arg1: tensor<1x4xi32>) -> tensor<1x4x768xf32> {
            %0 = "ttir.constant"() <{value = dense<0x7FC00000> : tensor<1xf32>}> : () -> tensor<1xf32>
            %1 = "ttir.constant"() <{value = dense<30521> : tensor<1xi32>}> : () -> tensor<1xi32>
            %2 = "ttir.constant"() <{value = dense<30522> : tensor<1xi32>}> : () -> tensor<1xi32>
            %3 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
            %4 = tensor.empty() : tensor<1x1xi32>
            %5 = "ttir.reshape"(%3, %4) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
            %6 = tensor.empty() : tensor<1x4xi32>
            %7 = "ttir.broadcast"(%5, %6) <{broadcast_dimensions = array<i64: 1, 4>}> : (tensor<1x1xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
            %8 = tensor.empty() : tensor<1x4xbf16>
            %9 = "ttir.lt"(%arg1, %7, %8) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xbf16>) -> tensor<1x4xbf16>
            %10 = tensor.empty() : tensor<1x1xi32>
            %11 = "ttir.reshape"(%2, %10) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
            %12 = tensor.empty() : tensor<1x4xi32>
            %13 = "ttir.broadcast"(%11, %12) <{broadcast_dimensions = array<i64: 1, 4>}> : (tensor<1x1xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
            %14 = tensor.empty() : tensor<1x4xi32>
            %15 = "ttir.add"(%arg1, %13, %14) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
            %16 = call @_where(%9, %15, %arg1) : (tensor<1x4xbf16>, tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
            %17 = tensor.empty() : tensor<1x4x1xi32>
            %18 = "ttir.reshape"(%16, %17) <{shape = [1 : i32, 4 : i32, 1 : i32]}> : (tensor<1x4xi32>, tensor<1x4x1xi32>) -> tensor<1x4x1xi32>
            %19 = tensor.empty() : tensor<1x4x1xi32>
            %20 = "ttir.broadcast"(%18, %19) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x4x1xi32>, tensor<1x4x1xi32>) -> tensor<1x4x1xi32>
            %21 = tensor.empty() : tensor<1x1x1xi32>
            %22 = "ttir.reshape"(%3, %21) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
            %23 = tensor.empty() : tensor<1x4x1xi32>
            %24 = "ttir.broadcast"(%22, %23) <{broadcast_dimensions = array<i64: 1, 4, 1>}> : (tensor<1x1x1xi32>, tensor<1x4x1xi32>) -> tensor<1x4x1xi32>
            %25 = tensor.empty() : tensor<1x4x1xbf16>
            %26 = "ttir.ge"(%20, %24, %25) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x4x1xi32>, tensor<1x4x1xi32>, tensor<1x4x1xbf16>) -> tensor<1x4x1xbf16>
            %27 = tensor.empty() : tensor<1x1x1xi32>
            %28 = "ttir.reshape"(%1, %27) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
            %29 = tensor.empty() : tensor<1x1x1xi32>
            %30 = "ttir.broadcast"(%28, %29) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
            %31 = tensor.empty() : tensor<1x4x1xi32>
            %32 = "ttir.broadcast"(%30, %31) <{broadcast_dimensions = array<i64: 1, 4, 1>}> : (tensor<1x1x1xi32>, tensor<1x4x1xi32>) -> tensor<1x4x1xi32>
            %33 = tensor.empty() : tensor<1x4x1xbf16>
            %34 = "ttir.le"(%20, %32, %33) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x4x1xi32>, tensor<1x4x1xi32>, tensor<1x4x1xbf16>) -> tensor<1x4x1xbf16>
            %35 = tensor.empty() : tensor<1x4x1xbf16>
            %36 = "ttir.logical_and"(%26, %34, %35) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x4x1xbf16>, tensor<1x4x1xbf16>, tensor<1x4x1xbf16>) -> tensor<1x4x1xbf16>
            %37 = tensor.empty() : tensor<1x4xbf16>
            %38 = "ttir.reduce_and"(%36, %37) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x4x1xbf16>, tensor<1x4xbf16>) -> tensor<1x4xbf16>
            %39 = tensor.empty() : tensor<1x4x768xf32>
            %40 = "ttir.gather"(%arg0, %20, %39) <{collapsed_slice_dims = array<i64: 0>, index_vector_dim = 2 : si64, indices_are_sorted = false, offset_dims = array<i64: 2>, operand_batching_dims = array<i64>, slice_sizes = array<i64: 1, 768>, start_index_map = array<i64: 0>, start_indices_batching_dims = array<i64>}> : (tensor<30522x768xf32>, tensor<1x4x1xi32>, tensor<1x4x768xf32>) -> tensor<1x4x768xf32>
            %41 = tensor.empty() : tensor<1x4x1xbf16>
            %42 = "ttir.reshape"(%38, %41) <{shape = [1 : i32, 4 : i32, 1 : i32]}> : (tensor<1x4xbf16>, tensor<1x4x1xbf16>) -> tensor<1x4x1xbf16>
            %43 = tensor.empty() : tensor<1x4x768xbf16>
            %44 = "ttir.broadcast"(%42, %43) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x4x1xbf16>, tensor<1x4x768xbf16>) -> tensor<1x4x768xbf16>
            %45 = tensor.empty() : tensor<1x1x1xf32>
            %46 = "ttir.reshape"(%0, %45) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
            %47 = tensor.empty() : tensor<1x4x768xf32>
            %48 = "ttir.broadcast"(%46, %47) <{broadcast_dimensions = array<i64: 1, 4, 768>}> : (tensor<1x1x1xf32>, tensor<1x4x768xf32>) -> tensor<1x4x768xf32>
            %49 = tensor.empty() : tensor<1x4x768xf32>
            %50 = "ttir.where"(%44, %40, %48, %49) <{operandSegmentSizes = array<i32: 3, 1>}> : (tensor<1x4x768xbf16>, tensor<1x4x768xf32>, tensor<1x4x768xf32>, tensor<1x4x768xf32>) -> tensor<1x4x768xf32>
            return %50 : tensor<1x4x768xf32>
        }
        func.func private @_take_0(%arg0: tensor<512x768xf32>, %arg1: tensor<1x4xi32>) -> tensor<1x4x768xf32> {
            %0 = "ttir.constant"() <{value = dense<0x7FC00000> : tensor<1xf32>}> : () -> tensor<1xf32>
            %1 = "ttir.constant"() <{value = dense<511> : tensor<1xi32>}> : () -> tensor<1xi32>
            %2 = "ttir.constant"() <{value = dense<512> : tensor<1xi32>}> : () -> tensor<1xi32>
            %3 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
            %4 = tensor.empty() : tensor<1x1xi32>
            %5 = "ttir.reshape"(%3, %4) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
            %6 = tensor.empty() : tensor<1x4xi32>
            %7 = "ttir.broadcast"(%5, %6) <{broadcast_dimensions = array<i64: 1, 4>}> : (tensor<1x1xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
            %8 = tensor.empty() : tensor<1x4xbf16>
            %9 = "ttir.lt"(%arg1, %7, %8) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xbf16>) -> tensor<1x4xbf16>
            %10 = tensor.empty() : tensor<1x1xi32>
            %11 = "ttir.reshape"(%2, %10) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
            %12 = tensor.empty() : tensor<1x4xi32>
            %13 = "ttir.broadcast"(%11, %12) <{broadcast_dimensions = array<i64: 1, 4>}> : (tensor<1x1xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
            %14 = tensor.empty() : tensor<1x4xi32>
            %15 = "ttir.add"(%arg1, %13, %14) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
            %16 = call @_where(%9, %15, %arg1) : (tensor<1x4xbf16>, tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
            %17 = tensor.empty() : tensor<1x4x1xi32>
            %18 = "ttir.reshape"(%16, %17) <{shape = [1 : i32, 4 : i32, 1 : i32]}> : (tensor<1x4xi32>, tensor<1x4x1xi32>) -> tensor<1x4x1xi32>
            %19 = tensor.empty() : tensor<1x4x1xi32>
            %20 = "ttir.broadcast"(%18, %19) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x4x1xi32>, tensor<1x4x1xi32>) -> tensor<1x4x1xi32>
            %21 = tensor.empty() : tensor<1x1x1xi32>
            %22 = "ttir.reshape"(%3, %21) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
            %23 = tensor.empty() : tensor<1x4x1xi32>
            %24 = "ttir.broadcast"(%22, %23) <{broadcast_dimensions = array<i64: 1, 4, 1>}> : (tensor<1x1x1xi32>, tensor<1x4x1xi32>) -> tensor<1x4x1xi32>
            %25 = tensor.empty() : tensor<1x4x1xbf16>
            %26 = "ttir.ge"(%20, %24, %25) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x4x1xi32>, tensor<1x4x1xi32>, tensor<1x4x1xbf16>) -> tensor<1x4x1xbf16>
            %27 = tensor.empty() : tensor<1x1x1xi32>
            %28 = "ttir.reshape"(%1, %27) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
            %29 = tensor.empty() : tensor<1x1x1xi32>
            %30 = "ttir.broadcast"(%28, %29) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
            %31 = tensor.empty() : tensor<1x4x1xi32>
            %32 = "ttir.broadcast"(%30, %31) <{broadcast_dimensions = array<i64: 1, 4, 1>}> : (tensor<1x1x1xi32>, tensor<1x4x1xi32>) -> tensor<1x4x1xi32>
            %33 = tensor.empty() : tensor<1x4x1xbf16>
            %34 = "ttir.le"(%20, %32, %33) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x4x1xi32>, tensor<1x4x1xi32>, tensor<1x4x1xbf16>) -> tensor<1x4x1xbf16>
            %35 = tensor.empty() : tensor<1x4x1xbf16>
            %36 = "ttir.logical_and"(%26, %34, %35) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x4x1xbf16>, tensor<1x4x1xbf16>, tensor<1x4x1xbf16>) -> tensor<1x4x1xbf16>
            %37 = tensor.empty() : tensor<1x4xbf16>
            %38 = "ttir.reduce_and"(%36, %37) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x4x1xbf16>, tensor<1x4xbf16>) -> tensor<1x4xbf16>
            %39 = tensor.empty() : tensor<1x4x768xf32>
            %40 = "ttir.gather"(%arg0, %20, %39) <{collapsed_slice_dims = array<i64: 0>, index_vector_dim = 2 : si64, indices_are_sorted = false, offset_dims = array<i64: 2>, operand_batching_dims = array<i64>, slice_sizes = array<i64: 1, 768>, start_index_map = array<i64: 0>, start_indices_batching_dims = array<i64>}> : (tensor<512x768xf32>, tensor<1x4x1xi32>, tensor<1x4x768xf32>) -> tensor<1x4x768xf32>
            %41 = tensor.empty() : tensor<1x4x1xbf16>
            %42 = "ttir.reshape"(%38, %41) <{shape = [1 : i32, 4 : i32, 1 : i32]}> : (tensor<1x4xbf16>, tensor<1x4x1xbf16>) -> tensor<1x4x1xbf16>
            %43 = tensor.empty() : tensor<1x4x768xbf16>
            %44 = "ttir.broadcast"(%42, %43) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x4x1xbf16>, tensor<1x4x768xbf16>) -> tensor<1x4x768xbf16>
            %45 = tensor.empty() : tensor<1x1x1xf32>
            %46 = "ttir.reshape"(%0, %45) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
            %47 = tensor.empty() : tensor<1x4x768xf32>
            %48 = "ttir.broadcast"(%46, %47) <{broadcast_dimensions = array<i64: 1, 4, 768>}> : (tensor<1x1x1xf32>, tensor<1x4x768xf32>) -> tensor<1x4x768xf32>
            %49 = tensor.empty() : tensor<1x4x768xf32>
            %50 = "ttir.where"(%44, %40, %48, %49) <{operandSegmentSizes = array<i32: 3, 1>}> : (tensor<1x4x768xbf16>, tensor<1x4x768xf32>, tensor<1x4x768xf32>, tensor<1x4x768xf32>) -> tensor<1x4x768xf32>
            return %50 : tensor<1x4x768xf32>
        }
        func.func private @_take_1(%arg0: tensor<2x768xf32>, %arg1: tensor<1x4xi32>) -> tensor<1x4x768xf32> {
            %0 = "ttir.constant"() <{value = dense<0x7FC00000> : tensor<1xf32>}> : () -> tensor<1xf32>
            %1 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
            %2 = "ttir.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
            %3 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
            %4 = tensor.empty() : tensor<1x1xi32>
            %5 = "ttir.reshape"(%3, %4) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
            %6 = tensor.empty() : tensor<1x4xi32>
            %7 = "ttir.broadcast"(%5, %6) <{broadcast_dimensions = array<i64: 1, 4>}> : (tensor<1x1xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
            %8 = tensor.empty() : tensor<1x4xbf16>
            %9 = "ttir.lt"(%arg1, %7, %8) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xbf16>) -> tensor<1x4xbf16>
            %10 = tensor.empty() : tensor<1x1xi32>
            %11 = "ttir.reshape"(%2, %10) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
            %12 = tensor.empty() : tensor<1x4xi32>
            %13 = "ttir.broadcast"(%11, %12) <{broadcast_dimensions = array<i64: 1, 4>}> : (tensor<1x1xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
            %14 = tensor.empty() : tensor<1x4xi32>
            %15 = "ttir.add"(%arg1, %13, %14) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
            %16 = call @_where(%9, %15, %arg1) : (tensor<1x4xbf16>, tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
            %17 = tensor.empty() : tensor<1x4x1xi32>
            %18 = "ttir.reshape"(%16, %17) <{shape = [1 : i32, 4 : i32, 1 : i32]}> : (tensor<1x4xi32>, tensor<1x4x1xi32>) -> tensor<1x4x1xi32>
            %19 = tensor.empty() : tensor<1x4x1xi32>
            %20 = "ttir.broadcast"(%18, %19) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x4x1xi32>, tensor<1x4x1xi32>) -> tensor<1x4x1xi32>
            %21 = tensor.empty() : tensor<1x1x1xi32>
            %22 = "ttir.reshape"(%3, %21) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
            %23 = tensor.empty() : tensor<1x4x1xi32>
            %24 = "ttir.broadcast"(%22, %23) <{broadcast_dimensions = array<i64: 1, 4, 1>}> : (tensor<1x1x1xi32>, tensor<1x4x1xi32>) -> tensor<1x4x1xi32>
            %25 = tensor.empty() : tensor<1x4x1xbf16>
            %26 = "ttir.ge"(%20, %24, %25) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x4x1xi32>, tensor<1x4x1xi32>, tensor<1x4x1xbf16>) -> tensor<1x4x1xbf16>
            %27 = tensor.empty() : tensor<1x1x1xi32>
            %28 = "ttir.reshape"(%1, %27) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
            %29 = tensor.empty() : tensor<1x1x1xi32>
            %30 = "ttir.broadcast"(%28, %29) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
            %31 = tensor.empty() : tensor<1x4x1xi32>
            %32 = "ttir.broadcast"(%30, %31) <{broadcast_dimensions = array<i64: 1, 4, 1>}> : (tensor<1x1x1xi32>, tensor<1x4x1xi32>) -> tensor<1x4x1xi32>
            %33 = tensor.empty() : tensor<1x4x1xbf16>
            %34 = "ttir.le"(%20, %32, %33) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x4x1xi32>, tensor<1x4x1xi32>, tensor<1x4x1xbf16>) -> tensor<1x4x1xbf16>
            %35 = tensor.empty() : tensor<1x4x1xbf16>
            %36 = "ttir.logical_and"(%26, %34, %35) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x4x1xbf16>, tensor<1x4x1xbf16>, tensor<1x4x1xbf16>) -> tensor<1x4x1xbf16>
            %37 = tensor.empty() : tensor<1x4xbf16>
            %38 = "ttir.reduce_and"(%36, %37) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x4x1xbf16>, tensor<1x4xbf16>) -> tensor<1x4xbf16>
            %39 = tensor.empty() : tensor<1x4x768xf32>
            %40 = "ttir.gather"(%arg0, %20, %39) <{collapsed_slice_dims = array<i64: 0>, index_vector_dim = 2 : si64, indices_are_sorted = false, offset_dims = array<i64: 2>, operand_batching_dims = array<i64>, slice_sizes = array<i64: 1, 768>, start_index_map = array<i64: 0>, start_indices_batching_dims = array<i64>}> : (tensor<2x768xf32>, tensor<1x4x1xi32>, tensor<1x4x768xf32>) -> tensor<1x4x768xf32>
            %41 = tensor.empty() : tensor<1x4x1xbf16>
            %42 = "ttir.reshape"(%38, %41) <{shape = [1 : i32, 4 : i32, 1 : i32]}> : (tensor<1x4xbf16>, tensor<1x4x1xbf16>) -> tensor<1x4x1xbf16>
            %43 = tensor.empty() : tensor<1x4x768xbf16>
            %44 = "ttir.broadcast"(%42, %43) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x4x1xbf16>, tensor<1x4x768xbf16>) -> tensor<1x4x768xbf16>
            %45 = tensor.empty() : tensor<1x1x1xf32>
            %46 = "ttir.reshape"(%0, %45) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
            %47 = tensor.empty() : tensor<1x4x768xf32>
            %48 = "ttir.broadcast"(%46, %47) <{broadcast_dimensions = array<i64: 1, 4, 768>}> : (tensor<1x1x1xf32>, tensor<1x4x768xf32>) -> tensor<1x4x768xf32>
            %49 = tensor.empty() : tensor<1x4x768xf32>
            %50 = "ttir.where"(%44, %40, %48, %49) <{operandSegmentSizes = array<i32: 3, 1>}> : (tensor<1x4x768xbf16>, tensor<1x4x768xf32>, tensor<1x4x768xf32>, tensor<1x4x768xf32>) -> tensor<1x4x768xf32>
            return %50 : tensor<1x4x768xf32>
        }
        func.func private @_where(%arg0: tensor<1x4xbf16>, %arg1: tensor<1x4xi32>, %arg2: tensor<1x4xi32>) -> tensor<1x4xi32> {
            %0 = tensor.empty() : tensor<1x4xi32>
            %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) <{operandSegmentSizes = array<i32: 3, 1>}> : (tensor<1x4xbf16>, tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
            return %1 : tensor<1x4xi32>
        }
    }
    """

    splitter: TTIRModuleSplitter = TTIRModuleSplitter.create_from_module_str(
        ttir_module_str
    )

    for m in splitter.get_sub_modules():
        print(str(m))


if __name__ == "__main__":
    test1()
    # test2()
