// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module @jit_scatter attributes {} {
    func.func public @test_scatter(%arg0: tensor<1x3x320x320xf32>, %arg1: tensor<1x1xi64>, %arg2: tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32> {
        // CHECK: [[VAL0:%[0-9]+]] = tensor.empty() : [[TENSOR_SIZE1:tensor<[0-9]+x[0-9]+x[0-9]+x[0-9]+xf[0-9]+>]]
        %result = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
        ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
            stablehlo.return %arg4 : tensor<f32>
        }) : (tensor<1x3x320x320xf32>, tensor<1x1xi64>, tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32>
        // CHECK: [[VAL1:%[0-9]+]] = "ttir.scatter"(%arg0, %arg1, %arg2, [[VAL0]]) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile, #any_device_tile], scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 1, 2, 3>}
        // CHECK: ([[TENSOR_SIZE1]], tensor<1x1xi32>, tensor<1x3x32x32xf32>, [[TENSOR_SIZE1]]) -> tensor<1x3x320x320xf32>
        return %result : tensor<1x3x320x320xf32>
        // CHECK: return [[VAL1]] : [[TENSOR_SIZE1]]
    }
}
