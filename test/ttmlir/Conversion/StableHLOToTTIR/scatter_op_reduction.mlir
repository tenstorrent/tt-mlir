// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline  -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_scatter_reduction attributes {} {
    func.func public @test_scatter_with_add_reduction(%arg0: tensor<151936x2048xbf16>, %arg1: tensor<128x1xi64>, %arg2: tensor<128x2048xbf16>) -> tensor<151936x2048xbf16> {
        // CHECK: [[VAL0:%[0-9]+]] = "ttir.repeat"(%arg1)
        // CHECK-SAME: <{repeat_dimensions = array<i64: 1, 2048>}> : (tensor<128x1xi64>) -> tensor<128x2048xi64>
        %result = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
        ^bb0(%arg3: tensor<bf16>, %arg4: tensor<bf16>):
            %add = stablehlo.maximum %arg3, %arg4 : tensor<bf16>
            stablehlo.return %add : tensor<bf16>
        }) : (tensor<151936x2048xbf16>, tensor<128x1xi64>, tensor<128x2048xbf16>) -> tensor<151936x2048xbf16>
        // CHECK: [[VAL1:%[0-9]+]] = "ttir.scatter"(%arg0, [[VAL0]], %arg2)
        // CHECK-SAME: <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<max>}>
        // CHECK-SAME: (tensor<151936x2048xbf16>, tensor<128x2048xi64>, tensor<128x2048xbf16>) -> tensor<151936x2048xbf16>
        return %result : tensor<151936x2048xbf16>
        // CHECK: return [[VAL1]] : tensor<151936x2048xbf16>
    }
}
