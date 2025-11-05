// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline  -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_scatter_reduction attributes {} {
    func.func public @test_scatter_with_add_reduction(%arg0: tensor<151936x2048xbf16>, %arg1: tensor<128x1xi64>, %arg2: tensor<128x2048xbf16>) -> tensor<151936x2048xbf16> {
        // CHECK: [[VAL0:%[0-9]+]] = ttir.empty() : [[TENSOR_SIZE1:tensor<[0-9]+x[0-9]+xbf16>]]
        %result = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
        ^bb0(%arg3: tensor<bf16>, %arg4: tensor<bf16>):
            %add = stablehlo.add %arg3, %arg4 : tensor<bf16>
            stablehlo.return %add : tensor<bf16>
        }) : (tensor<151936x2048xbf16>, tensor<128x1xi64>, tensor<128x2048xbf16>) -> tensor<151936x2048xbf16>
        // CHECK: [[VAL1:%[0-9]+]] = "ttir.scatter"(%arg0, %arg1, %arg2, [[VAL0]])
        // CHECK-SAME: scatter_reduce_type = #ttcore.reduce_type<sum>
        return %result : tensor<151936x2048xbf16>
        // CHECK: return [[VAL1]] : [[TENSOR_SIZE1]]
    }
}
