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

    func.func public @test_scatter_with_add_reduction_multidim_i32(
        %arg0: tensor<200x100x300xf32>, %arg1: tensor<10x2xi32>,
        %arg2: tensor<10x300xf32>) -> tensor<200x100x300xf32> {
        %result = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
        ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
            %add = stablehlo.add %arg3, %arg4 : tensor<f32>
            stablehlo.return %add : tensor<f32>
        }) : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) -> tensor<200x100x300xf32>
        // CHECK-LABEL: func.func public @test_scatter_with_add_reduction_multidim_i32(
        // CHECK: [[VAL0:%[0-9]+]] = "ttir.repeat"(%{{[0-9]+}})
        // CHECK-SAME: <{repeat_dimensions = array<i64: 1, 300, 1>}> : (tensor<10x1x2xi32>) -> tensor<10x300x2xi32>
        // CHECK: [[VAL1:%[0-9]+]] = "ttir.scatter"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}})
        // CHECK-SAME: <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<sum>}>
        // CHECK-SAME: (tensor<6000000xf32>, tensor<3000xi32>, tensor<3000xf32>) -> tensor<6000000xf32>
        // CHECK: [[VAL2:%[0-9]+]] = "ttir.reshape"([[VAL1]]) <{shape = [200 : i32, 100 : i32, 300 : i32]}>
        // CHECK: return [[VAL2]] : tensor<200x100x300xf32>
        return %result : tensor<200x100x300xf32>
    }
}
