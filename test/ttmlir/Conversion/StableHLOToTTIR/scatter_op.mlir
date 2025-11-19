// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_scatter attributes {} {
    func.func public @test_scatter(%arg0: tensor<1x3x320x320xf32>, %arg1: tensor<1x1xi64>, %arg2: tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32> {
        // CHECK: [[VAL1:%[0-9]+]] = "ttir.reshape"(%arg1)
        // CHECK: [[VAL3:%[0-9]+]] = "ttir.repeat"([[VAL1]])
        %result = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
        ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
            stablehlo.return %arg4 : tensor<f32>
        }) : (tensor<1x3x320x320xf32>, tensor<1x1xi64>, tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32>
        // CHECK: [[VAL5:%[0-9]+]] = "ttir.scatter"(%arg0, [[VAL3]], %arg2) <{dim = 0 : i32}>
        // CHECK-SAME: (tensor<1x3x320x320xf32>, tensor<1x3x32x32xi64>, tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32>
        return %result : tensor<1x3x320x320xf32>
        // CHECK: return [[VAL5]] : tensor<1x3x320x320xf32>
    }

    func.func public @test_scatter_simple_1(%arg0: tensor<1x3x320x320xf32>, %arg1: tensor<1x1xi32>, %arg2: tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32> {
        // CHECK-LABEL: func.func public @test_scatter_simple_1
        // CHECK: "ttir.scatter"
        // CHECK-SAME: <{dim = 0 : i32}>
        %result = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
        ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
            stablehlo.return %arg4 : tensor<f32>
        }) : (tensor<1x3x320x320xf32>, tensor<1x1xi32>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
        return %result : tensor<1x3x320x320xf32>
    }

    func.func public @test_scatter_simple_2(%arg0: tensor<32x32xi32>, %arg1: tensor<1x1xi32>, %arg2: tensor<1x32xi32>) -> tensor<32x32xi32> {
        // CHECK-LABEL: func.func public @test_scatter_simple_2
        // CHECK: "ttir.scatter"
        // CHECK-SAME: <{dim = 0 : i32}>
        %result = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
        ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
            stablehlo.return %arg4 : tensor<i32>
        }) : (tensor<32x32xi32>, tensor<1x1xi32>, tensor<1x32xi32>) -> tensor<32x32xi32>
        return %result : tensor<32x32xi32>
    }

    func.func public @test_scatter_simple_3(%arg0: tensor<1000x32xf32>, %arg1: tensor<1x1xi64>, %arg2: tensor<1x32xf32>) -> tensor<1000x32xf32> {
        // CHECK-LABEL: func.func public @test_scatter_simple_3
        // CHECK: "ttir.scatter"
        // CHECK-SAME: <{dim = 0 : i32}>
        %result = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
        ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
            stablehlo.return %arg4 : tensor<f32>
        }) : (tensor<1000x32xf32>, tensor<1x1xi64>, tensor<1x32xf32>) -> tensor<1000x32xf32>
        return %result : tensor<1000x32xf32>
    }
}
