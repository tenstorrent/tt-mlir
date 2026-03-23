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
        // CHECK: [[VAL5:%[0-9]+]] = "ttir.scatter"(%arg0, [[VAL3]], %arg2) <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}>
        // CHECK-SAME: (tensor<1x3x320x320xf32>, tensor<1x3x32x32xi64>, tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32>
        return %result : tensor<1x3x320x320xf32>
        // CHECK: return [[VAL5]] : tensor<1x3x320x320xf32>
    }

    func.func public @test_scatter_simple_1(%arg0: tensor<1x3x320x320xf32>, %arg1: tensor<1x1xi32>, %arg2: tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32> {
        // CHECK-LABEL: func.func public @test_scatter_simple_1
        // CHECK: "ttir.scatter"
        // CHECK-SAME: <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}>
        %result = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
        ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
            stablehlo.return %arg4 : tensor<f32>
        }) : (tensor<1x3x320x320xf32>, tensor<1x1xi32>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
        return %result : tensor<1x3x320x320xf32>
    }

    func.func public @test_scatter_simple_2(%arg0: tensor<32x32xi32>, %arg1: tensor<1x1xi32>, %arg2: tensor<1x32xi32>) -> tensor<32x32xi32> {
        // CHECK-LABEL: func.func public @test_scatter_simple_2
        // CHECK: "ttir.scatter"
        // CHECK-SAME: <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}>
        %result = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
        ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
            stablehlo.return %arg4 : tensor<i32>
        }) : (tensor<32x32xi32>, tensor<1x1xi32>, tensor<1x32xi32>) -> tensor<32x32xi32>
        return %result : tensor<32x32xi32>
    }

    func.func public @test_scatter_simple_3(%arg0: tensor<1000x32xf32>, %arg1: tensor<1x1xi64>, %arg2: tensor<1x32xf32>) -> tensor<1000x32xf32> {
        // CHECK-LABEL: func.func public @test_scatter_simple_3
        // CHECK: "ttir.scatter"
        // CHECK-SAME: <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}>
        %result = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
        ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
            stablehlo.return %arg4 : tensor<f32>
        }) : (tensor<1000x32xf32>, tensor<1x1xi64>, tensor<1x32xf32>) -> tensor<1000x32xf32>
        return %result : tensor<1000x32xf32>
    }

    func.func public @test_multidim_point_scatter(%arg0: tensor<1x18xf32>, %arg1: tensor<11x2xi64>, %arg2: tensor<11xf32>) -> tensor<1x18xf32> {
        // CHECK-LABEL: func.func public @test_multidim_point_scatter
        // CHECK: "ttir.scatter"
        // CHECK-SAME: <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}>
        %result = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
        ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
            stablehlo.return %arg4 : tensor<f32>
        }) : (tensor<1x18xf32>, tensor<11x2xi64>, tensor<11xf32>) -> tensor<1x18xf32>
        return %result : tensor<1x18xf32>
    }

    func.func @test_multidim_scatter_with_window(%updates: tensor<1x2xbf16>) -> tensor<1x7x2xbf16> {
        // CHECK: "ttir.scatter"
        // CHECK-SAME: <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<sum>}>
        // Operand: zeros tensor to scatter into
        %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<1x7x2xbf16>
        // Indices: scatter at position [0, 3] (batch=0, row=3)
        %indices = stablehlo.constant dense<[[0, 3]]> : tensor<1x2xi64>
        %result = "stablehlo.scatter"(%cst_16, %indices, %updates) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
        ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
            %sum = stablehlo.add %arg0, %arg1 : tensor<bf16>
            stablehlo.return %sum : tensor<bf16>
        }) : (tensor<1x7x2xbf16>, tensor<1x2xi64>, tensor<1x2xbf16>) -> tensor<1x7x2xbf16>
        return %result : tensor<1x7x2xbf16>
    }

    func.func @test_multidim_scatter_with_window_extracted_from_model(%arg186: tensor<1x2xbf16>, %arg187: tensor<1xi64>, %arg188: tensor<1xi64>) -> (tensor<1x7x2xbf16>) {
        // CHECK: "ttir.scatter"
        // CHECK-SAME: <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<sum>}>
        %c_13 = stablehlo.constant dense<7> : tensor<1xi64>
        %c_14 = stablehlo.constant dense<1> : tensor<1xi64>
        %c_15 = stablehlo.constant dense<0> : tensor<1xi64>
        %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<1x7x2xbf16>
        %3 = stablehlo.compare LT, %arg188, %c_15 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
        %4 = stablehlo.add %arg188, %c_14 : tensor<1xi64>
        %5 = stablehlo.select %3, %4, %arg188 : tensor<1xi1>, tensor<1xi64>
        %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<1x1xi64>
        %7 = stablehlo.compare LT, %arg187, %c_15 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
        %8 = stablehlo.add %arg187, %c_13 : tensor<1xi64>
        %9 = stablehlo.select %7, %8, %arg187 : tensor<1xi1>, tensor<1xi64>
        %10 = stablehlo.reshape %9 : (tensor<1xi64>) -> tensor<1x1xi64>
        %11 = stablehlo.concatenate %6, %10, dim = 1 : (tensor<1x1xi64>, tensor<1x1xi64>) -> tensor<1x2xi64>
        %12 = "stablehlo.scatter"(%cst_16, %11, %arg186) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
        ^bb0(%arg373: tensor<bf16>, %arg374: tensor<bf16>):
        %result = stablehlo.add %arg373, %arg374 : tensor<bf16>
        stablehlo.return %result : tensor<bf16>
        }) : (tensor<1x7x2xbf16>, tensor<1x2xi64>, tensor<1x2xbf16>) -> tensor<1x7x2xbf16>
        return %12 : tensor<1x7x2xbf16>
  }
}
