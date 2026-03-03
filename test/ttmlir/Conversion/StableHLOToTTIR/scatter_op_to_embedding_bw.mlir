// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @test_embedding_backward(%weight: tensor<2x128xf32>, %indices: tensor<1x10x1xsi32>, %gradient: tensor<1x10x128xf32>) -> tensor<2x128xf32> {
    %c0 = stablehlo.constant dense<0.0> : tensor<2x128xf32>
    // CHECK: "ttir.embedding_backward"
    // CHECK: (tensor<1x10x1xsi32>, tensor<2x128xf32>, tensor<1x10x128xf32>) -> tensor<2x128xf32>
    %result = "stablehlo.scatter"(%weight, %indices, %gradient) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %sum = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) {
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [2],
        inserted_window_dims = [0],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 2
      >,
      indices_are_sorted = false,
      unique_indices = false
    } : (tensor<2x128xf32>, tensor<1x10x1xsi32>, tensor<1x10x128xf32>) -> tensor<2x128xf32>
    return %result : tensor<2x128xf32>
  }

  func.func @test_flattened_indices(%arg0: tensor<50257x768xf32>, %arg1: tensor<256x1xi64>, %arg2: tensor<256x768xf32>) -> tensor<50257x768xf32> {
  // CHECK: "ttir.reshape"
  // CHECK: (tensor<256x1xi64>) -> tensor<1x256x1xi64>
  // CHECK: "ttir.embedding_backward"
  // CHECK: (tensor<1x256x1xi64>, tensor<50257x768xf32>, tensor<256x768xf32>) -> tensor<50257x768xf32>
  %result = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
      ^bb0(%argL: tensor<f32>, %argR: tensor<f32>):
        %sum = stablehlo.add %argL, %argR : tensor<f32>
        stablehlo.return %sum : tensor<f32>
      }) : (tensor<50257x768xf32>, tensor<256x1xi64>, tensor<256x768xf32>) -> tensor<50257x768xf32>
    return %result : tensor<50257x768xf32>
  }
}
