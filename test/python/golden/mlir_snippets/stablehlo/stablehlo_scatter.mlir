module @jit_scatter attributes {} {
  func.func public @test_scatter_simple(%input: tensor<8xf32>) -> tensor<8xf32> {
    %indices = stablehlo.constant dense<[1, 3, 5]> : tensor<3xi32>
    %updates = stablehlo.constant dense<[10.0, 20.0, 30.0]> : tensor<3xf32>
    %0 = "stablehlo.scatter"(%input, %indices, %updates) <{
      scatter_dimension_numbers = #stablehlo.scatter<
        inserted_window_dims = [0],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 1
      >,
      indices_are_sorted = false,
      unique_indices = false
    }> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      stablehlo.return %arg1 : tensor<f32>
    }) : (tensor<8xf32>, tensor<3xi32>, tensor<3xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }

  func.func public @test_scatter_2d(%input: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %indices = stablehlo.constant dense<[[0], [2]]> : tensor<2x1xi32>
    %updates = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                                         [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]]> : tensor<2x8xf32>
    %0 = "stablehlo.scatter"(%input, %indices, %updates) <{
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [1],
        inserted_window_dims = [0],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 1
      >,
      indices_are_sorted = false,
      unique_indices = false
    }> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      stablehlo.return %arg1 : tensor<f32>
    }) : (tensor<4x8xf32>, tensor<2x1xi32>, tensor<2x8xf32>) -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }

  func.func public @test_scatter_add(%input: tensor<8xf32>) -> tensor<8xf32> {
    %indices = stablehlo.constant dense<[1, 3, 5]> : tensor<3xi32>
    %updates = stablehlo.constant dense<[10.0, 20.0, 30.0]> : tensor<3xf32>
    %0 = "stablehlo.scatter"(%input, %indices, %updates) <{
      scatter_dimension_numbers = #stablehlo.scatter<
        inserted_window_dims = [0],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 1
      >,
      indices_are_sorted = false,
      unique_indices = false
    }> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %sum = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) : (tensor<8xf32>, tensor<3xi32>, tensor<3xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
