module {
  func.func @test_small_vocab_embedding_bw(%weight: tensor<2x128xf32>, %indices: tensor<1x10x1xsi32>, %gradient: tensor<1x10x128xf32>) -> tensor<2x128xf32> {
    %c0 = stablehlo.constant dense<0.0> : tensor<2x128xf32>
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

  func.func @test_medium_vocab_embedding_bw(%weight: tensor<512x128xf32>, %indices: tensor<1x10x1xsi32>, %gradient: tensor<1x10x128xf32>) -> tensor<512x128xf32> {
    %c0 = stablehlo.constant dense<0.0> : tensor<512x128xf32>
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
    } : (tensor<512x128xf32>, tensor<1x10x1xsi32>, tensor<1x10x128xf32>) -> tensor<512x128xf32>
    return %result : tensor<512x128xf32>
  }

  func.func @test_large_vocab_embedding_bw(%weight: tensor<30000x128xf32>, %indices: tensor<1x10x1xsi32>, %gradient: tensor<1x10x128xf32>) -> tensor<30000x128xf32> {
    %c0 = stablehlo.constant dense<0.0> : tensor<30000x128xf32>
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
    } : (tensor<30000x128xf32>, tensor<1x10x1xsi32>, tensor<1x10x128xf32>) -> tensor<30000x128xf32>
    return %result : tensor<30000x128xf32>
  }

  func.func @test_roberta_embedding_bw(%weight: tensor<50265x768xbf16>, %indices: tensor<1x13x1xsi32>, %gradient: tensor<1x13x768xbf16>) -> tensor<50265x768xbf16> {
    %c0 = stablehlo.constant dense<0.0> : tensor<50265x768xbf16>
    %result = "stablehlo.scatter"(%weight, %indices, %gradient) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %sum = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %sum : tensor<bf16>
    }) {
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [2],
        inserted_window_dims = [0],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 2
      >,
      indices_are_sorted = false,
      unique_indices = false
    } : (tensor<50265x768xbf16>, tensor<1x13x1xsi32>, tensor<1x13x768xbf16>) -> tensor<50265x768xbf16>
    return %result : tensor<50265x768xbf16>
  }

  func.func @test_bert_embedding_bw(%weight: tensor<30522x768xf32>, %indices: tensor<1x9x1xsi32>, %gradient: tensor<1x9x768xf32>) -> tensor<30522x768xf32> {
    %c0 = stablehlo.constant dense<0.0> : tensor<30522x768xf32>
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
    } : (tensor<30522x768xf32>, tensor<1x9x1xsi32>, tensor<1x9x768xf32>) -> tensor<30522x768xf32>
    return %result : tensor<30522x768xf32>
  }

  func.func @test_gpt2_embedding_bw(%weight: tensor<50257x768xf32>, %indices: tensor<1x4x1xsi32>, %gradient: tensor<1x4x768xf32>) -> tensor<50257x768xf32> {
    %c0 = stablehlo.constant dense<0.0> : tensor<50257x768xf32>
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
    } : (tensor<50257x768xf32>, tensor<1x4x1xsi32>, tensor<1x4x768xf32>) -> tensor<50257x768xf32>
    return %result : tensor<50257x768xf32>
  }

  func.func @test_position_embedding_bw(%weight: tensor<1024x768xf32>, %indices: tensor<1x4x1xsi32>, %gradient: tensor<1x4x768xf32>) -> tensor<1024x768xf32> {
    %c0 = stablehlo.constant dense<0.0> : tensor<1024x768xf32>
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
    } : (tensor<1024x768xf32>, tensor<1x4x1xsi32>, tensor<1x4x768xf32>) -> tensor<1024x768xf32>
    return %result : tensor<1024x768xf32>
  }

  func.func @test_roberta_position_embedding_bw(%weight: tensor<1026x768xbf16>, %indices: tensor<1x13x1xsi32>, %gradient: tensor<1x13x768xbf16>) -> tensor<1026x768xbf16> {
    %c0 = stablehlo.constant dense<0.0> : tensor<1026x768xbf16>
    %result = "stablehlo.scatter"(%weight, %indices, %gradient) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %sum = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %sum : tensor<bf16>
    }) {
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [2],
        inserted_window_dims = [0],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 2
      >,
      indices_are_sorted = false,
      unique_indices = false
    } : (tensor<1026x768xbf16>, tensor<1x13x1xsi32>, tensor<1x13x768xbf16>) -> tensor<1026x768xbf16>
    return %result : tensor<1026x768xbf16>
  }
  func.func @test_token_type_embedding_bw(%weight: tensor<2x768xf32>, %indices: tensor<1x9x1xsi32>, %gradient: tensor<1x9x768xf32>) -> tensor<2x768xf32> {
    %c0 = stablehlo.constant dense<0.0> : tensor<2x768xf32>
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
    } : (tensor<2x768xf32>, tensor<1x9x1xsi32>, tensor<1x9x768xf32>) -> tensor<2x768xf32>
    return %result : tensor<2x768xf32>
  }
}
