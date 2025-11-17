// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Tests for StableHLOToBatchNormGradOpConversionPattern
module @jit_batch_norm_grad attributes {} {
  // Test basic batch norm grad conversion
  func.func public @test_batch_norm_grad(%operand: tensor<2x2x2x2xf32>,
                                         %scale: tensor<2xf32>,
                                         %mean: tensor<2xf32>,
                                         %variance: tensor<2xf32>,
                                         %grad_output: tensor<2x2x2x2xf32>)
                                         -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) {
    %grad_operand, %grad_scale, %grad_offset = "stablehlo.batch_norm_grad"(%operand, %scale, %mean, %variance, %grad_output) {
      epsilon = 1.0e-3 : f32,
      feature_index = 1 : i64
    } : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>)
        -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)

    // CHECK: "ttir.subtract"
    // CHECK: "ttir.add"
    // CHECK: "ttir.sqrt"
    // CHECK: "ttir.div"
    // CHECK: "ttir.multiply"
    // CHECK: "ttir.sum"

    return %grad_operand, %grad_scale, %grad_offset : tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>
  }


  // Test with different feature index
  func.func public @test_batch_norm_grad_feature_index_3(%operand: tensor<2x3x4x5xf32>,
                                                         %scale: tensor<5xf32>,
                                                         %mean: tensor<5xf32>,
                                                         %variance: tensor<5xf32>,
                                                         %grad_output: tensor<2x3x4x5xf32>)
                                                         -> (tensor<2x3x4x5xf32>, tensor<5xf32>, tensor<5xf32>) {
    %grad_operand, %grad_scale, %grad_offset = "stablehlo.batch_norm_grad"(%operand, %scale, %mean, %variance, %grad_output) {
      epsilon = 0.001 : f32,
      feature_index = 3 : i64
    } : (tensor<2x3x4x5xf32>, tensor<5xf32>, tensor<5xf32>, tensor<5xf32>, tensor<2x3x4x5xf32>)
        -> (tensor<2x3x4x5xf32>, tensor<5xf32>, tensor<5xf32>)

    // Verify sum uses correct dimensions for feature_index=3
    // CHECK: "ttir.sum"{{.*}}dim_arg = [0 : i32, 1 : i32, 2 : i32]

    return %grad_operand, %grad_scale, %grad_offset : tensor<2x3x4x5xf32>, tensor<5xf32>, tensor<5xf32>
  }


  // Test with 2D input (minimum supported rank)
  func.func public @test_batch_norm_grad_2d(%operand: tensor<4x8xf32>,
                                            %scale: tensor<8xf32>,
                                            %mean: tensor<8xf32>,
                                            %variance: tensor<8xf32>,
                                            %grad_output: tensor<4x8xf32>)
                                            -> (tensor<4x8xf32>, tensor<8xf32>, tensor<8xf32>) {
    %grad_operand, %grad_scale, %grad_offset = "stablehlo.batch_norm_grad"(%operand, %scale, %mean, %variance, %grad_output) {
      epsilon = 1.0e-5 : f32,
      feature_index = 1 : i64
    } : (tensor<4x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<4x8xf32>)
        -> (tensor<4x8xf32>, tensor<8xf32>, tensor<8xf32>)

    // CHECK: "ttir.subtract"
    // CHECK: "ttir.add"
    // CHECK: "ttir.sqrt"
    // CHECK: "ttir.div"
    // CHECK: "ttir.multiply"
    // CHECK: "ttir.sum"{{.*}}dim_arg = [0 : i32]

    return %grad_operand, %grad_scale, %grad_offset : tensor<4x8xf32>, tensor<8xf32>, tensor<8xf32>
  }


}
