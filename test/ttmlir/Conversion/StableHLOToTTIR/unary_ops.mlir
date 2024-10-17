// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_eltwise_abs attributes {} {
  func.func public @test_abs(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.abs %arg0 : tensor<13x21x3xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.abs"[[C:.*]]
    return %0 : tensor<13x21x3xf32>
  }

  func.func public @test_ceil(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.ceil %arg0 : tensor<13x21x3xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.ceil"[[C:.*]]
    return %0 : tensor<13x21x3xf32>
  }

  func.func public @test_cos(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.cosine %arg0 : tensor<13x21x3xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.cos"[[C:.*]]
    return %0 : tensor<13x21x3xf32>
  }

  func.func public @test_exp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.exponential %arg0 : tensor<13x21x3xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.exp"[[C:.*]]
    return %0 : tensor<13x21x3xf32>
  }

  func.func public @test_neg(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.negate %arg0 : tensor<13x21x3xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.neg"[[C:.*]]
    return %0 : tensor<13x21x3xf32>
  }

  func.func public @test_sin(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.sine %arg0 : tensor<13x21x3xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.sin"[[C:.*]]
    return %0 : tensor<13x21x3xf32>
  }

  func.func public @test_sqrt(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.sqrt %arg0 : tensor<13x21x3xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.sqrt"[[C:.*]]
    return %0 : tensor<13x21x3xf32>
  }

  func.func public @test_transpose(%arg0: tensor<64x128xf32>) -> tensor<128x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1,0] : (tensor<64x128xf32>) -> tensor<128x64xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.transpose"[[C:.*]]
    return %0 : tensor<128x64xf32>
  }
}
