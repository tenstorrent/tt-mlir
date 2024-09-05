// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline="enable-remove-dead-values=true" %s | FileCheck %s
module attributes {} {
  func.func public @test_reduce_add_opt(%arg0: tensor<128x10xf32>, %cst_0: tensor<f32>) -> tensor<128xf32> {
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.sum"[[C:.*]]
    %1 = tensor.empty() : tensor<64x96xbf16>
    // CHECK-NOT: %[[C:.*]] = tensor.empty[[C:.*]]
    return %0 : tensor<128xf32>
  }
}
