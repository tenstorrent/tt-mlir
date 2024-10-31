// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
// XFAIL: true
module attributes {torch.debug_module_name = "GraphModule"} {
  func.func @forward(%arg0: tensor<128x64xf32>, %arg1: tensor<128xf32>, %arg2: tensor<32x64xf32>) -> (tensor<32x128xf32>, tensor<32x64xf32>) {
    %0 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    // CHECK: %[[C:.*]] = "ttir.transpose"[[C:.*]]
    %1 = tosa.transpose %arg0, %0 : (tensor<128x64xf32>, tensor<2xi32>) -> tensor<64x128xf32>
    // CHECK: %[[C:.*]] = "ttir.reshape"[[C:.*]]
    %2 = tosa.reshape %arg2 {new_shape = array<i64: 1, 32, 64>} : (tensor<32x64xf32>) -> tensor<1x32x64xf32>
    %3 = tosa.reshape %1 {new_shape = array<i64: 1, 64, 128>} : (tensor<64x128xf32>) -> tensor<1x64x128xf32>
    // CHECK: %[[C:.*]] = "ttir.matmul"[[C:.*]]
    %4 = tosa.matmul %2, %3 : (tensor<1x32x64xf32>, tensor<1x64x128xf32>) -> tensor<1x32x128xf32>
    // CHECK: %[[C:.*]] = "ttir.reshape"[[C:.*]]
    %5 = tosa.reshape %4 {new_shape = array<i64: 32, 128>} : (tensor<1x32x128xf32>) -> tensor<32x128xf32>
    %6 = tosa.reshape %arg1 {new_shape = array<i64: 1, 128>} : (tensor<128xf32>) -> tensor<1x128xf32>
    // CHECK: %[[C:.*]] = "ttir.add"[[C:.*]]
    %7 = tosa.add %6, %5 : (tensor<1x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
    return %7, %arg2 : tensor<32x128xf32>, tensor<32x64xf32>
  }
}
