// RUN: ttmlir-opt --ttir-erase-inverse-ops="enable-commute-downwards=false" %s | FileCheck %s

module {
  func.func @test_reshape_concat_commute_upwards(%arg0: tensor<1x64x64x1xbf16>, %arg1: tensor<1x64x64x1xbf16>) -> tensor<1x4096x2xbf16> {
    // CHECK: %[[RESHAPE1:[0-9]+]] = "ttir.reshape"(%arg0
    // CHECK: %[[RESHAPE2:[0-9]+]] = "ttir.reshape"(%arg1
    // CHECK: %[[CONCAT:[0-9]+]] = "ttir.concat"(%[[RESHAPE1]], %[[RESHAPE2]]
    // CHECK: dim = 2
    // CHECK: return %[[CONCAT]]
    %0 = tensor.empty() : tensor<1x64x64x2xbf16>
    %1 = "ttir.concat"(%arg0, %arg1, %0) <{dim = 3 : si32}> : (tensor<1x64x64x1xbf16>, tensor<1x64x64x1xbf16>, tensor<1x64x64x2xbf16>) -> tensor<1x64x64x2xbf16>
    %2 = tensor.empty() : tensor<1x4096x2xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [1: i32, 4096: i32, 2: i32]}> : (tensor<1x64x64x2xbf16>, tensor<1x4096x2xbf16>) -> tensor<1x4096x2xbf16>
    return %3 : tensor<1x4096x2xbf16>
  }
}
