// RUN: ttmlir-opt --ttir-erase-inverse-ops="enable-commute-downwards=false" %s | FileCheck %s

module {
  func.func @test_permute_concat_commute_upwards(%arg0: tensor<32x64xbf16>, %arg1: tensor<32x64xbf16>) -> tensor<64x64xbf16> {
    // CHECK: %[[PERMUTE1:[0-9]+]] = "ttir.permute"(%arg0, %{{[0-9]+}}) <{permutation = array<i64: 1, 0>}>
    // CHECK: %[[PERMUTE2:[0-9]+]] = "ttir.permute"(%arg1, %{{[0-9]+}}) <{permutation = array<i64: 1, 0>}>
    // CHECK: %[[CONCAT:[0-9]+]] = "ttir.concat"(%[[PERMUTE1]], %[[PERMUTE2]]
    // CHECK dim = 1
    %0 = tensor.empty() : tensor<64x64xbf16>
    %1 = "ttir.concat"(%arg0, %arg1, %0) <{dim = 0 : si32}> : (tensor<32x64xbf16>, tensor<32x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %2 = tensor.empty() : tensor<64x64xbf16>
    %3 = "ttir.permute"(%1, %2) <{permutation = array<i64: 1, 0>}> : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>

    return %3 : tensor<64x64xbf16>
  }
}
