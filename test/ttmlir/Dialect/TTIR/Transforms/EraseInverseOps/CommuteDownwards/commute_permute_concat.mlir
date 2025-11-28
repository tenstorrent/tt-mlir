// RUN: ttmlir-opt --ttir-erase-inverse-ops="force=true enable-commute-upwards=false" -o %t %s
// RUN: FileCheck --input-file=%t %s

module {
  func.func @test_permute_concat_commute_downwards(%arg0: tensor<32x64xbf16>, %arg1: tensor<32x64xbf16>) -> tensor<64x64xbf16> {
    // CHECK: %[[CONCAT:[0-9]+]] = "ttir.concat"(%arg0, %arg1
    // CHECK: dim = 0
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%[[CONCAT]]
    // CHECK: permutation = array<i64: 1, 0>
    // CHECK: return %[[PERMUTE]]

    %1 = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0>}> : (tensor<32x64xbf16>) -> tensor<64x32xbf16>
    %3 = "ttir.permute"(%arg1) <{permutation = array<i64: 1, 0>}> : (tensor<32x64xbf16>) -> tensor<64x32xbf16>
    %5 = "ttir.concat"(%1, %3) <{dim = 1 : si32}> : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x64xbf16>
    return %5 : tensor<64x64xbf16>
  }
}
