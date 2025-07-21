// RUN: ttmlir-opt --ttir-erase-inverse-ops="enable-commute-downwards=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @test_permute_concat_commute_upwards(%arg0: tensor<1x160x160x160xbf16>) -> tensor<1x160x160x80xbf16> {
    // CHECK: %[[PERMUTE1:[0-9]+]] = "ttir.permute"(%arg0, %{{[0-9]+}}) <{permutation = array<i64: 0, 2, 3, 1>}>
    // CHECK: %[[CONCAT:[0-9]+]] = "ttir.slice"(%[[PERMUTE1]]
    // CHECK: begins = [0 : i32, 0 : i32, 0 : i32, 80 : i32]
    // CHECK: ends = [1 : i32, 160 : i32, 160 : i32, 160 : i32]
    // CHECK: step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]
    %0 = tensor.empty() : tensor<1x80x160x160xbf16>
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0 : i32, 80 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 160 : i32, 160 : i32, 160 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x160x160x160xbf16>, tensor<1x80x160x160xbf16>) -> tensor<1x80x160x160xbf16>
    %2 = tensor.empty() : tensor<1x160x160x80xbf16>
    %3 = "ttir.permute"(%1, %2) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x80x160x160xbf16>, tensor<1x160x160x80xbf16>) -> tensor<1x160x160x80xbf16>
    return %3 : tensor<1x160x160x80xbf16>
  }
}
