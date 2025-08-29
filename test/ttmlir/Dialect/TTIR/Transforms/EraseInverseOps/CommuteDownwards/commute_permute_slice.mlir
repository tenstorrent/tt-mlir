// RUN: ttmlir-opt --ttir-erase-inverse-ops="enable-commute-upwards=false" -o %t %s
// RUN: FileCheck --input-file=%t %s

module {
  func.func @test_permute_concat_commute_downwards(%arg0: tensor<1x160x160x160xbf16>) -> tensor<1x160x160x80xbf16> {
    // CHECK: %[[CONCAT:[0-9]+]] = "ttir.slice_static"(%arg0,
    // CHECK: begins = [0 : i32, 80 : i32, 0 : i32, 0 : i32]
    // CHECK: ends = [1 : i32, 160 : i32, 160 : i32, 160 : i32]
    // CHECK: step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%[[CONCAT]]
    // CHECK: permutation = array<i64: 0, 2, 3, 1>
    // CHECK: return %[[PERMUTE]]

    %0 = ttir.empty() : tensor<1x160x160x160xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x160x160x160xbf16>, tensor<1x160x160x160xbf16>) -> tensor<1x160x160x160xbf16>
    %2 = ttir.empty() : tensor<1x160x160x80xbf16>
    %3 = "ttir.slice_static"(%1, %2) <{begins = [0 : i32, 0 : i32, 0 : i32, 80 : i32], ends = [1 : i32, 160 : i32, 160 : i32, 160 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x160x160x160xbf16>, tensor<1x160x160x80xbf16>) -> tensor<1x160x160x80xbf16>
    return %3 : tensor<1x160x160x80xbf16>
  }
}
