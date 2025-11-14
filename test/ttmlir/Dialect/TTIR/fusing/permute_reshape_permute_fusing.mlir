// RUN: ttmlir-opt --canonicalize --ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @test_nhwc_nchw_reshape_permute(%arg0: tensor<12x7x7x1152xbf16>) -> tensor<12x1x49x1152xbf16> {
    // CHECK-LABEL: func.func @test_nhwc_nchw_reshape_permute
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0, %{{[0-9]+}}) <{shape = [12 : i32, 1 : i32, 49 : i32, 1152 : i32]}>
    // CHECK-NOT: "ttir.permute"
    // CHECK: return %[[RESHAPE]]
    %0 = ttir.empty() : tensor<12x1152x7x7xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<12x7x7x1152xbf16>, tensor<12x1152x7x7xbf16>) -> tensor<12x1152x7x7xbf16>
    %2 = ttir.empty() : tensor<12x1152x1x49xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [12 : i32, 1152 : i32, 1 : i32, 49 : i32]}> : (tensor<12x1152x7x7xbf16>, tensor<12x1152x1x49xbf16>) -> tensor<12x1152x1x49xbf16>
    %4 = ttir.empty() : tensor<12x1x49x1152xbf16>
    %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<12x1152x1x49xbf16>, tensor<12x1x49x1152xbf16>) -> tensor<12x1x49x1152xbf16>
    return %5 : tensor<12x1x49x1152xbf16>
  }

  func.func @test_no_fusion_not_inverse(%arg0: tensor<12x7x7x1152xbf16>) -> tensor<12x49x1152x1xbf16> {
    // CHECK-LABEL: func.func @test_no_fusion_not_inverse
    // CHECK: "ttir.permute"
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.permute"
    %0 = ttir.empty() : tensor<12x1152x7x7xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<12x7x7x1152xbf16>, tensor<12x1152x7x7xbf16>) -> tensor<12x1152x7x7xbf16>
    %2 = ttir.empty() : tensor<12x1152x1x49xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [12 : i32, 1152 : i32, 1 : i32, 49 : i32]}> : (tensor<12x1152x7x7xbf16>, tensor<12x1152x1x49xbf16>) -> tensor<12x1152x1x49xbf16>
    %4 = ttir.empty() : tensor<12x49x1152x1xbf16>
    %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<12x1152x1x49xbf16>, tensor<12x49x1152x1xbf16>) -> tensor<12x49x1152x1xbf16>
    return %5 : tensor<12x49x1152x1xbf16>
  }

  func.func @test_no_fusion_first_permute_multiple_uses(%arg0: tensor<12x7x7x1152xbf16>) -> (tensor<12x1x49x1152xbf16>, tensor<12x1152x7x7xbf16>) {
    // CHECK-LABEL: func.func @test_no_fusion_first_permute_multiple_uses
    // CHECK: "ttir.permute"
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.permute"
    // CHECK: "ttir.add"
    %0 = ttir.empty() : tensor<12x1152x7x7xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<12x7x7x1152xbf16>, tensor<12x1152x7x7xbf16>) -> tensor<12x1152x7x7xbf16>
    %2 = ttir.empty() : tensor<12x1152x1x49xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [12 : i32, 1152 : i32, 1 : i32, 49 : i32]}> : (tensor<12x1152x7x7xbf16>, tensor<12x1152x1x49xbf16>) -> tensor<12x1152x1x49xbf16>
    %4 = ttir.empty() : tensor<12x1x49x1152xbf16>
    %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<12x1152x1x49xbf16>, tensor<12x1x49x1152xbf16>) -> tensor<12x1x49x1152xbf16>
    %6 = ttir.empty() : tensor<12x1152x7x7xbf16>
    %7 = "ttir.add"(%1, %1, %6) : (tensor<12x1152x7x7xbf16>, tensor<12x1152x7x7xbf16>, tensor<12x1152x7x7xbf16>) -> tensor<12x1152x7x7xbf16>
    return %5, %7 : tensor<12x1x49x1152xbf16>, tensor<12x1152x7x7xbf16>
  }

  func.func @test_no_fusion_reshape_multiple_uses(%arg0: tensor<12x7x7x1152xbf16>) -> (tensor<12x1x49x1152xbf16>, tensor<12x1152x1x49xbf16>) {
    // CHECK-LABEL: func.func @test_no_fusion_reshape_multiple_uses
    // CHECK: "ttir.permute"
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.permute"
    // CHECK: "ttir.add"
    %0 = ttir.empty() : tensor<12x1152x7x7xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<12x7x7x1152xbf16>, tensor<12x1152x7x7xbf16>) -> tensor<12x1152x7x7xbf16>
    %2 = ttir.empty() : tensor<12x1152x1x49xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [12 : i32, 1152 : i32, 1 : i32, 49 : i32]}> : (tensor<12x1152x7x7xbf16>, tensor<12x1152x1x49xbf16>) -> tensor<12x1152x1x49xbf16>
    %4 = ttir.empty() : tensor<12x1x49x1152xbf16>
    %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<12x1152x1x49xbf16>, tensor<12x1x49x1152xbf16>) -> tensor<12x1x49x1152xbf16>
    %6 = ttir.empty() : tensor<12x1152x1x49xbf16>
    %7 = "ttir.add"(%3, %3, %6) : (tensor<12x1152x1x49xbf16>, tensor<12x1152x1x49xbf16>, tensor<12x1152x1x49xbf16>) -> tensor<12x1152x1x49xbf16>
    return %5, %7 : tensor<12x1x49x1152xbf16>, tensor<12x1152x1x49xbf16>
  }
}
