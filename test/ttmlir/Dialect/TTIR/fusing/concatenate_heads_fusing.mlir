// RUN: ttmlir-opt %s --ttir-fusing | FileCheck %s

module {
  // CHECK-LABEL: func.func @concatenate_heads_fusion_1
  func.func @concatenate_heads_fusion_1(%arg0: tensor<1x24x32x128xbf16> ) -> tensor<1x32x3072xbf16> {
    // CHECK: %[[RESULT:.*]] = "ttir.concatenate_heads"(%arg0, %{{.*}}) : (tensor<1x24x32x128xbf16>, tensor<1x32x3072xbf16>) -> tensor<1x32x3072xbf16>
    // CHECK-NOT: ttir.reshape
    // CHECK-NOT: ttir.permute
    // CHECK: return %[[RESULT]]

    %0 = ttir.empty() : tensor<1x32x24x128xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x32x128xbf16>, tensor<1x32x24x128xbf16>) -> tensor<1x32x24x128xbf16>

    %2 = ttir.empty() : tensor<1x32x3072xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [1 : i32, 32 : i32, 3072 : i32]}> : (tensor<1x32x24x128xbf16>, tensor<1x32x3072xbf16>) -> tensor<1x32x3072xbf16>

    return %3 : tensor<1x32x3072xbf16>
  }
}

module {
  // CHECK-LABEL: func.func @concatenate_heads_fusion_2
  func.func @concatenate_heads_fusion_2(%arg0: tensor<1x24x32x128xbf16> ) -> tensor<32x3072xbf16> {
    // CHECK-NOT: ttir.permute
    // CHECK: %0 = ttir.empty() : tensor<1x32x3072xbf16>
    // CHECK: %[[CONCAT_RESULT:.*]] = "ttir.concatenate_heads"(%arg0, %0) : (tensor<1x24x32x128xbf16>, tensor<1x32x3072xbf16>) -> tensor<1x32x3072xbf16>
    // CHECK: %2 = ttir.empty() : tensor<32x3072xbf16>
    // CHECK: %[[RESHAPE_RESULT:.*]] = "ttir.reshape"(%[[CONCAT_RESULT]], %2) <{shape = [32 : i32, 3072 : i32]}> : (tensor<1x32x3072xbf16>, tensor<32x3072xbf16>) -> tensor<32x3072xbf16>
    // CHECK: return %[[RESHAPE_RESULT]]

    %0 = ttir.empty() : tensor<1x32x24x128xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x32x128xbf16>, tensor<1x32x24x128xbf16>) -> tensor<1x32x24x128xbf16>

    %2 = ttir.empty() : tensor<32x3072xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [32 : i32, 3072 : i32]}> : (tensor<1x32x24x128xbf16>, tensor<32x3072xbf16>) -> tensor<32x3072xbf16>

    return %3 : tensor<32x3072xbf16>
  }
}
