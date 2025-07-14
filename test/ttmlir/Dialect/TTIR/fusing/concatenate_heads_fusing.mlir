// RUN: ttmlir-opt %s --ttir-fusing | FileCheck %s
// Check that the fusing of permute and reshape into concatenate_heads works correctly.
// Example from Meta Llama Attention Layer.
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

// Check that the fusing of permute and reshape into concatenate_heads works correctly with reshape fold.
// Example from Meta Llama Attention Layer.
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

// Check that the fusing of permute and reshape into concatenate_heads works correctly.
// Example from Meta Llama Attention Layer with batch size 2.
module {
  // CHECK-LABEL: func.func @concatenate_heads_fusion_3
  func.func @concatenate_heads_fusion_3(%arg0: tensor<2x24x32x128xbf16> ) -> tensor<2x32x3072xbf16> {
    // CHECK: %[[RESULT:.*]] = "ttir.concatenate_heads"(%arg0, %{{.*}}) : (tensor<2x24x32x128xbf16>, tensor<2x32x3072xbf16>) -> tensor<2x32x3072xbf16>
    // CHECK-NOT: ttir.reshape
    // CHECK-NOT: ttir.permute
    // CHECK: return %[[RESULT]]

    %0 = ttir.empty() : tensor<2x32x24x128xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<2x24x32x128xbf16>, tensor<2x32x24x128xbf16>) -> tensor<2x32x24x128xbf16>

    %2 = ttir.empty() : tensor<2x32x3072xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [2 : i32, 32 : i32, 3072 : i32]}> : (tensor<2x32x24x128xbf16>, tensor<2x32x3072xbf16>) -> tensor<2x32x3072xbf16>

    return %3 : tensor<2x32x3072xbf16>
  }
}

// Check that the fusing of permute and reshape into concatenate_heads works correctly with reshape fold.
// Example from Meta Llama Attention Layer with batch size 2.
module {
  // CHECK-LABEL: func.func @concatenate_heads_fusion_4
  func.func @concatenate_heads_fusion_4(%arg0: tensor<2x24x32x128xbf16> ) -> tensor<64x3072xbf16> {
    // CHECK-NOT: ttir.permute
    // CHECK: %0 = ttir.empty() : tensor<2x32x3072xbf16>
    // CHECK: %[[CONCAT_RESULT:.*]] = "ttir.concatenate_heads"(%arg0, %0) : (tensor<2x24x32x128xbf16>, tensor<2x32x3072xbf16>) -> tensor<2x32x3072xbf16>
    // CHECK: %2 = ttir.empty() : tensor<64x3072xbf16>
    // CHECK: %[[RESHAPE_RESULT:.*]] = "ttir.reshape"(%[[CONCAT_RESULT]], %2) <{shape = [64 : i32, 3072 : i32]}> : (tensor<2x32x3072xbf16>, tensor<64x3072xbf16>) -> tensor<64x3072xbf16>
    // CHECK: return %[[RESHAPE_RESULT]]

    %0 = ttir.empty() : tensor<2x32x24x128xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<2x24x32x128xbf16>, tensor<2x32x24x128xbf16>) -> tensor<2x32x24x128xbf16>

    %2 = ttir.empty() : tensor<64x3072xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [64 : i32, 3072 : i32]}> : (tensor<2x32x24x128xbf16>, tensor<64x3072xbf16>) -> tensor<64x3072xbf16>

    return %3 : tensor<64x3072xbf16>
  }
}

// Check that the fusing of permute and reshape into concatenate_heads works correctly.
// Example from BERT base Attention Layer.
module {
  // CHECK-LABEL: func.func @concatenate_heads_fusion_5
  func.func @concatenate_heads_fusion_5(%arg0: tensor<1x12x256x64xbf16>) -> tensor<1x256x768xbf16> {
    // CHECK: %[[RESULT:.*]] = "ttir.concatenate_heads"(%arg0, %{{.*}}) : (tensor<1x12x256x64xbf16>, tensor<1x256x768xbf16>) -> tensor<1x256x768xbf16>
    // CHECK-NOT: ttir.reshape
    // CHECK-NOT: ttir.permute
    // CHECK: return %[[RESULT]]

    %0 = ttir.empty() : tensor<1x256x12x64xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x12x256x64xbf16>, tensor<1x256x12x64xbf16>) -> tensor<1x256x12x64xbf16>

    %2 = ttir.empty() : tensor<1x256x768xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [1 : i32, 256 : i32, 768 : i32]}> : (tensor<1x256x12x64xbf16>, tensor<1x256x768xbf16>) -> tensor<1x256x768xbf16>

    return %3 : tensor<1x256x768xbf16>
  }
}

// Check that the fusing of permute and reshape into concatenate_heads works correctly.
// Example from ViT Attention Layer.
module {
  // CHECK-LABEL: func.func @concatenate_heads_fusion_6
  func.func @concatenate_heads_fusion_6(%arg0: tensor<1x12x197x64xbf16>) -> tensor<1x197x768xbf16> {
    // CHECK: %[[RESULT:.*]] = "ttir.concatenate_heads"(%arg0, %{{.*}}) : (tensor<1x12x197x64xbf16>, tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>
    // CHECK-NOT: ttir.reshape
    // CHECK-NOT: ttir.permute
    // CHECK: return %[[RESULT]]

    %0 = ttir.empty() : tensor<1x197x12x64xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x12x197x64xbf16>, tensor<1x197x12x64xbf16>) -> tensor<1x197x12x64xbf16>

    %2 = ttir.empty() : tensor<1x197x768xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [1 : i32, 197 : i32, 768 : i32]}> : (tensor<1x197x12x64xbf16>, tensor<1x197x768xbf16>) -> tensor<1x197x768xbf16>

    return %3 : tensor<1x197x768xbf16>
  }
}

// Check that the fusing of permute and reshape into concatenate_heads works correctly with batch_size != 1.
// Example from ViT Attention Layer.
module {
  // CHECK-LABEL: func.func @concatenate_heads_fusion_7
  func.func @concatenate_heads_fusion_7(%arg0: tensor<2x12x197x64xbf16>) -> tensor<2x197x768xbf16> {
    // CHECK: %[[RESULT:.*]] = "ttir.concatenate_heads"(%arg0, %{{.*}}) : (tensor<2x12x197x64xbf16>, tensor<2x197x768xbf16>) -> tensor<2x197x768xbf16>
    // CHECK-NOT: ttir.reshape
    // CHECK-NOT: ttir.permute
    // CHECK: return %[[RESULT]]

    %0 = ttir.empty() : tensor<2x197x12x64xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<2x12x197x64xbf16>, tensor<2x197x12x64xbf16>) -> tensor<2x197x12x64xbf16>

    %2 = ttir.empty() : tensor<2x197x768xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [2 : i32, 197 : i32, 768 : i32]}> : (tensor<2x197x12x64xbf16>, tensor<2x197x768xbf16>) -> tensor<2x197x768xbf16>

    return %3 : tensor<2x197x768xbf16>
  }
}
