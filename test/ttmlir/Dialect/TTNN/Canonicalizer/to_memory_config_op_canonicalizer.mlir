// RUN: ttmlir-opt --ttcore-register-device --canonicalize --mlir-print-local-scope -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// ttnn.to_memory_config folds when it leaves the tensor in the memory
// configuration it already had: either directly (identity), or as the second
// half of a round trip that returns to the producer's input type.

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout_dram = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<3x2304x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_l1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<3x2304x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>

module {
  // An identity move folds away.
  // CHECK-LABEL: func.func @identity
  func.func @identity(%arg0: tensor<96x73728xbf16, #ttnn_layout_dram>) -> tensor<96x73728xbf16, #ttnn_layout_dram> {
    // CHECK-NOT: "ttnn.to_memory_config"
    // CHECK: return %arg0
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<96x73728xbf16, #ttnn_layout_dram>) -> tensor<96x73728xbf16, #ttnn_layout_dram>
    return %0 : tensor<96x73728xbf16, #ttnn_layout_dram>
  }

  // DRAM -> L1 -> DRAM returns the tensor to where it started, and folding it
  // removes an L1 allocation rather than creating one.
  // CHECK-LABEL: func.func @dram_l1_dram_folds
  func.func @dram_l1_dram_folds(%arg0: tensor<96x73728xbf16, #ttnn_layout_dram>) -> tensor<96x73728xbf16, #ttnn_layout_dram> {
    // CHECK-NOT: "ttnn.to_memory_config"
    // CHECK: return %arg0
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<96x73728xbf16, #ttnn_layout_dram>) -> tensor<96x73728xbf16, #ttnn_layout_l1>
    %1 = "ttnn.to_memory_config"(%0) : (tensor<96x73728xbf16, #ttnn_layout_l1>) -> tensor<96x73728xbf16, #ttnn_layout_dram>
    return %1 : tensor<96x73728xbf16, #ttnn_layout_dram>
  }

  // L1 -> DRAM -> L1 must NOT fold: the DRAM hop may be deliberate staging,
  // and folding would keep the value resident in L1 across the ops in between.
  // CHECK-LABEL: func.func @l1_dram_l1_does_not_fold
  func.func @l1_dram_l1_does_not_fold(%arg0: tensor<96x73728xbf16, #ttnn_layout_l1>) -> tensor<96x73728xbf16, #ttnn_layout_l1> {
    // CHECK: "ttnn.to_memory_config"
    // CHECK: "ttnn.to_memory_config"
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<96x73728xbf16, #ttnn_layout_l1>) -> tensor<96x73728xbf16, #ttnn_layout_dram>
    %1 = "ttnn.to_memory_config"(%0) : (tensor<96x73728xbf16, #ttnn_layout_dram>) -> tensor<96x73728xbf16, #ttnn_layout_l1>
    return %1 : tensor<96x73728xbf16, #ttnn_layout_l1>
  }

  // A genuine DRAM -> L1 move stays.
  // CHECK-LABEL: func.func @dram_to_l1_stays
  func.func @dram_to_l1_stays(%arg0: tensor<96x73728xbf16, #ttnn_layout_dram>) -> tensor<96x73728xbf16, #ttnn_layout_l1> {
    // CHECK: "ttnn.to_memory_config"
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<96x73728xbf16, #ttnn_layout_dram>) -> tensor<96x73728xbf16, #ttnn_layout_l1>
    return %0 : tensor<96x73728xbf16, #ttnn_layout_l1>
  }
}
