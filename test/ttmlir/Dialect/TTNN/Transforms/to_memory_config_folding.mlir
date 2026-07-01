// RUN: ttmlir-opt --ttnn-fold-to-memory-config -o %t %s
// RUN: FileCheck %s --input-file=%t

// With the escape-hatch option off, no to_memory_config op is folded away.
// RUN: ttmlir-opt --ttnn-fold-to-memory-config="enable-to-memory-config-folding=false" -o %t.disabled %s
// RUN: FileCheck %s --check-prefix=DISABLED --input-file=%t.disabled

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

#res_l1_bs = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 1024 + d1, d2), <8x11>, memref<4x12x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,7)>]>>
#res_dram = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 1024 + d1, d2), <1x1>, memref<32x128x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#res_l1_il = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 1024 + d1, d2), <10x11>, memref<1x38x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#res_l1_bs2 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 1024 + d1, d2), <8x4>, memref<4x32x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (3,7)>]>>

module attributes {} {

  // CHECK-LABEL: func.func @identity_to_memory_config
  // DISABLED-LABEL: func.func @identity_to_memory_config
  func.func @identity_to_memory_config(%arg0: tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_l1_bs> {
    // DISABLED: "ttnn.to_memory_config"
    // CHECK-NOT: "ttnn.to_memory_config"
    // CHECK: return %arg0
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_l1_bs>
    return %0 : tensor<1x1024x4096xbf16, #res_l1_bs>
  }

  // CHECK-LABEL: func.func @consecutive_collapse
  // CHECK-SAME: -> tensor<1x1024x4096xbf16, [[OUT:#[a-z0-9_]+]]>
  func.func @consecutive_collapse(%arg0: tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_l1_il> {
    // CHECK: %[[R:.*]] = "ttnn.to_memory_config"(%arg0){{.*}}-> tensor<1x1024x4096xbf16, [[OUT]]>
    // CHECK-NOT: "ttnn.to_memory_config"
    // CHECK: return %[[R]]
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_dram>
    %1 = "ttnn.to_memory_config"(%0) : (tensor<1x1024x4096xbf16, #res_dram>) -> tensor<1x1024x4096xbf16, #res_l1_il>
    return %1 : tensor<1x1024x4096xbf16, #res_l1_il>
  }

  // CHECK-LABEL: func.func @residual_roundtrip_adjacent
  func.func @residual_roundtrip_adjacent(%arg0: tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_l1_bs> {
    // CHECK-NOT: "ttnn.to_memory_config"
    // CHECK: return %arg0
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_dram>
    %1 = "ttnn.to_memory_config"(%0) : (tensor<1x1024x4096xbf16, #res_dram>) -> tensor<1x1024x4096xbf16, #res_l1_bs>
    return %1 : tensor<1x1024x4096xbf16, #res_l1_bs>
  }

  // CHECK-LABEL: func.func @residual_roundtrip_op_between
  func.func @residual_roundtrip_op_between(%arg0: tensor<1x1024x4096xbf16, #res_l1_bs>, %arg1: tensor<1x1024x4096xbf16, #res_l1_bs>) -> (tensor<1x1024x4096xbf16, #res_l1_bs>, tensor<1x1024x4096xbf16, #res_l1_bs>) {
    // CHECK-NOT: "ttnn.to_memory_config"
    // CHECK: %[[A:.*]] = "ttnn.abs"(%arg1)
    // CHECK: return %arg0, %[[A]]
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_dram>
    %1 = "ttnn.abs"(%arg1) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_l1_bs>
    %2 = "ttnn.to_memory_config"(%0) : (tensor<1x1024x4096xbf16, #res_dram>) -> tensor<1x1024x4096xbf16, #res_l1_bs>
    return %2, %1 : tensor<1x1024x4096xbf16, #res_l1_bs>, tensor<1x1024x4096xbf16, #res_l1_bs>
  }

  // CHECK-LABEL: func.func @staging_preserved_op_between
  func.func @staging_preserved_op_between(%arg0: tensor<1x1024x4096xbf16, #res_l1_il>, %arg1: tensor<1x1024x4096xbf16, #res_l1_bs>) -> (tensor<1x1024x4096xbf16, #res_l1_bs>, tensor<1x1024x4096xbf16, #res_l1_bs>) {
    // CHECK: "ttnn.to_memory_config"
    // CHECK: "ttnn.abs"
    // CHECK: "ttnn.to_memory_config"
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1024x4096xbf16, #res_l1_il>) -> tensor<1x1024x4096xbf16, #res_dram>
    %1 = "ttnn.abs"(%arg1) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_l1_bs>
    %2 = "ttnn.to_memory_config"(%0) : (tensor<1x1024x4096xbf16, #res_dram>) -> tensor<1x1024x4096xbf16, #res_l1_bs>
    return %2, %1 : tensor<1x1024x4096xbf16, #res_l1_bs>, tensor<1x1024x4096xbf16, #res_l1_bs>
  }

  // CHECK-LABEL: func.func @staging_preserved_different_grid
  func.func @staging_preserved_different_grid(%arg0: tensor<1x1024x4096xbf16, #res_l1_bs>, %arg1: tensor<1x1024x4096xbf16, #res_l1_bs>) -> (tensor<1x1024x4096xbf16, #res_l1_bs2>, tensor<1x1024x4096xbf16, #res_l1_bs>) {
    // CHECK: "ttnn.to_memory_config"
    // CHECK: "ttnn.abs"
    // CHECK: "ttnn.to_memory_config"
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_dram>
    %1 = "ttnn.abs"(%arg1) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_l1_bs>
    %2 = "ttnn.to_memory_config"(%0) : (tensor<1x1024x4096xbf16, #res_dram>) -> tensor<1x1024x4096xbf16, #res_l1_bs2>
    return %2, %1 : tensor<1x1024x4096xbf16, #res_l1_bs2>, tensor<1x1024x4096xbf16, #res_l1_bs>
  }

  // CHECK-LABEL: func.func @multi_use_dram_intermediate
  func.func @multi_use_dram_intermediate(%arg0: tensor<1x1024x4096xbf16, #res_l1_bs>) -> (tensor<1x1024x4096xbf16, #res_l1_bs>, tensor<1x1024x4096xbf16, #res_dram>) {
    // CHECK: %[[P:.*]] = "ttnn.to_memory_config"(%arg0)
    // CHECK: %[[A:.*]] = "ttnn.abs"(%[[P]])
    // CHECK-NOT: "ttnn.to_memory_config"
    // CHECK: return %arg0, %[[A]]
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_dram>
    %1 = "ttnn.abs"(%0) : (tensor<1x1024x4096xbf16, #res_dram>) -> tensor<1x1024x4096xbf16, #res_dram>
    %2 = "ttnn.to_memory_config"(%0) : (tensor<1x1024x4096xbf16, #res_dram>) -> tensor<1x1024x4096xbf16, #res_l1_bs>
    return %2, %1 : tensor<1x1024x4096xbf16, #res_l1_bs>, tensor<1x1024x4096xbf16, #res_dram>
  }

  // CHECK-LABEL: func.func @dram_l1_dram_roundtrip
  func.func @dram_l1_dram_roundtrip(%arg0: tensor<1x1024x4096xbf16, #res_dram>, %arg1: tensor<1x1024x4096xbf16, #res_l1_bs>) -> (tensor<1x1024x4096xbf16, #res_dram>, tensor<1x1024x4096xbf16, #res_l1_bs>) {
    // CHECK-NOT: "ttnn.to_memory_config"
    // CHECK: %[[A:.*]] = "ttnn.abs"(%arg1)
    // CHECK: return %arg0, %[[A]]
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1024x4096xbf16, #res_dram>) -> tensor<1x1024x4096xbf16, #res_l1_bs>
    %1 = "ttnn.abs"(%arg1) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_l1_bs>
    %2 = "ttnn.to_memory_config"(%0) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_dram>
    return %2, %1 : tensor<1x1024x4096xbf16, #res_dram>, tensor<1x1024x4096xbf16, #res_l1_bs>
  }
}
