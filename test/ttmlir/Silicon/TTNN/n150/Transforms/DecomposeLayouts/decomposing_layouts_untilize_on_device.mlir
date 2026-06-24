// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttcore-mark-functions-as-forward --ttnn-decompose-layouts -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// CHECK-DAG: #[[DRAM_RM_UI16:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<32x2048xui16, #dram>
// CHECK-DAG: #[[L1_RM_UI16:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<1x1024xui16, #l1>
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout_device_l1_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout_device_l1_rm_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x1024xbf16, #l1>, <interleaved>>
#ttnn_layout_device_dram_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_device_dram_rm_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x2048xbf16, #dram>, <interleaved>>
#ttnn_layout_device_dram_tile_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_device_dram_rm_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x2048xf32, #dram>, <interleaved>>
#ttnn_layout_device_l1_tile_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x2x!ttcore.tile<32x32, f32>, #l1>, <interleaved>>
#ttnn_layout_device_l1_rm_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x1024xf32, #l1>, <interleaved>>
#ttnn_layout_device_l1_block_sharded_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <4x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (7,3)>]>>
#ttnn_layout_device_dram_rm_bf16_small = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x256xbf16, #dram>, <interleaved>>
#ttnn_layout_device_dram_tile_ui16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, u16>, #dram>, <interleaved>>
#ttnn_layout_device_dram_rm_ui16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x2048xui16, #dram>, <interleaved>>
#ttnn_layout_device_l1_rm_ui16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x1024xui16, #l1>, <interleaved>>
module attributes {} {

    // Test: L1 interleaved tile -> DRAM interleaved row-major (bf16).
    // Verify we move to DRAM first (toMemoryConfig), then untilize (toLayout).
    // This avoids creating a large row-major intermediate in L1.
    func.func @from_l1_tile_to_dram_rm_bf16(%arg0: tensor<32x2048xbf16, #ttnn_layout_device_l1_tile_bf16>) -> tensor<32x2048xbf16, #ttnn_layout_device_dram_rm_bf16> {
        // CHECK-LABEL: func.func @from_l1_tile_to_dram_rm_bf16
        // CHECK: %[[MEM_CONFIG:.*]] = "ttnn.to_memory_config"(%arg0)
        // CHECK: %[[TO_LAYOUT:.*]] = "ttnn.to_layout"(%[[MEM_CONFIG]])
        // CHECK: return %[[TO_LAYOUT]]
        %0 = "ttnn.to_layout"(%arg0)  : (tensor<32x2048xbf16, #ttnn_layout_device_l1_tile_bf16>) -> tensor<32x2048xbf16, #ttnn_layout_device_dram_rm_bf16>
        return %0 : tensor<32x2048xbf16, #ttnn_layout_device_dram_rm_bf16>
    }

    // Test: L1 interleaved tile -> DRAM interleaved row-major (f32).
    // Same ordering: toMemoryConfig first, then toLayout.
    func.func @from_l1_tile_to_dram_rm_f32(%arg0: tensor<32x2048xf32, #ttnn_layout_device_l1_tile_f32>) -> tensor<32x2048xf32, #ttnn_layout_device_dram_rm_f32> {
        // CHECK-LABEL: func.func @from_l1_tile_to_dram_rm_f32
        // CHECK: %[[MEM_CONFIG:.*]] = "ttnn.to_memory_config"(%arg0)
        // CHECK: %[[TO_LAYOUT:.*]] = "ttnn.to_layout"(%[[MEM_CONFIG]])
        // CHECK: return %[[TO_LAYOUT]]
        %0 = "ttnn.to_layout"(%arg0)  : (tensor<32x2048xf32, #ttnn_layout_device_l1_tile_f32>) -> tensor<32x2048xf32, #ttnn_layout_device_dram_rm_f32>
        return %0 : tensor<32x2048xf32, #ttnn_layout_device_dram_rm_f32>
    }

    // Test: DRAM interleaved tile -> L1 interleaved row-major (bf16).
    // Verify we untilize first (toLayout in DRAM), then move to L1 (toMemoryConfig).
    func.func @from_dram_tile_to_l1_rm_bf16(%arg0: tensor<32x2048xbf16, #ttnn_layout_device_dram_tile_bf16>) -> tensor<32x2048xbf16, #ttnn_layout_device_l1_rm_bf16> {
        // CHECK-LABEL: func.func @from_dram_tile_to_l1_rm_bf16
        // CHECK: %[[TO_LAYOUT:.*]] = "ttnn.to_layout"(%arg0)
        // CHECK: %[[MEM_CONFIG:.*]] = "ttnn.to_memory_config"(%[[TO_LAYOUT]])
        // CHECK: return %[[MEM_CONFIG]]
        %0 = "ttnn.to_layout"(%arg0)  : (tensor<32x2048xbf16, #ttnn_layout_device_dram_tile_bf16>) -> tensor<32x2048xbf16, #ttnn_layout_device_l1_rm_bf16>
        return %0 : tensor<32x2048xbf16, #ttnn_layout_device_l1_rm_bf16>
    }

    // Test: DRAM tile -> DRAM row-major (no memory config change, just untilize).
    func.func @from_dram_tile_to_dram_rm_bf16(%arg0: tensor<32x2048xbf16, #ttnn_layout_device_dram_tile_bf16>) -> tensor<32x2048xbf16, #ttnn_layout_device_dram_rm_bf16> {
        // CHECK-LABEL: func.func @from_dram_tile_to_dram_rm_bf16
        // CHECK: %[[TO_LAYOUT:.*]] = "ttnn.to_layout"(%arg0)
        // CHECK: return %[[TO_LAYOUT]]
        %0 = "ttnn.to_layout"(%arg0)  : (tensor<32x2048xbf16, #ttnn_layout_device_dram_tile_bf16>) -> tensor<32x2048xbf16, #ttnn_layout_device_dram_rm_bf16>
        return %0 : tensor<32x2048xbf16, #ttnn_layout_device_dram_rm_bf16>
    }

    // Test: L1 block sharded tile -> DRAM interleaved row-major (bf16).
    // Verify we unshard to DRAM first (toMemoryConfig), then untilize (toLayout).
    func.func @from_l1_sharded_tile_to_dram_rm_bf16(%arg0: tensor<32x256xbf16, #ttnn_layout_device_l1_block_sharded_tile_bf16>) -> tensor<32x256xbf16, #ttnn_layout_device_dram_rm_bf16_small> {
        // CHECK-LABEL: func.func @from_l1_sharded_tile_to_dram_rm_bf16
        // CHECK: %[[MEM_CONFIG:.*]] = "ttnn.to_memory_config"(%arg0)
        // CHECK: %[[TO_LAYOUT:.*]] = "ttnn.to_layout"(%[[MEM_CONFIG]])
        // CHECK: return %[[TO_LAYOUT]]
        %0 = "ttnn.to_layout"(%arg0)  : (tensor<32x256xbf16, #ttnn_layout_device_l1_block_sharded_tile_bf16>) -> tensor<32x256xbf16, #ttnn_layout_device_dram_rm_bf16_small>
        return %0 : tensor<32x256xbf16, #ttnn_layout_device_dram_rm_bf16_small>
    }

    // Test: DRAM tile -> DRAM row-major (ui16). uint16 untilize is supported on
    // device, so this must stay on device (a single toLayout) rather than
    // falling back to a host round-trip (from_device / to_device).
    func.func @from_dram_tile_to_dram_rm_ui16(%arg0: tensor<32x2048xui16, #ttnn_layout_device_dram_tile_ui16>) -> tensor<32x2048xui16, #ttnn_layout_device_dram_rm_ui16> {
        // CHECK-LABEL: func.func @from_dram_tile_to_dram_rm_ui16
        // CHECK-NOT: "ttnn.from_device"
        // CHECK: %[[TO_LAYOUT:.*]] = "ttnn.to_layout"(%arg0)
        // CHECK-SAME: -> tensor<32x2048xui16, #[[DRAM_RM_UI16]]>
        // CHECK-NOT: "ttnn.from_device"
        // CHECK: return %[[TO_LAYOUT]]
        %0 = "ttnn.to_layout"(%arg0) : (tensor<32x2048xui16, #ttnn_layout_device_dram_tile_ui16>) -> tensor<32x2048xui16, #ttnn_layout_device_dram_rm_ui16>
        return %0 : tensor<32x2048xui16, #ttnn_layout_device_dram_rm_ui16>
    }

    // Test: DRAM tile -> L1 row-major (ui16). Untilize stays on device (toLayout),
    // then the DRAM -> L1 move goes through the existing ui16 to_memory_config
    // typecast workaround (ui16 -> u32 -> u32 -> ui16), since ttnn.copy does not
    // support ui16. The key point is no host round-trip (no from_device).
    func.func @from_dram_tile_to_l1_rm_ui16(%arg0: tensor<32x2048xui16, #ttnn_layout_device_dram_tile_ui16>) -> tensor<32x2048xui16, #ttnn_layout_device_l1_rm_ui16> {
        // CHECK-LABEL: func.func @from_dram_tile_to_l1_rm_ui16
        // CHECK-NOT: "ttnn.from_device"
        // CHECK: %[[TO_LAYOUT:.*]] = "ttnn.to_layout"(%arg0)
        // CHECK-SAME: -> tensor<32x2048xui16, #[[DRAM_RM_UI16]]>
        // CHECK: %[[TYPECAST_U32:.*]] = "ttnn.typecast"(%[[TO_LAYOUT]])
        // CHECK-SAME: -> tensor<{{.*}}ui32
        // CHECK: %[[MEM_CONFIG:.*]] = "ttnn.to_memory_config"(%[[TYPECAST_U32]])
        // CHECK: %[[TYPECAST_U16:.*]] = "ttnn.typecast"(%[[MEM_CONFIG]])
        // CHECK-SAME: -> tensor<32x2048xui16, #[[L1_RM_UI16]]>
        // CHECK-NOT: "ttnn.from_device"
        // CHECK: return %[[TYPECAST_U16]]
        %0 = "ttnn.to_layout"(%arg0) : (tensor<32x2048xui16, #ttnn_layout_device_dram_tile_ui16>) -> tensor<32x2048xui16, #ttnn_layout_device_l1_rm_ui16>
        return %0 : tensor<32x2048xui16, #ttnn_layout_device_l1_rm_ui16>
    }
}
