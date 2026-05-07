// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttcore-mark-functions-as-forward --ttnn-decompose-layouts -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Tests for device-to-device layout transformations in TTNNDecomposeLayouts.

#l1 = #ttnn.buffer_type<l1>
#dram = #ttnn.buffer_type<dram>

#ttnn_layout_l1_rm_ui16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xui16, #l1>, <interleaved>>
#ttnn_layout_dram_tile_ui16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, u16>, #dram>, <interleaved>>
#ttnn_layout_dram_width_sharded_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <width_sharded>, core_ranges = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (7,0)>]>>
#ttnn_layout_dram_interleaved_tile_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_dram_width_sharded_tile_bf16_grid4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x4>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <width_sharded>, core_ranges = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (3,0)>]>>
#ttnn_layout_l1_block_sharded_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (7,0)>]>>

module attributes {} {

    // Verify that when moving a ui16 tensor from device L1 row-major to device
    // DRAM tiled, a typecast workaround is inserted around to_memory_config
    // because ttnn.copy (used internally by ttnn.to_memory_config) does not
    // support ui16. The expected decomposition is:
    //   to_layout (tilize on device, L1 rm ui16 -> L1 tile ui16)
    //   typecast (ui16 -> u32, L1 tile)
    //   to_memory_config (L1 tile u32 -> DRAM tile u32)
    //   typecast (u32 -> ui16, DRAM tile)
    func.func @device_l1_rm_ui16_to_dram_tile_ui16_inserts_typecast_workaround(%arg0: tensor<64x128xui16, #ttnn_layout_l1_rm_ui16>) -> tensor<64x128xui16, #ttnn_layout_dram_tile_ui16> {
        // CHECK-LABEL: func.func @device_l1_rm_ui16_to_dram_tile_ui16_inserts_typecast_workaround
        // CHECK: %[[TO_LAYOUT:.*]] = "ttnn.to_layout"(%arg0)
        // CHECK-SAME: layout = #ttnn.layout<tile>
        // CHECK-NEXT: %[[TYPECAST_U32:.*]] = "ttnn.typecast"(%[[TO_LAYOUT]])
        // CHECK-SAME: dtype = #ttcore.supportedDataTypes<u32>
        // CHECK-NEXT: %[[TO_MEM_CONFIG:.*]] = "ttnn.to_memory_config"(%[[TYPECAST_U32]])
        // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
        // CHECK-NEXT: %[[TYPECAST_U16:.*]] = "ttnn.typecast"(%[[TO_MEM_CONFIG]])
        // CHECK-SAME: dtype = #ttcore.supportedDataTypes<u16>
        // CHECK-NEXT: return %[[TYPECAST_U16]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<u16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xui16, #ttnn_layout_l1_rm_ui16>) -> tensor<64x128xui16, #ttnn_layout_dram_tile_ui16>
        return %0 : tensor<64x128xui16, #ttnn_layout_dram_tile_ui16>
    }

    // Verify that for a DRAM-sharded tile input being typecast to a DRAM
    // interleaved tile output of a different dtype (no layout change), the
    // unshard (to_memory_config) is emitted *before* the typecast.
    // tt-metal's typecast requires input/output memory layouts to match, so
    // converting memory first sidesteps the sharded-input mismatch.
    func.func @device_dram_sharded_tile_bf16_to_dram_interleaved_tile_f32_unshards_before_typecast(%arg0: tensor<32x256xbf16, #ttnn_layout_dram_width_sharded_tile_bf16>) -> tensor<32x256xf32, #ttnn_layout_dram_interleaved_tile_f32> {
        // CHECK-LABEL: func.func @device_dram_sharded_tile_bf16_to_dram_interleaved_tile_f32_unshards_before_typecast
        // CHECK: %[[TO_MEM_CONFIG:.*]] = "ttnn.to_memory_config"(%arg0)
        // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
        // CHECK-NEXT: %[[TYPECAST:.*]] = "ttnn.typecast"(%[[TO_MEM_CONFIG]])
        // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
        // CHECK-NEXT: return %[[TYPECAST]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x256xbf16, #ttnn_layout_dram_width_sharded_tile_bf16>) -> tensor<32x256xf32, #ttnn_layout_dram_interleaved_tile_f32>
        return %0 : tensor<32x256xf32, #ttnn_layout_dram_interleaved_tile_f32>
    }

    // Verify that a DRAM-sharded -> DRAM-sharded reshard with a different
    // grid (1x8 -> 1x4) decomposes to a single to_memory_config (no layout
    // or dtype change involved).
    func.func @device_dram_sharded_to_dram_sharded_different_grid_emits_to_memory_config(%arg0: tensor<32x256xbf16, #ttnn_layout_dram_width_sharded_tile_bf16>) -> tensor<32x256xbf16, #ttnn_layout_dram_width_sharded_tile_bf16_grid4> {
        // CHECK-LABEL: func.func @device_dram_sharded_to_dram_sharded_different_grid_emits_to_memory_config
        // CHECK: %[[TO_MEM_CONFIG:.*]] = "ttnn.to_memory_config"(%arg0)
        // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <width_sharded>
        // CHECK-NOT: ttnn.to_layout
        // CHECK-NOT: ttnn.typecast
        // CHECK: return %[[TO_MEM_CONFIG]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <width_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0, 0), (3, 0)>]>, <32x64>, <row_major>>>}> : (tensor<32x256xbf16, #ttnn_layout_dram_width_sharded_tile_bf16>) -> tensor<32x256xbf16, #ttnn_layout_dram_width_sharded_tile_bf16_grid4>
        return %0 : tensor<32x256xbf16, #ttnn_layout_dram_width_sharded_tile_bf16_grid4>
    }

    // Verify that an L1-sharded -> DRAM-sharded reshard decomposes to a
    // single to_memory_config.
    func.func @device_l1_sharded_to_dram_sharded_emits_to_memory_config(%arg0: tensor<32x256xbf16, #ttnn_layout_l1_block_sharded_tile_bf16>) -> tensor<32x256xbf16, #ttnn_layout_dram_width_sharded_tile_bf16> {
        // CHECK-LABEL: func.func @device_l1_sharded_to_dram_sharded_emits_to_memory_config
        // CHECK: %[[TO_MEM_CONFIG:.*]] = "ttnn.to_memory_config"(%arg0)
        // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <width_sharded>
        // CHECK-NOT: ttnn.to_layout
        // CHECK-NOT: ttnn.typecast
        // CHECK: return %[[TO_MEM_CONFIG]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <width_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0, 0), (7, 0)>]>, <32x32>, <row_major>>>}> : (tensor<32x256xbf16, #ttnn_layout_l1_block_sharded_tile_bf16>) -> tensor<32x256xbf16, #ttnn_layout_dram_width_sharded_tile_bf16>
        return %0 : tensor<32x256xbf16, #ttnn_layout_dram_width_sharded_tile_bf16>
    }
}
