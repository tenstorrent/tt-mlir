// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-decompose-layouts -o %t %s
// RUN: FileCheck %s --input-file=%t

// Tests for device-to-device layout transformations in TTNNDecomposeLayouts.

#l1 = #ttnn.buffer_type<l1>
#dram = #ttnn.buffer_type<dram>

#ttnn_layout_l1_rm_ui16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xui16, #l1>, <interleaved>>
#ttnn_layout_dram_tile_ui16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, u16>, #dram>, <interleaved>>

// ROW_MAJOR HEIGHT_SHARDED L1 input: 32x4 bf16 (last dim 4 is NOT tile-aligned).
#ttnn_layout_l1_hs_rm_bf16_nontile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x4xbf16, #l1>, <height_sharded>>
// TILE HEIGHT_SHARDED L1 output (same shape logically, padded internally to 1x1 tiles).
#ttnn_layout_l1_hs_tile_bf16_nontile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>>

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

    // Workaround for tt-metal#30541: ttnn.tilize does not support HEIGHT_SHARDED
    // L1 tensors with non-tile-aligned shard shapes. Verify the decomposition
    // produces: unshard (to_memory_config to DRAM INTERLEAVED) → pad → tilize
    // → slice → reshard (to_memory_config back to HEIGHT_SHARDED L1).
    func.func @device_l1_hs_rm_bf16_nontile_to_tile_inserts_pad_slice_workaround(%arg0: tensor<32x4xbf16, #ttnn_layout_l1_hs_rm_bf16_nontile>) -> tensor<32x4xbf16, #ttnn_layout_l1_hs_tile_bf16_nontile> {
        // CHECK-LABEL: func.func @device_l1_hs_rm_bf16_nontile_to_tile_inserts_pad_slice_workaround
        // CHECK: %[[UNSHARD:.*]] = "ttnn.to_memory_config"(%arg0)
        // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
        // CHECK-NEXT: %[[PAD:.*]] = "ttnn.pad"(%[[UNSHARD]])
        // CHECK-NEXT: %[[TILIZE:.*]] = "ttnn.to_layout"(%[[PAD]])
        // CHECK-SAME: layout = #ttnn.layout<tile>
        // CHECK-NEXT: %[[SLICE:.*]] = "ttnn.slice_static"(%[[TILIZE]])
        // CHECK-NEXT: %[[RESHARD:.*]] = "ttnn.to_memory_config"(%[[SLICE]])
        // CHECK-SAME: memory_config = #ttnn.memory_config<#l1, <height_sharded>
        // CHECK-NEXT: return %[[RESHARD]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#l1, <height_sharded>, #ttnn.shard_spec<#ttnn.core_range_set<[#ttnn.core_range<(0,0), (0,0)>]>, <32x4>, <row_major>>>}> : (tensor<32x4xbf16, #ttnn_layout_l1_hs_rm_bf16_nontile>) -> tensor<32x4xbf16, #ttnn_layout_l1_hs_tile_bf16_nontile>
        return %0 : tensor<32x4xbf16, #ttnn_layout_l1_hs_tile_bf16_nontile>
    }
}
