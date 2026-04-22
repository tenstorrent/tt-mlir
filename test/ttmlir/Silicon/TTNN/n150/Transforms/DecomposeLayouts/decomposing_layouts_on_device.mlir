// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttcore-mark-functions-as-forward --ttnn-decompose-layouts -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Tests for device-to-device layout transformations in TTNNDecomposeLayouts.

#l1 = #ttnn.buffer_type<l1>
#dram = #ttnn.buffer_type<dram>

#ttnn_layout_l1_rm_ui16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xui16, #l1>, <interleaved>>
#ttnn_layout_dram_tile_ui16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, u16>, #dram>, <interleaved>>

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
}
