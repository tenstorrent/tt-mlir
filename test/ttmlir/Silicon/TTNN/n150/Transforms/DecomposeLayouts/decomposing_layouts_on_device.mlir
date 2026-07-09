// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttcore-mark-functions-as-forward --ttnn-decompose-layouts -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Tests for device-to-device layout transformations in TTNNDecomposeLayouts.

#l1 = #ttnn.buffer_type<l1>
#dram = #ttnn.buffer_type<dram>

#ttnn_layout_l1_rm_ui16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xui16, #l1>, <interleaved>>
#ttnn_layout_dram_tile_ui16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, u16>, #dram>, <interleaved>>

module attributes {} {

    // Moving a ui16 tensor from device L1 row-major to device DRAM tiled: tilize
    // on device in the (preferred) L1 input memory, then a single to_memory_config
    // moves the tilized tensor to DRAM. uint16 to_memory_config works like every
    // other dtype, so no typecast workaround is needed.
    func.func @device_l1_rm_ui16_to_dram_tile_ui16(%arg0: tensor<64x128xui16, #ttnn_layout_l1_rm_ui16>) -> tensor<64x128xui16, #ttnn_layout_dram_tile_ui16> {
        // CHECK-LABEL: func.func @device_l1_rm_ui16_to_dram_tile_ui16
        // CHECK: %[[TO_LAYOUT:.*]] = "ttnn.to_layout"(%arg0)
        // CHECK-NEXT: %[[TO_MEM_CONFIG:.*]] = "ttnn.to_memory_config"(%[[TO_LAYOUT]])
        // CHECK-NOT: "ttnn.typecast"
        // CHECK: return %[[TO_MEM_CONFIG]]
        %0 = "ttnn.to_layout"(%arg0) : (tensor<64x128xui16, #ttnn_layout_l1_rm_ui16>) -> tensor<64x128xui16, #ttnn_layout_dram_tile_ui16>
        return %0 : tensor<64x128xui16, #ttnn_layout_dram_tile_ui16>
    }
}
