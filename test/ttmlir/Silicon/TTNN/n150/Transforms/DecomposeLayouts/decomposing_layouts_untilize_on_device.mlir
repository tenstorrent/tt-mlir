// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttcore-mark-functions-as-forward --ttnn-decompose-layouts -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout_device_l1_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout_device_l1_rm_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x2048xbf16, #l1>, <interleaved>>
#ttnn_layout_device_dram_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_device_dram_rm_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x2048xbf16, #dram>, <interleaved>>
#ttnn_layout_device_dram_tile_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_device_dram_rm_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x2048xf32, #dram>, <interleaved>>
#ttnn_layout_device_l1_tile_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x2x!ttcore.tile<32x32, f32>, #l1>, <interleaved>>
#ttnn_layout_device_l1_rm_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x2048xf32, #l1>, <interleaved>>
#ttnn_layout_device_l1_block_sharded_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <4x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>
#ttnn_layout_device_dram_rm_bf16_small = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x256xbf16, #dram>, <interleaved>>
module attributes {} {

    // Test: L1 interleaved tile -> DRAM interleaved row-major (bf16).
    // Verify we move to DRAM first (toMemoryConfig), then untilize (toLayout).
    // This avoids creating a large row-major intermediate in L1.
    func.func @from_l1_tile_to_dram_rm_bf16(%arg0: tensor<32x2048xbf16, #ttnn_layout_device_l1_tile_bf16>) -> tensor<32x2048xbf16, #ttnn_layout_device_dram_rm_bf16> {
        // CHECK-LABEL: func.func @from_l1_tile_to_dram_rm_bf16
        // CHECK: %[[MEM_CONFIG:.*]] = "ttnn.to_memory_config"(%arg0)
        // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
        // CHECK: %[[TO_LAYOUT:.*]] = "ttnn.to_layout"(%[[MEM_CONFIG]])
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK: return %[[TO_LAYOUT]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x2048xbf16, #ttnn_layout_device_l1_tile_bf16>) -> tensor<32x2048xbf16, #ttnn_layout_device_dram_rm_bf16>
        return %0 : tensor<32x2048xbf16, #ttnn_layout_device_dram_rm_bf16>
    }

    // Test: L1 interleaved tile -> DRAM interleaved row-major (f32).
    // Same ordering: toMemoryConfig first, then toLayout.
    func.func @from_l1_tile_to_dram_rm_f32(%arg0: tensor<32x2048xf32, #ttnn_layout_device_l1_tile_f32>) -> tensor<32x2048xf32, #ttnn_layout_device_dram_rm_f32> {
        // CHECK-LABEL: func.func @from_l1_tile_to_dram_rm_f32
        // CHECK: %[[MEM_CONFIG:.*]] = "ttnn.to_memory_config"(%arg0)
        // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
        // CHECK: %[[TO_LAYOUT:.*]] = "ttnn.to_layout"(%[[MEM_CONFIG]])
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK: return %[[TO_LAYOUT]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x2048xf32, #ttnn_layout_device_l1_tile_f32>) -> tensor<32x2048xf32, #ttnn_layout_device_dram_rm_f32>
        return %0 : tensor<32x2048xf32, #ttnn_layout_device_dram_rm_f32>
    }

    // Test: DRAM interleaved tile -> L1 interleaved row-major (bf16).
    // Verify we untilize first (toLayout in DRAM), then move to L1 (toMemoryConfig).
    func.func @from_dram_tile_to_l1_rm_bf16(%arg0: tensor<32x2048xbf16, #ttnn_layout_device_dram_tile_bf16>) -> tensor<32x2048xbf16, #ttnn_layout_device_l1_rm_bf16> {
        // CHECK-LABEL: func.func @from_dram_tile_to_l1_rm_bf16
        // CHECK: %[[TO_LAYOUT:.*]] = "ttnn.to_layout"(%arg0)
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK: %[[MEM_CONFIG:.*]] = "ttnn.to_memory_config"(%[[TO_LAYOUT]])
        // CHECK-SAME: memory_config = #ttnn.memory_config<#l1, <interleaved>>
        // CHECK: return %[[MEM_CONFIG]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#l1, <interleaved>>}> : (tensor<32x2048xbf16, #ttnn_layout_device_dram_tile_bf16>) -> tensor<32x2048xbf16, #ttnn_layout_device_l1_rm_bf16>
        return %0 : tensor<32x2048xbf16, #ttnn_layout_device_l1_rm_bf16>
    }

    // Test: DRAM tile -> DRAM row-major (no memory config change, just untilize).
    func.func @from_dram_tile_to_dram_rm_bf16(%arg0: tensor<32x2048xbf16, #ttnn_layout_device_dram_tile_bf16>) -> tensor<32x2048xbf16, #ttnn_layout_device_dram_rm_bf16> {
        // CHECK-LABEL: func.func @from_dram_tile_to_dram_rm_bf16
        // CHECK: %[[TO_LAYOUT:.*]] = "ttnn.to_layout"(%arg0)
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK: return %[[TO_LAYOUT]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x2048xbf16, #ttnn_layout_device_dram_tile_bf16>) -> tensor<32x2048xbf16, #ttnn_layout_device_dram_rm_bf16>
        return %0 : tensor<32x2048xbf16, #ttnn_layout_device_dram_rm_bf16>
    }

    // Test: L1 block sharded tile -> DRAM interleaved row-major (bf16).
    // Verify we unshard to DRAM first (toMemoryConfig), then untilize (toLayout).
    func.func @from_l1_sharded_tile_to_dram_rm_bf16(%arg0: tensor<32x256xbf16, #ttnn_layout_device_l1_block_sharded_tile_bf16>) -> tensor<32x256xbf16, #ttnn_layout_device_dram_rm_bf16_small> {
        // CHECK-LABEL: func.func @from_l1_sharded_tile_to_dram_rm_bf16
        // CHECK: %[[MEM_CONFIG:.*]] = "ttnn.to_memory_config"(%arg0)
        // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
        // CHECK: %[[TO_LAYOUT:.*]] = "ttnn.to_layout"(%[[MEM_CONFIG]])
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK: return %[[TO_LAYOUT]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x256xbf16, #ttnn_layout_device_l1_block_sharded_tile_bf16>) -> tensor<32x256xbf16, #ttnn_layout_device_dram_rm_bf16_small>
        return %0 : tensor<32x256xbf16, #ttnn_layout_device_dram_rm_bf16_small>
    }
}
