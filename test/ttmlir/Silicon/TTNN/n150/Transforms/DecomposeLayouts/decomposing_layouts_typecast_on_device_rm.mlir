// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttcore-mark-functions-as-forward --ttnn-decompose-layouts --mlir-print-local-scope -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Tests for device-to-device row-major typecasts in TTNNDecomposeLayouts
// (handleDeviceInputNoLayoutTypecast, RM-output path).

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

#dram_il_rm_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x128xbf16, #dram>, <interleaved>>
#dram_il_rm_f32  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x128xf32,  #dram>, <interleaved>>

// L1 interleaved row-major is stored in linearized (1 x N) form per shard, so
// for grid <1x1> a 32x128 tensor becomes memref<1x4096, #l1>.
#l1_il_rm_bf16   = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4096xbf16, #l1>,   <interleaved>>
#l1_il_rm_f32    = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4096xf32,  #l1>,   <interleaved>>

module attributes {} {

    // DRAM interleaved RM bf16 -> DRAM interleaved RM f32. Same memory config,
    // just a dtype change. The typecast consumes %arg0 directly and its result
    // tensor stays in DRAM.
    func.func @dram_il_rm_bf16_to_dram_il_rm_f32(%arg0: tensor<32x128xbf16, #dram_il_rm_bf16>) -> tensor<32x128xf32, #dram_il_rm_f32> {
        // CHECK-LABEL: func.func @dram_il_rm_bf16_to_dram_il_rm_f32
        // CHECK: %[[CAST:.*]] = "ttnn.typecast"(%arg0)
        // CHECK-SAME: -> tensor<32x128xf32, {{.*}}#ttnn.buffer_type<dram>
        // CHECK-NEXT: return %[[CAST]]
        %0 = "ttnn.to_layout"(%arg0)  : (tensor<32x128xbf16, #dram_il_rm_bf16>) -> tensor<32x128xf32, #dram_il_rm_f32>
        return %0 : tensor<32x128xf32, #dram_il_rm_f32>
    }

    // DRAM interleaved RM f32 -> DRAM interleaved RM bf16. Opposite dtype
    // direction; same single-typecast expectation, result stays in DRAM.
    func.func @dram_il_rm_f32_to_dram_il_rm_bf16(%arg0: tensor<32x128xf32, #dram_il_rm_f32>) -> tensor<32x128xbf16, #dram_il_rm_bf16> {
        // CHECK-LABEL: func.func @dram_il_rm_f32_to_dram_il_rm_bf16
        // CHECK: %[[CAST:.*]] = "ttnn.typecast"(%arg0)
        // CHECK-SAME: -> tensor<32x128xbf16, {{.*}}#ttnn.buffer_type<dram>
        // CHECK-NEXT: return %[[CAST]]
        %0 = "ttnn.to_layout"(%arg0)  : (tensor<32x128xf32, #dram_il_rm_f32>) -> tensor<32x128xbf16, #dram_il_rm_bf16>
        return %0 : tensor<32x128xbf16, #dram_il_rm_bf16>
    }

    // L1 interleaved RM bf16 -> L1 interleaved RM f32. Same memory config,
    // just a dtype change. Result stays in L1.
    func.func @l1_il_rm_bf16_to_l1_il_rm_f32(%arg0: tensor<32x128xbf16, #l1_il_rm_bf16>) -> tensor<32x128xf32, #l1_il_rm_f32> {
        // CHECK-LABEL: func.func @l1_il_rm_bf16_to_l1_il_rm_f32
        // CHECK: %[[CAST:.*]] = "ttnn.typecast"(%arg0)
        // CHECK-SAME: -> tensor<32x128xf32, {{.*}}#ttnn.buffer_type<l1>
        // CHECK-NEXT: return %[[CAST]]
        %0 = "ttnn.to_layout"(%arg0)  : (tensor<32x128xbf16, #l1_il_rm_bf16>) -> tensor<32x128xf32, #l1_il_rm_f32>
        return %0 : tensor<32x128xf32, #l1_il_rm_f32>
    }

    // DRAM interleaved RM bf16 -> L1 interleaved RM f32. Dtype change plus a
    // memory move. The typecast runs on the DRAM input, then a single
    // to_memory_config moves the (already-typecast) result to L1; no host
    // round-trip.
    func.func @dram_il_rm_bf16_to_l1_il_rm_f32(%arg0: tensor<32x128xbf16, #dram_il_rm_bf16>) -> tensor<32x128xf32, #l1_il_rm_f32> {
        // CHECK-LABEL: func.func @dram_il_rm_bf16_to_l1_il_rm_f32
        // CHECK: %[[CAST:.*]] = "ttnn.typecast"(%arg0)
        // CHECK-SAME: -> tensor<32x128xf32, {{.*}}#ttnn.buffer_type<dram>
        // CHECK-NEXT: %[[MEM:.*]] = "ttnn.to_memory_config"(%[[CAST]])
        // CHECK-SAME: -> tensor<32x128xf32, {{.*}}#ttnn.buffer_type<l1>
        // CHECK-NEXT: return %[[MEM]]
        %0 = "ttnn.to_layout"(%arg0)  : (tensor<32x128xbf16, #dram_il_rm_bf16>) -> tensor<32x128xf32, #l1_il_rm_f32>
        return %0 : tensor<32x128xf32, #l1_il_rm_f32>
    }

    // L1 interleaved RM f32 -> DRAM interleaved RM bf16. Opposite memory
    // direction with the smaller dtype on the output side. The typecast runs
    // on the L1 input (so its result is in L1), then a single to_memory_config
    // moves it to DRAM; no host round-trip.
    func.func @l1_il_rm_f32_to_dram_il_rm_bf16(%arg0: tensor<32x128xf32, #l1_il_rm_f32>) -> tensor<32x128xbf16, #dram_il_rm_bf16> {
        // CHECK-LABEL: func.func @l1_il_rm_f32_to_dram_il_rm_bf16
        // CHECK: %[[CAST:.*]] = "ttnn.typecast"(%arg0)
        // CHECK-SAME: -> tensor<32x128xbf16, {{.*}}#ttnn.buffer_type<l1>
        // CHECK-NEXT: %[[MEM:.*]] = "ttnn.to_memory_config"(%[[CAST]])
        // CHECK-SAME: -> tensor<32x128xbf16, {{.*}}#ttnn.buffer_type<dram>
        // CHECK-NEXT: return %[[MEM]]
        %0 = "ttnn.to_layout"(%arg0)  : (tensor<32x128xf32, #l1_il_rm_f32>) -> tensor<32x128xbf16, #dram_il_rm_bf16>
        return %0 : tensor<32x128xbf16, #dram_il_rm_bf16>
    }
}
