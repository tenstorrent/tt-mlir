// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=false enable-trace=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// The runtime test relies on the exact layouts below; if any of these checks
// break, revisit test_trace_recapture_slot_reuse_arena's sizing assumptions.
//
// The persistent input slot must be row-major DRAM (write_tensor copies raw
// host bytes into it) while compute happens on tiled DRAM - it is this layout
// pair that forces the on-device tilize inside the traced body:
// CHECK-DAG: #[[RM_DRAM:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2048x4096xbf16, #dram>, <interleaved>>
// CHECK-DAG: #[[TILED_DRAM:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// The traced body must tilize the row-major input slot into a large
// capture-time intermediate and free it inside the body - that freed region
// is the "unsafe" hole the runtime test aims a faulty recapture into:
// CHECK-LABEL: func.func private @trace_0_large_neg
// CHECK-SAME: %arg0: tensor<2048x4096xbf16, #[[RM_DRAM]]>
// CHECK: %[[TILIZED:.*]] = "ttnn.to_layout"(%arg0)
// CHECK-SAME: -> tensor<2048x4096xbf16, #[[TILED_DRAM]]>
// CHECK: "ttnn.neg"(%[[TILIZED]])
// CHECK: "ttnn.deallocate"(%[[TILIZED]])

module {
  func.func @large_neg(%arg0: tensor<2048x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<2048x4096xbf16> {
    // CHECK: ttnn.capture_or_execute_trace
    %0 = "ttir.neg"(%arg0) : (tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16>
    return %0 : tensor<2048x4096xbf16>
  }
}
