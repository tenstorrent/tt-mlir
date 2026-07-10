// RUN: ttmlir-opt --ttnn-to-ttnn-l1-advisor="system-desc-path=%system_desc_path% optimization-level=2" %s | FileCheck %s

// The direct-TTNN advisor pipeline: input is already TTNN (layouts assigned by
// the producer). The pipeline must NOT run any TTIR->TTNN lowering; it marks
// the function forward, registers a device, and runs the greedy L1 optimizer.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x96x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 3072 + d1 * 3072 + d2, d3), <1x1>, memref<96x96x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// The greedy optimizer must reshape a layout into an L1 width-sharded one.
// CHECK: #ttnn_layout{{[0-9]*}} = #ttnn.ttnn_layout{{.*}}#l1{{.*}}width_sharded
module {
  // CHECK: ttcore.device_module
  // CHECK: ttcore.device @default_device
  // CHECK-LABEL: func.func @spike
  // CHECK-SAME: tt.function_type = "forward_device"
  func.func @spike(%arg0: tensor<1x1x1x3072xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<1x1x3072x3072xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<1x1x1x3072xbf16, #ttnn_layout> {
    // No TTIR->TTNN lowering runs; the matmul stays a ttnn.matmul and the
    // optimizer picks a program config for it.
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: matmul_program_config
    // CHECK-NOT: "ttir.matmul"
    %0 = "ttnn.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1x1x1x3072xbf16, #ttnn_layout>, tensor<1x1x3072x3072xbf16, #ttnn_layout1>) -> tensor<1x1x1x3072xbf16, #ttnn_layout>
    return %0 : tensor<1x1x1x3072xbf16, #ttnn_layout>
  }
}
