// REQUIRES: opmodel
// RUN: not ttmlir-opt --ttcore-register-device --ttnn-operation-validation-and-fallback="d2m-fallback-enabled=false tensor-l1-usage-cap=0.001" %s 2>&1 | FileCheck %s

// with tensor-l1-usage-cap=0.001 and d2m fallback disabled,
// this should fail compilation with an error.

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout_l1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 262144 + d1 * 512 + d2, d3), <1x1>, memref<16x16x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>

module attributes {"ttnn.tensor_l1_usage_cap" = 0.001 : f32} {
  func.func @add_with_l1_oom(%arg0: tensor<1x1x512x512xbf16, #ttnn_layout_l1>, %arg1: tensor<1x1x512x512xbf16, #ttnn_layout_l1>) -> tensor<1x1x512x512xbf16, #ttnn_layout_l1> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device


    // CHECK: error: OperationValidationAndFallback: Operation ttnn.add failed validation
    %1 = "ttnn.add"(%arg0, %arg1) <{operandSegmentSizes = array<i32: 2, 0>, dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x512x512xbf16, #ttnn_layout_l1>, tensor<1x1x512x512xbf16, #ttnn_layout_l1>) -> tensor<1x1x512x512xbf16, #ttnn_layout_l1>

    return %1 : tensor<1x1x512x512xbf16, #ttnn_layout_l1>
  }
}
