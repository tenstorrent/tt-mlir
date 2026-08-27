// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // The target enters the pipeline tiled, but the TTML kernel requires
  // row-major uint32. OpModel validates all three inputs and derives the tiled,
  // DRAM-interleaved output layout from the host-side composition.
  func.func @cross_entropy_bw(
      %input: tensor<4x1x32x64xbf16>,
      %target: tensor<4x32xui32>,
      %grad: tensor<1x1x1x1xbf16>) -> tensor<4x1x32x64xbf16> {
    // CHECK-DAG: #[[INPUT_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<4x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
    // CHECK-DAG: #[[TARGET_TILED_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
    // CHECK-DAG: #[[TARGET_RM_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<4x32xui32, #dram>, <interleaved>>

    // CHECK-LABEL: func.func @cross_entropy_bw(
    // CHECK-SAME: %[[INPUT:[0-9a-z_]+]]: tensor<4x1x32x64xbf16, #[[INPUT_LAYOUT]]>
    // CHECK-SAME: %[[TARGET:[0-9a-z_]+]]: tensor<4x32xui32, #[[TARGET_TILED_LAYOUT]]>
    // CHECK-SAME: %[[GRAD:[0-9a-z_]+]]: tensor<1x1x1x1xbf16,
    // CHECK-SAME: -> tensor<4x1x32x64xbf16, #[[INPUT_LAYOUT]]>
    // CHECK: %[[TARGET_RM:[0-9a-z_]+]] = "ttnn.to_layout"(%[[TARGET]])
    // CHECK-SAME: -> tensor<4x32xui32, #[[TARGET_RM_LAYOUT]]>
    // CHECK: %[[RESULT:[0-9a-z_]+]] = "ttnn.cross_entropy_bw"(%[[INPUT]], %[[TARGET_RM]], %[[GRAD]])
    // CHECK-SAME: scaler = 3.125000e-02 : f32
    // CHECK-SAME: -> tensor<4x1x32x64xbf16, #[[INPUT_LAYOUT]]>
    // CHECK: return %[[RESULT]]
    %result = "ttir.cross_entropy_bw"(%input, %target, %grad) <{
        scaler = 3.125e-02 : f32}>
        : (tensor<4x1x32x64xbf16>, tensor<4x32xui32>,
           tensor<1x1x1x1xbf16>) -> tensor<4x1x32x64xbf16>
    return %result : tensor<4x1x32x64xbf16>
  }

  // Exercise operand workarounds together with OpModel validation when the
  // function boundary uses f32 logits and grad plus signed targets. The
  // host-side multiply accepts the f32 grad directly.
  func.func @cross_entropy_bw_wrong_types(
      %input: tensor<4x1x32x64xf32>,
      %target: tensor<4x32xi32>,
      %grad: tensor<1x1x1x1xf32>) -> tensor<4x1x32x64xf32> {
    // CHECK-LABEL: func.func @cross_entropy_bw_wrong_types(
    // CHECK-SAME: %[[INPUT_F32:[0-9a-z_]+]]: tensor<4x1x32x64xf32
    // CHECK-SAME: %[[TARGET_I32:[0-9a-z_]+]]: tensor<4x32xsi32
    // CHECK-SAME: %[[GRAD_F32:[0-9a-z_]+]]: tensor<1x1x1x1xf32
    // CHECK: %[[INPUT_BF16:[0-9a-z_]+]] = "ttnn.typecast"(%[[INPUT_F32]])
    // CHECK-SAME: -> tensor<4x1x32x64xbf16
    // CHECK: %[[TARGET_U32:[0-9a-z_]+]] = "ttnn.typecast"(%[[TARGET_I32]])
    // CHECK-SAME: -> tensor<4x32xui32
    // CHECK: %[[TARGET_U32_RM:[0-9a-z_]+]] = "ttnn.to_layout"(%[[TARGET_U32]])
    // CHECK-SAME: -> tensor<4x32xui32, #[[TARGET_RM_LAYOUT]]>
    // CHECK: %[[RESULT_BF16:[0-9a-z_]+]] = "ttnn.cross_entropy_bw"(%[[INPUT_BF16]], %[[TARGET_U32_RM]], %[[GRAD_F32]])
    // CHECK-SAME: scaler = 3.125000e-02 : f32
    // CHECK-SAME: -> tensor<4x1x32x64xbf16, #[[INPUT_LAYOUT]]>
    // CHECK: %[[RESULT_F32:[0-9a-z_]+]] = "ttnn.typecast"(%[[RESULT_BF16]])
    // CHECK-SAME: -> tensor<4x1x32x64xf32
    // CHECK: return %[[RESULT_F32]]
    %result = "ttir.cross_entropy_bw"(%input, %target, %grad) <{
        scaler = 3.125e-02 : f32}>
        : (tensor<4x1x32x64xf32>, tensor<4x32xi32>,
           tensor<1x1x1x1xf32>) -> tensor<4x1x32x64xf32>
    return %result : tensor<4x1x32x64xf32>
  }
}
