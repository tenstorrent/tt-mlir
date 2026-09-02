// RUN: ttmlir-opt --ttir-to-ttnn-runtime-pipeline="composite-resolution=force-promote" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // The logits and grad stay tiled bf16, but ttml::metal::cross_entropy_bw reads
  // the class indices as row-major uint32, so the operand workarounds must insert
  // a to_layout that untilizes target.
  func.func @cross_entropy_bw(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x32xui32>, %grad: tensor<1x1x1x1xbf16>)
      -> tensor<4x1x32x64xbf16> {
    // CHECK-DAG: #[[INPUT_TILED:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<4x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
    // CHECK-DAG: #[[TARGET_RM:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<4x32xui32, #dram>, <interleaved>>

    // The target arrives tiled and gets untilized to row-major ui32.
    // CHECK: %[[TARGET:[0-9]+]] = "ttnn.to_layout"(%arg1)
    // CHECK-SAME: -> tensor<4x32xui32, #[[TARGET_RM]]>

    // Unlike cross_entropy_fw, the result keeps input's shape.
    // CHECK: "ttnn.cross_entropy_bw"(%arg0, %[[TARGET]], %arg2)
    // CHECK-SAME: scaler = 3.125000e-02 : f32
    // CHECK-SAME: (tensor<4x1x32x64xbf16, #[[INPUT_TILED]]>, tensor<4x32xui32, #[[TARGET_RM]]>, tensor<1x1x1x1xbf16
    // CHECK-SAME: -> tensor<4x1x32x64xbf16
    %0 = "ttcore.composite"(%input, %target, %grad) <{
      composite_name = "cross_entropy_bw",
      decomposition = @cross_entropy_bw_decomposition,
      composite_attributes = {scaler = 3.125e-02 : f32}}>
        : (tensor<4x1x32x64xbf16>, tensor<4x32xui32>, tensor<1x1x1x1xbf16>) -> tensor<4x1x32x64xbf16>
    return %0 : tensor<4x1x32x64xbf16>
  }
  func.func private @cross_entropy_bw_decomposition(
      %input: tensor<4x1x32x64xbf16>, %target: tensor<4x32xui32>,
      %grad: tensor<1x1x1x1xbf16>) -> tensor<4x1x32x64xbf16> {
    return %input : tensor<4x1x32x64xbf16>
  }
}
