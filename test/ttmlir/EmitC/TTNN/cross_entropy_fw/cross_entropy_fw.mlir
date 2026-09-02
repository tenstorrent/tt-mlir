// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir | FileCheck %s

module {
  func.func @cross_entropy_fw(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x32xui32>)
      -> tensor<4x1x32x1xbf16> {
    // CHECK: ::ttnn::Tensor {{[a-z0-9]+}} = ttml::metal::cross_entropy_fw(
    %0 = "ttcore.composite"(%input, %target) <{
      composite_name = "cross_entropy_fw",
      decomposition = @cross_entropy_fw_decomposition}>
        : (tensor<4x1x32x64xbf16>, tensor<4x32xui32>) -> tensor<4x1x32x1xbf16>
    return %0 : tensor<4x1x32x1xbf16>
  }
  func.func private @cross_entropy_fw_decomposition(
      %input: tensor<4x1x32x64xbf16>,
      %target: tensor<4x32xui32>) -> tensor<4x1x32x1xbf16> {
    %0 = "ttir.zeros"() <{shape = array<i32: 4, 1, 32, 1>}> : () -> tensor<4x1x32x1xbf16>
    return %0 : tensor<4x1x32x1xbf16>
  }
}
