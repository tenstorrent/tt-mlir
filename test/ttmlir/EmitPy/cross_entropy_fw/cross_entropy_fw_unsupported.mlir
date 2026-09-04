// RUN: not ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" %s 2>&1 | FileCheck %s

// EmitPy lowering of ttnn.cross_entropy_fw is deliberately unsupported. ttml does
// not expose the metal::cross_entropy_fw primitive through its Python bindings.

module {
  func.func @cross_entropy_fw(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x32xui32>)
      -> tensor<4x1x32x1xbf16> {
    // CHECK: failed to legalize operation 'ttnn.cross_entropy_fw'
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
