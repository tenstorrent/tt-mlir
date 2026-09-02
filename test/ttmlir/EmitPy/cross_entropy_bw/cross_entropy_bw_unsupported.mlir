// RUN: not ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" %s 2>&1 | FileCheck %s

// EmitPy lowering of ttnn.cross_entropy_bw is deliberately unsupported. ttml does
// not expose the metal::cross_entropy_bw primitive through its Python bindings.

module {
  func.func @cross_entropy_bw(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x32xui32>, %grad: tensor<1x1x1x1xbf16>)
      -> tensor<4x1x32x64xbf16> {
    // CHECK: failed to legalize operation 'ttnn.cross_entropy_bw'
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
