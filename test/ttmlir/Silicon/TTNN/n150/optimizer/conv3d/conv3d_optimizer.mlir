// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1 enable-greedy-optimizer=false" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: ttrt run %t.ttnn

// Silicon smoke for the optimizer-enabled vertical slice: lower a single
// Conv3dOp through the optimizer-enabled pipeline, prepare its weights,
// and execute on n150 hardware.
//
module {
  func.func @conv3d_silicon_smoke(
      %arg0: tensor<1x8x28x28x128xbf16>,
      %arg1: tensor<32x128x3x3x3xbf16>)
      -> tensor<1x6x26x26x32xbf16> {
    %0 = "ttir.conv3d"(%arg0, %arg1) <{
        stride = array<i32: 1, 1, 1>,
        padding = array<i32: 0, 0, 0>,
        groups = 1 : i32,
        padding_mode = "zeros"
      }> : (tensor<1x8x28x28x128xbf16>, tensor<32x128x3x3x3xbf16>)
        -> tensor<1x6x26x26x32xbf16>
    return %0 : tensor<1x6x26x26x32xbf16>
  }
}
