// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module @jit_convolution {
  func.func public @test_NCHW_IOHW_to_NHWC_OIHW_conv2d(%arg0: tensor<1x3x100x100xbf16>, %arg1: tensor<7x3x3x3xbf16>) -> tensor<1x7x100x100xbf16> {
    %0 = "ttir.conv2d"(%arg0, %arg1) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>, batch_dim = 0 : i64, channel_dim = 1 : i64, height_dim = 2 : i64, width_dim = 3 : i64}> : (tensor<1x3x100x100xbf16>, tensor<7x3x3x3xbf16>) -> tensor<1x7x100x100xbf16>
    return %0 : tensor<1x7x100x100xbf16>
  }
}
