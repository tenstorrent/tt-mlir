// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s

module @Conv2DBasic attributes {} {
  func.func @forward(%arg0: tensor<1x32x32x64xbf16> {ttir.name = "input"}, %arg1: tensor<64x64x3x3xbf16> {ttir.name = "weights"}, %arg2: tensor<1x1x1x64xbf16> {ttir.name = "bias"}) -> (tensor<1x32x32x64xbf16> {ttir.name = "Conv2DBasic.output"}) {
    %0 = ttir.empty() : tensor<1x32x32x64xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = array<i32: 1, 1>,
              padding = array<i32: 1, 1>,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x32x32x64xbf16>) -> tensor<1x32x32x64xbf16> loc(#loc1)
    return %1 : tensor<1x32x32x64xbf16>
  }
}
#loc1 = loc("conv2d_1")
