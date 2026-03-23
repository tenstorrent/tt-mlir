// RUN: ttmlir-opt --ttir-to-ttir-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  func.func @test_full_maxpool2d_fusing() -> tensor<1x64x64x32xbf16> {
    // CHECK: %[[FULL:.*]] = "ttir.full"() <{fill_value = 1.000000e+00 : f32, shape = array<i32: 1, 64, 64, 32>}> : () -> tensor<1x64x64x32xbf16>
    // CHECK-NOT: "ttir.max_pool2d"
    // CHECK: return %[[FULL]] : tensor<1x64x64x32xbf16>
    %cst = "ttir.full"() <{fill_value = 1.0 : f32, shape = array<i32: 1, 128, 128, 32>}> : () -> tensor<1x128x128x32xbf16>
    %0 = "ttir.max_pool2d"(%cst) <{
        kernel = array<i32: 2, 2>,
        stride = array<i32: 2, 2>,
        dilation = array<i32: 1, 1>,
        ceil_mode = false,
        padding = array<i32: 0, 0, 0, 0>}> : (tensor<1x128x128x32xbf16>) -> tensor<1x64x64x32xbf16>
    return %0 : tensor<1x64x64x32xbf16>
  }

  func.func @test_full_avgpool2d_fusing() -> tensor<1x64x64x32xbf16> {
    // CHECK: %[[FULL:.*]] = "ttir.full"() <{fill_value = 2.000000e+00 : f32, shape = array<i32: 1, 64, 64, 32>}> : () -> tensor<1x64x64x32xbf16>
    // CHECK-NOT: "ttir.avg_pool2d"
    // CHECK: return %[[FULL]] : tensor<1x64x64x32xbf16>
    %cst = "ttir.full"() <{fill_value = 2.0 : f32, shape = array<i32: 1, 128, 128, 32>}> : () -> tensor<1x128x128x32xbf16>
    %0 = "ttir.avg_pool2d"(%cst) <{
        kernel = array<i32: 2, 2>,
        stride = array<i32: 2, 2>,
        dilation = array<i32: 1, 1>,
        ceil_mode = false,
        padding = array<i32: 0, 0, 0, 0>}> : (tensor<1x128x128x32xbf16>) -> tensor<1x64x64x32xbf16>
    return %0 : tensor<1x64x64x32xbf16>
  }
}
