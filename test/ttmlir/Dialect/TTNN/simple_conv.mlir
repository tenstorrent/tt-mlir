// RUN: ttmlir-opt --ttir-load-system-desc --ttir-layout --ttnn-open-device --convert-ttir-to-ttnn %s | FileCheck %s
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<1x32x32x1xbf16>, %arg1: tensor<1x1x3x3xbf16>, %arg2: tensor<1x32x32x1xbf16>) -> tensor<1x32x32x1xbf16> {
    %0 = tensor.empty() : tensor<1x32x32x1xbf16> 
    // CHECK: %[[C:.*]] = "ttnn.conv2d"[[C:.*]]
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0) <{kernel_height=3: si32, kernel_width=3: si32, stride_height=1: si32, stride_width=1: si32, dilation=1: si32, groups=1: si32, padding_left=1: si32, padding_right=1: si32, padding_top=1: si32, padding_bottom=1: si32, is_convtranspose2d=0: si32, output_height_transpose=0: si32, output_width_transpose=0: si32, stride_transpose=0: si32, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<1x32x32x1xbf16>, tensor<1x1x3x3xbf16>, tensor<1x32x32x1xbf16>, tensor<1x32x32x1xbf16>) -> tensor<1x32x32x1xbf16>
    return %1 : tensor<1x32x32x1xbf16> 
  }
}