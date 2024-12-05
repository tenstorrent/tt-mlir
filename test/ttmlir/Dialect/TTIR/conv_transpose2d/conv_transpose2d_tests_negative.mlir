// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for slice operation

// Verify that the parsing fails if the begins attribute is not a 3D tensor
module attributes {} {
  func.func @conv_transpose2d_simple(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x10x10x256xbf16> {
    %0 = tensor.empty() : tensor<1x10x10x256xbf16>
    // CHECK: error: 'ttir.slice' op Input must be at least a 1D tensor
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0) 
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32,
              operand_constraints = [#any_device, #any_device, #any_device, #any_device]}
            > : (tensor<1x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<1x10x10x256xbf16>) -> tensor<1x10x10x256xbf16>
    return %1 : tensor<1x10x10x256xbf16>
  }
}