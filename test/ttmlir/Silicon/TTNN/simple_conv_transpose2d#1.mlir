// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#any_device = #tt.operand_constraint<any_device>
#l1_block_sharded = #tt.operand_constraint<l1_block_sharded>

module attributes {} {
  func.func @forward(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x8x8x256xbf16> {
    %0 = tensor.empty() : tensor<1x8x8x256xbf16>
    // CHECK: %[[C:.*]] = "ttnn.conv_transpose2d"[[C:.*]]
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0) 
            <{
                stride_height=1: si32,
                stride_width=1: si32,
                dilation_height=1: si32,
                dilation_width=1: si32,
                groups=1: si32,
                padding_height=1: si32,
                padding_width=1: si32,
                output_padding_height=0: si32,
                output_padding_width=0: si32,
                operand_constraints = [#any_device, #any_device, #any_device, #any_device]}
            > : (tensor<1x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<1x8x8x256xbf16>) -> tensor<1x8x8x256xbf16>
    return %1 : tensor<1x8x8x256xbf16>
  }
}
