// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-modify-signatures-for-dylib --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @conv2d(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = tensor.empty() : tensor<1x30x30x64xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
                stride = 1: i32,
                padding = 0: i32,
                dilation = 1: i32,
                groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
}
