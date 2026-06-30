// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
//
// RUN: ttmlir-opt --ttnn-common-to-runtime-pipeline -o %t_rt.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_rt.mlir
//
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module {
  func.func @conv1d_with_bias(%arg0: tensor<1x32x64xbf16>, %arg1: tensor<64x64x3xbf16>, %arg2: tensor<1x1x64xbf16>) -> tensor<1x30x64xbf16> {
    %0 = "ttir.conv1d"(%arg0, %arg1, %arg2)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x64xbf16>, tensor<64x64x3xbf16>, tensor<1x1x64xbf16>) -> tensor<1x30x64xbf16>
    return %0 : tensor<1x30x64xbf16>
  }
}
