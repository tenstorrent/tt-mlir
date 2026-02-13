// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-flatbuffer-pipeline -o %t_fb.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_fb.mlir
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module {
  func.func @conv3d_with_bias(%arg0: tensor<1x8x28x28x32xbf16>, %arg1: tensor<32x32x3x3x3xbf16>, %arg2: tensor<1x1x1x1x32xbf16>) -> tensor<1x6x26x26x32xbf16> {
    %0 = "ttir.conv3d"(%arg0, %arg1, %arg2)
            <{
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              padding_mode = "zeros",
              groups = 1: i32
            }> : (tensor<1x8x28x28x32xbf16>, tensor<32x32x3x3x3xbf16>, tensor<1x1x1x1x32xbf16>) -> tensor<1x6x26x26x32xbf16>
    return %0 : tensor<1x6x26x26x32xbf16>
  }
}
