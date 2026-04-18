// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
//
// RUN: ttmlir-opt --ttnn-common-to-runtime-pipeline -o %t_rt.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_rt.mlir
//
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module attributes {} {
  func.func @forward(%arg0: tensor<3x4xbf16>, %arg1: tensor<3x2xui32>) -> tensor<3x2xbf16> {
    %0 = "ttir.gather_dim"(%arg0, %arg1) <{dim = 1 : i32}> : (tensor<3x4xbf16>, tensor<3x2xui32>) -> tensor<3x2xbf16>
    return %0 : tensor<3x2xbf16>
  }
}
