// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @convert_to_hwc(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x1x8x3xf32> {
    %1 = "ttir.convert_to_hwc"(%arg0) : (tensor<1x2x3x4xf32>) -> tensor<1x1x8x3xf32>
    return %1 : tensor<1x1x8x3xf32>
}
