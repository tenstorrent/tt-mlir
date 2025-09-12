// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

// UNSUPPORTED: true
func.func @dynamic_slice(%arg0: tensor<4x32x32xbf16>, %arg1: tensor<3xui32>, %arg2: tensor<3xui32>) -> tensor<2x16x16xbf16> {
    %0 = ttir.empty() : tensor<2x16x16xbf16>
    %1 = "ttir.slice_dynamic"(%arg0, %arg1, %arg2, %0) <{step = [1: i32, 1: i32, 1: i32]}> : (tensor<4x32x32xbf16>, tensor<3xui32>, tensor<3xui32>, tensor<2x16x16xbf16>) -> tensor<2x16x16xbf16>
    return %1 : tensor<2x16x16xbf16>
}
