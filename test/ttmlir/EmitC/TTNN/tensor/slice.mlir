// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-to-emitc-device-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @slice(%arg0: tensor<4x32x32xbf16>) -> tensor<2x16x16xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0: i32, 0: i32, 0: i32], ends = [2: i32, 16: i32, 16: i32], step = [1: i32, 1: i32, 1: i32]}> : (tensor<4x32x32xbf16>) -> tensor<2x16x16xbf16>
    return %0 : tensor<2x16x16xbf16>
}
