// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-to-emitc-device-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir
// UNSUPPORTED: true
// EmitC lowering generates ttnn::constant()/ttnn::full() calls for scale/zero_point tensors, but these functions currently don't exist in TTNN C++ API.
module {
  func.func @forward(%arg0: tensor<1x3x320x320xf32>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 0.1>> {
    %1 = "ttir.quantize"(%arg0) : (tensor<1x3x320x320xf32>, tensor<1x3x320x320x!quant.uniform<i32:f32) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 0.1>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32, 0.1>>
  }
}
