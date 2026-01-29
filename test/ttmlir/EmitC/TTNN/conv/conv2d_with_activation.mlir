// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-to-emitc-device-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module {
    func.func @conv2d_with_activation(%arg0: tensor<3x32x32x8xf32>, %arg1: tensor<16x8x3x3xf32>, %arg2: tensor<1x1x1x16xf32>) -> tensor<3x15x15x16xf32> {
        %0 = "ttir.conv2d"(%arg0, %arg1, %arg2)
                <{
                    stride = 2: i32,
                    padding = 0: i32,
                    dilation = 1: i32,
                    groups = 1: i32
                }> : (tensor<3x32x32x8xf32>, tensor<16x8x3x3xf32>, tensor<1x1x1x16xf32>) -> tensor<3x15x15x16xf32>
        %1 = "ttir.silu"(%0) : (tensor<3x15x15x16xf32>) -> tensor<3x15x15x16xf32>
        return %1 : tensor<3x15x15x16xf32>
    }
}
