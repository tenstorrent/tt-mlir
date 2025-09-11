// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

// TODO (#2507): conv2d currently fails when run in group. Merge this with
// conv2d_bf16 once resolved

module {
    func.func @conv2d_f32(%arg0: tensor<3x32x32x8xf32>, %arg1: tensor<16x8x3x3xf32>, %arg2: tensor<1x1x1x16xf32>) -> tensor<3x15x15x16xf32> {
        %0 = ttir.empty() : tensor<3x15x15x16xf32>
        %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
                <{
                    stride = 2: i32,
                    padding = 0: i32,
                    dilation = 1: i32,
                    groups = 1: i32
                }> : (tensor<3x32x32x8xf32>, tensor<16x8x3x3xf32>, tensor<1x1x1x16xf32>, tensor<3x15x15x16xf32>) -> tensor<3x15x15x16xf32>
        return %1 : tensor<3x15x15x16xf32>
    }
}
