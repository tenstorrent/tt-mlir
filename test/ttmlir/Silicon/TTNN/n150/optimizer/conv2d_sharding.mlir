// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true" %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

func.func @conv2d_sharding(%arg0: tensor<16x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<16x32x32x64xbf16> {
    // CHECK-DAG: #[[LAYOUT_0:.*]] = #ttnn.ttnn_layout{{.*}}sharded{{.*}}
    %0 = ttir.empty() : tensor<16x32x32x64xbf16>
    // CHECK: = "ttnn.conv2d"{{.*}} -> {{.*}}#[[LAYOUT_0]]{{.*}}
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
                stride = 1: i32,
                padding = 1: i32,
                dilation = 1: i32,
                groups = 1: i32
            }> : (tensor<16x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<16x32x32x64xbf16>) -> tensor<16x32x32x64xbf16>
    return %1 : tensor<16x32x32x64xbf16>
}
