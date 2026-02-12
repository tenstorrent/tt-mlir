// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true max-legal-layouts=32" -o output_file.mlir %s
// RUN: FileCheck %s --input-file=output_file.mlir
module attributes {} {
  func.func @forward(%arg0: tensor<16x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<16x32x32x64xbf16> {
    // CHECK-DAG: #[[CONV_LAYOUT:.*]] = #ttnn.ttnn_layout<{{.*}}#l1>, <height_sharded>>
    // CHECK: {{.*}} = "ttnn.conv2d"{{.*}}-> tensor<{{.*}}, #[[CONV_LAYOUT]]>
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2)
            <{
                stride = 1: i32,
                padding = 1: i32,
                dilation = 1: i32,
                groups = 1: i32
            }> : (tensor<16x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>) -> tensor<16x32x32x64xbf16>
    %3 = "ttir.add"(%1, %1) : (tensor<16x32x32x64xbf16>, tensor<16x32x32x64xbf16>) -> tensor<16x32x32x64xbf16>
    return %3 : tensor<16x32x32x64xbf16>
  }
}
