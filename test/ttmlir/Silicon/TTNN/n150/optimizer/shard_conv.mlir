// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true" -o shard_conv_ttnn.mlir %s
// RUN: FileCheck %s --input-file=shard_conv_ttnn.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer shard_conv_ttnn.mlir > %t.ttnn

#loc = loc("ConvTest":0:0)
module @ConvTest attributes {} {
  func.func @forward(%arg0: tensor<1x224x224x3xf32> {ttir.name = "input"} loc("ConvTest":0:0), 
                     %arg1: tensor<64x3x7x7xf32> {ttir.name = "conv_weight"} loc("ConvTest":0:0)) 
                     -> (tensor<1x112x112x64xf32> {ttir.name = "conv_output"}) {
    // CHECK-DAG: #[[SHARDED_LAYOUT:.*]] = #ttnn.ttnn_layout<{{.*}}_sharded
    // CHECK: %{{.*}} = "ttnn.conv2d"{{.*}}#[[SHARDED_LAYOUT]]>
    %0 = "ttir.empty"() : () -> tensor<1x112x112x64xf32> loc(#loc1)
    %1 = "ttir.conv2d"(%arg0, %arg1, %0) {dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 3, 3, 3, 3>, stride = array<i32: 2, 2>, channel_last = 1 : si32} : (tensor<1x224x224x3xf32>, tensor<64x3x7x7xf32>, tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32> loc(#loc1)
    return %1 : tensor<1x112x112x64xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("conv2d_op"(#loc))
