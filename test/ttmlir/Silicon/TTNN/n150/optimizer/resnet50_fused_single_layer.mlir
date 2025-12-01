// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true max-legal-layouts=32 override-conv2d-config=conv1=weights_dtype#bf16:activation#relu:deallocate_activation#false" -o resnet50_fused_single_layer_ttnn.mlir %s
// RUN: FileCheck %s --input-file=resnet50_fused_single_layer_ttnn.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn resnet50_fused_single_layer_ttnn.mlir

#loc = loc("ResNetForImageClassification")
module @ResNetBlock attributes {} {
  func.func @forward(%arg0: tensor<8x56x56x256xbf16> {ttir.name = "input"},
                     %arg1: tensor<64x256x1x1xbf16> {ttir.name = "conv1.weight"},
                     %arg2: tensor<1x1x1x64xbf16> {ttir.name = "conv1.scale"},
                     %arg3: tensor<1x1x1x64xbf16> {ttir.name = "conv1.bias"},
                     %arg4: tensor<64x64x3x3xbf16> {ttir.name = "conv2.weight"},
                     %arg5: tensor<1x1x1x64xbf16> {ttir.name = "conv2.scale"},
                     %arg6: tensor<1x1x1x64xbf16> {ttir.name = "conv2.bias"},
                     %arg7: tensor<256x64x1x1xbf16> {ttir.name = "conv3.weight"},
                     %arg8: tensor<1x1x1x256xbf16> {ttir.name = "conv3.scale"},
                     %arg9: tensor<1x1x1x256xbf16> {ttir.name = "conv3.bias"})
                     -> (tensor<8x56x56x256xbf16> {ttir.name = "output"}) {
    %0 = "ttir.relu"(%arg0) : (tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc1)
    // CHECK-DAG: #[[SHARDED_LAYOUT:.*]] = #ttnn.ttnn_layout<{{.*}}_sharded
    // CHECK: %{{.*}}conv2d_config = #ttnn.conv2d_config<weights_dtype = bf16, activation = <op_type = relu>, deallocate_activation = false
    %1 = "ttir.conv2d"(%0, %arg1) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x256xbf16>, tensor<64x256x1x1xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc2)
    %2 = "ttir.conv2d"(%1, %arg4) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<64x64x3x3xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc3)
    %3 = "ttir.conv2d"(%2, %arg7) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<256x64x1x1xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc4)
    %4 = "ttir.add"(%3, %0) : (tensor<8x56x56x256xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc5)
    return %4 : tensor<8x56x56x256xbf16> loc(#loc6)
  }
}
#loc1 = loc("initial_relu")
#loc2 = loc("conv1")
#loc3 = loc("conv2")
#loc4 = loc("conv3")
#loc5 = loc("residual_add")
#loc6 = loc("output")
