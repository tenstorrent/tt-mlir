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
    %0 = ttir.empty() : tensor<8x56x56x256xbf16> loc(#loc1)
    %1 = "ttir.relu"(%arg0, %0) : (tensor<8x56x56x256xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc2)
    %2 = ttir.empty() : tensor<8x56x56x64xbf16> loc(#loc3)
    // CHECK-DAG: #[[SHARDED_LAYOUT:.*]] = #ttnn.ttnn_layout<{{.*}}_sharded
    // CHECK: "ttnn.reshape"([[INPUT1:%.+]]) <{shape = [25088 : i32, 256 : i32]}>
    // CHECK: "ttnn.matmul"([[RESHAPED_INPUT1:%.+]], [[WEIGHTS1:%.+]]) <{transpose_a = false, transpose_b = false}>
    %3 = "ttir.conv2d"(%1, %arg1, %2) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x256xbf16>, tensor<64x256x1x1xbf16>, tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc4)
    %10 = ttir.empty() : tensor<8x56x56x64xbf16> loc(#loc11)
    %11 = "ttir.conv2d"(%3, %arg4, %10) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<64x64x3x3xbf16>, tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc12)
    %18 = ttir.empty() : tensor<8x56x56x256xbf16> loc(#loc19)
    %19 = "ttir.conv2d"(%11, %arg7, %18) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<256x64x1x1xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc20)
    %24 = ttir.empty() : tensor<8x56x56x256xbf16> loc(#loc25)
    %25 = "ttir.add"(%19, %1, %24) : (tensor<8x56x56x256xbf16>, tensor<8x56x56x256xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc26)
    return %25 : tensor<8x56x56x256xbf16> loc(#loc27)
  }
}
#loc1 = loc("empty_for_initial_relu")
#loc2 = loc("initial_relu")
#loc3 = loc("empty_for_conv1")
#loc4 = loc("conv1")
#loc11 = loc("empty_for_conv2")
#loc12 = loc("conv2")
#loc19 = loc("empty_for_conv3")
#loc20 = loc("conv3")
#loc25 = loc("empty_for_residual")
#loc26 = loc("residual_add")
#loc27 = loc("output")
