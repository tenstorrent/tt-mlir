// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=false override-conv2d-config=conv1=shard_layout#height_sharded,conv2=shard_layout#height_sharded" -o resnet50_layer1_module2_ttnn.mlir %s --mlir-print-debuginfo
// RUN: FileCheck %s --input-file=resnet50_layer1_module2_ttnn.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn resnet50_layer1_module2_ttnn.mlir
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
    // CHECK: %{{.*}} = "ttnn.conv2d"{{.*}} conv2d_config = #ttnn.conv2d_config<{{.*}}shard_layout = height_sharded{{.*}}>
    %1 = "ttir.conv2d"(%0, %arg1) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x256xbf16>, tensor<64x256x1x1xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc2)
    %2 = "ttir.multiply"(%1, %arg2) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc3)
    %3 = "ttir.add"(%2, %arg3) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc4)
    %4 = "ttir.relu"(%3) : (tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc5)
    // CHECK: %{{.*}} = "ttnn.conv2d"{{.*}} conv2d_config = #ttnn.conv2d_config<{{.*}}shard_layout = height_sharded{{.*}}>
    %5 = "ttir.conv2d"(%4, %arg4) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<64x64x3x3xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc6)
    %6 = "ttir.multiply"(%5, %arg5) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc7)
    %7 = "ttir.add"(%6, %arg6) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc8)
    %8 = "ttir.relu"(%7) : (tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc9)
    %9 = "ttir.conv2d"(%8, %arg7) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<256x64x1x1xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc10)
    %10 = "ttir.multiply"(%9, %arg8) : (tensor<8x56x56x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc11)
    %11 = "ttir.add"(%10, %arg9) : (tensor<8x56x56x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc12)
    %12 = "ttir.add"(%11, %0) : (tensor<8x56x56x256xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc13)
    return %12 : tensor<8x56x56x256xbf16>
  }
}
#loc1 = loc("initial_relu")
#loc2 = loc("conv1")
#loc3 = loc("conv1_scale")
#loc4 = loc("conv1_bias")
#loc5 = loc("conv1_relu")
#loc6 = loc("conv2")
#loc7 = loc("conv2_scale")
#loc8 = loc("conv2_bias")
#loc9 = loc("conv2_relu")
#loc10 = loc("conv3")
#loc11 = loc("conv3_scale")
#loc12 = loc("conv3_bias")
#loc13 = loc("residual_add")
