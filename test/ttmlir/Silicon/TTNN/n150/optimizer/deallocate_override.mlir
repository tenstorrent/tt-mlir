// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=false override-conv2d-config=conv1=deallocate_activation#true,conv2=deallocate_activation#true" -o deallocate_override_ttnn.mlir %s --mlir-print-debuginfo
// RUN: FileCheck %s --input-file=deallocate_override_ttnn.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn deallocate_override_ttnn.mlir

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
    %1 = "ttir.relu"(%arg0) : (tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc1)
    // CHECK: "ttnn.conv2d"([[INPUT1:%.+]], [[WEIGHTS1:%.+]], {{%.*}}) {{.*}} conv2d_config = #ttnn.conv2d_config<{{.*}}deallocate_activation = true{{.*}}>
    %2 = "ttir.conv2d"(%1, %arg1) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x256xbf16>, tensor<64x256x1x1xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc2)
    // CHECK-NOT: "ttnn.deallocate"([[INPUT1]])
    // CHECK: "ttnn.deallocate"([[WEIGHTS1]])
    %3 = "ttir.multiply"(%2, %arg2) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc3)
    %4 = "ttir.add"(%3, %arg3) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc4)
    %5 = "ttir.relu"(%4) : (tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc5)
    // CHECK: "ttnn.conv2d"([[INPUT2:%.+]], [[WEIGHTS2:%.+]], {{%.*}}) {{.*}} conv2d_config = #ttnn.conv2d_config<{{.*}}deallocate_activation = true{{.*}}>
    %6 = "ttir.conv2d"(%5, %arg4) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<64x64x3x3xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc6)
    // CHECK-NOT: "ttnn.deallocate"([[INPUT2]])
    // CHECK: "ttnn.deallocate"([[WEIGHTS2]])
    %7 = "ttir.multiply"(%6, %arg5) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc7)
    %8 = "ttir.add"(%7, %arg6) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc8)
    %9 = "ttir.relu"(%8) : (tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc9)
    %10 = "ttir.conv2d"(%9, %arg7) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<256x64x1x1xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc10)
    %11 = "ttir.multiply"(%10, %arg8) : (tensor<8x56x56x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc11)
    %12 = "ttir.add"(%11, %arg9) : (tensor<8x56x56x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc12)
    %13 = "ttir.add"(%12, %1) : (tensor<8x56x56x256xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc13)
    return %13 : tensor<8x56x56x256xbf16>
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
