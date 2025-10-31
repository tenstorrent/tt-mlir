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
    %0 = ttir.empty() : tensor<8x56x56x256xbf16> loc(#loc1)
    %1 = "ttir.relu"(%arg0, %0) : (tensor<8x56x56x256xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc2)
    %2 = ttir.empty() : tensor<8x56x56x64xbf16> loc(#loc3)
    // CHECK: "ttnn.conv2d"([[INPUT1:%.+]], [[WEIGHTS1:%.+]], {{%.*}}) {{.*}} conv2d_config = #ttnn.conv2d_config<{{.*}}deallocate_activation = true{{.*}}>
    %3 = "ttir.conv2d"(%1, %arg1, %2) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x256xbf16>, tensor<64x256x1x1xbf16>, tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc4)
    // CHECK-NOT: "ttnn.deallocate"([[INPUT1]])
    // CHECK: "ttnn.deallocate"([[WEIGHTS1]])
    %4 = ttir.empty() : tensor<8x56x56x64xbf16> loc(#loc5)
    %5 = "ttir.multiply"(%3, %arg2, %4) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>, tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc6)
    %6 = ttir.empty() : tensor<8x56x56x64xbf16> loc(#loc7)
    %7 = "ttir.add"(%5, %arg3, %6) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>, tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc8)
    %8 = ttir.empty() : tensor<8x56x56x64xbf16> loc(#loc9)
    %9 = "ttir.relu"(%7, %8) : (tensor<8x56x56x64xbf16>, tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc10)
    %10 = ttir.empty() : tensor<8x56x56x64xbf16> loc(#loc11)
    // CHECK: "ttnn.conv2d"([[INPUT2:%.+]], [[WEIGHTS2:%.+]], {{%.*}}) {{.*}} conv2d_config = #ttnn.conv2d_config<{{.*}}deallocate_activation = true{{.*}}>
    %11 = "ttir.conv2d"(%9, %arg4, %10) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<64x64x3x3xbf16>, tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc12)
    // CHECK-NOT: "ttnn.deallocate"([[INPUT2]])
    // CHECK: "ttnn.deallocate"([[WEIGHTS2]])
    %12 = ttir.empty() : tensor<8x56x56x64xbf16> loc(#loc13)
    %13 = "ttir.multiply"(%11, %arg5, %12) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>, tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc14)
    %14 = ttir.empty() : tensor<8x56x56x64xbf16> loc(#loc15)
    %15 = "ttir.add"(%13, %arg6, %14) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>, tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc16)
    %16 = ttir.empty() : tensor<8x56x56x64xbf16> loc(#loc17)
    %17 = "ttir.relu"(%15, %16) : (tensor<8x56x56x64xbf16>, tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc18)
    %18 = ttir.empty() : tensor<8x56x56x256xbf16> loc(#loc19)
    %19 = "ttir.conv2d"(%17, %arg7, %18) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<256x64x1x1xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc20)
    %20 = ttir.empty() : tensor<8x56x56x256xbf16> loc(#loc21)
    %21 = "ttir.multiply"(%19, %arg8, %20) : (tensor<8x56x56x256xbf16>, tensor<1x1x1x256xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc22)
    %22 = ttir.empty() : tensor<8x56x56x256xbf16> loc(#loc23)
    %23 = "ttir.add"(%21, %arg9, %22) : (tensor<8x56x56x256xbf16>, tensor<1x1x1x256xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc24)
    %24 = ttir.empty() : tensor<8x56x56x256xbf16> loc(#loc25)
    %25 = "ttir.add"(%23, %1, %24) : (tensor<8x56x56x256xbf16>, tensor<8x56x56x256xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc26)
    return %25 : tensor<8x56x56x256xbf16>
  }
}
#loc1 = loc("empty_for_initial_relu")
#loc2 = loc("initial_relu")
#loc3 = loc("empty_for_conv1")
#loc4 = loc("conv1")
#loc5 = loc("empty_for_conv1_scale")
#loc6 = loc("conv1_scale")
#loc7 = loc("empty_for_conv1_bias")
#loc8 = loc("conv1_bias")
#loc9 = loc("empty_for_conv1_relu")
#loc10 = loc("conv1_relu")
#loc11 = loc("empty_for_conv2")
#loc12 = loc("conv2")
#loc13 = loc("empty_for_conv2_scale")
#loc14 = loc("conv2_scale")
#loc15 = loc("empty_for_conv2_bias")
#loc16 = loc("conv2_bias")
#loc17 = loc("empty_for_conv2_relu")
#loc18 = loc("conv2_relu")
#loc19 = loc("empty_for_conv3")
#loc20 = loc("conv3")
#loc21 = loc("empty_for_conv3_scale")
#loc22 = loc("conv3_scale")
#loc23 = loc("empty_for_conv3_bias")
#loc24 = loc("conv3_bias")
#loc25 = loc("empty_for_residual")
#loc26 = loc("residual_add")
