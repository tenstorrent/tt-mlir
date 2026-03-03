// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true max-legal-layouts=32" -o resnet50_layer2_module1_ttnn.mlir %s --mlir-print-debuginfo
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn resnet50_layer2_module1_ttnn.mlir
#loc = loc("ResNetForImageClassification")
module @ResNetLayer2Module1 attributes {} {
  func.func @forward(%arg0: tensor<8x56x56x256xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input"} , %arg1: tensor<128x256x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.1.layers.0.layer.0.convolution.weight"} , %arg2: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_multiply_183"} , %arg3: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_multiply_183_fork_clone1065"} , %arg4: tensor<128x128x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.1.layers.0.layer.1.convolution.weight"} , %arg5: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_multiply_199"} , %arg6: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_multiply_199_fork_clone962"} , %arg7: tensor<512x128x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.1.layers.0.layer.2.convolution.weight"} , %arg8: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_multiply_215"} , %arg9: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_multiply_215_fork_clone853"} , %arg10: tensor<512x256x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.1.layers.0.shortcut.convolution.weight"} , %arg11: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_multiply_230"} , %arg12: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_multiply_230_fork_clone857"} ) -> (tensor<8x28x28x512xbf16> {ttir.name = "output"}) {
    %0 = "ttir.relu"(%arg0) : (tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc1)
    %1 = "ttir.conv2d"(%0, %arg1) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x256xbf16>, tensor<128x256x1x1xbf16>) -> tensor<8x56x56x128xbf16> loc(#loc2)
    %2 = "ttir.multiply"(%1, %arg2) : (tensor<8x56x56x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x56x56x128xbf16> loc(#loc3)
    %3 = "ttir.add"(%2, %arg3) : (tensor<8x56x56x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x56x56x128xbf16> loc(#loc4)
    %4 = "ttir.relu"(%3) : (tensor<8x56x56x128xbf16>) -> tensor<8x56x56x128xbf16> loc(#loc5)
    %5 = "ttir.conv2d"(%4, %arg4) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<8x56x56x128xbf16>, tensor<128x128x3x3xbf16>) -> tensor<8x28x28x128xbf16> loc(#loc6)
    %6 = "ttir.multiply"(%5, %arg5) : (tensor<8x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x28x28x128xbf16> loc(#loc7)
    %7 = "ttir.add"(%6, %arg6) : (tensor<8x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x28x28x128xbf16> loc(#loc8)
    %8 = "ttir.relu"(%7) : (tensor<8x28x28x128xbf16>) -> tensor<8x28x28x128xbf16> loc(#loc9)
    %9 = "ttir.conv2d"(%8, %arg7) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x28x28x128xbf16>, tensor<512x128x1x1xbf16>) -> tensor<8x28x28x512xbf16> loc(#loc10)
    %10 = "ttir.multiply"(%9, %arg8) : (tensor<8x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x28x28x512xbf16> loc(#loc11)
    %11 = "ttir.add"(%10, %arg9) : (tensor<8x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x28x28x512xbf16> loc(#loc12)
    %12 = "ttir.conv2d"(%0, %arg10) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<8x56x56x256xbf16>, tensor<512x256x1x1xbf16>) -> tensor<8x28x28x512xbf16> loc(#loc13)
    %13 = "ttir.multiply"(%12, %arg11) : (tensor<8x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x28x28x512xbf16> loc(#loc14)
    %14 = "ttir.add"(%13, %arg12) : (tensor<8x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x28x28x512xbf16> loc(#loc15)
    %15 = "ttir.add"(%11, %14) : (tensor<8x28x28x512xbf16>, tensor<8x28x28x512xbf16>) -> tensor<8x28x28x512xbf16> loc(#loc16)
    return %15 : tensor<8x28x28x512xbf16>
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
#loc13 = loc("shortcut")
#loc14 = loc("shortcut_scale")
#loc15 = loc("shortcut_bias")
#loc16 = loc("residual_add")
