// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true max-legal-layouts=32" -o resnet50_layer2_module2_ttnn.mlir %s --mlir-print-debuginfo
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn resnet50_layer2_module2_ttnn.mlir
#loc = loc("ResNetForImageClassification")
module @ResNetLayer2Module2 attributes {} {
  func.func @forward(%arg0: tensor<8x28x28x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input"}, %arg1: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_multiply_247"}, %arg2: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_multiply_247_fork_clone1007"}, %arg3: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_multiply_263"}, %arg4: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_multiply_263_fork_clone901"}, %arg5: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_multiply_279"}, %arg6: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_multiply_279_fork_clone791"}, %arg7: tensor<128x512x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.1.layers.1.layer.0.convolution.weight"}, %arg8: tensor<128x128x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.1.layers.1.layer.1.convolution.weight"}, %arg9: tensor<512x128x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.1.layers.1.layer.2.convolution.weight"}) -> (tensor<8x28x28x512xbf16> {ttir.name = "output"}) {
    %0 = "ttir.relu"(%arg0) : (tensor<8x28x28x512xbf16>) -> tensor<8x28x28x512xbf16> loc(#loc1)
    %1 = "ttir.conv2d"(%0, %arg7) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x28x28x512xbf16>, tensor<128x512x1x1xbf16>) -> tensor<8x28x28x128xbf16> loc(#loc2)
    %2 = "ttir.multiply"(%1, %arg1) : (tensor<8x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x28x28x128xbf16> loc(#loc3)
    %3 = "ttir.add"(%2, %arg2) : (tensor<8x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x28x28x128xbf16> loc(#loc4)
    %4 = "ttir.relu"(%3) : (tensor<8x28x28x128xbf16>) -> tensor<8x28x28x128xbf16> loc(#loc5)
    %5 = "ttir.conv2d"(%4, %arg8) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x28x28x128xbf16>, tensor<128x128x3x3xbf16>) -> tensor<8x28x28x128xbf16> loc(#loc6)
    %6 = "ttir.multiply"(%5, %arg3) : (tensor<8x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x28x28x128xbf16> loc(#loc7)
    %7 = "ttir.add"(%6, %arg4) : (tensor<8x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x28x28x128xbf16> loc(#loc8)
    %8 = "ttir.relu"(%7) : (tensor<8x28x28x128xbf16>) -> tensor<8x28x28x128xbf16> loc(#loc9)
    %9 = "ttir.conv2d"(%8, %arg9) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x28x28x128xbf16>, tensor<512x128x1x1xbf16>) -> tensor<8x28x28x512xbf16> loc(#loc10)
    %10 = "ttir.multiply"(%9, %arg5) : (tensor<8x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x28x28x512xbf16> loc(#loc11)
    %11 = "ttir.add"(%10, %arg6) : (tensor<8x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x28x28x512xbf16> loc(#loc12)
    %12 = "ttir.add"(%11, %0) : (tensor<8x28x28x512xbf16>, tensor<8x28x28x512xbf16>) -> tensor<8x28x28x512xbf16> loc(#loc13)
    return %12 : tensor<8x28x28x512xbf16> loc(#loc14)
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
#loc14 = loc("residual_add")
