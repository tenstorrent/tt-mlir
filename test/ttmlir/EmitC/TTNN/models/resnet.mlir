// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" --tt-unwrap-device-module %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-modify-signatures-for-dylib --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp
//
// UNSUPPORTED: true
// Not supported as several ops don't have proper emitc conversions yet

// https://huggingface.co/microsoft/resnet-50
module @ResNetForImageClassification attributes {} {
  func.func @forward(%arg0: tensor<1x3x224x224xf32> {ttir.name = "pixel_values"}, %arg1: tensor<1x1x1x64xf32> {ttir.name = "input_1_add_2"}, %arg2: tensor<1x1x1x64xf32> {ttir.name = "input_1_add_2_fork_clone1229"}, %arg3: tensor<1x1x1x64xf32> {ttir.name = "input_1_add_19"}, %arg4: tensor<1x1x1x64xf32> {ttir.name = "input_1_add_19_fork_clone1271"}, %arg5: tensor<1x1x1x64xf32> {ttir.name = "input_1_add_35"}, %arg6: tensor<1x1x1x64xf32> {ttir.name = "input_1_add_35_fork_clone1204"}, %arg7: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_51"}, %arg8: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_51_fork_clone1108"}, %arg9: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_66"}, %arg10: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_66_fork_clone1112"}, %arg11: tensor<1x1x1x64xf32> {ttir.name = "input_1_add_83"}, %arg12: tensor<1x1x1x64xf32> {ttir.name = "input_1_add_83_fork_clone1238"}, %arg13: tensor<1x1x1x64xf32> {ttir.name = "input_1_add_99"}, %arg14: tensor<1x1x1x64xf32> {ttir.name = "input_1_add_99_fork_clone1152"}, %arg15: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_115"}, %arg16: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_115_fork_clone1051"}, %arg17: tensor<1x1x1x64xf32> {ttir.name = "input_1_add_132"}, %arg18: tensor<1x1x1x64xf32> {ttir.name = "input_1_add_132_fork_clone1192"}, %arg19: tensor<1x1x1x64xf32> {ttir.name = "input_1_add_148"}, %arg20: tensor<1x1x1x64xf32> {ttir.name = "input_1_add_148_fork_clone1096"}, %arg21: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_164"}, %arg22: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_164_fork_clone992"}, %arg23: tensor<1x1x1x128xf32> {ttir.name = "input_1_add_181"}, %arg24: tensor<1x1x1x128xf32> {ttir.name = "input_1_add_181_fork_clone1065"}, %arg25: tensor<1x1x1x128xf32> {ttir.name = "input_1_add_197"}, %arg26: tensor<1x1x1x128xf32> {ttir.name = "input_1_add_197_fork_clone962"}, %arg27: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_213"}, %arg28: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_213_fork_clone853"}, %arg29: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_228"}, %arg30: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_228_fork_clone857"}, %arg31: tensor<1x1x1x128xf32> {ttir.name = "input_1_add_245"}, %arg32: tensor<1x1x1x128xf32> {ttir.name = "input_1_add_245_fork_clone1007"}, %arg33: tensor<1x1x1x128xf32> {ttir.name = "input_1_add_261"}, %arg34: tensor<1x1x1x128xf32> {ttir.name = "input_1_add_261_fork_clone901"}, %arg35: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_277"}, %arg36: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_277_fork_clone791"}, %arg37: tensor<1x1x1x128xf32> {ttir.name = "input_1_add_294"}, %arg38: tensor<1x1x1x128xf32> {ttir.name = "input_1_add_294_fork_clone950"}, %arg39: tensor<1x1x1x128xf32> {ttir.name = "input_1_add_310"}, %arg40: tensor<1x1x1x128xf32> {ttir.name = "input_1_add_310_fork_clone841"}, %arg41: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_326"}, %arg42: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_326_fork_clone735"}, %arg43: tensor<1x1x1x128xf32> {ttir.name = "input_1_add_343"}, %arg44: tensor<1x1x1x128xf32> {ttir.name = "input_1_add_343_fork_clone889"}, %arg45: tensor<1x1x1x128xf32> {ttir.name = "input_1_add_359"}, %arg46: tensor<1x1x1x128xf32> {ttir.name = "input_1_add_359_fork_clone779"}, %arg47: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_375"}, %arg48: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_375_fork_clone677"}, %arg49: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_392"}, %arg50: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_392_fork_clone748"}, %arg51: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_408"}, %arg52: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_408_fork_clone645"}, %arg53: tensor<1x1x1x1024xf32> {ttir.name = "input_1_add_424"}, %arg54: tensor<1x1x1x1024xf32> {ttir.name = "input_1_add_424_fork_clone524"}, %arg55: tensor<1x1x1x1024xf32> {ttir.name = "input_1_add_439"}, %arg56: tensor<1x1x1x1024xf32> {ttir.name = "input_1_add_439_fork_clone528"}, %arg57: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_456"}, %arg58: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_456_fork_clone692"}, %arg59: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_472"}, %arg60: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_472_fork_clone580"}, %arg61: tensor<1x1x1x1024xf32> {ttir.name = "input_1_add_488"}, %arg62: tensor<1x1x1x1024xf32> {ttir.name = "input_1_add_488_fork_clone453"}, %arg63: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_505"}, %arg64: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_505_fork_clone633"}, %arg65: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_521"}, %arg66: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_521_fork_clone512"}, %arg67: tensor<1x1x1x1024xf32> {ttir.name = "input_1_add_537"}, %arg68: tensor<1x1x1x1024xf32> {ttir.name = "input_1_add_537_fork_clone389"}, %arg69: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_554"}, %arg70: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_554_fork_clone568"}, %arg71: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_570"}, %arg72: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_570_fork_clone441"}, %arg73: tensor<1x1x1x1024xf32> {ttir.name = "input_1_add_586"}, %arg74: tensor<1x1x1x1024xf32> {ttir.name = "input_1_add_586_fork_clone329"}, %arg75: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_603"}, %arg76: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_603_fork_clone500"}, %arg77: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_619"}, %arg78: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_619_fork_clone377"}, %arg79: tensor<1x1x1x1024xf32> {ttir.name = "input_1_add_635"}, %arg80: tensor<1x1x1x1024xf32> {ttir.name = "input_1_add_635_fork_clone274"}, %arg81: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_652"}, %arg82: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_652_fork_clone429"}, %arg83: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_668"}, %arg84: tensor<1x1x1x256xf32> {ttir.name = "input_1_add_668_fork_clone317"}, %arg85: tensor<1x1x1x1024xf32> {ttir.name = "input_1_add_684"}, %arg86: tensor<1x1x1x1024xf32> {ttir.name = "input_1_add_684_fork_clone219"}, %arg87: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_701"}, %arg88: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_701_fork_clone287"}, %arg89: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_717"}, %arg90: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_717_fork_clone190"}, %arg91: tensor<1x2048x1x1xf32> {ttir.name = "input_1_add_733"}, %arg92: tensor<1x2048x1x1xf32> {ttir.name = "input_1_add_733_fork_clone101"}, %arg93: tensor<1x2048x1x1xf32> {ttir.name = "input_1_add_748"}, %arg94: tensor<1x2048x1x1xf32> {ttir.name = "input_1_add_748_fork_clone105"}, %arg95: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_765"}, %arg96: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_765_fork_clone233"}, %arg97: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_781"}, %arg98: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_781_fork_clone138"}, %arg99: tensor<1x2048x1x1xf32> {ttir.name = "input_1_add_797"}, %arg100: tensor<1x2048x1x1xf32> {ttir.name = "input_1_add_797_fork_clone61"}, %arg101: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_814"}, %arg102: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_814_fork_clone178"}, %arg103: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_830"}, %arg104: tensor<1x1x1x512xf32> {ttir.name = "input_1_add_830_fork_clone89"}, %arg105: tensor<1x2048x1x1xf32> {ttir.name = "input_1_add_846"}, %arg106: tensor<1x2048x1x1xf32> {ttir.name = "input_1_add_846_fork_clone32"}, %arg107: tensor<64x3x7x7xf32> {ttir.name = "resnet.embedder.embedder.convolution.weight"}, %arg108: tensor<64x64x1x1xf32> {ttir.name = "resnet.encoder.stages.0.layers.0.layer.0.convolution.weight"}, %arg109: tensor<64x64x3x3xf32> {ttir.name = "resnet.encoder.stages.0.layers.0.layer.1.convolution.weight"}, %arg110: tensor<256x64x1x1xf32> {ttir.name = "resnet.encoder.stages.0.layers.0.layer.2.convolution.weight"}, %arg111: tensor<256x64x1x1xf32> {ttir.name = "resnet.encoder.stages.0.layers.0.shortcut.convolution.weight"}, %arg112: tensor<64x256x1x1xf32> {ttir.name = "resnet.encoder.stages.0.layers.1.layer.0.convolution.weight"}, %arg113: tensor<64x64x3x3xf32> {ttir.name = "resnet.encoder.stages.0.layers.1.layer.1.convolution.weight"}, %arg114: tensor<256x64x1x1xf32> {ttir.name = "resnet.encoder.stages.0.layers.1.layer.2.convolution.weight"}, %arg115: tensor<64x256x1x1xf32> {ttir.name = "resnet.encoder.stages.0.layers.2.layer.0.convolution.weight"}, %arg116: tensor<64x64x3x3xf32> {ttir.name = "resnet.encoder.stages.0.layers.2.layer.1.convolution.weight"}, %arg117: tensor<256x64x1x1xf32> {ttir.name = "resnet.encoder.stages.0.layers.2.layer.2.convolution.weight"}, %arg118: tensor<128x256x1x1xf32> {ttir.name = "resnet.encoder.stages.1.layers.0.layer.0.convolution.weight"}, %arg119: tensor<128x128x3x3xf32> {ttir.name = "resnet.encoder.stages.1.layers.0.layer.1.convolution.weight"}, %arg120: tensor<512x128x1x1xf32> {ttir.name = "resnet.encoder.stages.1.layers.0.layer.2.convolution.weight"}, %arg121: tensor<512x256x1x1xf32> {ttir.name = "resnet.encoder.stages.1.layers.0.shortcut.convolution.weight"}, %arg122: tensor<128x512x1x1xf32> {ttir.name = "resnet.encoder.stages.1.layers.1.layer.0.convolution.weight"}, %arg123: tensor<128x128x3x3xf32> {ttir.name = "resnet.encoder.stages.1.layers.1.layer.1.convolution.weight"}, %arg124: tensor<512x128x1x1xf32> {ttir.name = "resnet.encoder.stages.1.layers.1.layer.2.convolution.weight"}, %arg125: tensor<128x512x1x1xf32> {ttir.name = "resnet.encoder.stages.1.layers.2.layer.0.convolution.weight"}, %arg126: tensor<128x128x3x3xf32> {ttir.name = "resnet.encoder.stages.1.layers.2.layer.1.convolution.weight"}, %arg127: tensor<512x128x1x1xf32> {ttir.name = "resnet.encoder.stages.1.layers.2.layer.2.convolution.weight"}, %arg128: tensor<128x512x1x1xf32> {ttir.name = "resnet.encoder.stages.1.layers.3.layer.0.convolution.weight"}, %arg129: tensor<128x128x3x3xf32> {ttir.name = "resnet.encoder.stages.1.layers.3.layer.1.convolution.weight"}, %arg130: tensor<512x128x1x1xf32> {ttir.name = "resnet.encoder.stages.1.layers.3.layer.2.convolution.weight"}, %arg131: tensor<256x512x1x1xf32> {ttir.name = "resnet.encoder.stages.2.layers.0.layer.0.convolution.weight"}, %arg132: tensor<256x256x3x3xf32> {ttir.name = "resnet.encoder.stages.2.layers.0.layer.1.convolution.weight"}, %arg133: tensor<1024x256x1x1xf32> {ttir.name = "resnet.encoder.stages.2.layers.0.layer.2.convolution.weight"}, %arg134: tensor<1024x512x1x1xf32> {ttir.name = "resnet.encoder.stages.2.layers.0.shortcut.convolution.weight"}, %arg135: tensor<256x1024x1x1xf32> {ttir.name = "resnet.encoder.stages.2.layers.1.layer.0.convolution.weight"}, %arg136: tensor<256x256x3x3xf32> {ttir.name = "resnet.encoder.stages.2.layers.1.layer.1.convolution.weight"}, %arg137: tensor<1024x256x1x1xf32> {ttir.name = "resnet.encoder.stages.2.layers.1.layer.2.convolution.weight"}, %arg138: tensor<256x1024x1x1xf32> {ttir.name = "resnet.encoder.stages.2.layers.2.layer.0.convolution.weight"}, %arg139: tensor<256x256x3x3xf32> {ttir.name = "resnet.encoder.stages.2.layers.2.layer.1.convolution.weight"}, %arg140: tensor<1024x256x1x1xf32> {ttir.name = "resnet.encoder.stages.2.layers.2.layer.2.convolution.weight"}, %arg141: tensor<256x1024x1x1xf32> {ttir.name = "resnet.encoder.stages.2.layers.3.layer.0.convolution.weight"}, %arg142: tensor<256x256x3x3xf32> {ttir.name = "resnet.encoder.stages.2.layers.3.layer.1.convolution.weight"}, %arg143: tensor<1024x256x1x1xf32> {ttir.name = "resnet.encoder.stages.2.layers.3.layer.2.convolution.weight"}, %arg144: tensor<256x1024x1x1xf32> {ttir.name = "resnet.encoder.stages.2.layers.4.layer.0.convolution.weight"}, %arg145: tensor<256x256x3x3xf32> {ttir.name = "resnet.encoder.stages.2.layers.4.layer.1.convolution.weight"}, %arg146: tensor<1024x256x1x1xf32> {ttir.name = "resnet.encoder.stages.2.layers.4.layer.2.convolution.weight"}, %arg147: tensor<256x1024x1x1xf32> {ttir.name = "resnet.encoder.stages.2.layers.5.layer.0.convolution.weight"}, %arg148: tensor<256x256x3x3xf32> {ttir.name = "resnet.encoder.stages.2.layers.5.layer.1.convolution.weight"}, %arg149: tensor<1024x256x1x1xf32> {ttir.name = "resnet.encoder.stages.2.layers.5.layer.2.convolution.weight"}, %arg150: tensor<512x1024x1x1xf32> {ttir.name = "resnet.encoder.stages.3.layers.0.layer.0.convolution.weight"}, %arg151: tensor<512x512x3x3xf32> {ttir.name = "resnet.encoder.stages.3.layers.0.layer.1.convolution.weight"}, %arg152: tensor<2048x512x1x1xf32> {ttir.name = "resnet.encoder.stages.3.layers.0.layer.2.convolution.weight"}, %arg153: tensor<2048x1024x1x1xf32> {ttir.name = "resnet.encoder.stages.3.layers.0.shortcut.convolution.weight"}, %arg154: tensor<512x2048x1x1xf32> {ttir.name = "resnet.encoder.stages.3.layers.1.layer.0.convolution.weight"}, %arg155: tensor<512x512x3x3xf32> {ttir.name = "resnet.encoder.stages.3.layers.1.layer.1.convolution.weight"}, %arg156: tensor<2048x512x1x1xf32> {ttir.name = "resnet.encoder.stages.3.layers.1.layer.2.convolution.weight"}, %arg157: tensor<512x2048x1x1xf32> {ttir.name = "resnet.encoder.stages.3.layers.2.layer.0.convolution.weight"}, %arg158: tensor<512x512x3x3xf32> {ttir.name = "resnet.encoder.stages.3.layers.2.layer.1.convolution.weight"}, %arg159: tensor<2048x512x1x1xf32> {ttir.name = "resnet.encoder.stages.3.layers.2.layer.2.convolution.weight"}, %arg160: tensor<2048x1000xf32> {ttir.name = "classifier.1.weight"}, %arg161: tensor<1000xf32> {ttir.name = "classifier.1.bias"}) -> (tensor<1x1000xf32> {ttir.name = "ResNetForImageClassification.output_add_868"}) {
    %0 = tensor.empty() : tensor<1x224x3x224xf32>
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x3x224x224xf32>, tensor<1x224x3x224xf32>) -> tensor<1x224x3x224xf32>
    %2 = tensor.empty() : tensor<1x224x224x3xf32>
    %3 = "ttir.transpose"(%1, %2) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x224x3x224xf32>, tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
    %4 = tensor.empty() : tensor<1x112x112x64xf32>
    %5 = "ttir.conv2d"(%3, %arg107, %4) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 3 : si32, padding_left = 3 : si32, padding_right = 3 : si32, padding_top = 3 : si32, stride_height = 2 : si32, stride_width = 2 : si32}> {channel_last = 1 : si32} : (tensor<1x224x224x3xf32>, tensor<64x3x7x7xf32>, tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
    %6 = tensor.empty() : tensor<1x112x112x64xf32>
    %7 = "ttir.multiply"(%5, %arg1, %6) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x112x112x64xf32>, tensor<1x1x1x64xf32>, tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
    %8 = tensor.empty() : tensor<1x112x112x64xf32>
    %9 = "ttir.add"(%7, %arg2, %8) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x112x112x64xf32>, tensor<1x1x1x64xf32>, tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
    %10 = tensor.empty() : tensor<1x112x112x64xf32>
    %11 = "ttir.relu"(%9, %10) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
    %12 = tensor.empty() : tensor<1x56x56x64xf32>
    %13 = "ttir.max_pool2d"(%11, %12) <{ceil_mode = false, dilation_height = 1 : si32, dilation_width = 1 : si32, kernel_height = 3 : si32, kernel_width = 3 : si32, padding_bottom = 1 : si32, padding_left = 1 : si32, padding_right = 1 : si32, padding_top = 1 : si32, stride_height = 2 : si32, stride_width = 2 : si32}> {channel_last = true} : (tensor<1x112x112x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %14 = tensor.empty() : tensor<1x56x56x64xf32>
    %15 = "ttir.conv2d"(%13, %arg108, %14) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x56x56x64xf32>, tensor<64x64x1x1xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %16 = tensor.empty() : tensor<1x56x56x64xf32>
    %17 = "ttir.multiply"(%15, %arg3, %16) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x64xf32>, tensor<1x1x1x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %18 = tensor.empty() : tensor<1x56x56x64xf32>
    %19 = "ttir.add"(%17, %arg4, %18) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x64xf32>, tensor<1x1x1x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %20 = tensor.empty() : tensor<1x56x56x64xf32>
    %21 = "ttir.relu"(%19, %20) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %22 = tensor.empty() : tensor<1x56x56x64xf32>
    %23 = "ttir.conv2d"(%21, %arg109, %22) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 1 : si32, padding_left = 1 : si32, padding_right = 1 : si32, padding_top = 1 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x56x56x64xf32>, tensor<64x64x3x3xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %24 = tensor.empty() : tensor<1x56x56x64xf32>
    %25 = "ttir.multiply"(%23, %arg5, %24) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x64xf32>, tensor<1x1x1x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %26 = tensor.empty() : tensor<1x56x56x64xf32>
    %27 = "ttir.add"(%25, %arg6, %26) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x64xf32>, tensor<1x1x1x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %28 = tensor.empty() : tensor<1x56x56x64xf32>
    %29 = "ttir.relu"(%27, %28) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %30 = tensor.empty() : tensor<1x56x56x256xf32>
    %31 = "ttir.conv2d"(%29, %arg110, %30) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x56x56x64xf32>, tensor<256x64x1x1xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %32 = tensor.empty() : tensor<1x56x56x256xf32>
    %33 = "ttir.multiply"(%31, %arg7, %32) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x256xf32>, tensor<1x1x1x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %34 = tensor.empty() : tensor<1x56x56x256xf32>
    %35 = "ttir.add"(%33, %arg8, %34) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x256xf32>, tensor<1x1x1x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %36 = tensor.empty() : tensor<1x56x56x256xf32>
    %37 = "ttir.conv2d"(%13, %arg111, %36) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x56x56x64xf32>, tensor<256x64x1x1xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %38 = tensor.empty() : tensor<1x56x56x256xf32>
    %39 = "ttir.multiply"(%37, %arg9, %38) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x256xf32>, tensor<1x1x1x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %40 = tensor.empty() : tensor<1x56x56x256xf32>
    %41 = "ttir.add"(%39, %arg10, %40) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x256xf32>, tensor<1x1x1x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %42 = tensor.empty() : tensor<1x56x56x256xf32>
    %43 = "ttir.add"(%35, %41, %42) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %44 = tensor.empty() : tensor<1x56x56x256xf32>
    %45 = "ttir.relu"(%43, %44) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %46 = tensor.empty() : tensor<1x56x56x64xf32>
    %47 = "ttir.conv2d"(%45, %arg112, %46) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x56x56x256xf32>, tensor<64x256x1x1xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %48 = tensor.empty() : tensor<1x56x56x64xf32>
    %49 = "ttir.multiply"(%47, %arg11, %48) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x64xf32>, tensor<1x1x1x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %50 = tensor.empty() : tensor<1x56x56x64xf32>
    %51 = "ttir.add"(%49, %arg12, %50) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x64xf32>, tensor<1x1x1x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %52 = tensor.empty() : tensor<1x56x56x64xf32>
    %53 = "ttir.relu"(%51, %52) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %54 = tensor.empty() : tensor<1x56x56x64xf32>
    %55 = "ttir.conv2d"(%53, %arg113, %54) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 1 : si32, padding_left = 1 : si32, padding_right = 1 : si32, padding_top = 1 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x56x56x64xf32>, tensor<64x64x3x3xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %56 = tensor.empty() : tensor<1x56x56x64xf32>
    %57 = "ttir.multiply"(%55, %arg13, %56) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x64xf32>, tensor<1x1x1x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %58 = tensor.empty() : tensor<1x56x56x64xf32>
    %59 = "ttir.add"(%57, %arg14, %58) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x64xf32>, tensor<1x1x1x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %60 = tensor.empty() : tensor<1x56x56x64xf32>
    %61 = "ttir.relu"(%59, %60) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %62 = tensor.empty() : tensor<1x56x56x256xf32>
    %63 = "ttir.conv2d"(%61, %arg114, %62) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x56x56x64xf32>, tensor<256x64x1x1xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %64 = tensor.empty() : tensor<1x56x56x256xf32>
    %65 = "ttir.multiply"(%63, %arg15, %64) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x256xf32>, tensor<1x1x1x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %66 = tensor.empty() : tensor<1x56x56x256xf32>
    %67 = "ttir.add"(%65, %arg16, %66) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x256xf32>, tensor<1x1x1x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %68 = tensor.empty() : tensor<1x56x56x256xf32>
    %69 = "ttir.add"(%67, %45, %68) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %70 = tensor.empty() : tensor<1x56x56x256xf32>
    %71 = "ttir.relu"(%69, %70) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %72 = tensor.empty() : tensor<1x56x56x64xf32>
    %73 = "ttir.conv2d"(%71, %arg115, %72) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x56x56x256xf32>, tensor<64x256x1x1xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %74 = tensor.empty() : tensor<1x56x56x64xf32>
    %75 = "ttir.multiply"(%73, %arg17, %74) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x64xf32>, tensor<1x1x1x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %76 = tensor.empty() : tensor<1x56x56x64xf32>
    %77 = "ttir.add"(%75, %arg18, %76) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x64xf32>, tensor<1x1x1x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %78 = tensor.empty() : tensor<1x56x56x64xf32>
    %79 = "ttir.relu"(%77, %78) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %80 = tensor.empty() : tensor<1x56x56x64xf32>
    %81 = "ttir.conv2d"(%79, %arg116, %80) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 1 : si32, padding_left = 1 : si32, padding_right = 1 : si32, padding_top = 1 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x56x56x64xf32>, tensor<64x64x3x3xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %82 = tensor.empty() : tensor<1x56x56x64xf32>
    %83 = "ttir.multiply"(%81, %arg19, %82) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x64xf32>, tensor<1x1x1x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %84 = tensor.empty() : tensor<1x56x56x64xf32>
    %85 = "ttir.add"(%83, %arg20, %84) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x64xf32>, tensor<1x1x1x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %86 = tensor.empty() : tensor<1x56x56x64xf32>
    %87 = "ttir.relu"(%85, %86) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %88 = tensor.empty() : tensor<1x56x56x256xf32>
    %89 = "ttir.conv2d"(%87, %arg117, %88) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x56x56x64xf32>, tensor<256x64x1x1xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %90 = tensor.empty() : tensor<1x56x56x256xf32>
    %91 = "ttir.multiply"(%89, %arg21, %90) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x256xf32>, tensor<1x1x1x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %92 = tensor.empty() : tensor<1x56x56x256xf32>
    %93 = "ttir.add"(%91, %arg22, %92) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x256xf32>, tensor<1x1x1x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %94 = tensor.empty() : tensor<1x56x56x256xf32>
    %95 = "ttir.add"(%93, %71, %94) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %96 = tensor.empty() : tensor<1x56x56x256xf32>
    %97 = "ttir.relu"(%95, %96) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %98 = tensor.empty() : tensor<1x56x56x128xf32>
    %99 = "ttir.conv2d"(%97, %arg118, %98) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x56x56x256xf32>, tensor<128x256x1x1xf32>, tensor<1x56x56x128xf32>) -> tensor<1x56x56x128xf32>
    %100 = tensor.empty() : tensor<1x56x56x128xf32>
    %101 = "ttir.multiply"(%99, %arg23, %100) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x128xf32>, tensor<1x1x1x128xf32>, tensor<1x56x56x128xf32>) -> tensor<1x56x56x128xf32>
    %102 = tensor.empty() : tensor<1x56x56x128xf32>
    %103 = "ttir.add"(%101, %arg24, %102) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x56x56x128xf32>, tensor<1x1x1x128xf32>, tensor<1x56x56x128xf32>) -> tensor<1x56x56x128xf32>
    %104 = tensor.empty() : tensor<1x56x56x128xf32>
    %105 = "ttir.relu"(%103, %104) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x56x56x128xf32>, tensor<1x56x56x128xf32>) -> tensor<1x56x56x128xf32>
    %106 = tensor.empty() : tensor<1x28x28x128xf32>
    %107 = "ttir.conv2d"(%105, %arg119, %106) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 1 : si32, padding_left = 1 : si32, padding_right = 1 : si32, padding_top = 1 : si32, stride_height = 2 : si32, stride_width = 2 : si32}> {channel_last = 1 : si32} : (tensor<1x56x56x128xf32>, tensor<128x128x3x3xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %108 = tensor.empty() : tensor<1x28x28x128xf32>
    %109 = "ttir.multiply"(%107, %arg25, %108) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x1x1x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %110 = tensor.empty() : tensor<1x28x28x128xf32>
    %111 = "ttir.add"(%109, %arg26, %110) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x1x1x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %112 = tensor.empty() : tensor<1x28x28x128xf32>
    %113 = "ttir.relu"(%111, %112) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %114 = tensor.empty() : tensor<1x28x28x512xf32>
    %115 = "ttir.conv2d"(%113, %arg120, %114) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x28x28x128xf32>, tensor<512x128x1x1xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %116 = tensor.empty() : tensor<1x28x28x512xf32>
    %117 = "ttir.multiply"(%115, %arg27, %116) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x512xf32>, tensor<1x1x1x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %118 = tensor.empty() : tensor<1x28x28x512xf32>
    %119 = "ttir.add"(%117, %arg28, %118) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x512xf32>, tensor<1x1x1x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %120 = tensor.empty() : tensor<1x28x28x512xf32>
    %121 = "ttir.conv2d"(%97, %arg121, %120) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 2 : si32, stride_width = 2 : si32}> {channel_last = 1 : si32} : (tensor<1x56x56x256xf32>, tensor<512x256x1x1xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %122 = tensor.empty() : tensor<1x28x28x512xf32>
    %123 = "ttir.multiply"(%121, %arg29, %122) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x512xf32>, tensor<1x1x1x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %124 = tensor.empty() : tensor<1x28x28x512xf32>
    %125 = "ttir.add"(%123, %arg30, %124) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x512xf32>, tensor<1x1x1x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %126 = tensor.empty() : tensor<1x28x28x512xf32>
    %127 = "ttir.add"(%119, %125, %126) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %128 = tensor.empty() : tensor<1x28x28x512xf32>
    %129 = "ttir.relu"(%127, %128) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %130 = tensor.empty() : tensor<1x28x28x128xf32>
    %131 = "ttir.conv2d"(%129, %arg122, %130) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x28x28x512xf32>, tensor<128x512x1x1xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %132 = tensor.empty() : tensor<1x28x28x128xf32>
    %133 = "ttir.multiply"(%131, %arg31, %132) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x1x1x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %134 = tensor.empty() : tensor<1x28x28x128xf32>
    %135 = "ttir.add"(%133, %arg32, %134) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x1x1x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %136 = tensor.empty() : tensor<1x28x28x128xf32>
    %137 = "ttir.relu"(%135, %136) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %138 = tensor.empty() : tensor<1x28x28x128xf32>
    %139 = "ttir.conv2d"(%137, %arg123, %138) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 1 : si32, padding_left = 1 : si32, padding_right = 1 : si32, padding_top = 1 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x28x28x128xf32>, tensor<128x128x3x3xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %140 = tensor.empty() : tensor<1x28x28x128xf32>
    %141 = "ttir.multiply"(%139, %arg33, %140) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x1x1x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %142 = tensor.empty() : tensor<1x28x28x128xf32>
    %143 = "ttir.add"(%141, %arg34, %142) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x1x1x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %144 = tensor.empty() : tensor<1x28x28x128xf32>
    %145 = "ttir.relu"(%143, %144) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %146 = tensor.empty() : tensor<1x28x28x512xf32>
    %147 = "ttir.conv2d"(%145, %arg124, %146) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x28x28x128xf32>, tensor<512x128x1x1xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %148 = tensor.empty() : tensor<1x28x28x512xf32>
    %149 = "ttir.multiply"(%147, %arg35, %148) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x512xf32>, tensor<1x1x1x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %150 = tensor.empty() : tensor<1x28x28x512xf32>
    %151 = "ttir.add"(%149, %arg36, %150) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x512xf32>, tensor<1x1x1x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %152 = tensor.empty() : tensor<1x28x28x512xf32>
    %153 = "ttir.add"(%151, %129, %152) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %154 = tensor.empty() : tensor<1x28x28x512xf32>
    %155 = "ttir.relu"(%153, %154) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %156 = tensor.empty() : tensor<1x28x28x128xf32>
    %157 = "ttir.conv2d"(%155, %arg125, %156) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x28x28x512xf32>, tensor<128x512x1x1xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %158 = tensor.empty() : tensor<1x28x28x128xf32>
    %159 = "ttir.multiply"(%157, %arg37, %158) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x1x1x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %160 = tensor.empty() : tensor<1x28x28x128xf32>
    %161 = "ttir.add"(%159, %arg38, %160) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x1x1x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %162 = tensor.empty() : tensor<1x28x28x128xf32>
    %163 = "ttir.relu"(%161, %162) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %164 = tensor.empty() : tensor<1x28x28x128xf32>
    %165 = "ttir.conv2d"(%163, %arg126, %164) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 1 : si32, padding_left = 1 : si32, padding_right = 1 : si32, padding_top = 1 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x28x28x128xf32>, tensor<128x128x3x3xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %166 = tensor.empty() : tensor<1x28x28x128xf32>
    %167 = "ttir.multiply"(%165, %arg39, %166) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x1x1x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %168 = tensor.empty() : tensor<1x28x28x128xf32>
    %169 = "ttir.add"(%167, %arg40, %168) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x1x1x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %170 = tensor.empty() : tensor<1x28x28x128xf32>
    %171 = "ttir.relu"(%169, %170) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %172 = tensor.empty() : tensor<1x28x28x512xf32>
    %173 = "ttir.conv2d"(%171, %arg127, %172) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x28x28x128xf32>, tensor<512x128x1x1xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %174 = tensor.empty() : tensor<1x28x28x512xf32>
    %175 = "ttir.multiply"(%173, %arg41, %174) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x512xf32>, tensor<1x1x1x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %176 = tensor.empty() : tensor<1x28x28x512xf32>
    %177 = "ttir.add"(%175, %arg42, %176) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x512xf32>, tensor<1x1x1x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %178 = tensor.empty() : tensor<1x28x28x512xf32>
    %179 = "ttir.add"(%177, %155, %178) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %180 = tensor.empty() : tensor<1x28x28x512xf32>
    %181 = "ttir.relu"(%179, %180) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %182 = tensor.empty() : tensor<1x28x28x128xf32>
    %183 = "ttir.conv2d"(%181, %arg128, %182) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x28x28x512xf32>, tensor<128x512x1x1xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %184 = tensor.empty() : tensor<1x28x28x128xf32>
    %185 = "ttir.multiply"(%183, %arg43, %184) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x1x1x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %186 = tensor.empty() : tensor<1x28x28x128xf32>
    %187 = "ttir.add"(%185, %arg44, %186) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x1x1x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %188 = tensor.empty() : tensor<1x28x28x128xf32>
    %189 = "ttir.relu"(%187, %188) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %190 = tensor.empty() : tensor<1x28x28x128xf32>
    %191 = "ttir.conv2d"(%189, %arg129, %190) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 1 : si32, padding_left = 1 : si32, padding_right = 1 : si32, padding_top = 1 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x28x28x128xf32>, tensor<128x128x3x3xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %192 = tensor.empty() : tensor<1x28x28x128xf32>
    %193 = "ttir.multiply"(%191, %arg45, %192) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x1x1x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %194 = tensor.empty() : tensor<1x28x28x128xf32>
    %195 = "ttir.add"(%193, %arg46, %194) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x1x1x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %196 = tensor.empty() : tensor<1x28x28x128xf32>
    %197 = "ttir.relu"(%195, %196) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %198 = tensor.empty() : tensor<1x28x28x512xf32>
    %199 = "ttir.conv2d"(%197, %arg130, %198) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x28x28x128xf32>, tensor<512x128x1x1xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %200 = tensor.empty() : tensor<1x28x28x512xf32>
    %201 = "ttir.multiply"(%199, %arg47, %200) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x512xf32>, tensor<1x1x1x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %202 = tensor.empty() : tensor<1x28x28x512xf32>
    %203 = "ttir.add"(%201, %arg48, %202) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x512xf32>, tensor<1x1x1x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %204 = tensor.empty() : tensor<1x28x28x512xf32>
    %205 = "ttir.add"(%203, %181, %204) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %206 = tensor.empty() : tensor<1x28x28x512xf32>
    %207 = "ttir.relu"(%205, %206) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %208 = tensor.empty() : tensor<1x28x28x256xf32>
    %209 = "ttir.conv2d"(%207, %arg131, %208) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x28x28x512xf32>, tensor<256x512x1x1xf32>, tensor<1x28x28x256xf32>) -> tensor<1x28x28x256xf32>
    %210 = tensor.empty() : tensor<1x28x28x256xf32>
    %211 = "ttir.multiply"(%209, %arg49, %210) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x256xf32>, tensor<1x1x1x256xf32>, tensor<1x28x28x256xf32>) -> tensor<1x28x28x256xf32>
    %212 = tensor.empty() : tensor<1x28x28x256xf32>
    %213 = "ttir.add"(%211, %arg50, %212) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x28x28x256xf32>, tensor<1x1x1x256xf32>, tensor<1x28x28x256xf32>) -> tensor<1x28x28x256xf32>
    %214 = tensor.empty() : tensor<1x28x28x256xf32>
    %215 = "ttir.relu"(%213, %214) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x28x28x256xf32>, tensor<1x28x28x256xf32>) -> tensor<1x28x28x256xf32>
    %216 = tensor.empty() : tensor<1x14x14x256xf32>
    %217 = "ttir.conv2d"(%215, %arg132, %216) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 1 : si32, padding_left = 1 : si32, padding_right = 1 : si32, padding_top = 1 : si32, stride_height = 2 : si32, stride_width = 2 : si32}> {channel_last = 1 : si32} : (tensor<1x28x28x256xf32>, tensor<256x256x3x3xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %218 = tensor.empty() : tensor<1x14x14x256xf32>
    %219 = "ttir.multiply"(%217, %arg51, %218) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %220 = tensor.empty() : tensor<1x14x14x256xf32>
    %221 = "ttir.add"(%219, %arg52, %220) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %222 = tensor.empty() : tensor<1x14x14x256xf32>
    %223 = "ttir.relu"(%221, %222) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %224 = tensor.empty() : tensor<1x14x14x1024xf32>
    %225 = "ttir.conv2d"(%223, %arg133, %224) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x14x14x256xf32>, tensor<1024x256x1x1xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %226 = tensor.empty() : tensor<1x14x14x1024xf32>
    %227 = "ttir.multiply"(%225, %arg53, %226) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x1x1x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %228 = tensor.empty() : tensor<1x14x14x1024xf32>
    %229 = "ttir.add"(%227, %arg54, %228) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x1x1x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %230 = tensor.empty() : tensor<1x14x14x1024xf32>
    %231 = "ttir.conv2d"(%207, %arg134, %230) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 2 : si32, stride_width = 2 : si32}> {channel_last = 1 : si32} : (tensor<1x28x28x512xf32>, tensor<1024x512x1x1xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %232 = tensor.empty() : tensor<1x14x14x1024xf32>
    %233 = "ttir.multiply"(%231, %arg55, %232) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x1x1x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %234 = tensor.empty() : tensor<1x14x14x1024xf32>
    %235 = "ttir.add"(%233, %arg56, %234) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x1x1x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %236 = tensor.empty() : tensor<1x14x14x1024xf32>
    %237 = "ttir.add"(%229, %235, %236) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %238 = tensor.empty() : tensor<1x14x14x1024xf32>
    %239 = "ttir.relu"(%237, %238) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %240 = tensor.empty() : tensor<1x14x14x256xf32>
    %241 = "ttir.conv2d"(%239, %arg135, %240) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xf32>, tensor<256x1024x1x1xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %242 = tensor.empty() : tensor<1x14x14x256xf32>
    %243 = "ttir.multiply"(%241, %arg57, %242) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %244 = tensor.empty() : tensor<1x14x14x256xf32>
    %245 = "ttir.add"(%243, %arg58, %244) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %246 = tensor.empty() : tensor<1x14x14x256xf32>
    %247 = "ttir.relu"(%245, %246) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %248 = tensor.empty() : tensor<1x14x14x256xf32>
    %249 = "ttir.conv2d"(%247, %arg136, %248) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 1 : si32, padding_left = 1 : si32, padding_right = 1 : si32, padding_top = 1 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x14x14x256xf32>, tensor<256x256x3x3xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %250 = tensor.empty() : tensor<1x14x14x256xf32>
    %251 = "ttir.multiply"(%249, %arg59, %250) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %252 = tensor.empty() : tensor<1x14x14x256xf32>
    %253 = "ttir.add"(%251, %arg60, %252) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %254 = tensor.empty() : tensor<1x14x14x256xf32>
    %255 = "ttir.relu"(%253, %254) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %256 = tensor.empty() : tensor<1x14x14x1024xf32>
    %257 = "ttir.conv2d"(%255, %arg137, %256) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x14x14x256xf32>, tensor<1024x256x1x1xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %258 = tensor.empty() : tensor<1x14x14x1024xf32>
    %259 = "ttir.multiply"(%257, %arg61, %258) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x1x1x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %260 = tensor.empty() : tensor<1x14x14x1024xf32>
    %261 = "ttir.add"(%259, %arg62, %260) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x1x1x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %262 = tensor.empty() : tensor<1x14x14x1024xf32>
    %263 = "ttir.add"(%261, %239, %262) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %264 = tensor.empty() : tensor<1x14x14x1024xf32>
    %265 = "ttir.relu"(%263, %264) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %266 = tensor.empty() : tensor<1x14x14x256xf32>
    %267 = "ttir.conv2d"(%265, %arg138, %266) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xf32>, tensor<256x1024x1x1xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %268 = tensor.empty() : tensor<1x14x14x256xf32>
    %269 = "ttir.multiply"(%267, %arg63, %268) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %270 = tensor.empty() : tensor<1x14x14x256xf32>
    %271 = "ttir.add"(%269, %arg64, %270) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %272 = tensor.empty() : tensor<1x14x14x256xf32>
    %273 = "ttir.relu"(%271, %272) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %274 = tensor.empty() : tensor<1x14x14x256xf32>
    %275 = "ttir.conv2d"(%273, %arg139, %274) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 1 : si32, padding_left = 1 : si32, padding_right = 1 : si32, padding_top = 1 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x14x14x256xf32>, tensor<256x256x3x3xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %276 = tensor.empty() : tensor<1x14x14x256xf32>
    %277 = "ttir.multiply"(%275, %arg65, %276) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %278 = tensor.empty() : tensor<1x14x14x256xf32>
    %279 = "ttir.add"(%277, %arg66, %278) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %280 = tensor.empty() : tensor<1x14x14x256xf32>
    %281 = "ttir.relu"(%279, %280) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %282 = tensor.empty() : tensor<1x14x14x1024xf32>
    %283 = "ttir.conv2d"(%281, %arg140, %282) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x14x14x256xf32>, tensor<1024x256x1x1xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %284 = tensor.empty() : tensor<1x14x14x1024xf32>
    %285 = "ttir.multiply"(%283, %arg67, %284) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x1x1x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %286 = tensor.empty() : tensor<1x14x14x1024xf32>
    %287 = "ttir.add"(%285, %arg68, %286) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x1x1x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %288 = tensor.empty() : tensor<1x14x14x1024xf32>
    %289 = "ttir.add"(%287, %265, %288) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %290 = tensor.empty() : tensor<1x14x14x1024xf32>
    %291 = "ttir.relu"(%289, %290) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %292 = tensor.empty() : tensor<1x14x14x256xf32>
    %293 = "ttir.conv2d"(%291, %arg141, %292) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xf32>, tensor<256x1024x1x1xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %294 = tensor.empty() : tensor<1x14x14x256xf32>
    %295 = "ttir.multiply"(%293, %arg69, %294) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %296 = tensor.empty() : tensor<1x14x14x256xf32>
    %297 = "ttir.add"(%295, %arg70, %296) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %298 = tensor.empty() : tensor<1x14x14x256xf32>
    %299 = "ttir.relu"(%297, %298) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %300 = tensor.empty() : tensor<1x14x14x256xf32>
    %301 = "ttir.conv2d"(%299, %arg142, %300) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 1 : si32, padding_left = 1 : si32, padding_right = 1 : si32, padding_top = 1 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x14x14x256xf32>, tensor<256x256x3x3xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %302 = tensor.empty() : tensor<1x14x14x256xf32>
    %303 = "ttir.multiply"(%301, %arg71, %302) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %304 = tensor.empty() : tensor<1x14x14x256xf32>
    %305 = "ttir.add"(%303, %arg72, %304) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %306 = tensor.empty() : tensor<1x14x14x256xf32>
    %307 = "ttir.relu"(%305, %306) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %308 = tensor.empty() : tensor<1x14x14x1024xf32>
    %309 = "ttir.conv2d"(%307, %arg143, %308) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x14x14x256xf32>, tensor<1024x256x1x1xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %310 = tensor.empty() : tensor<1x14x14x1024xf32>
    %311 = "ttir.multiply"(%309, %arg73, %310) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x1x1x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %312 = tensor.empty() : tensor<1x14x14x1024xf32>
    %313 = "ttir.add"(%311, %arg74, %312) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x1x1x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %314 = tensor.empty() : tensor<1x14x14x1024xf32>
    %315 = "ttir.add"(%313, %291, %314) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %316 = tensor.empty() : tensor<1x14x14x1024xf32>
    %317 = "ttir.relu"(%315, %316) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %318 = tensor.empty() : tensor<1x14x14x256xf32>
    %319 = "ttir.conv2d"(%317, %arg144, %318) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xf32>, tensor<256x1024x1x1xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %320 = tensor.empty() : tensor<1x14x14x256xf32>
    %321 = "ttir.multiply"(%319, %arg75, %320) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %322 = tensor.empty() : tensor<1x14x14x256xf32>
    %323 = "ttir.add"(%321, %arg76, %322) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %324 = tensor.empty() : tensor<1x14x14x256xf32>
    %325 = "ttir.relu"(%323, %324) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %326 = tensor.empty() : tensor<1x14x14x256xf32>
    %327 = "ttir.conv2d"(%325, %arg145, %326) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 1 : si32, padding_left = 1 : si32, padding_right = 1 : si32, padding_top = 1 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x14x14x256xf32>, tensor<256x256x3x3xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %328 = tensor.empty() : tensor<1x14x14x256xf32>
    %329 = "ttir.multiply"(%327, %arg77, %328) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %330 = tensor.empty() : tensor<1x14x14x256xf32>
    %331 = "ttir.add"(%329, %arg78, %330) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %332 = tensor.empty() : tensor<1x14x14x256xf32>
    %333 = "ttir.relu"(%331, %332) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %334 = tensor.empty() : tensor<1x14x14x1024xf32>
    %335 = "ttir.conv2d"(%333, %arg146, %334) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x14x14x256xf32>, tensor<1024x256x1x1xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %336 = tensor.empty() : tensor<1x14x14x1024xf32>
    %337 = "ttir.multiply"(%335, %arg79, %336) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x1x1x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %338 = tensor.empty() : tensor<1x14x14x1024xf32>
    %339 = "ttir.add"(%337, %arg80, %338) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x1x1x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %340 = tensor.empty() : tensor<1x14x14x1024xf32>
    %341 = "ttir.add"(%339, %317, %340) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %342 = tensor.empty() : tensor<1x14x14x1024xf32>
    %343 = "ttir.relu"(%341, %342) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %344 = tensor.empty() : tensor<1x14x14x256xf32>
    %345 = "ttir.conv2d"(%343, %arg147, %344) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xf32>, tensor<256x1024x1x1xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %346 = tensor.empty() : tensor<1x14x14x256xf32>
    %347 = "ttir.multiply"(%345, %arg81, %346) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %348 = tensor.empty() : tensor<1x14x14x256xf32>
    %349 = "ttir.add"(%347, %arg82, %348) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %350 = tensor.empty() : tensor<1x14x14x256xf32>
    %351 = "ttir.relu"(%349, %350) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %352 = tensor.empty() : tensor<1x14x14x256xf32>
    %353 = "ttir.conv2d"(%351, %arg148, %352) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 1 : si32, padding_left = 1 : si32, padding_right = 1 : si32, padding_top = 1 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x14x14x256xf32>, tensor<256x256x3x3xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %354 = tensor.empty() : tensor<1x14x14x256xf32>
    %355 = "ttir.multiply"(%353, %arg83, %354) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %356 = tensor.empty() : tensor<1x14x14x256xf32>
    %357 = "ttir.add"(%355, %arg84, %356) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x1x1x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %358 = tensor.empty() : tensor<1x14x14x256xf32>
    %359 = "ttir.relu"(%357, %358) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %360 = tensor.empty() : tensor<1x14x14x1024xf32>
    %361 = "ttir.conv2d"(%359, %arg149, %360) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x14x14x256xf32>, tensor<1024x256x1x1xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %362 = tensor.empty() : tensor<1x14x14x1024xf32>
    %363 = "ttir.multiply"(%361, %arg85, %362) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x1x1x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %364 = tensor.empty() : tensor<1x14x14x1024xf32>
    %365 = "ttir.add"(%363, %arg86, %364) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x1x1x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %366 = tensor.empty() : tensor<1x14x14x1024xf32>
    %367 = "ttir.add"(%365, %343, %366) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %368 = tensor.empty() : tensor<1x14x14x1024xf32>
    %369 = "ttir.relu"(%367, %368) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %370 = tensor.empty() : tensor<1x14x14x512xf32>
    %371 = "ttir.conv2d"(%369, %arg150, %370) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xf32>, tensor<512x1024x1x1xf32>, tensor<1x14x14x512xf32>) -> tensor<1x14x14x512xf32>
    %372 = tensor.empty() : tensor<1x14x14x512xf32>
    %373 = "ttir.multiply"(%371, %arg87, %372) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x512xf32>, tensor<1x1x1x512xf32>, tensor<1x14x14x512xf32>) -> tensor<1x14x14x512xf32>
    %374 = tensor.empty() : tensor<1x14x14x512xf32>
    %375 = "ttir.add"(%373, %arg88, %374) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x14x14x512xf32>, tensor<1x1x1x512xf32>, tensor<1x14x14x512xf32>) -> tensor<1x14x14x512xf32>
    %376 = tensor.empty() : tensor<1x14x14x512xf32>
    %377 = "ttir.relu"(%375, %376) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x14x14x512xf32>, tensor<1x14x14x512xf32>) -> tensor<1x14x14x512xf32>
    %378 = tensor.empty() : tensor<1x7x7x512xf32>
    %379 = "ttir.conv2d"(%377, %arg151, %378) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 1 : si32, padding_left = 1 : si32, padding_right = 1 : si32, padding_top = 1 : si32, stride_height = 2 : si32, stride_width = 2 : si32}> {channel_last = 1 : si32} : (tensor<1x14x14x512xf32>, tensor<512x512x3x3xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %380 = tensor.empty() : tensor<1x7x7x512xf32>
    %381 = "ttir.multiply"(%379, %arg89, %380) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x7x7x512xf32>, tensor<1x1x1x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %382 = tensor.empty() : tensor<1x7x7x512xf32>
    %383 = "ttir.add"(%381, %arg90, %382) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x7x7x512xf32>, tensor<1x1x1x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %384 = tensor.empty() : tensor<1x7x7x512xf32>
    %385 = "ttir.relu"(%383, %384) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %386 = tensor.empty() : tensor<1x7x7x2048xf32>
    %387 = "ttir.conv2d"(%385, %arg152, %386) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x7x7x512xf32>, tensor<2048x512x1x1xf32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %388 = tensor.empty() : tensor<1x7x2048x7xf32>
    %389 = "ttir.transpose"(%387, %388) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x7x7x2048xf32>, tensor<1x7x2048x7xf32>) -> tensor<1x7x2048x7xf32>
    %390 = tensor.empty() : tensor<1x2048x7x7xf32>
    %391 = "ttir.transpose"(%389, %390) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x7x2048x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %392 = tensor.empty() : tensor<1x2048x7x7xf32>
    %393 = "ttir.multiply"(%391, %arg91, %392) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xf32>, tensor<1x2048x1x1xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %394 = tensor.empty() : tensor<1x2048x7x7xf32>
    %395 = "ttir.add"(%393, %arg92, %394) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xf32>, tensor<1x2048x1x1xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %396 = tensor.empty() : tensor<1x7x7x2048xf32>
    %397 = "ttir.conv2d"(%369, %arg153, %396) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 2 : si32, stride_width = 2 : si32}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xf32>, tensor<2048x1024x1x1xf32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %398 = tensor.empty() : tensor<1x7x2048x7xf32>
    %399 = "ttir.transpose"(%397, %398) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x7x7x2048xf32>, tensor<1x7x2048x7xf32>) -> tensor<1x7x2048x7xf32>
    %400 = tensor.empty() : tensor<1x2048x7x7xf32>
    %401 = "ttir.transpose"(%399, %400) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x7x2048x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %402 = tensor.empty() : tensor<1x2048x7x7xf32>
    %403 = "ttir.multiply"(%401, %arg93, %402) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xf32>, tensor<1x2048x1x1xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %404 = tensor.empty() : tensor<1x2048x7x7xf32>
    %405 = "ttir.add"(%403, %arg94, %404) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xf32>, tensor<1x2048x1x1xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %406 = tensor.empty() : tensor<1x2048x7x7xf32>
    %407 = "ttir.add"(%395, %405, %406) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %408 = tensor.empty() : tensor<1x2048x7x7xf32>
    %409 = "ttir.relu"(%407, %408) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %410 = tensor.empty() : tensor<1x7x2048x7xf32>
    %411 = "ttir.transpose"(%409, %410) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x2048x7x7xf32>, tensor<1x7x2048x7xf32>) -> tensor<1x7x2048x7xf32>
    %412 = tensor.empty() : tensor<1x7x7x2048xf32>
    %413 = "ttir.transpose"(%411, %412) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x7x2048x7xf32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %414 = tensor.empty() : tensor<1x7x7x512xf32>
    %415 = "ttir.conv2d"(%413, %arg154, %414) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x7x7x2048xf32>, tensor<512x2048x1x1xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %416 = tensor.empty() : tensor<1x7x7x512xf32>
    %417 = "ttir.multiply"(%415, %arg95, %416) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x7x7x512xf32>, tensor<1x1x1x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %418 = tensor.empty() : tensor<1x7x7x512xf32>
    %419 = "ttir.add"(%417, %arg96, %418) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x7x7x512xf32>, tensor<1x1x1x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %420 = tensor.empty() : tensor<1x7x7x512xf32>
    %421 = "ttir.relu"(%419, %420) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %422 = tensor.empty() : tensor<1x7x7x512xf32>
    %423 = "ttir.conv2d"(%421, %arg155, %422) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 1 : si32, padding_left = 1 : si32, padding_right = 1 : si32, padding_top = 1 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x7x7x512xf32>, tensor<512x512x3x3xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %424 = tensor.empty() : tensor<1x7x7x512xf32>
    %425 = "ttir.multiply"(%423, %arg97, %424) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x7x7x512xf32>, tensor<1x1x1x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %426 = tensor.empty() : tensor<1x7x7x512xf32>
    %427 = "ttir.add"(%425, %arg98, %426) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x7x7x512xf32>, tensor<1x1x1x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %428 = tensor.empty() : tensor<1x7x7x512xf32>
    %429 = "ttir.relu"(%427, %428) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %430 = tensor.empty() : tensor<1x7x7x2048xf32>
    %431 = "ttir.conv2d"(%429, %arg156, %430) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x7x7x512xf32>, tensor<2048x512x1x1xf32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %432 = tensor.empty() : tensor<1x7x2048x7xf32>
    %433 = "ttir.transpose"(%431, %432) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x7x7x2048xf32>, tensor<1x7x2048x7xf32>) -> tensor<1x7x2048x7xf32>
    %434 = tensor.empty() : tensor<1x2048x7x7xf32>
    %435 = "ttir.transpose"(%433, %434) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x7x2048x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %436 = tensor.empty() : tensor<1x2048x7x7xf32>
    %437 = "ttir.multiply"(%435, %arg99, %436) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xf32>, tensor<1x2048x1x1xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %438 = tensor.empty() : tensor<1x2048x7x7xf32>
    %439 = "ttir.add"(%437, %arg100, %438) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xf32>, tensor<1x2048x1x1xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %440 = tensor.empty() : tensor<1x2048x7x7xf32>
    %441 = "ttir.add"(%439, %409, %440) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %442 = tensor.empty() : tensor<1x2048x7x7xf32>
    %443 = "ttir.relu"(%441, %442) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %444 = tensor.empty() : tensor<1x7x2048x7xf32>
    %445 = "ttir.transpose"(%443, %444) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x2048x7x7xf32>, tensor<1x7x2048x7xf32>) -> tensor<1x7x2048x7xf32>
    %446 = tensor.empty() : tensor<1x7x7x2048xf32>
    %447 = "ttir.transpose"(%445, %446) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x7x2048x7xf32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %448 = tensor.empty() : tensor<1x7x7x512xf32>
    %449 = "ttir.conv2d"(%447, %arg157, %448) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x7x7x2048xf32>, tensor<512x2048x1x1xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %450 = tensor.empty() : tensor<1x7x7x512xf32>
    %451 = "ttir.multiply"(%449, %arg101, %450) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x7x7x512xf32>, tensor<1x1x1x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %452 = tensor.empty() : tensor<1x7x7x512xf32>
    %453 = "ttir.add"(%451, %arg102, %452) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x7x7x512xf32>, tensor<1x1x1x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %454 = tensor.empty() : tensor<1x7x7x512xf32>
    %455 = "ttir.relu"(%453, %454) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %456 = tensor.empty() : tensor<1x7x7x512xf32>
    %457 = "ttir.conv2d"(%455, %arg158, %456) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 1 : si32, padding_left = 1 : si32, padding_right = 1 : si32, padding_top = 1 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x7x7x512xf32>, tensor<512x512x3x3xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %458 = tensor.empty() : tensor<1x7x7x512xf32>
    %459 = "ttir.multiply"(%457, %arg103, %458) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x7x7x512xf32>, tensor<1x1x1x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %460 = tensor.empty() : tensor<1x7x7x512xf32>
    %461 = "ttir.add"(%459, %arg104, %460) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x7x7x512xf32>, tensor<1x1x1x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %462 = tensor.empty() : tensor<1x7x7x512xf32>
    %463 = "ttir.relu"(%461, %462) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %464 = tensor.empty() : tensor<1x7x7x2048xf32>
    %465 = "ttir.conv2d"(%463, %arg159, %464) <{dilation_height = 1 : si32, dilation_width = 1 : si32, groups = 1 : si32, padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32, stride_height = 1 : si32, stride_width = 1 : si32}> {channel_last = 1 : si32} : (tensor<1x7x7x512xf32>, tensor<2048x512x1x1xf32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %466 = tensor.empty() : tensor<1x7x2048x7xf32>
    %467 = "ttir.transpose"(%465, %466) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x7x7x2048xf32>, tensor<1x7x2048x7xf32>) -> tensor<1x7x2048x7xf32>
    %468 = tensor.empty() : tensor<1x2048x7x7xf32>
    %469 = "ttir.transpose"(%467, %468) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x7x2048x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %470 = tensor.empty() : tensor<1x2048x7x7xf32>
    %471 = "ttir.multiply"(%469, %arg105, %470) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xf32>, tensor<1x2048x1x1xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %472 = tensor.empty() : tensor<1x2048x7x7xf32>
    %473 = "ttir.add"(%471, %arg106, %472) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xf32>, tensor<1x2048x1x1xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %474 = tensor.empty() : tensor<1x2048x7x7xf32>
    %475 = "ttir.add"(%473, %443, %474) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %476 = tensor.empty() : tensor<1x2048x7x7xf32>
    %477 = "ttir.relu"(%475, %476) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %478 = tensor.empty() : tensor<1x1x2048x49xf32>
    %479 = "ttir.reshape"(%477, %478) <{shape = [1 : i32, 1 : i32, 2048 : i32, 49 : i32]}> : (tensor<1x2048x7x7xf32>, tensor<1x1x2048x49xf32>) -> tensor<1x1x2048x49xf32>
    %480 = tensor.empty() : tensor<1x1x49x2048xf32>
    %481 = "ttir.transpose"(%479, %480) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x1x2048x49xf32>, tensor<1x1x49x2048xf32>) -> tensor<1x1x49x2048xf32>
    %482 = tensor.empty() : tensor<1x1x1x2048xf32>
    %483 = "ttir.mean"(%481, %482) <{dim_arg = [-2 : i32], keep_dim = true}> : (tensor<1x1x49x2048xf32>, tensor<1x1x1x2048xf32>) -> tensor<1x1x1x2048xf32>
    %484 = tensor.empty() : tensor<1x1x2048x1xf32>
    %485 = "ttir.transpose"(%483, %484) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x1x1x2048xf32>, tensor<1x1x2048x1xf32>) -> tensor<1x1x2048x1xf32>
    %486 = tensor.empty() : tensor<1x2048x1xf32>
    %487 = "ttir.squeeze"(%485, %486) <{dim = 0 : si32}> : (tensor<1x1x2048x1xf32>, tensor<1x2048x1xf32>) -> tensor<1x2048x1xf32>
    %488 = tensor.empty() : tensor<1x2048xf32>
    %489 = "ttir.squeeze"(%487, %488) <{dim = -1 : si32}> : (tensor<1x2048x1xf32>, tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %490 = tensor.empty() : tensor<1x1000xf32>
    %491 = "ttir.matmul"(%489, %arg160, %490) : (tensor<1x2048xf32>, tensor<2048x1000xf32>, tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %492 = tensor.empty() : tensor<1x1000xf32>
    %493 = "ttir.add"(%491, %arg161, %492) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1000xf32>, tensor<1000xf32>, tensor<1x1000xf32>) -> tensor<1x1000xf32>
    return %493 : tensor<1x1000xf32>
  }
}
