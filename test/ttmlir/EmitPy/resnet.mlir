// RUN: ttmlir-opt --ttir-to-emitpy-pipeline -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir

module @ResNetForImageClassification attributes {} {
  func.func @forward(%arg0: tensor<1x3x224x224xbf16> {ttir.name = "pixel_values"}, %arg1: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_2"}, %arg2: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_2_fork_clone1229"}, %arg3: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_19"}, %arg4: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_19_fork_clone1271"}, %arg5: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_35"}, %arg6: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_35_fork_clone1204"}, %arg7: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_51"}, %arg8: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_51_fork_clone1108"}, %arg9: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_66"}, %arg10: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_66_fork_clone1112"}, %arg11: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_83"}, %arg12: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_83_fork_clone1238"}, %arg13: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_99"}, %arg14: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_99_fork_clone1152"}, %arg15: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_115"}, %arg16: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_115_fork_clone1051"}, %arg17: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_132"}, %arg18: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_132_fork_clone1192"}, %arg19: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_148"}, %arg20: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_148_fork_clone1096"}, %arg21: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_164"}, %arg22: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_164_fork_clone992"}, %arg23: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_181"}, %arg24: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_181_fork_clone1065"}, %arg25: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_197"}, %arg26: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_197_fork_clone962"}, %arg27: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_213"}, %arg28: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_213_fork_clone853"}, %arg29: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_228"}, %arg30: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_228_fork_clone857"}, %arg31: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_245"}, %arg32: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_245_fork_clone1007"}, %arg33: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_261"}, %arg34: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_261_fork_clone901"}, %arg35: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_277"}, %arg36: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_277_fork_clone791"}, %arg37: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_294"}, %arg38: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_294_fork_clone950"}, %arg39: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_310"}, %arg40: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_310_fork_clone841"}, %arg41: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_326"}, %arg42: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_326_fork_clone735"}, %arg43: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_343"}, %arg44: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_343_fork_clone889"}, %arg45: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_359"}, %arg46: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_359_fork_clone779"}, %arg47: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_375"}, %arg48: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_375_fork_clone677"}, %arg49: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_392"}, %arg50: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_392_fork_clone748"}, %arg51: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_408"}, %arg52: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_408_fork_clone645"}, %arg53: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_424"}, %arg54: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_424_fork_clone524"}, %arg55: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_439"}, %arg56: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_439_fork_clone528"}, %arg57: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_456"}, %arg58: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_456_fork_clone692"}, %arg59: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_472"}, %arg60: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_472_fork_clone580"}, %arg61: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_488"}, %arg62: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_488_fork_clone453"}, %arg63: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_505"}, %arg64: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_505_fork_clone633"}, %arg65: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_521"}, %arg66: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_521_fork_clone512"}, %arg67: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_537"}, %arg68: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_537_fork_clone389"}, %arg69: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_554"}, %arg70: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_554_fork_clone568"}, %arg71: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_570"}, %arg72: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_570_fork_clone441"}, %arg73: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_586"}, %arg74: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_586_fork_clone329"}, %arg75: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_603"}, %arg76: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_603_fork_clone500"}, %arg77: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_619"}, %arg78: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_619_fork_clone377"}, %arg79: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_635"}, %arg80: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_635_fork_clone274"}, %arg81: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_652"}, %arg82: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_652_fork_clone429"}, %arg83: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_668"}, %arg84: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_668_fork_clone317"}, %arg85: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_684"}, %arg86: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_684_fork_clone219"}, %arg87: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_701"}, %arg88: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_701_fork_clone287"}, %arg89: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_717"}, %arg90: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_717_fork_clone190"}, %arg91: tensor<1x2048x1x1xbf16> {ttir.name = "input_1_add_733"}, %arg92: tensor<1x2048x1x1xbf16> {ttir.name = "input_1_add_733_fork_clone101"}, %arg93: tensor<1x2048x1x1xbf16> {ttir.name = "input_1_add_748"}, %arg94: tensor<1x2048x1x1xbf16> {ttir.name = "input_1_add_748_fork_clone105"}, %arg95: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_765"}, %arg96: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_765_fork_clone233"}, %arg97: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_781"}, %arg98: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_781_fork_clone138"}, %arg99: tensor<1x2048x1x1xbf16> {ttir.name = "input_1_add_797"}, %arg100: tensor<1x2048x1x1xbf16> {ttir.name = "input_1_add_797_fork_clone61"}, %arg101: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_814"}, %arg102: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_814_fork_clone178"}, %arg103: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_830"}, %arg104: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_830_fork_clone89"}, %arg105: tensor<1x2048x1x1xbf16> {ttir.name = "input_1_add_846"}, %arg106: tensor<1x2048x1x1xbf16> {ttir.name = "input_1_add_846_fork_clone32"}, %arg107: tensor<64x3x7x7xbf16> {ttir.name = "resnet.embedder.embedder.convolution.weight"}, %arg108: tensor<64x64x1x1xbf16> {ttir.name = "resnet.encoder.stages.0.layers.0.layer.0.convolution.weight"}, %arg109: tensor<64x64x3x3xbf16> {ttir.name = "resnet.encoder.stages.0.layers.0.layer.1.convolution.weight"}, %arg110: tensor<256x64x1x1xbf16> {ttir.name = "resnet.encoder.stages.0.layers.0.layer.2.convolution.weight"}, %arg111: tensor<256x64x1x1xbf16> {ttir.name = "resnet.encoder.stages.0.layers.0.shortcut.convolution.weight"}, %arg112: tensor<64x256x1x1xbf16> {ttir.name = "resnet.encoder.stages.0.layers.1.layer.0.convolution.weight"}, %arg113: tensor<64x64x3x3xbf16> {ttir.name = "resnet.encoder.stages.0.layers.1.layer.1.convolution.weight"}, %arg114: tensor<256x64x1x1xbf16> {ttir.name = "resnet.encoder.stages.0.layers.1.layer.2.convolution.weight"}, %arg115: tensor<64x256x1x1xbf16> {ttir.name = "resnet.encoder.stages.0.layers.2.layer.0.convolution.weight"}, %arg116: tensor<64x64x3x3xbf16> {ttir.name = "resnet.encoder.stages.0.layers.2.layer.1.convolution.weight"}, %arg117: tensor<256x64x1x1xbf16> {ttir.name = "resnet.encoder.stages.0.layers.2.layer.2.convolution.weight"}, %arg118: tensor<128x256x1x1xbf16> {ttir.name = "resnet.encoder.stages.1.layers.0.layer.0.convolution.weight"}, %arg119: tensor<128x128x3x3xbf16> {ttir.name = "resnet.encoder.stages.1.layers.0.layer.1.convolution.weight"}, %arg120: tensor<512x128x1x1xbf16> {ttir.name = "resnet.encoder.stages.1.layers.0.layer.2.convolution.weight"}, %arg121: tensor<512x256x1x1xbf16> {ttir.name = "resnet.encoder.stages.1.layers.0.shortcut.convolution.weight"}, %arg122: tensor<128x512x1x1xbf16> {ttir.name = "resnet.encoder.stages.1.layers.1.layer.0.convolution.weight"}, %arg123: tensor<128x128x3x3xbf16> {ttir.name = "resnet.encoder.stages.1.layers.1.layer.1.convolution.weight"}, %arg124: tensor<512x128x1x1xbf16> {ttir.name = "resnet.encoder.stages.1.layers.1.layer.2.convolution.weight"}, %arg125: tensor<128x512x1x1xbf16> {ttir.name = "resnet.encoder.stages.1.layers.2.layer.0.convolution.weight"}, %arg126: tensor<128x128x3x3xbf16> {ttir.name = "resnet.encoder.stages.1.layers.2.layer.1.convolution.weight"}, %arg127: tensor<512x128x1x1xbf16> {ttir.name = "resnet.encoder.stages.1.layers.2.layer.2.convolution.weight"}, %arg128: tensor<128x512x1x1xbf16> {ttir.name = "resnet.encoder.stages.1.layers.3.layer.0.convolution.weight"}, %arg129: tensor<128x128x3x3xbf16> {ttir.name = "resnet.encoder.stages.1.layers.3.layer.1.convolution.weight"}, %arg130: tensor<512x128x1x1xbf16> {ttir.name = "resnet.encoder.stages.1.layers.3.layer.2.convolution.weight"}, %arg131: tensor<256x512x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.0.layer.0.convolution.weight"}, %arg132: tensor<256x256x3x3xbf16> {ttir.name = "resnet.encoder.stages.2.layers.0.layer.1.convolution.weight"}, %arg133: tensor<1024x256x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.0.layer.2.convolution.weight"}, %arg134: tensor<1024x512x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.0.shortcut.convolution.weight"}, %arg135: tensor<256x1024x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.1.layer.0.convolution.weight"}, %arg136: tensor<256x256x3x3xbf16> {ttir.name = "resnet.encoder.stages.2.layers.1.layer.1.convolution.weight"}, %arg137: tensor<1024x256x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.1.layer.2.convolution.weight"}, %arg138: tensor<256x1024x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.2.layer.0.convolution.weight"}, %arg139: tensor<256x256x3x3xbf16> {ttir.name = "resnet.encoder.stages.2.layers.2.layer.1.convolution.weight"}, %arg140: tensor<1024x256x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.2.layer.2.convolution.weight"}, %arg141: tensor<256x1024x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.3.layer.0.convolution.weight"}, %arg142: tensor<256x256x3x3xbf16> {ttir.name = "resnet.encoder.stages.2.layers.3.layer.1.convolution.weight"}, %arg143: tensor<1024x256x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.3.layer.2.convolution.weight"}, %arg144: tensor<256x1024x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.4.layer.0.convolution.weight"}, %arg145: tensor<256x256x3x3xbf16> {ttir.name = "resnet.encoder.stages.2.layers.4.layer.1.convolution.weight"}, %arg146: tensor<1024x256x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.4.layer.2.convolution.weight"}, %arg147: tensor<256x1024x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.5.layer.0.convolution.weight"}, %arg148: tensor<256x256x3x3xbf16> {ttir.name = "resnet.encoder.stages.2.layers.5.layer.1.convolution.weight"}, %arg149: tensor<1024x256x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.5.layer.2.convolution.weight"}, %arg150: tensor<512x1024x1x1xbf16> {ttir.name = "resnet.encoder.stages.3.layers.0.layer.0.convolution.weight"}, %arg151: tensor<512x512x3x3xbf16> {ttir.name = "resnet.encoder.stages.3.layers.0.layer.1.convolution.weight"}, %arg152: tensor<2048x512x1x1xbf16> {ttir.name = "resnet.encoder.stages.3.layers.0.layer.2.convolution.weight"}, %arg153: tensor<2048x1024x1x1xbf16> {ttir.name = "resnet.encoder.stages.3.layers.0.shortcut.convolution.weight"}, %arg154: tensor<512x2048x1x1xbf16> {ttir.name = "resnet.encoder.stages.3.layers.1.layer.0.convolution.weight"}, %arg155: tensor<512x512x3x3xbf16> {ttir.name = "resnet.encoder.stages.3.layers.1.layer.1.convolution.weight"}, %arg156: tensor<2048x512x1x1xbf16> {ttir.name = "resnet.encoder.stages.3.layers.1.layer.2.convolution.weight"}, %arg157: tensor<512x2048x1x1xbf16> {ttir.name = "resnet.encoder.stages.3.layers.2.layer.0.convolution.weight"}, %arg158: tensor<512x512x3x3xbf16> {ttir.name = "resnet.encoder.stages.3.layers.2.layer.1.convolution.weight"}, %arg159: tensor<2048x512x1x1xbf16> {ttir.name = "resnet.encoder.stages.3.layers.2.layer.2.convolution.weight"}, %arg160: tensor<2048x1000xbf16> {ttir.name = "classifier.1.weight"}, %arg161: tensor<1000xbf16> {ttir.name = "classifier.1.bias"}) -> (tensor<1x1000xbf16> {ttir.name = "ResNetForImageClassification.output_add_868"}) {
    %0 = ttir.empty() : tensor<1x224x3x224xbf16>
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x3x224x224xbf16>, tensor<1x224x3x224xbf16>) -> tensor<1x224x3x224xbf16>
    %2 = ttir.empty() : tensor<1x224x224x3xbf16>
    %3 = "ttir.transpose"(%1, %2) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x224x3x224xbf16>, tensor<1x224x224x3xbf16>) -> tensor<1x224x224x3xbf16>
    %4 = ttir.empty() : tensor<1x112x112x64xbf16>
    %5 = "ttir.conv2d"(%3, %arg107, %4) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 3, 3, 3, 3>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x224x224x3xbf16>, tensor<64x3x7x7xbf16>, tensor<1x112x112x64xbf16>) -> tensor<1x112x112x64xbf16>
    %6 = ttir.empty() : tensor<1x112x112x64xbf16>
    %7 = "ttir.multiply"(%5, %arg1, %6) : (tensor<1x112x112x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x112x112x64xbf16>) -> tensor<1x112x112x64xbf16>
    %8 = ttir.empty() : tensor<1x112x112x64xbf16>
    %9 = "ttir.add"(%7, %arg2, %8) : (tensor<1x112x112x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x112x112x64xbf16>) -> tensor<1x112x112x64xbf16>
    %10 = ttir.empty() : tensor<1x112x112x64xbf16>
    %11 = "ttir.relu"(%9, %10) : (tensor<1x112x112x64xbf16>, tensor<1x112x112x64xbf16>) -> tensor<1x112x112x64xbf16>
    %12 = ttir.empty() : tensor<1x56x56x64xbf16>
    %13 = "ttir.max_pool2d"(%11, %12) <{kernel = array<i32: 3, 3>, stride = array<i32: 2, 2>, dilation = array<i32: 1, 1>, padding = array<i32: 1, 1, 1, 1>, ceil_mode = false}> {channel_last = true} : (tensor<1x112x112x64xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %14 = ttir.empty() : tensor<1x56x56x64xbf16>
    %15 = "ttir.conv2d"(%13, %arg108, %14) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x64xbf16>, tensor<64x64x1x1xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %16 = ttir.empty() : tensor<1x56x56x64xbf16>
    %17 = "ttir.multiply"(%15, %arg3, %16) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %18 = ttir.empty() : tensor<1x56x56x64xbf16>
    %19 = "ttir.add"(%17, %arg4, %18) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %20 = ttir.empty() : tensor<1x56x56x64xbf16>
    %21 = "ttir.relu"(%19, %20) : (tensor<1x56x56x64xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %22 = ttir.empty() : tensor<1x56x56x64xbf16>
    %23 = "ttir.conv2d"(%21, %arg109, %22) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %24 = ttir.empty() : tensor<1x56x56x64xbf16>
    %25 = "ttir.multiply"(%23, %arg5, %24) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %26 = ttir.empty() : tensor<1x56x56x64xbf16>
    %27 = "ttir.add"(%25, %arg6, %26) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %28 = ttir.empty() : tensor<1x56x56x64xbf16>
    %29 = "ttir.relu"(%27, %28) : (tensor<1x56x56x64xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %30 = ttir.empty() : tensor<1x56x56x256xbf16>
    %31 = "ttir.conv2d"(%29, %arg110, %30) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x64xbf16>, tensor<256x64x1x1xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %32 = ttir.empty() : tensor<1x56x56x256xbf16>
    %33 = "ttir.multiply"(%31, %arg7, %32) : (tensor<1x56x56x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %34 = ttir.empty() : tensor<1x56x56x256xbf16>
    %35 = "ttir.add"(%33, %arg8, %34) : (tensor<1x56x56x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %36 = ttir.empty() : tensor<1x56x56x256xbf16>
    %37 = "ttir.conv2d"(%13, %arg111, %36) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x64xbf16>, tensor<256x64x1x1xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %38 = ttir.empty() : tensor<1x56x56x256xbf16>
    %39 = "ttir.multiply"(%37, %arg9, %38) : (tensor<1x56x56x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %40 = ttir.empty() : tensor<1x56x56x256xbf16>
    %41 = "ttir.add"(%39, %arg10, %40) : (tensor<1x56x56x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %42 = ttir.empty() : tensor<1x56x56x256xbf16>
    %43 = "ttir.add"(%35, %41, %42) : (tensor<1x56x56x256xbf16>, tensor<1x56x56x256xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %44 = ttir.empty() : tensor<1x56x56x256xbf16>
    %45 = "ttir.relu"(%43, %44) : (tensor<1x56x56x256xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %46 = ttir.empty() : tensor<1x56x56x64xbf16>
    %47 = "ttir.conv2d"(%45, %arg112, %46) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x256xbf16>, tensor<64x256x1x1xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %48 = ttir.empty() : tensor<1x56x56x64xbf16>
    %49 = "ttir.multiply"(%47, %arg11, %48) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %50 = ttir.empty() : tensor<1x56x56x64xbf16>
    %51 = "ttir.add"(%49, %arg12, %50) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %52 = ttir.empty() : tensor<1x56x56x64xbf16>
    %53 = "ttir.relu"(%51, %52) : (tensor<1x56x56x64xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %54 = ttir.empty() : tensor<1x56x56x64xbf16>
    %55 = "ttir.conv2d"(%53, %arg113, %54) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %56 = ttir.empty() : tensor<1x56x56x64xbf16>
    %57 = "ttir.multiply"(%55, %arg13, %56) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %58 = ttir.empty() : tensor<1x56x56x64xbf16>
    %59 = "ttir.add"(%57, %arg14, %58) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %60 = ttir.empty() : tensor<1x56x56x64xbf16>
    %61 = "ttir.relu"(%59, %60) : (tensor<1x56x56x64xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %62 = ttir.empty() : tensor<1x56x56x256xbf16>
    %63 = "ttir.conv2d"(%61, %arg114, %62) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x64xbf16>, tensor<256x64x1x1xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %64 = ttir.empty() : tensor<1x56x56x256xbf16>
    %65 = "ttir.multiply"(%63, %arg15, %64) : (tensor<1x56x56x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %66 = ttir.empty() : tensor<1x56x56x256xbf16>
    %67 = "ttir.add"(%65, %arg16, %66) : (tensor<1x56x56x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %68 = ttir.empty() : tensor<1x56x56x256xbf16>
    %69 = "ttir.add"(%67, %45, %68) : (tensor<1x56x56x256xbf16>, tensor<1x56x56x256xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %70 = ttir.empty() : tensor<1x56x56x256xbf16>
    %71 = "ttir.relu"(%69, %70) : (tensor<1x56x56x256xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %72 = ttir.empty() : tensor<1x56x56x64xbf16>
    %73 = "ttir.conv2d"(%71, %arg115, %72) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x256xbf16>, tensor<64x256x1x1xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %74 = ttir.empty() : tensor<1x56x56x64xbf16>
    %75 = "ttir.multiply"(%73, %arg17, %74) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %76 = ttir.empty() : tensor<1x56x56x64xbf16>
    %77 = "ttir.add"(%75, %arg18, %76) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %78 = ttir.empty() : tensor<1x56x56x64xbf16>
    %79 = "ttir.relu"(%77, %78) : (tensor<1x56x56x64xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %80 = ttir.empty() : tensor<1x56x56x64xbf16>
    %81 = "ttir.conv2d"(%79, %arg116, %80) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %82 = ttir.empty() : tensor<1x56x56x64xbf16>
    %83 = "ttir.multiply"(%81, %arg19, %82) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %84 = ttir.empty() : tensor<1x56x56x64xbf16>
    %85 = "ttir.add"(%83, %arg20, %84) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %86 = ttir.empty() : tensor<1x56x56x64xbf16>
    %87 = "ttir.relu"(%85, %86) : (tensor<1x56x56x64xbf16>, tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %88 = ttir.empty() : tensor<1x56x56x256xbf16>
    %89 = "ttir.conv2d"(%87, %arg117, %88) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x64xbf16>, tensor<256x64x1x1xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %90 = ttir.empty() : tensor<1x56x56x256xbf16>
    %91 = "ttir.multiply"(%89, %arg21, %90) : (tensor<1x56x56x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %92 = ttir.empty() : tensor<1x56x56x256xbf16>
    %93 = "ttir.add"(%91, %arg22, %92) : (tensor<1x56x56x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %94 = ttir.empty() : tensor<1x56x56x256xbf16>
    %95 = "ttir.add"(%93, %71, %94) : (tensor<1x56x56x256xbf16>, tensor<1x56x56x256xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %96 = ttir.empty() : tensor<1x56x56x256xbf16>
    %97 = "ttir.relu"(%95, %96) : (tensor<1x56x56x256xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %98 = ttir.empty() : tensor<1x56x56x128xbf16>
    %99 = "ttir.conv2d"(%97, %arg118, %98) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x256xbf16>, tensor<128x256x1x1xbf16>, tensor<1x56x56x128xbf16>) -> tensor<1x56x56x128xbf16>
    %100 = ttir.empty() : tensor<1x56x56x128xbf16>
    %101 = "ttir.multiply"(%99, %arg23, %100) : (tensor<1x56x56x128xbf16>, tensor<1x1x1x128xbf16>, tensor<1x56x56x128xbf16>) -> tensor<1x56x56x128xbf16>
    %102 = ttir.empty() : tensor<1x56x56x128xbf16>
    %103 = "ttir.add"(%101, %arg24, %102) : (tensor<1x56x56x128xbf16>, tensor<1x1x1x128xbf16>, tensor<1x56x56x128xbf16>) -> tensor<1x56x56x128xbf16>
    %104 = ttir.empty() : tensor<1x56x56x128xbf16>
    %105 = "ttir.relu"(%103, %104) : (tensor<1x56x56x128xbf16>, tensor<1x56x56x128xbf16>) -> tensor<1x56x56x128xbf16>
    %106 = ttir.empty() : tensor<1x28x28x128xbf16>
    %107 = "ttir.conv2d"(%105, %arg119, %106) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x56x56x128xbf16>, tensor<128x128x3x3xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %108 = ttir.empty() : tensor<1x28x28x128xbf16>
    %109 = "ttir.multiply"(%107, %arg25, %108) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %110 = ttir.empty() : tensor<1x28x28x128xbf16>
    %111 = "ttir.add"(%109, %arg26, %110) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %112 = ttir.empty() : tensor<1x28x28x128xbf16>
    %113 = "ttir.relu"(%111, %112) : (tensor<1x28x28x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %114 = ttir.empty() : tensor<1x28x28x512xbf16>
    %115 = "ttir.conv2d"(%113, %arg120, %114) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x128xbf16>, tensor<512x128x1x1xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %116 = ttir.empty() : tensor<1x28x28x512xbf16>
    %117 = "ttir.multiply"(%115, %arg27, %116) : (tensor<1x28x28x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %118 = ttir.empty() : tensor<1x28x28x512xbf16>
    %119 = "ttir.add"(%117, %arg28, %118) : (tensor<1x28x28x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %120 = ttir.empty() : tensor<1x28x28x512xbf16>
    %121 = "ttir.conv2d"(%97, %arg121, %120) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x56x56x256xbf16>, tensor<512x256x1x1xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %122 = ttir.empty() : tensor<1x28x28x512xbf16>
    %123 = "ttir.multiply"(%121, %arg29, %122) : (tensor<1x28x28x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %124 = ttir.empty() : tensor<1x28x28x512xbf16>
    %125 = "ttir.add"(%123, %arg30, %124) : (tensor<1x28x28x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %126 = ttir.empty() : tensor<1x28x28x512xbf16>
    %127 = "ttir.add"(%119, %125, %126) : (tensor<1x28x28x512xbf16>, tensor<1x28x28x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %128 = ttir.empty() : tensor<1x28x28x512xbf16>
    %129 = "ttir.relu"(%127, %128) : (tensor<1x28x28x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %130 = ttir.empty() : tensor<1x28x28x128xbf16>
    %131 = "ttir.conv2d"(%129, %arg122, %130) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x512xbf16>, tensor<128x512x1x1xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %132 = ttir.empty() : tensor<1x28x28x128xbf16>
    %133 = "ttir.multiply"(%131, %arg31, %132) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %134 = ttir.empty() : tensor<1x28x28x128xbf16>
    %135 = "ttir.add"(%133, %arg32, %134) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %136 = ttir.empty() : tensor<1x28x28x128xbf16>
    %137 = "ttir.relu"(%135, %136) : (tensor<1x28x28x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %138 = ttir.empty() : tensor<1x28x28x128xbf16>
    %139 = "ttir.conv2d"(%137, %arg123, %138) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x128xbf16>, tensor<128x128x3x3xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %140 = ttir.empty() : tensor<1x28x28x128xbf16>
    %141 = "ttir.multiply"(%139, %arg33, %140) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %142 = ttir.empty() : tensor<1x28x28x128xbf16>
    %143 = "ttir.add"(%141, %arg34, %142) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %144 = ttir.empty() : tensor<1x28x28x128xbf16>
    %145 = "ttir.relu"(%143, %144) : (tensor<1x28x28x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %146 = ttir.empty() : tensor<1x28x28x512xbf16>
    %147 = "ttir.conv2d"(%145, %arg124, %146) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x128xbf16>, tensor<512x128x1x1xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %148 = ttir.empty() : tensor<1x28x28x512xbf16>
    %149 = "ttir.multiply"(%147, %arg35, %148) : (tensor<1x28x28x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %150 = ttir.empty() : tensor<1x28x28x512xbf16>
    %151 = "ttir.add"(%149, %arg36, %150) : (tensor<1x28x28x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %152 = ttir.empty() : tensor<1x28x28x512xbf16>
    %153 = "ttir.add"(%151, %129, %152) : (tensor<1x28x28x512xbf16>, tensor<1x28x28x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %154 = ttir.empty() : tensor<1x28x28x512xbf16>
    %155 = "ttir.relu"(%153, %154) : (tensor<1x28x28x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %156 = ttir.empty() : tensor<1x28x28x128xbf16>
    %157 = "ttir.conv2d"(%155, %arg125, %156) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x512xbf16>, tensor<128x512x1x1xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %158 = ttir.empty() : tensor<1x28x28x128xbf16>
    %159 = "ttir.multiply"(%157, %arg37, %158) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %160 = ttir.empty() : tensor<1x28x28x128xbf16>
    %161 = "ttir.add"(%159, %arg38, %160) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %162 = ttir.empty() : tensor<1x28x28x128xbf16>
    %163 = "ttir.relu"(%161, %162) : (tensor<1x28x28x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %164 = ttir.empty() : tensor<1x28x28x128xbf16>
    %165 = "ttir.conv2d"(%163, %arg126, %164) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x128xbf16>, tensor<128x128x3x3xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %166 = ttir.empty() : tensor<1x28x28x128xbf16>
    %167 = "ttir.multiply"(%165, %arg39, %166) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %168 = ttir.empty() : tensor<1x28x28x128xbf16>
    %169 = "ttir.add"(%167, %arg40, %168) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %170 = ttir.empty() : tensor<1x28x28x128xbf16>
    %171 = "ttir.relu"(%169, %170) : (tensor<1x28x28x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %172 = ttir.empty() : tensor<1x28x28x512xbf16>
    %173 = "ttir.conv2d"(%171, %arg127, %172) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x128xbf16>, tensor<512x128x1x1xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %174 = ttir.empty() : tensor<1x28x28x512xbf16>
    %175 = "ttir.multiply"(%173, %arg41, %174) : (tensor<1x28x28x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %176 = ttir.empty() : tensor<1x28x28x512xbf16>
    %177 = "ttir.add"(%175, %arg42, %176) : (tensor<1x28x28x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %178 = ttir.empty() : tensor<1x28x28x512xbf16>
    %179 = "ttir.add"(%177, %155, %178) : (tensor<1x28x28x512xbf16>, tensor<1x28x28x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %180 = ttir.empty() : tensor<1x28x28x512xbf16>
    %181 = "ttir.relu"(%179, %180) : (tensor<1x28x28x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %182 = ttir.empty() : tensor<1x28x28x128xbf16>
    %183 = "ttir.conv2d"(%181, %arg128, %182) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x512xbf16>, tensor<128x512x1x1xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %184 = ttir.empty() : tensor<1x28x28x128xbf16>
    %185 = "ttir.multiply"(%183, %arg43, %184) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %186 = ttir.empty() : tensor<1x28x28x128xbf16>
    %187 = "ttir.add"(%185, %arg44, %186) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %188 = ttir.empty() : tensor<1x28x28x128xbf16>
    %189 = "ttir.relu"(%187, %188) : (tensor<1x28x28x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %190 = ttir.empty() : tensor<1x28x28x128xbf16>
    %191 = "ttir.conv2d"(%189, %arg129, %190) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x128xbf16>, tensor<128x128x3x3xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %192 = ttir.empty() : tensor<1x28x28x128xbf16>
    %193 = "ttir.multiply"(%191, %arg45, %192) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %194 = ttir.empty() : tensor<1x28x28x128xbf16>
    %195 = "ttir.add"(%193, %arg46, %194) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %196 = ttir.empty() : tensor<1x28x28x128xbf16>
    %197 = "ttir.relu"(%195, %196) : (tensor<1x28x28x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %198 = ttir.empty() : tensor<1x28x28x512xbf16>
    %199 = "ttir.conv2d"(%197, %arg130, %198) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x128xbf16>, tensor<512x128x1x1xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %200 = ttir.empty() : tensor<1x28x28x512xbf16>
    %201 = "ttir.multiply"(%199, %arg47, %200) : (tensor<1x28x28x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %202 = ttir.empty() : tensor<1x28x28x512xbf16>
    %203 = "ttir.add"(%201, %arg48, %202) : (tensor<1x28x28x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %204 = ttir.empty() : tensor<1x28x28x512xbf16>
    %205 = "ttir.add"(%203, %181, %204) : (tensor<1x28x28x512xbf16>, tensor<1x28x28x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %206 = ttir.empty() : tensor<1x28x28x512xbf16>
    %207 = "ttir.relu"(%205, %206) : (tensor<1x28x28x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %208 = ttir.empty() : tensor<1x28x28x256xbf16>
    %209 = "ttir.conv2d"(%207, %arg131, %208) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x512xbf16>, tensor<256x512x1x1xbf16>, tensor<1x28x28x256xbf16>) -> tensor<1x28x28x256xbf16>
    %210 = ttir.empty() : tensor<1x28x28x256xbf16>
    %211 = "ttir.multiply"(%209, %arg49, %210) : (tensor<1x28x28x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x28x28x256xbf16>) -> tensor<1x28x28x256xbf16>
    %212 = ttir.empty() : tensor<1x28x28x256xbf16>
    %213 = "ttir.add"(%211, %arg50, %212) : (tensor<1x28x28x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x28x28x256xbf16>) -> tensor<1x28x28x256xbf16>
    %214 = ttir.empty() : tensor<1x28x28x256xbf16>
    %215 = "ttir.relu"(%213, %214) : (tensor<1x28x28x256xbf16>, tensor<1x28x28x256xbf16>) -> tensor<1x28x28x256xbf16>
    %216 = ttir.empty() : tensor<1x14x14x256xbf16>
    %217 = "ttir.conv2d"(%215, %arg132, %216) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x28x28x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %218 = ttir.empty() : tensor<1x14x14x256xbf16>
    %219 = "ttir.multiply"(%217, %arg51, %218) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %220 = ttir.empty() : tensor<1x14x14x256xbf16>
    %221 = "ttir.add"(%219, %arg52, %220) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %222 = ttir.empty() : tensor<1x14x14x256xbf16>
    %223 = "ttir.relu"(%221, %222) : (tensor<1x14x14x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %224 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %225 = "ttir.conv2d"(%223, %arg133, %224) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<1024x256x1x1xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %226 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %227 = "ttir.multiply"(%225, %arg53, %226) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %228 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %229 = "ttir.add"(%227, %arg54, %228) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %230 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %231 = "ttir.conv2d"(%207, %arg134, %230) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x28x28x512xbf16>, tensor<1024x512x1x1xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %232 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %233 = "ttir.multiply"(%231, %arg55, %232) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %234 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %235 = "ttir.add"(%233, %arg56, %234) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %236 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %237 = "ttir.add"(%229, %235, %236) : (tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %238 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %239 = "ttir.relu"(%237, %238) : (tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %240 = ttir.empty() : tensor<1x14x14x256xbf16>
    %241 = "ttir.conv2d"(%239, %arg135, %240) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xbf16>, tensor<256x1024x1x1xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %242 = ttir.empty() : tensor<1x14x14x256xbf16>
    %243 = "ttir.multiply"(%241, %arg57, %242) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %244 = ttir.empty() : tensor<1x14x14x256xbf16>
    %245 = "ttir.add"(%243, %arg58, %244) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %246 = ttir.empty() : tensor<1x14x14x256xbf16>
    %247 = "ttir.relu"(%245, %246) : (tensor<1x14x14x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %248 = ttir.empty() : tensor<1x14x14x256xbf16>
    %249 = "ttir.conv2d"(%247, %arg136, %248) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %250 = ttir.empty() : tensor<1x14x14x256xbf16>
    %251 = "ttir.multiply"(%249, %arg59, %250) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %252 = ttir.empty() : tensor<1x14x14x256xbf16>
    %253 = "ttir.add"(%251, %arg60, %252) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %254 = ttir.empty() : tensor<1x14x14x256xbf16>
    %255 = "ttir.relu"(%253, %254) : (tensor<1x14x14x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %256 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %257 = "ttir.conv2d"(%255, %arg137, %256) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<1024x256x1x1xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %258 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %259 = "ttir.multiply"(%257, %arg61, %258) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %260 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %261 = "ttir.add"(%259, %arg62, %260) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %262 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %263 = "ttir.add"(%261, %239, %262) : (tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %264 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %265 = "ttir.relu"(%263, %264) : (tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %266 = ttir.empty() : tensor<1x14x14x256xbf16>
    %267 = "ttir.conv2d"(%265, %arg138, %266) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xbf16>, tensor<256x1024x1x1xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %268 = ttir.empty() : tensor<1x14x14x256xbf16>
    %269 = "ttir.multiply"(%267, %arg63, %268) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %270 = ttir.empty() : tensor<1x14x14x256xbf16>
    %271 = "ttir.add"(%269, %arg64, %270) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %272 = ttir.empty() : tensor<1x14x14x256xbf16>
    %273 = "ttir.relu"(%271, %272) : (tensor<1x14x14x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %274 = ttir.empty() : tensor<1x14x14x256xbf16>
    %275 = "ttir.conv2d"(%273, %arg139, %274) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %276 = ttir.empty() : tensor<1x14x14x256xbf16>
    %277 = "ttir.multiply"(%275, %arg65, %276) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %278 = ttir.empty() : tensor<1x14x14x256xbf16>
    %279 = "ttir.add"(%277, %arg66, %278) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %280 = ttir.empty() : tensor<1x14x14x256xbf16>
    %281 = "ttir.relu"(%279, %280) : (tensor<1x14x14x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %282 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %283 = "ttir.conv2d"(%281, %arg140, %282) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<1024x256x1x1xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %284 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %285 = "ttir.multiply"(%283, %arg67, %284) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %286 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %287 = "ttir.add"(%285, %arg68, %286) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %288 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %289 = "ttir.add"(%287, %265, %288) : (tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %290 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %291 = "ttir.relu"(%289, %290) : (tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %292 = ttir.empty() : tensor<1x14x14x256xbf16>
    %293 = "ttir.conv2d"(%291, %arg141, %292) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xbf16>, tensor<256x1024x1x1xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %294 = ttir.empty() : tensor<1x14x14x256xbf16>
    %295 = "ttir.multiply"(%293, %arg69, %294) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %296 = ttir.empty() : tensor<1x14x14x256xbf16>
    %297 = "ttir.add"(%295, %arg70, %296) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %298 = ttir.empty() : tensor<1x14x14x256xbf16>
    %299 = "ttir.relu"(%297, %298) : (tensor<1x14x14x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %300 = ttir.empty() : tensor<1x14x14x256xbf16>
    %301 = "ttir.conv2d"(%299, %arg142, %300) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %302 = ttir.empty() : tensor<1x14x14x256xbf16>
    %303 = "ttir.multiply"(%301, %arg71, %302) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %304 = ttir.empty() : tensor<1x14x14x256xbf16>
    %305 = "ttir.add"(%303, %arg72, %304) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %306 = ttir.empty() : tensor<1x14x14x256xbf16>
    %307 = "ttir.relu"(%305, %306) : (tensor<1x14x14x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %308 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %309 = "ttir.conv2d"(%307, %arg143, %308) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<1024x256x1x1xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %310 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %311 = "ttir.multiply"(%309, %arg73, %310) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %312 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %313 = "ttir.add"(%311, %arg74, %312) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %314 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %315 = "ttir.add"(%313, %291, %314) : (tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %316 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %317 = "ttir.relu"(%315, %316) : (tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %318 = ttir.empty() : tensor<1x14x14x256xbf16>
    %319 = "ttir.conv2d"(%317, %arg144, %318) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xbf16>, tensor<256x1024x1x1xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %320 = ttir.empty() : tensor<1x14x14x256xbf16>
    %321 = "ttir.multiply"(%319, %arg75, %320) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %322 = ttir.empty() : tensor<1x14x14x256xbf16>
    %323 = "ttir.add"(%321, %arg76, %322) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %324 = ttir.empty() : tensor<1x14x14x256xbf16>
    %325 = "ttir.relu"(%323, %324) : (tensor<1x14x14x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %326 = ttir.empty() : tensor<1x14x14x256xbf16>
    %327 = "ttir.conv2d"(%325, %arg145, %326) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %328 = ttir.empty() : tensor<1x14x14x256xbf16>
    %329 = "ttir.multiply"(%327, %arg77, %328) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %330 = ttir.empty() : tensor<1x14x14x256xbf16>
    %331 = "ttir.add"(%329, %arg78, %330) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %332 = ttir.empty() : tensor<1x14x14x256xbf16>
    %333 = "ttir.relu"(%331, %332) : (tensor<1x14x14x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %334 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %335 = "ttir.conv2d"(%333, %arg146, %334) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<1024x256x1x1xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %336 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %337 = "ttir.multiply"(%335, %arg79, %336) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %338 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %339 = "ttir.add"(%337, %arg80, %338) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %340 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %341 = "ttir.add"(%339, %317, %340) : (tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %342 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %343 = "ttir.relu"(%341, %342) : (tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %344 = ttir.empty() : tensor<1x14x14x256xbf16>
    %345 = "ttir.conv2d"(%343, %arg147, %344) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xbf16>, tensor<256x1024x1x1xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %346 = ttir.empty() : tensor<1x14x14x256xbf16>
    %347 = "ttir.multiply"(%345, %arg81, %346) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %348 = ttir.empty() : tensor<1x14x14x256xbf16>
    %349 = "ttir.add"(%347, %arg82, %348) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %350 = ttir.empty() : tensor<1x14x14x256xbf16>
    %351 = "ttir.relu"(%349, %350) : (tensor<1x14x14x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %352 = ttir.empty() : tensor<1x14x14x256xbf16>
    %353 = "ttir.conv2d"(%351, %arg148, %352) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %354 = ttir.empty() : tensor<1x14x14x256xbf16>
    %355 = "ttir.multiply"(%353, %arg83, %354) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %356 = ttir.empty() : tensor<1x14x14x256xbf16>
    %357 = "ttir.add"(%355, %arg84, %356) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %358 = ttir.empty() : tensor<1x14x14x256xbf16>
    %359 = "ttir.relu"(%357, %358) : (tensor<1x14x14x256xbf16>, tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %360 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %361 = "ttir.conv2d"(%359, %arg149, %360) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<1024x256x1x1xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %362 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %363 = "ttir.multiply"(%361, %arg85, %362) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %364 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %365 = "ttir.add"(%363, %arg86, %364) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %366 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %367 = "ttir.add"(%365, %343, %366) : (tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %368 = ttir.empty() : tensor<1x14x14x1024xbf16>
    %369 = "ttir.relu"(%367, %368) : (tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %370 = ttir.empty() : tensor<1x14x14x512xbf16>
    %371 = "ttir.conv2d"(%369, %arg150, %370) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xbf16>, tensor<512x1024x1x1xbf16>, tensor<1x14x14x512xbf16>) -> tensor<1x14x14x512xbf16>
    %372 = ttir.empty() : tensor<1x14x14x512xbf16>
    %373 = "ttir.multiply"(%371, %arg87, %372) : (tensor<1x14x14x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x14x14x512xbf16>) -> tensor<1x14x14x512xbf16>
    %374 = ttir.empty() : tensor<1x14x14x512xbf16>
    %375 = "ttir.add"(%373, %arg88, %374) : (tensor<1x14x14x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x14x14x512xbf16>) -> tensor<1x14x14x512xbf16>
    %376 = ttir.empty() : tensor<1x14x14x512xbf16>
    %377 = "ttir.relu"(%375, %376) : (tensor<1x14x14x512xbf16>, tensor<1x14x14x512xbf16>) -> tensor<1x14x14x512xbf16>
    %378 = ttir.empty() : tensor<1x7x7x512xbf16>
    %379 = "ttir.conv2d"(%377, %arg151, %378) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x14x14x512xbf16>, tensor<512x512x3x3xbf16>, tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %380 = ttir.empty() : tensor<1x7x7x512xbf16>
    %381 = "ttir.multiply"(%379, %arg89, %380) : (tensor<1x7x7x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %382 = ttir.empty() : tensor<1x7x7x512xbf16>
    %383 = "ttir.add"(%381, %arg90, %382) : (tensor<1x7x7x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %384 = ttir.empty() : tensor<1x7x7x512xbf16>
    %385 = "ttir.relu"(%383, %384) : (tensor<1x7x7x512xbf16>, tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %386 = ttir.empty() : tensor<1x7x7x2048xbf16>
    %387 = "ttir.conv2d"(%385, %arg152, %386) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x7x7x512xbf16>, tensor<2048x512x1x1xbf16>, tensor<1x7x7x2048xbf16>) -> tensor<1x7x7x2048xbf16>
    %388 = ttir.empty() : tensor<1x7x2048x7xbf16>
    %389 = "ttir.transpose"(%387, %388) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x7x7x2048xbf16>, tensor<1x7x2048x7xbf16>) -> tensor<1x7x2048x7xbf16>
    %390 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %391 = "ttir.transpose"(%389, %390) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x7x2048x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %392 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %393 = "ttir.multiply"(%391, %arg91, %392) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %394 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %395 = "ttir.add"(%393, %arg92, %394) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %396 = ttir.empty() : tensor<1x7x7x2048xbf16>
    %397 = "ttir.conv2d"(%369, %arg153, %396) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xbf16>, tensor<2048x1024x1x1xbf16>, tensor<1x7x7x2048xbf16>) -> tensor<1x7x7x2048xbf16>
    %398 = ttir.empty() : tensor<1x7x2048x7xbf16>
    %399 = "ttir.transpose"(%397, %398) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x7x7x2048xbf16>, tensor<1x7x2048x7xbf16>) -> tensor<1x7x2048x7xbf16>
    %400 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %401 = "ttir.transpose"(%399, %400) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x7x2048x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %402 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %403 = "ttir.multiply"(%401, %arg93, %402) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %404 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %405 = "ttir.add"(%403, %arg94, %404) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %406 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %407 = "ttir.add"(%395, %405, %406) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %408 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %409 = "ttir.relu"(%407, %408) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %410 = ttir.empty() : tensor<1x7x2048x7xbf16>
    %411 = "ttir.transpose"(%409, %410) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x2048x7x7xbf16>, tensor<1x7x2048x7xbf16>) -> tensor<1x7x2048x7xbf16>
    %412 = ttir.empty() : tensor<1x7x7x2048xbf16>
    %413 = "ttir.transpose"(%411, %412) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x7x2048x7xbf16>, tensor<1x7x7x2048xbf16>) -> tensor<1x7x7x2048xbf16>
    %414 = ttir.empty() : tensor<1x7x7x512xbf16>
    %415 = "ttir.conv2d"(%413, %arg154, %414) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x7x7x2048xbf16>, tensor<512x2048x1x1xbf16>, tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %416 = ttir.empty() : tensor<1x7x7x512xbf16>
    %417 = "ttir.multiply"(%415, %arg95, %416) : (tensor<1x7x7x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %418 = ttir.empty() : tensor<1x7x7x512xbf16>
    %419 = "ttir.add"(%417, %arg96, %418) : (tensor<1x7x7x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %420 = ttir.empty() : tensor<1x7x7x512xbf16>
    %421 = "ttir.relu"(%419, %420) : (tensor<1x7x7x512xbf16>, tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %422 = ttir.empty() : tensor<1x7x7x512xbf16>
    %423 = "ttir.conv2d"(%421, %arg155, %422) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x7x7x512xbf16>, tensor<512x512x3x3xbf16>, tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %424 = ttir.empty() : tensor<1x7x7x512xbf16>
    %425 = "ttir.multiply"(%423, %arg97, %424) : (tensor<1x7x7x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %426 = ttir.empty() : tensor<1x7x7x512xbf16>
    %427 = "ttir.add"(%425, %arg98, %426) : (tensor<1x7x7x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %428 = ttir.empty() : tensor<1x7x7x512xbf16>
    %429 = "ttir.relu"(%427, %428) : (tensor<1x7x7x512xbf16>, tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %430 = ttir.empty() : tensor<1x7x7x2048xbf16>
    %431 = "ttir.conv2d"(%429, %arg156, %430) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x7x7x512xbf16>, tensor<2048x512x1x1xbf16>, tensor<1x7x7x2048xbf16>) -> tensor<1x7x7x2048xbf16>
    %432 = ttir.empty() : tensor<1x7x2048x7xbf16>
    %433 = "ttir.transpose"(%431, %432) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x7x7x2048xbf16>, tensor<1x7x2048x7xbf16>) -> tensor<1x7x2048x7xbf16>
    %434 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %435 = "ttir.transpose"(%433, %434) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x7x2048x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %436 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %437 = "ttir.multiply"(%435, %arg99, %436) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %438 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %439 = "ttir.add"(%437, %arg100, %438) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %440 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %441 = "ttir.add"(%439, %409, %440) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %442 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %443 = "ttir.relu"(%441, %442) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %444 = ttir.empty() : tensor<1x7x2048x7xbf16>
    %445 = "ttir.transpose"(%443, %444) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x2048x7x7xbf16>, tensor<1x7x2048x7xbf16>) -> tensor<1x7x2048x7xbf16>
    %446 = ttir.empty() : tensor<1x7x7x2048xbf16>
    %447 = "ttir.transpose"(%445, %446) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x7x2048x7xbf16>, tensor<1x7x7x2048xbf16>) -> tensor<1x7x7x2048xbf16>
    %448 = ttir.empty() : tensor<1x7x7x512xbf16>
    %449 = "ttir.conv2d"(%447, %arg157, %448) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x7x7x2048xbf16>, tensor<512x2048x1x1xbf16>, tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %450 = ttir.empty() : tensor<1x7x7x512xbf16>
    %451 = "ttir.multiply"(%449, %arg101, %450) : (tensor<1x7x7x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %452 = ttir.empty() : tensor<1x7x7x512xbf16>
    %453 = "ttir.add"(%451, %arg102, %452) : (tensor<1x7x7x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %454 = ttir.empty() : tensor<1x7x7x512xbf16>
    %455 = "ttir.relu"(%453, %454) : (tensor<1x7x7x512xbf16>, tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %456 = ttir.empty() : tensor<1x7x7x512xbf16>
    %457 = "ttir.conv2d"(%455, %arg158, %456) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x7x7x512xbf16>, tensor<512x512x3x3xbf16>, tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %458 = ttir.empty() : tensor<1x7x7x512xbf16>
    %459 = "ttir.multiply"(%457, %arg103, %458) : (tensor<1x7x7x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %460 = ttir.empty() : tensor<1x7x7x512xbf16>
    %461 = "ttir.add"(%459, %arg104, %460) : (tensor<1x7x7x512xbf16>, tensor<1x1x1x512xbf16>, tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %462 = ttir.empty() : tensor<1x7x7x512xbf16>
    %463 = "ttir.relu"(%461, %462) : (tensor<1x7x7x512xbf16>, tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %464 = ttir.empty() : tensor<1x7x7x2048xbf16>
    %465 = "ttir.conv2d"(%463, %arg159, %464) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x7x7x512xbf16>, tensor<2048x512x1x1xbf16>, tensor<1x7x7x2048xbf16>) -> tensor<1x7x7x2048xbf16>
    %466 = ttir.empty() : tensor<1x7x2048x7xbf16>
    %467 = "ttir.transpose"(%465, %466) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x7x7x2048xbf16>, tensor<1x7x2048x7xbf16>) -> tensor<1x7x2048x7xbf16>
    %468 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %469 = "ttir.transpose"(%467, %468) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x7x2048x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %470 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %471 = "ttir.multiply"(%469, %arg105, %470) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %472 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %473 = "ttir.add"(%471, %arg106, %472) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %474 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %475 = "ttir.add"(%473, %443, %474) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %476 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %477 = "ttir.relu"(%475, %476) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %478 = ttir.empty() : tensor<1x1x2048x49xbf16>
    %479 = "ttir.reshape"(%477, %478) <{shape = [1 : i32, 1 : i32, 2048 : i32, 49 : i32]}> : (tensor<1x2048x7x7xbf16>, tensor<1x1x2048x49xbf16>) -> tensor<1x1x2048x49xbf16>
    %480 = ttir.empty() : tensor<1x1x49x2048xbf16>
    %481 = "ttir.transpose"(%479, %480) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x1x2048x49xbf16>, tensor<1x1x49x2048xbf16>) -> tensor<1x1x49x2048xbf16>
    %482 = ttir.empty() : tensor<1x1x1x2048xbf16>
    %483 = "ttir.mean"(%481, %482) <{dim_arg = [-2 : i32], keep_dim = true}> : (tensor<1x1x49x2048xbf16>, tensor<1x1x1x2048xbf16>) -> tensor<1x1x1x2048xbf16>
    %484 = ttir.empty() : tensor<1x1x2048x1xbf16>
    %485 = "ttir.transpose"(%483, %484) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x1x1x2048xbf16>, tensor<1x1x2048x1xbf16>) -> tensor<1x1x2048x1xbf16>
    %486 = ttir.empty() : tensor<1x2048x1xbf16>
    %487 = "ttir.squeeze"(%485, %486) <{dim = 0 : si32}> : (tensor<1x1x2048x1xbf16>, tensor<1x2048x1xbf16>) -> tensor<1x2048x1xbf16>
    %488 = ttir.empty() : tensor<1x2048xbf16>
    %489 = "ttir.squeeze"(%487, %488) <{dim = -1 : si32}> : (tensor<1x2048x1xbf16>, tensor<1x2048xbf16>) -> tensor<1x2048xbf16>
    %490 = ttir.empty() : tensor<1x1000xbf16>
    %491 = "ttir.matmul"(%489, %arg160, %490) <{transpose_a = false, transpose_b = false}> : (tensor<1x2048xbf16>, tensor<2048x1000xbf16>, tensor<1x1000xbf16>) -> tensor<1x1000xbf16>
    %492 = ttir.empty() : tensor<1x1000xbf16>
    %493 = "ttir.add"(%491, %arg161, %492) : (tensor<1x1000xbf16>, tensor<1000xbf16>, tensor<1x1000xbf16>) -> tensor<1x1000xbf16>
    return %493 : tensor<1x1000xbf16>
  }
}
