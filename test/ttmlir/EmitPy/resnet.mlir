// RUN: ttmlir-opt --ttir-to-emitpy-pipeline -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir

module @ResNetForImageClassification attributes {} {
  func.func @forward(%arg0: tensor<1x3x224x224xbf16> {ttir.name = "pixel_values"}, %arg1: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_2"}, %arg2: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_2_fork_clone1229"}, %arg3: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_19"}, %arg4: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_19_fork_clone1271"}, %arg5: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_35"}, %arg6: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_35_fork_clone1204"}, %arg7: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_51"}, %arg8: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_51_fork_clone1108"}, %arg9: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_66"}, %arg10: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_66_fork_clone1112"}, %arg11: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_83"}, %arg12: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_83_fork_clone1238"}, %arg13: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_99"}, %arg14: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_99_fork_clone1152"}, %arg15: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_115"}, %arg16: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_115_fork_clone1051"}, %arg17: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_132"}, %arg18: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_132_fork_clone1192"}, %arg19: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_148"}, %arg20: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_148_fork_clone1096"}, %arg21: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_164"}, %arg22: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_164_fork_clone992"}, %arg23: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_181"}, %arg24: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_181_fork_clone1065"}, %arg25: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_197"}, %arg26: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_197_fork_clone962"}, %arg27: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_213"}, %arg28: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_213_fork_clone853"}, %arg29: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_228"}, %arg30: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_228_fork_clone857"}, %arg31: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_245"}, %arg32: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_245_fork_clone1007"}, %arg33: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_261"}, %arg34: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_261_fork_clone901"}, %arg35: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_277"}, %arg36: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_277_fork_clone791"}, %arg37: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_294"}, %arg38: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_294_fork_clone950"}, %arg39: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_310"}, %arg40: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_310_fork_clone841"}, %arg41: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_326"}, %arg42: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_326_fork_clone735"}, %arg43: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_343"}, %arg44: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_343_fork_clone889"}, %arg45: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_359"}, %arg46: tensor<1x1x1x128xbf16> {ttir.name = "input_1_add_359_fork_clone779"}, %arg47: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_375"}, %arg48: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_375_fork_clone677"}, %arg49: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_392"}, %arg50: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_392_fork_clone748"}, %arg51: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_408"}, %arg52: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_408_fork_clone645"}, %arg53: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_424"}, %arg54: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_424_fork_clone524"}, %arg55: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_439"}, %arg56: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_439_fork_clone528"}, %arg57: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_456"}, %arg58: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_456_fork_clone692"}, %arg59: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_472"}, %arg60: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_472_fork_clone580"}, %arg61: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_488"}, %arg62: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_488_fork_clone453"}, %arg63: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_505"}, %arg64: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_505_fork_clone633"}, %arg65: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_521"}, %arg66: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_521_fork_clone512"}, %arg67: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_537"}, %arg68: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_537_fork_clone389"}, %arg69: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_554"}, %arg70: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_554_fork_clone568"}, %arg71: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_570"}, %arg72: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_570_fork_clone441"}, %arg73: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_586"}, %arg74: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_586_fork_clone329"}, %arg75: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_603"}, %arg76: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_603_fork_clone500"}, %arg77: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_619"}, %arg78: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_619_fork_clone377"}, %arg79: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_635"}, %arg80: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_635_fork_clone274"}, %arg81: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_652"}, %arg82: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_652_fork_clone429"}, %arg83: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_668"}, %arg84: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_668_fork_clone317"}, %arg85: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_684"}, %arg86: tensor<1x1x1x1024xbf16> {ttir.name = "input_1_add_684_fork_clone219"}, %arg87: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_701"}, %arg88: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_701_fork_clone287"}, %arg89: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_717"}, %arg90: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_717_fork_clone190"}, %arg91: tensor<1x2048x1x1xbf16> {ttir.name = "input_1_add_733"}, %arg92: tensor<1x2048x1x1xbf16> {ttir.name = "input_1_add_733_fork_clone101"}, %arg93: tensor<1x2048x1x1xbf16> {ttir.name = "input_1_add_748"}, %arg94: tensor<1x2048x1x1xbf16> {ttir.name = "input_1_add_748_fork_clone105"}, %arg95: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_765"}, %arg96: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_765_fork_clone233"}, %arg97: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_781"}, %arg98: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_781_fork_clone138"}, %arg99: tensor<1x2048x1x1xbf16> {ttir.name = "input_1_add_797"}, %arg100: tensor<1x2048x1x1xbf16> {ttir.name = "input_1_add_797_fork_clone61"}, %arg101: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_814"}, %arg102: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_814_fork_clone178"}, %arg103: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_830"}, %arg104: tensor<1x1x1x512xbf16> {ttir.name = "input_1_add_830_fork_clone89"}, %arg105: tensor<1x2048x1x1xbf16> {ttir.name = "input_1_add_846"}, %arg106: tensor<1x2048x1x1xbf16> {ttir.name = "input_1_add_846_fork_clone32"}, %arg107: tensor<64x3x7x7xbf16> {ttir.name = "resnet.embedder.embedder.convolution.weight"}, %arg108: tensor<64x64x1x1xbf16> {ttir.name = "resnet.encoder.stages.0.layers.0.layer.0.convolution.weight"}, %arg109: tensor<64x64x3x3xbf16> {ttir.name = "resnet.encoder.stages.0.layers.0.layer.1.convolution.weight"}, %arg110: tensor<256x64x1x1xbf16> {ttir.name = "resnet.encoder.stages.0.layers.0.layer.2.convolution.weight"}, %arg111: tensor<256x64x1x1xbf16> {ttir.name = "resnet.encoder.stages.0.layers.0.shortcut.convolution.weight"}, %arg112: tensor<64x256x1x1xbf16> {ttir.name = "resnet.encoder.stages.0.layers.1.layer.0.convolution.weight"}, %arg113: tensor<64x64x3x3xbf16> {ttir.name = "resnet.encoder.stages.0.layers.1.layer.1.convolution.weight"}, %arg114: tensor<256x64x1x1xbf16> {ttir.name = "resnet.encoder.stages.0.layers.1.layer.2.convolution.weight"}, %arg115: tensor<64x256x1x1xbf16> {ttir.name = "resnet.encoder.stages.0.layers.2.layer.0.convolution.weight"}, %arg116: tensor<64x64x3x3xbf16> {ttir.name = "resnet.encoder.stages.0.layers.2.layer.1.convolution.weight"}, %arg117: tensor<256x64x1x1xbf16> {ttir.name = "resnet.encoder.stages.0.layers.2.layer.2.convolution.weight"}, %arg118: tensor<128x256x1x1xbf16> {ttir.name = "resnet.encoder.stages.1.layers.0.layer.0.convolution.weight"}, %arg119: tensor<128x128x3x3xbf16> {ttir.name = "resnet.encoder.stages.1.layers.0.layer.1.convolution.weight"}, %arg120: tensor<512x128x1x1xbf16> {ttir.name = "resnet.encoder.stages.1.layers.0.layer.2.convolution.weight"}, %arg121: tensor<512x256x1x1xbf16> {ttir.name = "resnet.encoder.stages.1.layers.0.shortcut.convolution.weight"}, %arg122: tensor<128x512x1x1xbf16> {ttir.name = "resnet.encoder.stages.1.layers.1.layer.0.convolution.weight"}, %arg123: tensor<128x128x3x3xbf16> {ttir.name = "resnet.encoder.stages.1.layers.1.layer.1.convolution.weight"}, %arg124: tensor<512x128x1x1xbf16> {ttir.name = "resnet.encoder.stages.1.layers.1.layer.2.convolution.weight"}, %arg125: tensor<128x512x1x1xbf16> {ttir.name = "resnet.encoder.stages.1.layers.2.layer.0.convolution.weight"}, %arg126: tensor<128x128x3x3xbf16> {ttir.name = "resnet.encoder.stages.1.layers.2.layer.1.convolution.weight"}, %arg127: tensor<512x128x1x1xbf16> {ttir.name = "resnet.encoder.stages.1.layers.2.layer.2.convolution.weight"}, %arg128: tensor<128x512x1x1xbf16> {ttir.name = "resnet.encoder.stages.1.layers.3.layer.0.convolution.weight"}, %arg129: tensor<128x128x3x3xbf16> {ttir.name = "resnet.encoder.stages.1.layers.3.layer.1.convolution.weight"}, %arg130: tensor<512x128x1x1xbf16> {ttir.name = "resnet.encoder.stages.1.layers.3.layer.2.convolution.weight"}, %arg131: tensor<256x512x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.0.layer.0.convolution.weight"}, %arg132: tensor<256x256x3x3xbf16> {ttir.name = "resnet.encoder.stages.2.layers.0.layer.1.convolution.weight"}, %arg133: tensor<1024x256x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.0.layer.2.convolution.weight"}, %arg134: tensor<1024x512x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.0.shortcut.convolution.weight"}, %arg135: tensor<256x1024x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.1.layer.0.convolution.weight"}, %arg136: tensor<256x256x3x3xbf16> {ttir.name = "resnet.encoder.stages.2.layers.1.layer.1.convolution.weight"}, %arg137: tensor<1024x256x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.1.layer.2.convolution.weight"}, %arg138: tensor<256x1024x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.2.layer.0.convolution.weight"}, %arg139: tensor<256x256x3x3xbf16> {ttir.name = "resnet.encoder.stages.2.layers.2.layer.1.convolution.weight"}, %arg140: tensor<1024x256x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.2.layer.2.convolution.weight"}, %arg141: tensor<256x1024x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.3.layer.0.convolution.weight"}, %arg142: tensor<256x256x3x3xbf16> {ttir.name = "resnet.encoder.stages.2.layers.3.layer.1.convolution.weight"}, %arg143: tensor<1024x256x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.3.layer.2.convolution.weight"}, %arg144: tensor<256x1024x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.4.layer.0.convolution.weight"}, %arg145: tensor<256x256x3x3xbf16> {ttir.name = "resnet.encoder.stages.2.layers.4.layer.1.convolution.weight"}, %arg146: tensor<1024x256x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.4.layer.2.convolution.weight"}, %arg147: tensor<256x1024x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.5.layer.0.convolution.weight"}, %arg148: tensor<256x256x3x3xbf16> {ttir.name = "resnet.encoder.stages.2.layers.5.layer.1.convolution.weight"}, %arg149: tensor<1024x256x1x1xbf16> {ttir.name = "resnet.encoder.stages.2.layers.5.layer.2.convolution.weight"}, %arg150: tensor<512x1024x1x1xbf16> {ttir.name = "resnet.encoder.stages.3.layers.0.layer.0.convolution.weight"}, %arg151: tensor<512x512x3x3xbf16> {ttir.name = "resnet.encoder.stages.3.layers.0.layer.1.convolution.weight"}, %arg152: tensor<2048x512x1x1xbf16> {ttir.name = "resnet.encoder.stages.3.layers.0.layer.2.convolution.weight"}, %arg153: tensor<2048x1024x1x1xbf16> {ttir.name = "resnet.encoder.stages.3.layers.0.shortcut.convolution.weight"}, %arg154: tensor<512x2048x1x1xbf16> {ttir.name = "resnet.encoder.stages.3.layers.1.layer.0.convolution.weight"}, %arg155: tensor<512x512x3x3xbf16> {ttir.name = "resnet.encoder.stages.3.layers.1.layer.1.convolution.weight"}, %arg156: tensor<2048x512x1x1xbf16> {ttir.name = "resnet.encoder.stages.3.layers.1.layer.2.convolution.weight"}, %arg157: tensor<512x2048x1x1xbf16> {ttir.name = "resnet.encoder.stages.3.layers.2.layer.0.convolution.weight"}, %arg158: tensor<512x512x3x3xbf16> {ttir.name = "resnet.encoder.stages.3.layers.2.layer.1.convolution.weight"}, %arg159: tensor<2048x512x1x1xbf16> {ttir.name = "resnet.encoder.stages.3.layers.2.layer.2.convolution.weight"}, %arg160: tensor<2048x1000xbf16> {ttir.name = "classifier.1.weight"}, %arg161: tensor<1000xbf16> {ttir.name = "classifier.1.bias"}) -> (tensor<1x1000xbf16> {ttir.name = "ResNetForImageClassification.output_add_868"}) {
    %0 = "ttir.transpose"(%arg0) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x3x224x224xbf16>) -> tensor<1x224x3x224xbf16>
    %1 = "ttir.transpose"(%0) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x224x3x224xbf16>) -> tensor<1x224x224x3xbf16>
    %2 = "ttir.conv2d"(%1, %arg107) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 3, 3, 3, 3>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x224x224x3xbf16>, tensor<64x3x7x7xbf16>) -> tensor<1x112x112x64xbf16>
    %3 = "ttir.multiply"(%2, %arg1) : (tensor<1x112x112x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x112x112x64xbf16>
    %4 = "ttir.add"(%3, %arg2) : (tensor<1x112x112x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x112x112x64xbf16>
    %5 = "ttir.relu"(%4) : (tensor<1x112x112x64xbf16>) -> tensor<1x112x112x64xbf16>
    %6 = "ttir.max_pool2d"(%5) <{kernel = array<i32: 3, 3>, stride = array<i32: 2, 2>, dilation = array<i32: 1, 1>, padding = array<i32: 1, 1, 1, 1>, ceil_mode = false}> {channel_last = true} : (tensor<1x112x112x64xbf16>) -> tensor<1x56x56x64xbf16>
    %7 = "ttir.conv2d"(%6, %arg108) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x64xbf16>, tensor<64x64x1x1xbf16>) -> tensor<1x56x56x64xbf16>
    %8 = "ttir.multiply"(%7, %arg3) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x56x56x64xbf16>
    %9 = "ttir.add"(%8, %arg4) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x56x56x64xbf16>
    %10 = "ttir.relu"(%9) : (tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %11 = "ttir.conv2d"(%10, %arg109) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x64xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x56x56x64xbf16>
    %12 = "ttir.multiply"(%11, %arg5) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x56x56x64xbf16>
    %13 = "ttir.add"(%12, %arg6) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x56x56x64xbf16>
    %14 = "ttir.relu"(%13) : (tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %15 = "ttir.conv2d"(%14, %arg110) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x64xbf16>, tensor<256x64x1x1xbf16>) -> tensor<1x56x56x256xbf16>
    %16 = "ttir.multiply"(%15, %arg7) : (tensor<1x56x56x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x56x56x256xbf16>
    %17 = "ttir.add"(%16, %arg8) : (tensor<1x56x56x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x56x56x256xbf16>
    %18 = "ttir.conv2d"(%6, %arg111) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x64xbf16>, tensor<256x64x1x1xbf16>) -> tensor<1x56x56x256xbf16>
    %19 = "ttir.multiply"(%18, %arg9) : (tensor<1x56x56x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x56x56x256xbf16>
    %20 = "ttir.add"(%19, %arg10) : (tensor<1x56x56x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x56x56x256xbf16>
    %21 = "ttir.add"(%17, %20) : (tensor<1x56x56x256xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %22 = "ttir.relu"(%21) : (tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %23 = "ttir.conv2d"(%22, %arg112) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x256xbf16>, tensor<64x256x1x1xbf16>) -> tensor<1x56x56x64xbf16>
    %24 = "ttir.multiply"(%23, %arg11) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x56x56x64xbf16>
    %25 = "ttir.add"(%24, %arg12) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x56x56x64xbf16>
    %26 = "ttir.relu"(%25) : (tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %27 = "ttir.conv2d"(%26, %arg113) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x64xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x56x56x64xbf16>
    %28 = "ttir.multiply"(%27, %arg13) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x56x56x64xbf16>
    %29 = "ttir.add"(%28, %arg14) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x56x56x64xbf16>
    %30 = "ttir.relu"(%29) : (tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %31 = "ttir.conv2d"(%30, %arg114) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x64xbf16>, tensor<256x64x1x1xbf16>) -> tensor<1x56x56x256xbf16>
    %32 = "ttir.multiply"(%31, %arg15) : (tensor<1x56x56x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x56x56x256xbf16>
    %33 = "ttir.add"(%32, %arg16) : (tensor<1x56x56x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x56x56x256xbf16>
    %34 = "ttir.add"(%33, %22) : (tensor<1x56x56x256xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %35 = "ttir.relu"(%34) : (tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %36 = "ttir.conv2d"(%35, %arg115) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x256xbf16>, tensor<64x256x1x1xbf16>) -> tensor<1x56x56x64xbf16>
    %37 = "ttir.multiply"(%36, %arg17) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x56x56x64xbf16>
    %38 = "ttir.add"(%37, %arg18) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x56x56x64xbf16>
    %39 = "ttir.relu"(%38) : (tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %40 = "ttir.conv2d"(%39, %arg116) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x64xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x56x56x64xbf16>
    %41 = "ttir.multiply"(%40, %arg19) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x56x56x64xbf16>
    %42 = "ttir.add"(%41, %arg20) : (tensor<1x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x56x56x64xbf16>
    %43 = "ttir.relu"(%42) : (tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %44 = "ttir.conv2d"(%43, %arg117) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x64xbf16>, tensor<256x64x1x1xbf16>) -> tensor<1x56x56x256xbf16>
    %45 = "ttir.multiply"(%44, %arg21) : (tensor<1x56x56x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x56x56x256xbf16>
    %46 = "ttir.add"(%45, %arg22) : (tensor<1x56x56x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x56x56x256xbf16>
    %47 = "ttir.add"(%46, %35) : (tensor<1x56x56x256xbf16>, tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %48 = "ttir.relu"(%47) : (tensor<1x56x56x256xbf16>) -> tensor<1x56x56x256xbf16>
    %49 = "ttir.conv2d"(%48, %arg118) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x256xbf16>, tensor<128x256x1x1xbf16>) -> tensor<1x56x56x128xbf16>
    %50 = "ttir.multiply"(%49, %arg23) : (tensor<1x56x56x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x56x56x128xbf16>
    %51 = "ttir.add"(%50, %arg24) : (tensor<1x56x56x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x56x56x128xbf16>
    %52 = "ttir.relu"(%51) : (tensor<1x56x56x128xbf16>) -> tensor<1x56x56x128xbf16>
    %53 = "ttir.conv2d"(%52, %arg119) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x56x56x128xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x28x28x128xbf16>
    %54 = "ttir.multiply"(%53, %arg25) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x28x28x128xbf16>
    %55 = "ttir.add"(%54, %arg26) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x28x28x128xbf16>
    %56 = "ttir.relu"(%55) : (tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %57 = "ttir.conv2d"(%56, %arg120) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x128xbf16>, tensor<512x128x1x1xbf16>) -> tensor<1x28x28x512xbf16>
    %58 = "ttir.multiply"(%57, %arg27) : (tensor<1x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x28x28x512xbf16>
    %59 = "ttir.add"(%58, %arg28) : (tensor<1x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x28x28x512xbf16>
    %60 = "ttir.conv2d"(%48, %arg121) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x56x56x256xbf16>, tensor<512x256x1x1xbf16>) -> tensor<1x28x28x512xbf16>
    %61 = "ttir.multiply"(%60, %arg29) : (tensor<1x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x28x28x512xbf16>
    %62 = "ttir.add"(%61, %arg30) : (tensor<1x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x28x28x512xbf16>
    %63 = "ttir.add"(%59, %62) : (tensor<1x28x28x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %64 = "ttir.relu"(%63) : (tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %65 = "ttir.conv2d"(%64, %arg122) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x512xbf16>, tensor<128x512x1x1xbf16>) -> tensor<1x28x28x128xbf16>
    %66 = "ttir.multiply"(%65, %arg31) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x28x28x128xbf16>
    %67 = "ttir.add"(%66, %arg32) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x28x28x128xbf16>
    %68 = "ttir.relu"(%67) : (tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %69 = "ttir.conv2d"(%68, %arg123) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x128xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x28x28x128xbf16>
    %70 = "ttir.multiply"(%69, %arg33) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x28x28x128xbf16>
    %71 = "ttir.add"(%70, %arg34) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x28x28x128xbf16>
    %72 = "ttir.relu"(%71) : (tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %73 = "ttir.conv2d"(%72, %arg124) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x128xbf16>, tensor<512x128x1x1xbf16>) -> tensor<1x28x28x512xbf16>
    %74 = "ttir.multiply"(%73, %arg35) : (tensor<1x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x28x28x512xbf16>
    %75 = "ttir.add"(%74, %arg36) : (tensor<1x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x28x28x512xbf16>
    %76 = "ttir.add"(%75, %64) : (tensor<1x28x28x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %77 = "ttir.relu"(%76) : (tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %78 = "ttir.conv2d"(%77, %arg125) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x512xbf16>, tensor<128x512x1x1xbf16>) -> tensor<1x28x28x128xbf16>
    %79 = "ttir.multiply"(%78, %arg37) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x28x28x128xbf16>
    %80 = "ttir.add"(%79, %arg38) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x28x28x128xbf16>
    %81 = "ttir.relu"(%80) : (tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %82 = "ttir.conv2d"(%81, %arg126) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x128xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x28x28x128xbf16>
    %83 = "ttir.multiply"(%82, %arg39) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x28x28x128xbf16>
    %84 = "ttir.add"(%83, %arg40) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x28x28x128xbf16>
    %85 = "ttir.relu"(%84) : (tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %86 = "ttir.conv2d"(%85, %arg127) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x128xbf16>, tensor<512x128x1x1xbf16>) -> tensor<1x28x28x512xbf16>
    %87 = "ttir.multiply"(%86, %arg41) : (tensor<1x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x28x28x512xbf16>
    %88 = "ttir.add"(%87, %arg42) : (tensor<1x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x28x28x512xbf16>
    %89 = "ttir.add"(%88, %77) : (tensor<1x28x28x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %90 = "ttir.relu"(%89) : (tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %91 = "ttir.conv2d"(%90, %arg128) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x512xbf16>, tensor<128x512x1x1xbf16>) -> tensor<1x28x28x128xbf16>
    %92 = "ttir.multiply"(%91, %arg43) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x28x28x128xbf16>
    %93 = "ttir.add"(%92, %arg44) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x28x28x128xbf16>
    %94 = "ttir.relu"(%93) : (tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %95 = "ttir.conv2d"(%94, %arg129) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x128xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x28x28x128xbf16>
    %96 = "ttir.multiply"(%95, %arg45) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x28x28x128xbf16>
    %97 = "ttir.add"(%96, %arg46) : (tensor<1x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x28x28x128xbf16>
    %98 = "ttir.relu"(%97) : (tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %99 = "ttir.conv2d"(%98, %arg130) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x128xbf16>, tensor<512x128x1x1xbf16>) -> tensor<1x28x28x512xbf16>
    %100 = "ttir.multiply"(%99, %arg47) : (tensor<1x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x28x28x512xbf16>
    %101 = "ttir.add"(%100, %arg48) : (tensor<1x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x28x28x512xbf16>
    %102 = "ttir.add"(%101, %90) : (tensor<1x28x28x512xbf16>, tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %103 = "ttir.relu"(%102) : (tensor<1x28x28x512xbf16>) -> tensor<1x28x28x512xbf16>
    %104 = "ttir.conv2d"(%103, %arg131) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x28x28x512xbf16>, tensor<256x512x1x1xbf16>) -> tensor<1x28x28x256xbf16>
    %105 = "ttir.multiply"(%104, %arg49) : (tensor<1x28x28x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x28x28x256xbf16>
    %106 = "ttir.add"(%105, %arg50) : (tensor<1x28x28x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x28x28x256xbf16>
    %107 = "ttir.relu"(%106) : (tensor<1x28x28x256xbf16>) -> tensor<1x28x28x256xbf16>
    %108 = "ttir.conv2d"(%107, %arg132) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x28x28x256xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x14x14x256xbf16>
    %109 = "ttir.multiply"(%108, %arg51) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %110 = "ttir.add"(%109, %arg52) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %111 = "ttir.relu"(%110) : (tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %112 = "ttir.conv2d"(%111, %arg133) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<1x14x14x1024xbf16>
    %113 = "ttir.multiply"(%112, %arg53) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %114 = "ttir.add"(%113, %arg54) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %115 = "ttir.conv2d"(%103, %arg134) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x28x28x512xbf16>, tensor<1024x512x1x1xbf16>) -> tensor<1x14x14x1024xbf16>
    %116 = "ttir.multiply"(%115, %arg55) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %117 = "ttir.add"(%116, %arg56) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %118 = "ttir.add"(%114, %117) : (tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %119 = "ttir.relu"(%118) : (tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %120 = "ttir.conv2d"(%119, %arg135) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<1x14x14x256xbf16>
    %121 = "ttir.multiply"(%120, %arg57) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %122 = "ttir.add"(%121, %arg58) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %123 = "ttir.relu"(%122) : (tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %124 = "ttir.conv2d"(%123, %arg136) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x14x14x256xbf16>
    %125 = "ttir.multiply"(%124, %arg59) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %126 = "ttir.add"(%125, %arg60) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %127 = "ttir.relu"(%126) : (tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %128 = "ttir.conv2d"(%127, %arg137) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<1x14x14x1024xbf16>
    %129 = "ttir.multiply"(%128, %arg61) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %130 = "ttir.add"(%129, %arg62) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %131 = "ttir.add"(%130, %119) : (tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %132 = "ttir.relu"(%131) : (tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %133 = "ttir.conv2d"(%132, %arg138) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<1x14x14x256xbf16>
    %134 = "ttir.multiply"(%133, %arg63) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %135 = "ttir.add"(%134, %arg64) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %136 = "ttir.relu"(%135) : (tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %137 = "ttir.conv2d"(%136, %arg139) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x14x14x256xbf16>
    %138 = "ttir.multiply"(%137, %arg65) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %139 = "ttir.add"(%138, %arg66) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %140 = "ttir.relu"(%139) : (tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %141 = "ttir.conv2d"(%140, %arg140) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<1x14x14x1024xbf16>
    %142 = "ttir.multiply"(%141, %arg67) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %143 = "ttir.add"(%142, %arg68) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %144 = "ttir.add"(%143, %132) : (tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %145 = "ttir.relu"(%144) : (tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %146 = "ttir.conv2d"(%145, %arg141) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<1x14x14x256xbf16>
    %147 = "ttir.multiply"(%146, %arg69) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %148 = "ttir.add"(%147, %arg70) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %149 = "ttir.relu"(%148) : (tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %150 = "ttir.conv2d"(%149, %arg142) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x14x14x256xbf16>
    %151 = "ttir.multiply"(%150, %arg71) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %152 = "ttir.add"(%151, %arg72) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %153 = "ttir.relu"(%152) : (tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %154 = "ttir.conv2d"(%153, %arg143) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<1x14x14x1024xbf16>
    %155 = "ttir.multiply"(%154, %arg73) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %156 = "ttir.add"(%155, %arg74) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %157 = "ttir.add"(%156, %145) : (tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %158 = "ttir.relu"(%157) : (tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %159 = "ttir.conv2d"(%158, %arg144) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<1x14x14x256xbf16>
    %160 = "ttir.multiply"(%159, %arg75) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %161 = "ttir.add"(%160, %arg76) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %162 = "ttir.relu"(%161) : (tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %163 = "ttir.conv2d"(%162, %arg145) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x14x14x256xbf16>
    %164 = "ttir.multiply"(%163, %arg77) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %165 = "ttir.add"(%164, %arg78) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %166 = "ttir.relu"(%165) : (tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %167 = "ttir.conv2d"(%166, %arg146) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<1x14x14x1024xbf16>
    %168 = "ttir.multiply"(%167, %arg79) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %169 = "ttir.add"(%168, %arg80) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %170 = "ttir.add"(%169, %158) : (tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %171 = "ttir.relu"(%170) : (tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %172 = "ttir.conv2d"(%171, %arg147) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<1x14x14x256xbf16>
    %173 = "ttir.multiply"(%172, %arg81) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %174 = "ttir.add"(%173, %arg82) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %175 = "ttir.relu"(%174) : (tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %176 = "ttir.conv2d"(%175, %arg148) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<256x256x3x3xbf16>) -> tensor<1x14x14x256xbf16>
    %177 = "ttir.multiply"(%176, %arg83) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %178 = "ttir.add"(%177, %arg84) : (tensor<1x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x14x14x256xbf16>
    %179 = "ttir.relu"(%178) : (tensor<1x14x14x256xbf16>) -> tensor<1x14x14x256xbf16>
    %180 = "ttir.conv2d"(%179, %arg149) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x256xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<1x14x14x1024xbf16>
    %181 = "ttir.multiply"(%180, %arg85) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %182 = "ttir.add"(%181, %arg86) : (tensor<1x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %183 = "ttir.add"(%182, %171) : (tensor<1x14x14x1024xbf16>, tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %184 = "ttir.relu"(%183) : (tensor<1x14x14x1024xbf16>) -> tensor<1x14x14x1024xbf16>
    %185 = "ttir.conv2d"(%184, %arg150) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xbf16>, tensor<512x1024x1x1xbf16>) -> tensor<1x14x14x512xbf16>
    %186 = "ttir.multiply"(%185, %arg87) : (tensor<1x14x14x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x14x14x512xbf16>
    %187 = "ttir.add"(%186, %arg88) : (tensor<1x14x14x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x14x14x512xbf16>
    %188 = "ttir.relu"(%187) : (tensor<1x14x14x512xbf16>) -> tensor<1x14x14x512xbf16>
    %189 = "ttir.conv2d"(%188, %arg151) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x14x14x512xbf16>, tensor<512x512x3x3xbf16>) -> tensor<1x7x7x512xbf16>
    %190 = "ttir.multiply"(%189, %arg89) : (tensor<1x7x7x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x7x7x512xbf16>
    %191 = "ttir.add"(%190, %arg90) : (tensor<1x7x7x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x7x7x512xbf16>
    %192 = "ttir.relu"(%191) : (tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %193 = "ttir.conv2d"(%192, %arg152) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x7x7x512xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<1x7x7x2048xbf16>
    %194 = "ttir.transpose"(%193) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x7x7x2048xbf16>) -> tensor<1x7x2048x7xbf16>
    %195 = "ttir.transpose"(%194) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x7x2048x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %196 = "ttir.multiply"(%195, %arg91) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x7x7xbf16>
    %197 = "ttir.add"(%196, %arg92) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x7x7xbf16>
    %198 = "ttir.conv2d"(%184, %arg153) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x14x14x1024xbf16>, tensor<2048x1024x1x1xbf16>) -> tensor<1x7x7x2048xbf16>
    %199 = "ttir.transpose"(%198) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x7x7x2048xbf16>) -> tensor<1x7x2048x7xbf16>
    %200 = "ttir.transpose"(%199) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x7x2048x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %201 = "ttir.multiply"(%200, %arg93) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x7x7xbf16>
    %202 = "ttir.add"(%201, %arg94) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x7x7xbf16>
    %203 = "ttir.add"(%197, %202) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %204 = "ttir.relu"(%203) : (tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %205 = "ttir.transpose"(%204) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x2048x7x7xbf16>) -> tensor<1x7x2048x7xbf16>
    %206 = "ttir.transpose"(%205) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x7x2048x7xbf16>) -> tensor<1x7x7x2048xbf16>
    %207 = "ttir.conv2d"(%206, %arg154) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x7x7x2048xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<1x7x7x512xbf16>
    %208 = "ttir.multiply"(%207, %arg95) : (tensor<1x7x7x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x7x7x512xbf16>
    %209 = "ttir.add"(%208, %arg96) : (tensor<1x7x7x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x7x7x512xbf16>
    %210 = "ttir.relu"(%209) : (tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %211 = "ttir.conv2d"(%210, %arg155) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x7x7x512xbf16>, tensor<512x512x3x3xbf16>) -> tensor<1x7x7x512xbf16>
    %212 = "ttir.multiply"(%211, %arg97) : (tensor<1x7x7x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x7x7x512xbf16>
    %213 = "ttir.add"(%212, %arg98) : (tensor<1x7x7x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x7x7x512xbf16>
    %214 = "ttir.relu"(%213) : (tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %215 = "ttir.conv2d"(%214, %arg156) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x7x7x512xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<1x7x7x2048xbf16>
    %216 = "ttir.transpose"(%215) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x7x7x2048xbf16>) -> tensor<1x7x2048x7xbf16>
    %217 = "ttir.transpose"(%216) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x7x2048x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %218 = "ttir.multiply"(%217, %arg99) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x7x7xbf16>
    %219 = "ttir.add"(%218, %arg100) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x7x7xbf16>
    %220 = "ttir.add"(%219, %204) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %221 = "ttir.relu"(%220) : (tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %222 = "ttir.transpose"(%221) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x2048x7x7xbf16>) -> tensor<1x7x2048x7xbf16>
    %223 = "ttir.transpose"(%222) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x7x2048x7xbf16>) -> tensor<1x7x7x2048xbf16>
    %224 = "ttir.conv2d"(%223, %arg157) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x7x7x2048xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<1x7x7x512xbf16>
    %225 = "ttir.multiply"(%224, %arg101) : (tensor<1x7x7x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x7x7x512xbf16>
    %226 = "ttir.add"(%225, %arg102) : (tensor<1x7x7x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x7x7x512xbf16>
    %227 = "ttir.relu"(%226) : (tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %228 = "ttir.conv2d"(%227, %arg158) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x7x7x512xbf16>, tensor<512x512x3x3xbf16>) -> tensor<1x7x7x512xbf16>
    %229 = "ttir.multiply"(%228, %arg103) : (tensor<1x7x7x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x7x7x512xbf16>
    %230 = "ttir.add"(%229, %arg104) : (tensor<1x7x7x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<1x7x7x512xbf16>
    %231 = "ttir.relu"(%230) : (tensor<1x7x7x512xbf16>) -> tensor<1x7x7x512xbf16>
    %232 = "ttir.conv2d"(%231, %arg159) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x7x7x512xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<1x7x7x2048xbf16>
    %233 = "ttir.transpose"(%232) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x7x7x2048xbf16>) -> tensor<1x7x2048x7xbf16>
    %234 = "ttir.transpose"(%233) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x7x2048x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %235 = "ttir.multiply"(%234, %arg105) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x7x7xbf16>
    %236 = "ttir.add"(%235, %arg106) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x7x7xbf16>
    %237 = "ttir.add"(%236, %221) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %238 = "ttir.relu"(%237) : (tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %239 = "ttir.reshape"(%238) <{shape = [1 : i32, 1 : i32, 2048 : i32, 49 : i32]}> : (tensor<1x2048x7x7xbf16>) -> tensor<1x1x2048x49xbf16>
    %240 = "ttir.transpose"(%239) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x1x2048x49xbf16>) -> tensor<1x1x49x2048xbf16>
    %241 = "ttir.mean"(%240) <{dim_arg = [-2 : i32], keep_dim = true}> : (tensor<1x1x49x2048xbf16>) -> tensor<1x1x1x2048xbf16>
    %242 = "ttir.transpose"(%241) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x1x1x2048xbf16>) -> tensor<1x1x2048x1xbf16>
    %243 = "ttir.squeeze"(%242) <{dim = 0 : si32}> : (tensor<1x1x2048x1xbf16>) -> tensor<1x2048x1xbf16>
    %244 = "ttir.squeeze"(%243) <{dim = -1 : si32}> : (tensor<1x2048x1xbf16>) -> tensor<1x2048xbf16>
    %245 = "ttir.matmul"(%244, %arg160) <{transpose_a = false, transpose_b = false}> : (tensor<1x2048xbf16>, tensor<2048x1000xbf16>) -> tensor<1x1000xbf16>
    %246 = "ttir.add"(%245, %arg161) : (tensor<1x1000xbf16>, tensor<1000xbf16>) -> tensor<1x1000xbf16>
    return %246 : tensor<1x1000xbf16>
  }
}
