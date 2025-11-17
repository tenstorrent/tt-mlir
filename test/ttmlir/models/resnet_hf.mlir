#loc = loc("ResNetForImageClassification":0:0)
module @ResNetForImageClassification {
  func.func @forward(%arg0: tensor<8x3x224x224xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "pixel_values"} loc("ResNetForImageClassification":0:0), %arg1: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_3"} loc("ResNetForImageClassification":0:0), %arg2: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_3_fork_clone1184"} loc("ResNetForImageClassification":0:0), %arg3: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_19"} loc("ResNetForImageClassification":0:0), %arg4: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_19_fork_clone1224"} loc("ResNetForImageClassification":0:0), %arg5: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_34"} loc("ResNetForImageClassification":0:0), %arg6: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_34_fork_clone1161"} loc("ResNetForImageClassification":0:0), %arg7: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_49"} loc("ResNetForImageClassification":0:0), %arg8: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_49_fork_clone1069"} loc("ResNetForImageClassification":0:0), %arg9: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_63"} loc("ResNetForImageClassification":0:0), %arg10: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_63_fork_clone1073"} loc("ResNetForImageClassification":0:0), %arg11: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_79"} loc("ResNetForImageClassification":0:0), %arg12: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_79_fork_clone1191"} loc("ResNetForImageClassification":0:0), %arg13: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_94"} loc("ResNetForImageClassification":0:0), %arg14: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_94_fork_clone1110"} loc("ResNetForImageClassification":0:0), %arg15: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_109"} loc("ResNetForImageClassification":0:0), %arg16: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_109_fork_clone1013"} loc("ResNetForImageClassification":0:0), %arg17: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_125"} loc("ResNetForImageClassification":0:0), %arg18: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_125_fork_clone1148"} loc("ResNetForImageClassification":0:0), %arg19: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_140"} loc("ResNetForImageClassification":0:0), %arg20: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_140_fork_clone1056"} loc("ResNetForImageClassification":0:0), %arg21: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_155"} loc("ResNetForImageClassification":0:0), %arg22: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_155_fork_clone958"} loc("ResNetForImageClassification":0:0), %arg23: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_171"} loc("ResNetForImageClassification":0:0), %arg24: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_171_fork_clone1028"} loc("ResNetForImageClassification":0:0), %arg25: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_186"} loc("ResNetForImageClassification":0:0), %arg26: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_186_fork_clone930"} loc("ResNetForImageClassification":0:0), %arg27: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_201"} loc("ResNetForImageClassification":0:0), %arg28: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_201_fork_clone825"} loc("ResNetForImageClassification":0:0), %arg29: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_215"} loc("ResNetForImageClassification":0:0), %arg30: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_215_fork_clone829"} loc("ResNetForImageClassification":0:0), %arg31: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_231"} loc("ResNetForImageClassification":0:0), %arg32: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_231_fork_clone972"} loc("ResNetForImageClassification":0:0), %arg33: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_246"} loc("ResNetForImageClassification":0:0), %arg34: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_246_fork_clone871"} loc("ResNetForImageClassification":0:0), %arg35: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_261"} loc("ResNetForImageClassification":0:0), %arg36: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_261_fork_clone765"} loc("ResNetForImageClassification":0:0), %arg37: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_277"} loc("ResNetForImageClassification":0:0), %arg38: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_277_fork_clone917"} loc("ResNetForImageClassification":0:0), %arg39: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_292"} loc("ResNetForImageClassification":0:0), %arg40: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_292_fork_clone812"} loc("ResNetForImageClassification":0:0), %arg41: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_307"} loc("ResNetForImageClassification":0:0), %arg42: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_307_fork_clone710"} loc("ResNetForImageClassification":0:0), %arg43: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_323"} loc("ResNetForImageClassification":0:0), %arg44: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_323_fork_clone858"} loc("ResNetForImageClassification":0:0), %arg45: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_338"} loc("ResNetForImageClassification":0:0), %arg46: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_338_fork_clone752"} loc("ResNetForImageClassification":0:0), %arg47: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_353"} loc("ResNetForImageClassification":0:0), %arg48: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_353_fork_clone656"} loc("ResNetForImageClassification":0:0), %arg49: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_369"} loc("ResNetForImageClassification":0:0), %arg50: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_369_fork_clone724"} loc("ResNetForImageClassification":0:0), %arg51: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_384"} loc("ResNetForImageClassification":0:0), %arg52: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_384_fork_clone628"} loc("ResNetForImageClassification":0:0), %arg53: tensor<1x1x1x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_399"} loc("ResNetForImageClassification":0:0), %arg54: tensor<1x1x1x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_399_fork_clone512"} loc("ResNetForImageClassification":0:0), %arg55: tensor<1x1x1x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_413"} loc("ResNetForImageClassification":0:0), %arg56: tensor<1x1x1x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_413_fork_clone516"} loc("ResNetForImageClassification":0:0), %arg57: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_429"} loc("ResNetForImageClassification":0:0), %arg58: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_429_fork_clone670"} loc("ResNetForImageClassification":0:0), %arg59: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_444"} loc("ResNetForImageClassification":0:0), %arg60: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_444_fork_clone566"} loc("ResNetForImageClassification":0:0), %arg61: tensor<1x1x1x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_459"} loc("ResNetForImageClassification":0:0), %arg62: tensor<1x1x1x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_459_fork_clone443"} loc("ResNetForImageClassification":0:0), %arg63: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_475"} loc("ResNetForImageClassification":0:0), %arg64: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_475_fork_clone615"} loc("ResNetForImageClassification":0:0), %arg65: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_490"} loc("ResNetForImageClassification":0:0), %arg66: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_490_fork_clone499"} loc("ResNetForImageClassification":0:0), %arg67: tensor<1x1x1x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_505"} loc("ResNetForImageClassification":0:0), %arg68: tensor<1x1x1x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_505_fork_clone380"} loc("ResNetForImageClassification":0:0), %arg69: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_521"} loc("ResNetForImageClassification":0:0), %arg70: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_521_fork_clone553"} loc("ResNetForImageClassification":0:0), %arg71: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_536"} loc("ResNetForImageClassification":0:0), %arg72: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_536_fork_clone430"} loc("ResNetForImageClassification":0:0), %arg73: tensor<1x1x1x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_551"} loc("ResNetForImageClassification":0:0), %arg74: tensor<1x1x1x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_551_fork_clone322"} loc("ResNetForImageClassification":0:0), %arg75: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_567"} loc("ResNetForImageClassification":0:0), %arg76: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_567_fork_clone486"} loc("ResNetForImageClassification":0:0), %arg77: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_582"} loc("ResNetForImageClassification":0:0), %arg78: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_582_fork_clone367"} loc("ResNetForImageClassification":0:0), %arg79: tensor<1x1x1x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_597"} loc("ResNetForImageClassification":0:0), %arg80: tensor<1x1x1x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_597_fork_clone268"} loc("ResNetForImageClassification":0:0), %arg81: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_613"} loc("ResNetForImageClassification":0:0), %arg82: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_613_fork_clone417"} loc("ResNetForImageClassification":0:0), %arg83: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_628"} loc("ResNetForImageClassification":0:0), %arg84: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_628_fork_clone309"} loc("ResNetForImageClassification":0:0), %arg85: tensor<1x1x1x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_643"} loc("ResNetForImageClassification":0:0), %arg86: tensor<1x1x1x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_643_fork_clone216"} loc("ResNetForImageClassification":0:0), %arg87: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_659"} loc("ResNetForImageClassification":0:0), %arg88: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_659_fork_clone282"} loc("ResNetForImageClassification":0:0), %arg89: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_674"} loc("ResNetForImageClassification":0:0), %arg90: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_674_fork_clone189"} loc("ResNetForImageClassification":0:0), %arg91: tensor<1x2048x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_689"} loc("ResNetForImageClassification":0:0), %arg92: tensor<1x2048x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_689_fork_clone102"} loc("ResNetForImageClassification":0:0), %arg93: tensor<1x2048x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_703"} loc("ResNetForImageClassification":0:0), %arg94: tensor<1x2048x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_703_fork_clone106"} loc("ResNetForImageClassification":0:0), %arg95: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_719"} loc("ResNetForImageClassification":0:0), %arg96: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_719_fork_clone228"} loc("ResNetForImageClassification":0:0), %arg97: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_734"} loc("ResNetForImageClassification":0:0), %arg98: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_734_fork_clone137"} loc("ResNetForImageClassification":0:0), %arg99: tensor<1x2048x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_749"} loc("ResNetForImageClassification":0:0), %arg100: tensor<1x2048x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_749_fork_clone61"} loc("ResNetForImageClassification":0:0), %arg101: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_765"} loc("ResNetForImageClassification":0:0), %arg102: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_765_fork_clone176"} loc("ResNetForImageClassification":0:0), %arg103: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_780"} loc("ResNetForImageClassification":0:0), %arg104: tensor<1x1x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_780_fork_clone89"} loc("ResNetForImageClassification":0:0), %arg105: tensor<1x2048x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_795"} loc("ResNetForImageClassification":0:0), %arg106: tensor<1x2048x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_795_fork_clone32"} loc("ResNetForImageClassification":0:0), %arg107: tensor<64x3x7x7xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.embedder.embedder.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg108: tensor<64x64x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.0.layers.0.layer.0.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg109: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.0.layers.0.layer.1.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg110: tensor<256x64x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.0.layers.0.layer.2.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg111: tensor<256x64x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.0.layers.0.shortcut.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg112: tensor<64x256x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.0.layers.1.layer.0.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg113: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.0.layers.1.layer.1.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg114: tensor<256x64x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.0.layers.1.layer.2.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg115: tensor<64x256x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.0.layers.2.layer.0.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg116: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.0.layers.2.layer.1.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg117: tensor<256x64x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.0.layers.2.layer.2.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg118: tensor<128x256x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.1.layers.0.layer.0.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg119: tensor<128x128x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.1.layers.0.layer.1.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg120: tensor<512x128x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.1.layers.0.layer.2.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg121: tensor<512x256x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.1.layers.0.shortcut.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg122: tensor<128x512x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.1.layers.1.layer.0.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg123: tensor<128x128x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.1.layers.1.layer.1.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg124: tensor<512x128x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.1.layers.1.layer.2.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg125: tensor<128x512x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.1.layers.2.layer.0.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg126: tensor<128x128x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.1.layers.2.layer.1.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg127: tensor<512x128x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.1.layers.2.layer.2.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg128: tensor<128x512x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.1.layers.3.layer.0.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg129: tensor<128x128x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.1.layers.3.layer.1.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg130: tensor<512x128x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.1.layers.3.layer.2.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg131: tensor<256x512x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.2.layers.0.layer.0.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg132: tensor<256x256x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.2.layers.0.layer.1.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg133: tensor<1024x256x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.2.layers.0.layer.2.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg134: tensor<1024x512x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.2.layers.0.shortcut.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg135: tensor<256x1024x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.2.layers.1.layer.0.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg136: tensor<256x256x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.2.layers.1.layer.1.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg137: tensor<1024x256x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.2.layers.1.layer.2.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg138: tensor<256x1024x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.2.layers.2.layer.0.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg139: tensor<256x256x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.2.layers.2.layer.1.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg140: tensor<1024x256x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.2.layers.2.layer.2.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg141: tensor<256x1024x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.2.layers.3.layer.0.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg142: tensor<256x256x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.2.layers.3.layer.1.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg143: tensor<1024x256x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.2.layers.3.layer.2.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg144: tensor<256x1024x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.2.layers.4.layer.0.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg145: tensor<256x256x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.2.layers.4.layer.1.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg146: tensor<1024x256x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.2.layers.4.layer.2.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg147: tensor<256x1024x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.2.layers.5.layer.0.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg148: tensor<256x256x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.2.layers.5.layer.1.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg149: tensor<1024x256x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.2.layers.5.layer.2.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg150: tensor<512x1024x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.3.layers.0.layer.0.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg151: tensor<512x512x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.3.layers.0.layer.1.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg152: tensor<2048x512x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.3.layers.0.layer.2.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg153: tensor<2048x1024x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.3.layers.0.shortcut.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg154: tensor<512x2048x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.3.layers.1.layer.0.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg155: tensor<512x512x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.3.layers.1.layer.1.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg156: tensor<2048x512x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.3.layers.1.layer.2.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg157: tensor<512x2048x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.3.layers.2.layer.0.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg158: tensor<512x512x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.3.layers.2.layer.1.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg159: tensor<2048x512x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "resnet.encoder.stages.3.layers.2.layer.2.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg160: tensor<2048x1000xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "classifier.1.weight"} loc("ResNetForImageClassification":0:0), %arg161: tensor<1000xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "classifier.1.bias"} loc("ResNetForImageClassification":0:0)) -> (tensor<8x1000xbf16> {ttir.name = "ResNetForImageClassification.output_add_814"}) {
    %0 = "ttir.transpose"(%arg0) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<8x3x224x224xbf16>) -> tensor<8x224x3x224xbf16>
    %1 = "ttir.transpose"(%0) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<8x224x3x224xbf16>) -> tensor<8x224x224x3xbf16>
    %2 = "ttir.conv2d"(%1, %arg107) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 3, 3, 3, 3>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<8x224x224x3xbf16>, tensor<64x3x7x7xbf16>) -> tensor<8x112x112x64xbf16>
    %3 = "ttir.multiply"(%2, %arg1) : (tensor<8x112x112x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x112x112x64xbf16>
    %4 = "ttir.add"(%3, %arg2) : (tensor<8x112x112x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x112x112x64xbf16>
    %5 = "ttir.relu"(%4) : (tensor<8x112x112x64xbf16>) -> tensor<8x112x112x64xbf16>
    %6 = "ttir.max_pool2d"(%5) <{ceil_mode = false, dilation = array<i32: 1, 1>, kernel = array<i32: 3, 3>, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = true} : (tensor<8x112x112x64xbf16>) -> tensor<8x56x56x64xbf16>
    %7 = "ttir.conv2d"(%6, %arg108) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<64x64x1x1xbf16>) -> tensor<8x56x56x64xbf16>
    %8 = "ttir.multiply"(%7, %arg3) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x56x56x64xbf16>
    %9 = "ttir.add"(%8, %arg4) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x56x56x64xbf16>
    %10 = "ttir.relu"(%9) : (tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16>
    %11 = "ttir.conv2d"(%10, %arg109) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<64x64x3x3xbf16>) -> tensor<8x56x56x64xbf16>
    %12 = "ttir.multiply"(%11, %arg5) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x56x56x64xbf16>
    %13 = "ttir.add"(%12, %arg6) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x56x56x64xbf16>
    %14 = "ttir.relu"(%13) : (tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16>
    %15 = "ttir.conv2d"(%14, %arg110) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<256x64x1x1xbf16>) -> tensor<8x56x56x256xbf16>
    %16 = "ttir.multiply"(%15, %arg7) : (tensor<8x56x56x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x56x56x256xbf16>
    %17 = "ttir.add"(%16, %arg8) : (tensor<8x56x56x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x56x56x256xbf16>
    %18 = "ttir.conv2d"(%6, %arg111) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<256x64x1x1xbf16>) -> tensor<8x56x56x256xbf16>
    %19 = "ttir.multiply"(%18, %arg9) : (tensor<8x56x56x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x56x56x256xbf16>
    %20 = "ttir.add"(%19, %arg10) : (tensor<8x56x56x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x56x56x256xbf16>
    %21 = "ttir.add"(%17, %20) : (tensor<8x56x56x256xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16>
    %22 = "ttir.relu"(%21) : (tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16>
    %23 = "ttir.conv2d"(%22, %arg112) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x256xbf16>, tensor<64x256x1x1xbf16>) -> tensor<8x56x56x64xbf16>
    %24 = "ttir.multiply"(%23, %arg11) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x56x56x64xbf16>
    %25 = "ttir.add"(%24, %arg12) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x56x56x64xbf16>
    %26 = "ttir.relu"(%25) : (tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16>
    %27 = "ttir.conv2d"(%26, %arg113) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<64x64x3x3xbf16>) -> tensor<8x56x56x64xbf16>
    %28 = "ttir.multiply"(%27, %arg13) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x56x56x64xbf16>
    %29 = "ttir.add"(%28, %arg14) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x56x56x64xbf16>
    %30 = "ttir.relu"(%29) : (tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16>
    %31 = "ttir.conv2d"(%30, %arg114) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<256x64x1x1xbf16>) -> tensor<8x56x56x256xbf16>
    %32 = "ttir.multiply"(%31, %arg15) : (tensor<8x56x56x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x56x56x256xbf16>
    %33 = "ttir.add"(%32, %arg16) : (tensor<8x56x56x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x56x56x256xbf16>
    %34 = "ttir.add"(%33, %22) : (tensor<8x56x56x256xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16>
    %35 = "ttir.relu"(%34) : (tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16>
    %36 = "ttir.conv2d"(%35, %arg115) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x256xbf16>, tensor<64x256x1x1xbf16>) -> tensor<8x56x56x64xbf16>
    %37 = "ttir.multiply"(%36, %arg17) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x56x56x64xbf16>
    %38 = "ttir.add"(%37, %arg18) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x56x56x64xbf16>
    %39 = "ttir.relu"(%38) : (tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16>
    %40 = "ttir.conv2d"(%39, %arg116) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<64x64x3x3xbf16>) -> tensor<8x56x56x64xbf16>
    %41 = "ttir.multiply"(%40, %arg19) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x56x56x64xbf16>
    %42 = "ttir.add"(%41, %arg20) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<8x56x56x64xbf16>
    %43 = "ttir.relu"(%42) : (tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16>
    %44 = "ttir.conv2d"(%43, %arg117) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<256x64x1x1xbf16>) -> tensor<8x56x56x256xbf16>
    %45 = "ttir.multiply"(%44, %arg21) : (tensor<8x56x56x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x56x56x256xbf16>
    %46 = "ttir.add"(%45, %arg22) : (tensor<8x56x56x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x56x56x256xbf16>
    %47 = "ttir.add"(%46, %35) : (tensor<8x56x56x256xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16>
    %48 = "ttir.relu"(%47) : (tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16>
    %49 = "ttir.conv2d"(%48, %arg118) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x256xbf16>, tensor<128x256x1x1xbf16>) -> tensor<8x56x56x128xbf16>
    %50 = "ttir.multiply"(%49, %arg23) : (tensor<8x56x56x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x56x56x128xbf16>
    %51 = "ttir.add"(%50, %arg24) : (tensor<8x56x56x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x56x56x128xbf16>
    %52 = "ttir.relu"(%51) : (tensor<8x56x56x128xbf16>) -> tensor<8x56x56x128xbf16>
    %53 = "ttir.conv2d"(%52, %arg119) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<8x56x56x128xbf16>, tensor<128x128x3x3xbf16>) -> tensor<8x28x28x128xbf16>
    %54 = "ttir.multiply"(%53, %arg25) : (tensor<8x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x28x28x128xbf16>
    %55 = "ttir.add"(%54, %arg26) : (tensor<8x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x28x28x128xbf16>
    %56 = "ttir.relu"(%55) : (tensor<8x28x28x128xbf16>) -> tensor<8x28x28x128xbf16>
    %57 = "ttir.conv2d"(%56, %arg120) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x28x28x128xbf16>, tensor<512x128x1x1xbf16>) -> tensor<8x28x28x512xbf16>
    %58 = "ttir.multiply"(%57, %arg27) : (tensor<8x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x28x28x512xbf16>
    %59 = "ttir.add"(%58, %arg28) : (tensor<8x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x28x28x512xbf16>
    %60 = "ttir.conv2d"(%48, %arg121) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<8x56x56x256xbf16>, tensor<512x256x1x1xbf16>) -> tensor<8x28x28x512xbf16>
    %61 = "ttir.multiply"(%60, %arg29) : (tensor<8x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x28x28x512xbf16>
    %62 = "ttir.add"(%61, %arg30) : (tensor<8x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x28x28x512xbf16>
    %63 = "ttir.add"(%59, %62) : (tensor<8x28x28x512xbf16>, tensor<8x28x28x512xbf16>) -> tensor<8x28x28x512xbf16>
    %64 = "ttir.relu"(%63) : (tensor<8x28x28x512xbf16>) -> tensor<8x28x28x512xbf16>
    %65 = "ttir.conv2d"(%64, %arg122) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x28x28x512xbf16>, tensor<128x512x1x1xbf16>) -> tensor<8x28x28x128xbf16>
    %66 = "ttir.multiply"(%65, %arg31) : (tensor<8x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x28x28x128xbf16>
    %67 = "ttir.add"(%66, %arg32) : (tensor<8x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x28x28x128xbf16>
    %68 = "ttir.relu"(%67) : (tensor<8x28x28x128xbf16>) -> tensor<8x28x28x128xbf16>
    %69 = "ttir.conv2d"(%68, %arg123) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x28x28x128xbf16>, tensor<128x128x3x3xbf16>) -> tensor<8x28x28x128xbf16>
    %70 = "ttir.multiply"(%69, %arg33) : (tensor<8x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x28x28x128xbf16>
    %71 = "ttir.add"(%70, %arg34) : (tensor<8x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x28x28x128xbf16>
    %72 = "ttir.relu"(%71) : (tensor<8x28x28x128xbf16>) -> tensor<8x28x28x128xbf16>
    %73 = "ttir.conv2d"(%72, %arg124) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x28x28x128xbf16>, tensor<512x128x1x1xbf16>) -> tensor<8x28x28x512xbf16>
    %74 = "ttir.multiply"(%73, %arg35) : (tensor<8x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x28x28x512xbf16>
    %75 = "ttir.add"(%74, %arg36) : (tensor<8x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x28x28x512xbf16>
    %76 = "ttir.add"(%75, %64) : (tensor<8x28x28x512xbf16>, tensor<8x28x28x512xbf16>) -> tensor<8x28x28x512xbf16>
    %77 = "ttir.relu"(%76) : (tensor<8x28x28x512xbf16>) -> tensor<8x28x28x512xbf16>
    %78 = "ttir.conv2d"(%77, %arg125) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x28x28x512xbf16>, tensor<128x512x1x1xbf16>) -> tensor<8x28x28x128xbf16>
    %79 = "ttir.multiply"(%78, %arg37) : (tensor<8x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x28x28x128xbf16>
    %80 = "ttir.add"(%79, %arg38) : (tensor<8x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x28x28x128xbf16>
    %81 = "ttir.relu"(%80) : (tensor<8x28x28x128xbf16>) -> tensor<8x28x28x128xbf16>
    %82 = "ttir.conv2d"(%81, %arg126) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x28x28x128xbf16>, tensor<128x128x3x3xbf16>) -> tensor<8x28x28x128xbf16>
    %83 = "ttir.multiply"(%82, %arg39) : (tensor<8x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x28x28x128xbf16>
    %84 = "ttir.add"(%83, %arg40) : (tensor<8x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x28x28x128xbf16>
    %85 = "ttir.relu"(%84) : (tensor<8x28x28x128xbf16>) -> tensor<8x28x28x128xbf16>
    %86 = "ttir.conv2d"(%85, %arg127) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x28x28x128xbf16>, tensor<512x128x1x1xbf16>) -> tensor<8x28x28x512xbf16>
    %87 = "ttir.multiply"(%86, %arg41) : (tensor<8x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x28x28x512xbf16>
    %88 = "ttir.add"(%87, %arg42) : (tensor<8x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x28x28x512xbf16>
    %89 = "ttir.add"(%88, %77) : (tensor<8x28x28x512xbf16>, tensor<8x28x28x512xbf16>) -> tensor<8x28x28x512xbf16>
    %90 = "ttir.relu"(%89) : (tensor<8x28x28x512xbf16>) -> tensor<8x28x28x512xbf16>
    %91 = "ttir.conv2d"(%90, %arg128) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x28x28x512xbf16>, tensor<128x512x1x1xbf16>) -> tensor<8x28x28x128xbf16>
    %92 = "ttir.multiply"(%91, %arg43) : (tensor<8x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x28x28x128xbf16>
    %93 = "ttir.add"(%92, %arg44) : (tensor<8x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x28x28x128xbf16>
    %94 = "ttir.relu"(%93) : (tensor<8x28x28x128xbf16>) -> tensor<8x28x28x128xbf16>
    %95 = "ttir.conv2d"(%94, %arg129) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x28x28x128xbf16>, tensor<128x128x3x3xbf16>) -> tensor<8x28x28x128xbf16>
    %96 = "ttir.multiply"(%95, %arg45) : (tensor<8x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x28x28x128xbf16>
    %97 = "ttir.add"(%96, %arg46) : (tensor<8x28x28x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<8x28x28x128xbf16>
    %98 = "ttir.relu"(%97) : (tensor<8x28x28x128xbf16>) -> tensor<8x28x28x128xbf16>
    %99 = "ttir.conv2d"(%98, %arg130) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x28x28x128xbf16>, tensor<512x128x1x1xbf16>) -> tensor<8x28x28x512xbf16>
    %100 = "ttir.multiply"(%99, %arg47) : (tensor<8x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x28x28x512xbf16>
    %101 = "ttir.add"(%100, %arg48) : (tensor<8x28x28x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x28x28x512xbf16>
    %102 = "ttir.add"(%101, %90) : (tensor<8x28x28x512xbf16>, tensor<8x28x28x512xbf16>) -> tensor<8x28x28x512xbf16>
    %103 = "ttir.relu"(%102) : (tensor<8x28x28x512xbf16>) -> tensor<8x28x28x512xbf16>
    %104 = "ttir.conv2d"(%103, %arg131) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x28x28x512xbf16>, tensor<256x512x1x1xbf16>) -> tensor<8x28x28x256xbf16>
    %105 = "ttir.multiply"(%104, %arg49) : (tensor<8x28x28x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x28x28x256xbf16>
    %106 = "ttir.add"(%105, %arg50) : (tensor<8x28x28x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x28x28x256xbf16>
    %107 = "ttir.relu"(%106) : (tensor<8x28x28x256xbf16>) -> tensor<8x28x28x256xbf16>
    %108 = "ttir.conv2d"(%107, %arg132) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<8x28x28x256xbf16>, tensor<256x256x3x3xbf16>) -> tensor<8x14x14x256xbf16>
    %109 = "ttir.multiply"(%108, %arg51) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %110 = "ttir.add"(%109, %arg52) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %111 = "ttir.relu"(%110) : (tensor<8x14x14x256xbf16>) -> tensor<8x14x14x256xbf16>
    %112 = "ttir.conv2d"(%111, %arg133) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x14x14x256xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<8x14x14x1024xbf16>
    %113 = "ttir.multiply"(%112, %arg53) : (tensor<8x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %114 = "ttir.add"(%113, %arg54) : (tensor<8x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %115 = "ttir.conv2d"(%103, %arg134) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<8x28x28x512xbf16>, tensor<1024x512x1x1xbf16>) -> tensor<8x14x14x1024xbf16>
    %116 = "ttir.multiply"(%115, %arg55) : (tensor<8x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %117 = "ttir.add"(%116, %arg56) : (tensor<8x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %118 = "ttir.add"(%114, %117) : (tensor<8x14x14x1024xbf16>, tensor<8x14x14x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %119 = "ttir.relu"(%118) : (tensor<8x14x14x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %120 = "ttir.conv2d"(%119, %arg135) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x14x14x1024xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<8x14x14x256xbf16>
    %121 = "ttir.multiply"(%120, %arg57) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %122 = "ttir.add"(%121, %arg58) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %123 = "ttir.relu"(%122) : (tensor<8x14x14x256xbf16>) -> tensor<8x14x14x256xbf16>
    %124 = "ttir.conv2d"(%123, %arg136) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x14x14x256xbf16>, tensor<256x256x3x3xbf16>) -> tensor<8x14x14x256xbf16>
    %125 = "ttir.multiply"(%124, %arg59) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %126 = "ttir.add"(%125, %arg60) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %127 = "ttir.relu"(%126) : (tensor<8x14x14x256xbf16>) -> tensor<8x14x14x256xbf16>
    %128 = "ttir.conv2d"(%127, %arg137) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x14x14x256xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<8x14x14x1024xbf16>
    %129 = "ttir.multiply"(%128, %arg61) : (tensor<8x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %130 = "ttir.add"(%129, %arg62) : (tensor<8x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %131 = "ttir.add"(%130, %119) : (tensor<8x14x14x1024xbf16>, tensor<8x14x14x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %132 = "ttir.relu"(%131) : (tensor<8x14x14x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %133 = "ttir.conv2d"(%132, %arg138) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x14x14x1024xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<8x14x14x256xbf16>
    %134 = "ttir.multiply"(%133, %arg63) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %135 = "ttir.add"(%134, %arg64) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %136 = "ttir.relu"(%135) : (tensor<8x14x14x256xbf16>) -> tensor<8x14x14x256xbf16>
    %137 = "ttir.conv2d"(%136, %arg139) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x14x14x256xbf16>, tensor<256x256x3x3xbf16>) -> tensor<8x14x14x256xbf16>
    %138 = "ttir.multiply"(%137, %arg65) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %139 = "ttir.add"(%138, %arg66) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %140 = "ttir.relu"(%139) : (tensor<8x14x14x256xbf16>) -> tensor<8x14x14x256xbf16>
    %141 = "ttir.conv2d"(%140, %arg140) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x14x14x256xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<8x14x14x1024xbf16>
    %142 = "ttir.multiply"(%141, %arg67) : (tensor<8x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %143 = "ttir.add"(%142, %arg68) : (tensor<8x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %144 = "ttir.add"(%143, %132) : (tensor<8x14x14x1024xbf16>, tensor<8x14x14x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %145 = "ttir.relu"(%144) : (tensor<8x14x14x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %146 = "ttir.conv2d"(%145, %arg141) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x14x14x1024xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<8x14x14x256xbf16>
    %147 = "ttir.multiply"(%146, %arg69) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %148 = "ttir.add"(%147, %arg70) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %149 = "ttir.relu"(%148) : (tensor<8x14x14x256xbf16>) -> tensor<8x14x14x256xbf16>
    %150 = "ttir.conv2d"(%149, %arg142) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x14x14x256xbf16>, tensor<256x256x3x3xbf16>) -> tensor<8x14x14x256xbf16>
    %151 = "ttir.multiply"(%150, %arg71) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %152 = "ttir.add"(%151, %arg72) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %153 = "ttir.relu"(%152) : (tensor<8x14x14x256xbf16>) -> tensor<8x14x14x256xbf16>
    %154 = "ttir.conv2d"(%153, %arg143) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x14x14x256xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<8x14x14x1024xbf16>
    %155 = "ttir.multiply"(%154, %arg73) : (tensor<8x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %156 = "ttir.add"(%155, %arg74) : (tensor<8x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %157 = "ttir.add"(%156, %145) : (tensor<8x14x14x1024xbf16>, tensor<8x14x14x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %158 = "ttir.relu"(%157) : (tensor<8x14x14x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %159 = "ttir.conv2d"(%158, %arg144) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x14x14x1024xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<8x14x14x256xbf16>
    %160 = "ttir.multiply"(%159, %arg75) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %161 = "ttir.add"(%160, %arg76) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %162 = "ttir.relu"(%161) : (tensor<8x14x14x256xbf16>) -> tensor<8x14x14x256xbf16>
    %163 = "ttir.conv2d"(%162, %arg145) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x14x14x256xbf16>, tensor<256x256x3x3xbf16>) -> tensor<8x14x14x256xbf16>
    %164 = "ttir.multiply"(%163, %arg77) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %165 = "ttir.add"(%164, %arg78) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %166 = "ttir.relu"(%165) : (tensor<8x14x14x256xbf16>) -> tensor<8x14x14x256xbf16>
    %167 = "ttir.conv2d"(%166, %arg146) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x14x14x256xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<8x14x14x1024xbf16>
    %168 = "ttir.multiply"(%167, %arg79) : (tensor<8x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %169 = "ttir.add"(%168, %arg80) : (tensor<8x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %170 = "ttir.add"(%169, %158) : (tensor<8x14x14x1024xbf16>, tensor<8x14x14x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %171 = "ttir.relu"(%170) : (tensor<8x14x14x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %172 = "ttir.conv2d"(%171, %arg147) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x14x14x1024xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<8x14x14x256xbf16>
    %173 = "ttir.multiply"(%172, %arg81) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %174 = "ttir.add"(%173, %arg82) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %175 = "ttir.relu"(%174) : (tensor<8x14x14x256xbf16>) -> tensor<8x14x14x256xbf16>
    %176 = "ttir.conv2d"(%175, %arg148) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x14x14x256xbf16>, tensor<256x256x3x3xbf16>) -> tensor<8x14x14x256xbf16>
    %177 = "ttir.multiply"(%176, %arg83) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %178 = "ttir.add"(%177, %arg84) : (tensor<8x14x14x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<8x14x14x256xbf16>
    %179 = "ttir.relu"(%178) : (tensor<8x14x14x256xbf16>) -> tensor<8x14x14x256xbf16>
    %180 = "ttir.conv2d"(%179, %arg149) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x14x14x256xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<8x14x14x1024xbf16>
    %181 = "ttir.multiply"(%180, %arg85) : (tensor<8x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %182 = "ttir.add"(%181, %arg86) : (tensor<8x14x14x1024xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %183 = "ttir.add"(%182, %171) : (tensor<8x14x14x1024xbf16>, tensor<8x14x14x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %184 = "ttir.relu"(%183) : (tensor<8x14x14x1024xbf16>) -> tensor<8x14x14x1024xbf16>
    %185 = "ttir.conv2d"(%184, %arg150) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x14x14x1024xbf16>, tensor<512x1024x1x1xbf16>) -> tensor<8x14x14x512xbf16>
    %186 = "ttir.multiply"(%185, %arg87) : (tensor<8x14x14x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x14x14x512xbf16>
    %187 = "ttir.add"(%186, %arg88) : (tensor<8x14x14x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x14x14x512xbf16>
    %188 = "ttir.relu"(%187) : (tensor<8x14x14x512xbf16>) -> tensor<8x14x14x512xbf16>
    %189 = "ttir.conv2d"(%188, %arg151) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<8x14x14x512xbf16>, tensor<512x512x3x3xbf16>) -> tensor<8x7x7x512xbf16>
    %190 = "ttir.multiply"(%189, %arg89) : (tensor<8x7x7x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x7x7x512xbf16>
    %191 = "ttir.add"(%190, %arg90) : (tensor<8x7x7x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x7x7x512xbf16>
    %192 = "ttir.relu"(%191) : (tensor<8x7x7x512xbf16>) -> tensor<8x7x7x512xbf16>
    %193 = "ttir.conv2d"(%192, %arg152) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x7x7x512xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<8x7x7x2048xbf16>
    %194 = "ttir.transpose"(%193) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<8x7x7x2048xbf16>) -> tensor<8x7x2048x7xbf16>
    %195 = "ttir.transpose"(%194) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<8x7x2048x7xbf16>) -> tensor<8x2048x7x7xbf16>
    %196 = "ttir.multiply"(%195, %arg91) : (tensor<8x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<8x2048x7x7xbf16>
    %197 = "ttir.add"(%196, %arg92) : (tensor<8x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<8x2048x7x7xbf16>
    %198 = "ttir.conv2d"(%184, %arg153) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<8x14x14x1024xbf16>, tensor<2048x1024x1x1xbf16>) -> tensor<8x7x7x2048xbf16>
    %199 = "ttir.transpose"(%198) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<8x7x7x2048xbf16>) -> tensor<8x7x2048x7xbf16>
    %200 = "ttir.transpose"(%199) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<8x7x2048x7xbf16>) -> tensor<8x2048x7x7xbf16>
    %201 = "ttir.multiply"(%200, %arg93) : (tensor<8x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<8x2048x7x7xbf16>
    %202 = "ttir.add"(%201, %arg94) : (tensor<8x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<8x2048x7x7xbf16>
    %203 = "ttir.add"(%197, %202) : (tensor<8x2048x7x7xbf16>, tensor<8x2048x7x7xbf16>) -> tensor<8x2048x7x7xbf16>
    %204 = "ttir.relu"(%203) : (tensor<8x2048x7x7xbf16>) -> tensor<8x2048x7x7xbf16>
    %205 = "ttir.transpose"(%204) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<8x2048x7x7xbf16>) -> tensor<8x7x2048x7xbf16>
    %206 = "ttir.transpose"(%205) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<8x7x2048x7xbf16>) -> tensor<8x7x7x2048xbf16>
    %207 = "ttir.conv2d"(%206, %arg154) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x7x7x2048xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<8x7x7x512xbf16>
    %208 = "ttir.multiply"(%207, %arg95) : (tensor<8x7x7x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x7x7x512xbf16>
    %209 = "ttir.add"(%208, %arg96) : (tensor<8x7x7x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x7x7x512xbf16>
    %210 = "ttir.relu"(%209) : (tensor<8x7x7x512xbf16>) -> tensor<8x7x7x512xbf16>
    %211 = "ttir.conv2d"(%210, %arg155) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x7x7x512xbf16>, tensor<512x512x3x3xbf16>) -> tensor<8x7x7x512xbf16>
    %212 = "ttir.multiply"(%211, %arg97) : (tensor<8x7x7x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x7x7x512xbf16>
    %213 = "ttir.add"(%212, %arg98) : (tensor<8x7x7x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x7x7x512xbf16>
    %214 = "ttir.relu"(%213) : (tensor<8x7x7x512xbf16>) -> tensor<8x7x7x512xbf16>
    %215 = "ttir.conv2d"(%214, %arg156) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x7x7x512xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<8x7x7x2048xbf16>
    %216 = "ttir.transpose"(%215) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<8x7x7x2048xbf16>) -> tensor<8x7x2048x7xbf16>
    %217 = "ttir.transpose"(%216) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<8x7x2048x7xbf16>) -> tensor<8x2048x7x7xbf16>
    %218 = "ttir.multiply"(%217, %arg99) : (tensor<8x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<8x2048x7x7xbf16>
    %219 = "ttir.add"(%218, %arg100) : (tensor<8x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<8x2048x7x7xbf16>
    %220 = "ttir.add"(%219, %204) : (tensor<8x2048x7x7xbf16>, tensor<8x2048x7x7xbf16>) -> tensor<8x2048x7x7xbf16>
    %221 = "ttir.relu"(%220) : (tensor<8x2048x7x7xbf16>) -> tensor<8x2048x7x7xbf16>
    %222 = "ttir.transpose"(%221) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<8x2048x7x7xbf16>) -> tensor<8x7x2048x7xbf16>
    %223 = "ttir.transpose"(%222) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<8x7x2048x7xbf16>) -> tensor<8x7x7x2048xbf16>
    %224 = "ttir.conv2d"(%223, %arg157) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x7x7x2048xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<8x7x7x512xbf16>
    %225 = "ttir.multiply"(%224, %arg101) : (tensor<8x7x7x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x7x7x512xbf16>
    %226 = "ttir.add"(%225, %arg102) : (tensor<8x7x7x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x7x7x512xbf16>
    %227 = "ttir.relu"(%226) : (tensor<8x7x7x512xbf16>) -> tensor<8x7x7x512xbf16>
    %228 = "ttir.conv2d"(%227, %arg158) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x7x7x512xbf16>, tensor<512x512x3x3xbf16>) -> tensor<8x7x7x512xbf16>
    %229 = "ttir.multiply"(%228, %arg103) : (tensor<8x7x7x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x7x7x512xbf16>
    %230 = "ttir.add"(%229, %arg104) : (tensor<8x7x7x512xbf16>, tensor<1x1x1x512xbf16>) -> tensor<8x7x7x512xbf16>
    %231 = "ttir.relu"(%230) : (tensor<8x7x7x512xbf16>) -> tensor<8x7x7x512xbf16>
    %232 = "ttir.conv2d"(%231, %arg159) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x7x7x512xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<8x7x7x2048xbf16>
    %233 = "ttir.transpose"(%232) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<8x7x7x2048xbf16>) -> tensor<8x7x2048x7xbf16>
    %234 = "ttir.transpose"(%233) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<8x7x2048x7xbf16>) -> tensor<8x2048x7x7xbf16>
    %235 = "ttir.multiply"(%234, %arg105) : (tensor<8x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<8x2048x7x7xbf16>
    %236 = "ttir.add"(%235, %arg106) : (tensor<8x2048x7x7xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<8x2048x7x7xbf16>
    %237 = "ttir.add"(%236, %221) : (tensor<8x2048x7x7xbf16>, tensor<8x2048x7x7xbf16>) -> tensor<8x2048x7x7xbf16>
    %238 = "ttir.relu"(%237) : (tensor<8x2048x7x7xbf16>) -> tensor<8x2048x7x7xbf16>
    %239 = "ttir.reshape"(%238) <{shape = [8 : i32, 1 : i32, 2048 : i32, 49 : i32]}> : (tensor<8x2048x7x7xbf16>) -> tensor<8x1x2048x49xbf16>
    %240 = "ttir.transpose"(%239) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<8x1x2048x49xbf16>) -> tensor<8x1x49x2048xbf16>
    %241 = "ttir.mean"(%240) <{dim_arg = [-2 : i32], keep_dim = true}> : (tensor<8x1x49x2048xbf16>) -> tensor<8x1x1x2048xbf16>
    %242 = "ttir.reshape"(%241) <{shape = [8 : i32, 2048 : i32]}> : (tensor<8x1x1x2048xbf16>) -> tensor<8x2048xbf16>
    %243 = "ttir.matmul"(%242, %arg160) <{transpose_a = false, transpose_b = false}> : (tensor<8x2048xbf16>, tensor<2048x1000xbf16>) -> tensor<8x1000xbf16>
    %244 = "ttir.add"(%243, %arg161) : (tensor<8x1000xbf16>, tensor<1000xbf16>) -> tensor<8x1000xbf16>
    return %244 : tensor<8x1000xbf16> loc(#loc109)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("transformers.models.resnet.modeling_resnet.ResNetForImageClassification::")
#loc2 = loc("multiply_7")
#loc3 = loc("add_13")
#loc4 = loc("multiply_23")
#loc5 = loc("add_29")
#loc6 = loc("multiply_38")
#loc7 = loc("add_44")
#loc8 = loc("multiply_53")
#loc9 = loc("add_59")
#loc10 = loc("multiply_67")
#loc11 = loc("add_73")
#loc12 = loc("multiply_83")
#loc13 = loc("add_89")
#loc14 = loc("multiply_98")
#loc15 = loc("add_104")
#loc16 = loc("multiply_113")
#loc17 = loc("add_119")
#loc18 = loc("multiply_129")
#loc19 = loc("add_135")
#loc20 = loc("multiply_144")
#loc21 = loc("add_150")
#loc22 = loc("multiply_159")
#loc23 = loc("add_165")
#loc24 = loc("multiply_175")
#loc25 = loc("add_181")
#loc26 = loc("multiply_190")
#loc27 = loc("add_196")
#loc28 = loc("multiply_205")
#loc29 = loc("add_211")
#loc30 = loc("multiply_219")
#loc31 = loc("add_225")
#loc32 = loc("multiply_235")
#loc33 = loc("add_241")
#loc34 = loc("multiply_250")
#loc35 = loc("add_256")
#loc36 = loc("multiply_265")
#loc37 = loc("add_271")
#loc38 = loc("multiply_281")
#loc39 = loc("add_287")
#loc40 = loc("multiply_296")
#loc41 = loc("add_302")
#loc42 = loc("multiply_311")
#loc43 = loc("add_317")
#loc44 = loc("multiply_327")
#loc45 = loc("add_333")
#loc46 = loc("multiply_342")
#loc47 = loc("add_348")
#loc48 = loc("multiply_357")
#loc49 = loc("add_363")
#loc50 = loc("multiply_373")
#loc51 = loc("add_379")
#loc52 = loc("multiply_388")
#loc53 = loc("add_394")
#loc54 = loc("multiply_403")
#loc55 = loc("add_409")
#loc56 = loc("multiply_417")
#loc57 = loc("add_423")
#loc58 = loc("multiply_433")
#loc59 = loc("add_439")
#loc60 = loc("multiply_448")
#loc61 = loc("add_454")
#loc62 = loc("multiply_463")
#loc63 = loc("add_469")
#loc64 = loc("multiply_479")
#loc65 = loc("add_485")
#loc66 = loc("multiply_494")
#loc67 = loc("add_500")
#loc68 = loc("multiply_509")
#loc69 = loc("add_515")
#loc70 = loc("multiply_525")
#loc71 = loc("add_531")
#loc72 = loc("multiply_540")
#loc73 = loc("add_546")
#loc74 = loc("multiply_555")
#loc75 = loc("add_561")
#loc76 = loc("multiply_571")
#loc77 = loc("add_577")
#loc78 = loc("multiply_586")
#loc79 = loc("add_592")
#loc80 = loc("multiply_601")
#loc81 = loc("add_607")
#loc82 = loc("multiply_617")
#loc83 = loc("add_623")
#loc84 = loc("multiply_632")
#loc85 = loc("add_638")
#loc86 = loc("multiply_647")
#loc87 = loc("add_653")
#loc88 = loc("multiply_663")
#loc89 = loc("add_669")
#loc90 = loc("multiply_678")
#loc91 = loc("add_684")
#loc92 = loc("multiply_693")
#loc93 = loc("add_699")
#loc94 = loc("multiply_707")
#loc95 = loc("add_713")
#loc96 = loc("multiply_723")
#loc97 = loc("add_729")
#loc98 = loc("multiply_738")
#loc99 = loc("add_744")
#loc100 = loc("multiply_753")
#loc101 = loc("add_759")
#loc102 = loc("multiply_769")
#loc103 = loc("add_775")
#loc104 = loc("multiply_784")
#loc105 = loc("add_790")
#loc106 = loc("multiply_799")
#loc107 = loc("add_805")
#loc108 = loc("add_814")
#loc109 = loc(unknown)
#loc110 = loc("transformers.models.resnet.modeling_resnet.ResNetModel::resnet"(#loc1))
#loc111 = loc("torch.nn.modules.container.Sequential::classifier"(#loc1))
#loc112 = loc("transformers.models.resnet.modeling_resnet.ResNetEmbeddings::embedder"(#loc110))
#loc113 = loc("transformers.models.resnet.modeling_resnet.ResNetEncoder::encoder"(#loc110))
#loc114 = loc("torch.nn.modules.pooling.AdaptiveAvgPool2d::pooler"(#loc110))
#loc115 = loc("torch.nn.modules.linear.Linear::1"(#loc111))
#loc116 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::embedder"(#loc112))
#loc117 = loc("torch.nn.modules.pooling.MaxPool2d::pooler"(#loc112))
#loc118 = loc("transformers.models.resnet.modeling_resnet.ResNetStage::stages.0"(#loc113))
#loc119 = loc("transformers.models.resnet.modeling_resnet.ResNetStage::stages.1"(#loc113))
#loc120 = loc("transformers.models.resnet.modeling_resnet.ResNetStage::stages.2"(#loc113))
#loc121 = loc("transformers.models.resnet.modeling_resnet.ResNetStage::stages.3"(#loc113))
#loc122 = loc("avg_pool2d_808.dc.reshape.0"(#loc114))
#loc123 = loc("avg_pool2d_808.dc.transpose.1"(#loc114))
#loc124 = loc("avg_pool2d_808.dc.reduce_avg.2"(#loc114))
#loc125 = loc("avg_pool2d_808.dc.reshape.4"(#loc114))
#loc126 = loc("matmul_813"(#loc115))
#loc127 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc116))
#loc128 = loc("torch.nn.modules.activation.ReLU::activation"(#loc116))
#loc129 = loc("max_pool2d_15.dc.max_pool2d.2"(#loc117))
#loc130 = loc("transformers.models.resnet.modeling_resnet.ResNetBottleNeckLayer::0"(#loc118))
#loc131 = loc("transformers.models.resnet.modeling_resnet.ResNetBottleNeckLayer::1"(#loc118))
#loc132 = loc("transformers.models.resnet.modeling_resnet.ResNetBottleNeckLayer::2"(#loc118))
#loc133 = loc("transformers.models.resnet.modeling_resnet.ResNetBottleNeckLayer::0"(#loc119))
#loc134 = loc("transformers.models.resnet.modeling_resnet.ResNetBottleNeckLayer::1"(#loc119))
#loc135 = loc("transformers.models.resnet.modeling_resnet.ResNetBottleNeckLayer::2"(#loc119))
#loc136 = loc("transformers.models.resnet.modeling_resnet.ResNetBottleNeckLayer::3"(#loc119))
#loc137 = loc("transformers.models.resnet.modeling_resnet.ResNetBottleNeckLayer::0"(#loc120))
#loc138 = loc("transformers.models.resnet.modeling_resnet.ResNetBottleNeckLayer::1"(#loc120))
#loc139 = loc("transformers.models.resnet.modeling_resnet.ResNetBottleNeckLayer::2"(#loc120))
#loc140 = loc("transformers.models.resnet.modeling_resnet.ResNetBottleNeckLayer::3"(#loc120))
#loc141 = loc("transformers.models.resnet.modeling_resnet.ResNetBottleNeckLayer::4"(#loc120))
#loc142 = loc("transformers.models.resnet.modeling_resnet.ResNetBottleNeckLayer::5"(#loc120))
#loc143 = loc("transformers.models.resnet.modeling_resnet.ResNetBottleNeckLayer::0"(#loc121))
#loc144 = loc("transformers.models.resnet.modeling_resnet.ResNetBottleNeckLayer::1"(#loc121))
#loc145 = loc("transformers.models.resnet.modeling_resnet.ResNetBottleNeckLayer::2"(#loc121))
#loc146 = loc("conv2d_0.dc.transpose.0"(#loc127))
#loc147 = loc("conv2d_0.dc.transpose.1"(#loc127))
#loc148 = loc("conv2d_0.dc.conv2d.2"(#loc127))
#loc149 = loc("relu_14"(#loc128))
#loc150 = loc("torch.nn.modules.container.Sequential::layer"(#loc130))
#loc151 = loc("transformers.models.resnet.modeling_resnet.ResNetShortCut::shortcut"(#loc130))
#loc152 = loc("add_74"(#loc130))
#loc153 = loc("torch.nn.modules.activation.ReLU::activation"(#loc130))
#loc154 = loc("torch.nn.modules.container.Sequential::layer"(#loc131))
#loc155 = loc("add_120"(#loc131))
#loc156 = loc("torch.nn.modules.activation.ReLU::activation"(#loc131))
#loc157 = loc("torch.nn.modules.container.Sequential::layer"(#loc132))
#loc158 = loc("add_166"(#loc132))
#loc159 = loc("torch.nn.modules.activation.ReLU::activation"(#loc132))
#loc160 = loc("torch.nn.modules.container.Sequential::layer"(#loc133))
#loc161 = loc("transformers.models.resnet.modeling_resnet.ResNetShortCut::shortcut"(#loc133))
#loc162 = loc("add_226"(#loc133))
#loc163 = loc("torch.nn.modules.activation.ReLU::activation"(#loc133))
#loc164 = loc("torch.nn.modules.container.Sequential::layer"(#loc134))
#loc165 = loc("add_272"(#loc134))
#loc166 = loc("torch.nn.modules.activation.ReLU::activation"(#loc134))
#loc167 = loc("torch.nn.modules.container.Sequential::layer"(#loc135))
#loc168 = loc("add_318"(#loc135))
#loc169 = loc("torch.nn.modules.activation.ReLU::activation"(#loc135))
#loc170 = loc("torch.nn.modules.container.Sequential::layer"(#loc136))
#loc171 = loc("add_364"(#loc136))
#loc172 = loc("torch.nn.modules.activation.ReLU::activation"(#loc136))
#loc173 = loc("torch.nn.modules.container.Sequential::layer"(#loc137))
#loc174 = loc("transformers.models.resnet.modeling_resnet.ResNetShortCut::shortcut"(#loc137))
#loc175 = loc("add_424"(#loc137))
#loc176 = loc("torch.nn.modules.activation.ReLU::activation"(#loc137))
#loc177 = loc("torch.nn.modules.container.Sequential::layer"(#loc138))
#loc178 = loc("add_470"(#loc138))
#loc179 = loc("torch.nn.modules.activation.ReLU::activation"(#loc138))
#loc180 = loc("torch.nn.modules.container.Sequential::layer"(#loc139))
#loc181 = loc("add_516"(#loc139))
#loc182 = loc("torch.nn.modules.activation.ReLU::activation"(#loc139))
#loc183 = loc("torch.nn.modules.container.Sequential::layer"(#loc140))
#loc184 = loc("add_562"(#loc140))
#loc185 = loc("torch.nn.modules.activation.ReLU::activation"(#loc140))
#loc186 = loc("torch.nn.modules.container.Sequential::layer"(#loc141))
#loc187 = loc("add_608"(#loc141))
#loc188 = loc("torch.nn.modules.activation.ReLU::activation"(#loc141))
#loc189 = loc("torch.nn.modules.container.Sequential::layer"(#loc142))
#loc190 = loc("add_654"(#loc142))
#loc191 = loc("torch.nn.modules.activation.ReLU::activation"(#loc142))
#loc192 = loc("torch.nn.modules.container.Sequential::layer"(#loc143))
#loc193 = loc("transformers.models.resnet.modeling_resnet.ResNetShortCut::shortcut"(#loc143))
#loc194 = loc("add_714"(#loc143))
#loc195 = loc("torch.nn.modules.activation.ReLU::activation"(#loc143))
#loc196 = loc("torch.nn.modules.container.Sequential::layer"(#loc144))
#loc197 = loc("add_760"(#loc144))
#loc198 = loc("torch.nn.modules.activation.ReLU::activation"(#loc144))
#loc199 = loc("torch.nn.modules.container.Sequential::layer"(#loc145))
#loc200 = loc("add_806"(#loc145))
#loc201 = loc("torch.nn.modules.activation.ReLU::activation"(#loc145))
#loc202 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::0"(#loc150))
#loc203 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::1"(#loc150))
#loc204 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::2"(#loc150))
#loc205 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc151))
#loc206 = loc("relu_75"(#loc153))
#loc207 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::0"(#loc154))
#loc208 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::1"(#loc154))
#loc209 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::2"(#loc154))
#loc210 = loc("relu_121"(#loc156))
#loc211 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::0"(#loc157))
#loc212 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::1"(#loc157))
#loc213 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::2"(#loc157))
#loc214 = loc("relu_167"(#loc159))
#loc215 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::0"(#loc160))
#loc216 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::1"(#loc160))
#loc217 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::2"(#loc160))
#loc218 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc161))
#loc219 = loc("relu_227"(#loc163))
#loc220 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::0"(#loc164))
#loc221 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::1"(#loc164))
#loc222 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::2"(#loc164))
#loc223 = loc("relu_273"(#loc166))
#loc224 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::0"(#loc167))
#loc225 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::1"(#loc167))
#loc226 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::2"(#loc167))
#loc227 = loc("relu_319"(#loc169))
#loc228 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::0"(#loc170))
#loc229 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::1"(#loc170))
#loc230 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::2"(#loc170))
#loc231 = loc("relu_365"(#loc172))
#loc232 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::0"(#loc173))
#loc233 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::1"(#loc173))
#loc234 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::2"(#loc173))
#loc235 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc174))
#loc236 = loc("relu_425"(#loc176))
#loc237 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::0"(#loc177))
#loc238 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::1"(#loc177))
#loc239 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::2"(#loc177))
#loc240 = loc("relu_471"(#loc179))
#loc241 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::0"(#loc180))
#loc242 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::1"(#loc180))
#loc243 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::2"(#loc180))
#loc244 = loc("relu_517"(#loc182))
#loc245 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::0"(#loc183))
#loc246 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::1"(#loc183))
#loc247 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::2"(#loc183))
#loc248 = loc("relu_563"(#loc185))
#loc249 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::0"(#loc186))
#loc250 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::1"(#loc186))
#loc251 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::2"(#loc186))
#loc252 = loc("relu_609"(#loc188))
#loc253 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::0"(#loc189))
#loc254 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::1"(#loc189))
#loc255 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::2"(#loc189))
#loc256 = loc("relu_655"(#loc191))
#loc257 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::0"(#loc192))
#loc258 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::1"(#loc192))
#loc259 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::2"(#loc192))
#loc260 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc193))
#loc261 = loc("relu_715"(#loc195))
#loc262 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::0"(#loc196))
#loc263 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::1"(#loc196))
#loc264 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::2"(#loc196))
#loc265 = loc("relu_761"(#loc198))
#loc266 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::0"(#loc199))
#loc267 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::1"(#loc199))
#loc268 = loc("transformers.models.resnet.modeling_resnet.ResNetConvLayer::2"(#loc199))
#loc269 = loc("relu_807"(#loc201))
#loc270 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc202))
#loc271 = loc("torch.nn.modules.activation.ReLU::activation"(#loc202))
#loc272 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc203))
#loc273 = loc("torch.nn.modules.activation.ReLU::activation"(#loc203))
#loc274 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc204))
#loc275 = loc("conv2d_60.dc.conv2d.2"(#loc205))
#loc276 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc207))
#loc277 = loc("torch.nn.modules.activation.ReLU::activation"(#loc207))
#loc278 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc208))
#loc279 = loc("torch.nn.modules.activation.ReLU::activation"(#loc208))
#loc280 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc209))
#loc281 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc211))
#loc282 = loc("torch.nn.modules.activation.ReLU::activation"(#loc211))
#loc283 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc212))
#loc284 = loc("torch.nn.modules.activation.ReLU::activation"(#loc212))
#loc285 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc213))
#loc286 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc215))
#loc287 = loc("torch.nn.modules.activation.ReLU::activation"(#loc215))
#loc288 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc216))
#loc289 = loc("torch.nn.modules.activation.ReLU::activation"(#loc216))
#loc290 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc217))
#loc291 = loc("conv2d_212.dc.conv2d.2"(#loc218))
#loc292 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc220))
#loc293 = loc("torch.nn.modules.activation.ReLU::activation"(#loc220))
#loc294 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc221))
#loc295 = loc("torch.nn.modules.activation.ReLU::activation"(#loc221))
#loc296 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc222))
#loc297 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc224))
#loc298 = loc("torch.nn.modules.activation.ReLU::activation"(#loc224))
#loc299 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc225))
#loc300 = loc("torch.nn.modules.activation.ReLU::activation"(#loc225))
#loc301 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc226))
#loc302 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc228))
#loc303 = loc("torch.nn.modules.activation.ReLU::activation"(#loc228))
#loc304 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc229))
#loc305 = loc("torch.nn.modules.activation.ReLU::activation"(#loc229))
#loc306 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc230))
#loc307 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc232))
#loc308 = loc("torch.nn.modules.activation.ReLU::activation"(#loc232))
#loc309 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc233))
#loc310 = loc("torch.nn.modules.activation.ReLU::activation"(#loc233))
#loc311 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc234))
#loc312 = loc("conv2d_410.dc.conv2d.2"(#loc235))
#loc313 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc237))
#loc314 = loc("torch.nn.modules.activation.ReLU::activation"(#loc237))
#loc315 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc238))
#loc316 = loc("torch.nn.modules.activation.ReLU::activation"(#loc238))
#loc317 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc239))
#loc318 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc241))
#loc319 = loc("torch.nn.modules.activation.ReLU::activation"(#loc241))
#loc320 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc242))
#loc321 = loc("torch.nn.modules.activation.ReLU::activation"(#loc242))
#loc322 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc243))
#loc323 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc245))
#loc324 = loc("torch.nn.modules.activation.ReLU::activation"(#loc245))
#loc325 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc246))
#loc326 = loc("torch.nn.modules.activation.ReLU::activation"(#loc246))
#loc327 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc247))
#loc328 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc249))
#loc329 = loc("torch.nn.modules.activation.ReLU::activation"(#loc249))
#loc330 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc250))
#loc331 = loc("torch.nn.modules.activation.ReLU::activation"(#loc250))
#loc332 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc251))
#loc333 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc253))
#loc334 = loc("torch.nn.modules.activation.ReLU::activation"(#loc253))
#loc335 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc254))
#loc336 = loc("torch.nn.modules.activation.ReLU::activation"(#loc254))
#loc337 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc255))
#loc338 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc257))
#loc339 = loc("torch.nn.modules.activation.ReLU::activation"(#loc257))
#loc340 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc258))
#loc341 = loc("torch.nn.modules.activation.ReLU::activation"(#loc258))
#loc342 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc259))
#loc343 = loc("conv2d_700.dc.conv2d.2"(#loc260))
#loc344 = loc("conv2d_700.dc.transpose.3"(#loc260))
#loc345 = loc("conv2d_700.dc.transpose.4"(#loc260))
#loc346 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc262))
#loc347 = loc("torch.nn.modules.activation.ReLU::activation"(#loc262))
#loc348 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc263))
#loc349 = loc("torch.nn.modules.activation.ReLU::activation"(#loc263))
#loc350 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc264))
#loc351 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc266))
#loc352 = loc("torch.nn.modules.activation.ReLU::activation"(#loc266))
#loc353 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc267))
#loc354 = loc("torch.nn.modules.activation.ReLU::activation"(#loc267))
#loc355 = loc("torch.nn.modules.conv.Conv2d::convolution"(#loc268))
#loc356 = loc("conv2d_16.dc.conv2d.2"(#loc270))
#loc357 = loc("relu_30"(#loc271))
#loc358 = loc("conv2d_31.dc.conv2d.2"(#loc272))
#loc359 = loc("relu_45"(#loc273))
#loc360 = loc("conv2d_46.dc.conv2d.2"(#loc274))
#loc361 = loc("conv2d_76.dc.conv2d.2"(#loc276))
#loc362 = loc("relu_90"(#loc277))
#loc363 = loc("conv2d_91.dc.conv2d.2"(#loc278))
#loc364 = loc("relu_105"(#loc279))
#loc365 = loc("conv2d_106.dc.conv2d.2"(#loc280))
#loc366 = loc("conv2d_122.dc.conv2d.2"(#loc281))
#loc367 = loc("relu_136"(#loc282))
#loc368 = loc("conv2d_137.dc.conv2d.2"(#loc283))
#loc369 = loc("relu_151"(#loc284))
#loc370 = loc("conv2d_152.dc.conv2d.2"(#loc285))
#loc371 = loc("conv2d_168.dc.conv2d.2"(#loc286))
#loc372 = loc("relu_182"(#loc287))
#loc373 = loc("conv2d_183.dc.conv2d.2"(#loc288))
#loc374 = loc("relu_197"(#loc289))
#loc375 = loc("conv2d_198.dc.conv2d.2"(#loc290))
#loc376 = loc("conv2d_228.dc.conv2d.2"(#loc292))
#loc377 = loc("relu_242"(#loc293))
#loc378 = loc("conv2d_243.dc.conv2d.2"(#loc294))
#loc379 = loc("relu_257"(#loc295))
#loc380 = loc("conv2d_258.dc.conv2d.2"(#loc296))
#loc381 = loc("conv2d_274.dc.conv2d.2"(#loc297))
#loc382 = loc("relu_288"(#loc298))
#loc383 = loc("conv2d_289.dc.conv2d.2"(#loc299))
#loc384 = loc("relu_303"(#loc300))
#loc385 = loc("conv2d_304.dc.conv2d.2"(#loc301))
#loc386 = loc("conv2d_320.dc.conv2d.2"(#loc302))
#loc387 = loc("relu_334"(#loc303))
#loc388 = loc("conv2d_335.dc.conv2d.2"(#loc304))
#loc389 = loc("relu_349"(#loc305))
#loc390 = loc("conv2d_350.dc.conv2d.2"(#loc306))
#loc391 = loc("conv2d_366.dc.conv2d.2"(#loc307))
#loc392 = loc("relu_380"(#loc308))
#loc393 = loc("conv2d_381.dc.conv2d.2"(#loc309))
#loc394 = loc("relu_395"(#loc310))
#loc395 = loc("conv2d_396.dc.conv2d.2"(#loc311))
#loc396 = loc("conv2d_426.dc.conv2d.2"(#loc313))
#loc397 = loc("relu_440"(#loc314))
#loc398 = loc("conv2d_441.dc.conv2d.2"(#loc315))
#loc399 = loc("relu_455"(#loc316))
#loc400 = loc("conv2d_456.dc.conv2d.2"(#loc317))
#loc401 = loc("conv2d_472.dc.conv2d.2"(#loc318))
#loc402 = loc("relu_486"(#loc319))
#loc403 = loc("conv2d_487.dc.conv2d.2"(#loc320))
#loc404 = loc("relu_501"(#loc321))
#loc405 = loc("conv2d_502.dc.conv2d.2"(#loc322))
#loc406 = loc("conv2d_518.dc.conv2d.2"(#loc323))
#loc407 = loc("relu_532"(#loc324))
#loc408 = loc("conv2d_533.dc.conv2d.2"(#loc325))
#loc409 = loc("relu_547"(#loc326))
#loc410 = loc("conv2d_548.dc.conv2d.2"(#loc327))
#loc411 = loc("conv2d_564.dc.conv2d.2"(#loc328))
#loc412 = loc("relu_578"(#loc329))
#loc413 = loc("conv2d_579.dc.conv2d.2"(#loc330))
#loc414 = loc("relu_593"(#loc331))
#loc415 = loc("conv2d_594.dc.conv2d.2"(#loc332))
#loc416 = loc("conv2d_610.dc.conv2d.2"(#loc333))
#loc417 = loc("relu_624"(#loc334))
#loc418 = loc("conv2d_625.dc.conv2d.2"(#loc335))
#loc419 = loc("relu_639"(#loc336))
#loc420 = loc("conv2d_640.dc.conv2d.2"(#loc337))
#loc421 = loc("conv2d_656.dc.conv2d.2"(#loc338))
#loc422 = loc("relu_670"(#loc339))
#loc423 = loc("conv2d_671.dc.conv2d.2"(#loc340))
#loc424 = loc("relu_685"(#loc341))
#loc425 = loc("conv2d_686.dc.conv2d.2"(#loc342))
#loc426 = loc("conv2d_686.dc.transpose.3"(#loc342))
#loc427 = loc("conv2d_686.dc.transpose.4"(#loc342))
#loc428 = loc("conv2d_716.dc.transpose.0"(#loc346))
#loc429 = loc("conv2d_716.dc.transpose.1"(#loc346))
#loc430 = loc("conv2d_716.dc.conv2d.2"(#loc346))
#loc431 = loc("relu_730"(#loc347))
#loc432 = loc("conv2d_731.dc.conv2d.2"(#loc348))
#loc433 = loc("relu_745"(#loc349))
#loc434 = loc("conv2d_746.dc.conv2d.2"(#loc350))
#loc435 = loc("conv2d_746.dc.transpose.3"(#loc350))
#loc436 = loc("conv2d_746.dc.transpose.4"(#loc350))
#loc437 = loc("conv2d_762.dc.transpose.0"(#loc351))
#loc438 = loc("conv2d_762.dc.transpose.1"(#loc351))
#loc439 = loc("conv2d_762.dc.conv2d.2"(#loc351))
#loc440 = loc("relu_776"(#loc352))
#loc441 = loc("conv2d_777.dc.conv2d.2"(#loc353))
#loc442 = loc("relu_791"(#loc354))
#loc443 = loc("conv2d_792.dc.conv2d.2"(#loc355))
#loc444 = loc("conv2d_792.dc.transpose.3"(#loc355))
#loc445 = loc("conv2d_792.dc.transpose.4"(#loc355))