#loc = loc("Segformer":0:0)
module @Segformer {
  func.func @forward(%arg0: tensor<1x3x512x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "pixel_values"} loc("Segformer":0:0), %arg1: tensor<1x16384x32xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_7.1"} loc("Segformer":0:0), %arg2: tensor<1x16384x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_7.6"} loc("Segformer":0:0), %arg3: tensor<1x16384x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_7.8"} loc("Segformer":0:0), %arg4: tensor<1x16384x32xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_8.1"} loc("Segformer":0:0), %arg5: tensor<1x16384x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_8.6"} loc("Segformer":0:0), %arg6: tensor<1x16384x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_8.8"} loc("Segformer":0:0), %arg7: tensor<1x256x32xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_22.1"} loc("Segformer":0:0), %arg8: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_22.6"} loc("Segformer":0:0), %arg9: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_22.8"} loc("Segformer":0:0), %arg10: tensor<1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_1_divide_33"} loc("Segformer":0:0), %arg11: tensor<1x16384x32xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_52.1"} loc("Segformer":0:0), %arg12: tensor<1x16384x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_52.6"} loc("Segformer":0:0), %arg13: tensor<1x16384x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_52.8"} loc("Segformer":0:0), %arg14: tensor<1x16384x32xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_73.1"} loc("Segformer":0:0), %arg15: tensor<1x16384x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_73.6"} loc("Segformer":0:0), %arg16: tensor<1x16384x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_73.8"} loc("Segformer":0:0), %arg17: tensor<1x256x32xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_87.1"} loc("Segformer":0:0), %arg18: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_87.6"} loc("Segformer":0:0), %arg19: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_87.8"} loc("Segformer":0:0), %arg20: tensor<1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_1_divide_98"} loc("Segformer":0:0), %arg21: tensor<1x16384x32xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_117.1"} loc("Segformer":0:0), %arg22: tensor<1x16384x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_117.6"} loc("Segformer":0:0), %arg23: tensor<1x16384x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_117.8"} loc("Segformer":0:0), %arg24: tensor<1x16384x32xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_138.1"} loc("Segformer":0:0), %arg25: tensor<1x16384x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_138.6"} loc("Segformer":0:0), %arg26: tensor<1x16384x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_138.8"} loc("Segformer":0:0), %arg27: tensor<1x4096x64xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_149.1"} loc("Segformer":0:0), %arg28: tensor<1x4096x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_149.6"} loc("Segformer":0:0), %arg29: tensor<1x4096x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_149.8"} loc("Segformer":0:0), %arg30: tensor<1x4096x64xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_150.1"} loc("Segformer":0:0), %arg31: tensor<1x4096x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_150.6"} loc("Segformer":0:0), %arg32: tensor<1x4096x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_150.8"} loc("Segformer":0:0), %arg33: tensor<1x256x64xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_165.1"} loc("Segformer":0:0), %arg34: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_165.6"} loc("Segformer":0:0), %arg35: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_165.8"} loc("Segformer":0:0), %arg36: tensor<1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_1_divide_177"} loc("Segformer":0:0), %arg37: tensor<1x4096x64xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_200.1"} loc("Segformer":0:0), %arg38: tensor<1x4096x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_200.6"} loc("Segformer":0:0), %arg39: tensor<1x4096x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_200.8"} loc("Segformer":0:0), %arg40: tensor<1x4096x64xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_221.1"} loc("Segformer":0:0), %arg41: tensor<1x4096x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_221.6"} loc("Segformer":0:0), %arg42: tensor<1x4096x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_221.8"} loc("Segformer":0:0), %arg43: tensor<1x256x64xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_236.1"} loc("Segformer":0:0), %arg44: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_236.6"} loc("Segformer":0:0), %arg45: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_236.8"} loc("Segformer":0:0), %arg46: tensor<1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_1_divide_248"} loc("Segformer":0:0), %arg47: tensor<1x4096x64xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_271.1"} loc("Segformer":0:0), %arg48: tensor<1x4096x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_271.6"} loc("Segformer":0:0), %arg49: tensor<1x4096x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_271.8"} loc("Segformer":0:0), %arg50: tensor<1x4096x64xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_292.1"} loc("Segformer":0:0), %arg51: tensor<1x4096x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_292.6"} loc("Segformer":0:0), %arg52: tensor<1x4096x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_292.8"} loc("Segformer":0:0), %arg53: tensor<1x1024x160xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_303.1"} loc("Segformer":0:0), %arg54: tensor<1x1024x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_303.6"} loc("Segformer":0:0), %arg55: tensor<1x1024x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_303.8"} loc("Segformer":0:0), %arg56: tensor<1x1024x160xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_304.1"} loc("Segformer":0:0), %arg57: tensor<1x1024x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_304.6"} loc("Segformer":0:0), %arg58: tensor<1x1024x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_304.8"} loc("Segformer":0:0), %arg59: tensor<1x256x160xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_319.1"} loc("Segformer":0:0), %arg60: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_319.6"} loc("Segformer":0:0), %arg61: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_319.8"} loc("Segformer":0:0), %arg62: tensor<1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_1_divide_331"} loc("Segformer":0:0), %arg63: tensor<1x1024x160xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_354.1"} loc("Segformer":0:0), %arg64: tensor<1x1024x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_354.6"} loc("Segformer":0:0), %arg65: tensor<1x1024x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_354.8"} loc("Segformer":0:0), %arg66: tensor<1x1024x160xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_375.1"} loc("Segformer":0:0), %arg67: tensor<1x1024x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_375.6"} loc("Segformer":0:0), %arg68: tensor<1x1024x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_375.8"} loc("Segformer":0:0), %arg69: tensor<1x256x160xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_390.1"} loc("Segformer":0:0), %arg70: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_390.6"} loc("Segformer":0:0), %arg71: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_390.8"} loc("Segformer":0:0), %arg72: tensor<1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_1_divide_402"} loc("Segformer":0:0), %arg73: tensor<1x1024x160xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_425.1"} loc("Segformer":0:0), %arg74: tensor<1x1024x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_425.6"} loc("Segformer":0:0), %arg75: tensor<1x1024x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_425.8"} loc("Segformer":0:0), %arg76: tensor<1x1024x160xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_446.1"} loc("Segformer":0:0), %arg77: tensor<1x1024x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_446.6"} loc("Segformer":0:0), %arg78: tensor<1x1024x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_446.8"} loc("Segformer":0:0), %arg79: tensor<1x256x256xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_457.1"} loc("Segformer":0:0), %arg80: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_457.6"} loc("Segformer":0:0), %arg81: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_457.8"} loc("Segformer":0:0), %arg82: tensor<1x256x256xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_458.1"} loc("Segformer":0:0), %arg83: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_458.6"} loc("Segformer":0:0), %arg84: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_458.8"} loc("Segformer":0:0), %arg85: tensor<1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_1_divide_477"} loc("Segformer":0:0), %arg86: tensor<1x256x256xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_500.1"} loc("Segformer":0:0), %arg87: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_500.6"} loc("Segformer":0:0), %arg88: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_500.8"} loc("Segformer":0:0), %arg89: tensor<1x256x256xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_521.1"} loc("Segformer":0:0), %arg90: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_521.6"} loc("Segformer":0:0), %arg91: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_521.8"} loc("Segformer":0:0), %arg92: tensor<1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_1_divide_540"} loc("Segformer":0:0), %arg93: tensor<1x256x256xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_563.1"} loc("Segformer":0:0), %arg94: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_563.6"} loc("Segformer":0:0), %arg95: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_563.8"} loc("Segformer":0:0), %arg96: tensor<1x256x256xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_584.1"} loc("Segformer":0:0), %arg97: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_584.6"} loc("Segformer":0:0), %arg98: tensor<1x256x1xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "dc.input_tensor.layernorm_584.8"} loc("Segformer":0:0), %arg99: tensor<32x3x7x7xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.patch_embeddings.0.proj.weight"} loc("Segformer":0:0), %arg100: tensor<1x1x1x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.patch_embeddings.0.proj.bias"} loc("Segformer":0:0), %arg101: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.patch_embeddings.0.layer_norm.weight"} loc("Segformer":0:0), %arg102: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.patch_embeddings.0.layer_norm.bias"} loc("Segformer":0:0), %arg103: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.layer_norm_1.weight"} loc("Segformer":0:0), %arg104: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.layer_norm_1.bias"} loc("Segformer":0:0), %arg105: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.attention.self.query.weight"} loc("Segformer":0:0), %arg106: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.attention.self.query.bias"} loc("Segformer":0:0), %arg107: tensor<32x32x8x8xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.attention.self.sr.weight"} loc("Segformer":0:0), %arg108: tensor<1x1x1x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.attention.self.sr.bias"} loc("Segformer":0:0), %arg109: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.attention.self.layer_norm.weight"} loc("Segformer":0:0), %arg110: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.attention.self.layer_norm.bias"} loc("Segformer":0:0), %arg111: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.attention.self.key.weight"} loc("Segformer":0:0), %arg112: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.attention.self.key.bias"} loc("Segformer":0:0), %arg113: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.attention.self.value.weight"} loc("Segformer":0:0), %arg114: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.attention.self.value.bias"} loc("Segformer":0:0), %arg115: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.attention.output.dense.weight"} loc("Segformer":0:0), %arg116: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.attention.output.dense.bias"} loc("Segformer":0:0), %arg117: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.layer_norm_2.weight"} loc("Segformer":0:0), %arg118: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.layer_norm_2.bias"} loc("Segformer":0:0), %arg119: tensor<32x128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.mlp.dense1.weight"} loc("Segformer":0:0), %arg120: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.mlp.dense1.bias"} loc("Segformer":0:0), %arg121: tensor<128x1x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.mlp.dwconv.dwconv.weight"} loc("Segformer":0:0), %arg122: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.mlp.dwconv.dwconv.bias"} loc("Segformer":0:0), %arg123: tensor<128x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.mlp.dense2.weight"} loc("Segformer":0:0), %arg124: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.0.mlp.dense2.bias"} loc("Segformer":0:0), %arg125: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.layer_norm_1.weight"} loc("Segformer":0:0), %arg126: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.layer_norm_1.bias"} loc("Segformer":0:0), %arg127: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.attention.self.query.weight"} loc("Segformer":0:0), %arg128: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.attention.self.query.bias"} loc("Segformer":0:0), %arg129: tensor<32x32x8x8xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.attention.self.sr.weight"} loc("Segformer":0:0), %arg130: tensor<1x1x1x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.attention.self.sr.bias"} loc("Segformer":0:0), %arg131: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.attention.self.layer_norm.weight"} loc("Segformer":0:0), %arg132: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.attention.self.layer_norm.bias"} loc("Segformer":0:0), %arg133: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.attention.self.key.weight"} loc("Segformer":0:0), %arg134: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.attention.self.key.bias"} loc("Segformer":0:0), %arg135: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.attention.self.value.weight"} loc("Segformer":0:0), %arg136: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.attention.self.value.bias"} loc("Segformer":0:0), %arg137: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.attention.output.dense.weight"} loc("Segformer":0:0), %arg138: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.attention.output.dense.bias"} loc("Segformer":0:0), %arg139: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.layer_norm_2.weight"} loc("Segformer":0:0), %arg140: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.layer_norm_2.bias"} loc("Segformer":0:0), %arg141: tensor<32x128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.mlp.dense1.weight"} loc("Segformer":0:0), %arg142: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.mlp.dense1.bias"} loc("Segformer":0:0), %arg143: tensor<128x1x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.mlp.dwconv.dwconv.weight"} loc("Segformer":0:0), %arg144: tensor<1x1x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.mlp.dwconv.dwconv.bias"} loc("Segformer":0:0), %arg145: tensor<128x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.mlp.dense2.weight"} loc("Segformer":0:0), %arg146: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.0.1.mlp.dense2.bias"} loc("Segformer":0:0), %arg147: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.layer_norm.0.weight"} loc("Segformer":0:0), %arg148: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.layer_norm.0.bias"} loc("Segformer":0:0), %arg149: tensor<64x32x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.patch_embeddings.1.proj.weight"} loc("Segformer":0:0), %arg150: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.patch_embeddings.1.proj.bias"} loc("Segformer":0:0), %arg151: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.patch_embeddings.1.layer_norm.weight"} loc("Segformer":0:0), %arg152: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.patch_embeddings.1.layer_norm.bias"} loc("Segformer":0:0), %arg153: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.layer_norm_1.weight"} loc("Segformer":0:0), %arg154: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.layer_norm_1.bias"} loc("Segformer":0:0), %arg155: tensor<64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.attention.self.query.weight"} loc("Segformer":0:0), %arg156: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.attention.self.query.bias"} loc("Segformer":0:0), %arg157: tensor<64x64x4x4xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.attention.self.sr.weight"} loc("Segformer":0:0), %arg158: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.attention.self.sr.bias"} loc("Segformer":0:0), %arg159: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.attention.self.layer_norm.weight"} loc("Segformer":0:0), %arg160: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.attention.self.layer_norm.bias"} loc("Segformer":0:0), %arg161: tensor<64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.attention.self.key.weight"} loc("Segformer":0:0), %arg162: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.attention.self.key.bias"} loc("Segformer":0:0), %arg163: tensor<64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.attention.self.value.weight"} loc("Segformer":0:0), %arg164: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.attention.self.value.bias"} loc("Segformer":0:0), %arg165: tensor<64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.attention.output.dense.weight"} loc("Segformer":0:0), %arg166: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.attention.output.dense.bias"} loc("Segformer":0:0), %arg167: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.layer_norm_2.weight"} loc("Segformer":0:0), %arg168: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.layer_norm_2.bias"} loc("Segformer":0:0), %arg169: tensor<64x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.mlp.dense1.weight"} loc("Segformer":0:0), %arg170: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.mlp.dense1.bias"} loc("Segformer":0:0), %arg171: tensor<256x1x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.mlp.dwconv.dwconv.weight"} loc("Segformer":0:0), %arg172: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.mlp.dwconv.dwconv.bias"} loc("Segformer":0:0), %arg173: tensor<256x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.mlp.dense2.weight"} loc("Segformer":0:0), %arg174: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.0.mlp.dense2.bias"} loc("Segformer":0:0), %arg175: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.layer_norm_1.weight"} loc("Segformer":0:0), %arg176: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.layer_norm_1.bias"} loc("Segformer":0:0), %arg177: tensor<64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.attention.self.query.weight"} loc("Segformer":0:0), %arg178: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.attention.self.query.bias"} loc("Segformer":0:0), %arg179: tensor<64x64x4x4xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.attention.self.sr.weight"} loc("Segformer":0:0), %arg180: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.attention.self.sr.bias"} loc("Segformer":0:0), %arg181: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.attention.self.layer_norm.weight"} loc("Segformer":0:0), %arg182: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.attention.self.layer_norm.bias"} loc("Segformer":0:0), %arg183: tensor<64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.attention.self.key.weight"} loc("Segformer":0:0), %arg184: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.attention.self.key.bias"} loc("Segformer":0:0), %arg185: tensor<64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.attention.self.value.weight"} loc("Segformer":0:0), %arg186: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.attention.self.value.bias"} loc("Segformer":0:0), %arg187: tensor<64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.attention.output.dense.weight"} loc("Segformer":0:0), %arg188: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.attention.output.dense.bias"} loc("Segformer":0:0), %arg189: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.layer_norm_2.weight"} loc("Segformer":0:0), %arg190: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.layer_norm_2.bias"} loc("Segformer":0:0), %arg191: tensor<64x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.mlp.dense1.weight"} loc("Segformer":0:0), %arg192: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.mlp.dense1.bias"} loc("Segformer":0:0), %arg193: tensor<256x1x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.mlp.dwconv.dwconv.weight"} loc("Segformer":0:0), %arg194: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.mlp.dwconv.dwconv.bias"} loc("Segformer":0:0), %arg195: tensor<256x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.mlp.dense2.weight"} loc("Segformer":0:0), %arg196: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.1.1.mlp.dense2.bias"} loc("Segformer":0:0), %arg197: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.layer_norm.1.weight"} loc("Segformer":0:0), %arg198: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.layer_norm.1.bias"} loc("Segformer":0:0), %arg199: tensor<160x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.patch_embeddings.2.proj.weight"} loc("Segformer":0:0), %arg200: tensor<1x1x1x160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.patch_embeddings.2.proj.bias"} loc("Segformer":0:0), %arg201: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.patch_embeddings.2.layer_norm.weight"} loc("Segformer":0:0), %arg202: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.patch_embeddings.2.layer_norm.bias"} loc("Segformer":0:0), %arg203: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.layer_norm_1.weight"} loc("Segformer":0:0), %arg204: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.layer_norm_1.bias"} loc("Segformer":0:0), %arg205: tensor<160x160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.attention.self.query.weight"} loc("Segformer":0:0), %arg206: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.attention.self.query.bias"} loc("Segformer":0:0), %arg207: tensor<160x160x2x2xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.attention.self.sr.weight"} loc("Segformer":0:0), %arg208: tensor<1x1x1x160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.attention.self.sr.bias"} loc("Segformer":0:0), %arg209: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.attention.self.layer_norm.weight"} loc("Segformer":0:0), %arg210: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.attention.self.layer_norm.bias"} loc("Segformer":0:0), %arg211: tensor<160x160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.attention.self.key.weight"} loc("Segformer":0:0), %arg212: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.attention.self.key.bias"} loc("Segformer":0:0), %arg213: tensor<160x160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.attention.self.value.weight"} loc("Segformer":0:0), %arg214: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.attention.self.value.bias"} loc("Segformer":0:0), %arg215: tensor<160x160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.attention.output.dense.weight"} loc("Segformer":0:0), %arg216: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.attention.output.dense.bias"} loc("Segformer":0:0), %arg217: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.layer_norm_2.weight"} loc("Segformer":0:0), %arg218: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.layer_norm_2.bias"} loc("Segformer":0:0), %arg219: tensor<160x640xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.mlp.dense1.weight"} loc("Segformer":0:0), %arg220: tensor<640xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.mlp.dense1.bias"} loc("Segformer":0:0), %arg221: tensor<640x1x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.mlp.dwconv.dwconv.weight"} loc("Segformer":0:0), %arg222: tensor<1x1x1x640xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.mlp.dwconv.dwconv.bias"} loc("Segformer":0:0), %arg223: tensor<640x160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.mlp.dense2.weight"} loc("Segformer":0:0), %arg224: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.0.mlp.dense2.bias"} loc("Segformer":0:0), %arg225: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.layer_norm_1.weight"} loc("Segformer":0:0), %arg226: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.layer_norm_1.bias"} loc("Segformer":0:0), %arg227: tensor<160x160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.attention.self.query.weight"} loc("Segformer":0:0), %arg228: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.attention.self.query.bias"} loc("Segformer":0:0), %arg229: tensor<160x160x2x2xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.attention.self.sr.weight"} loc("Segformer":0:0), %arg230: tensor<1x1x1x160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.attention.self.sr.bias"} loc("Segformer":0:0), %arg231: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.attention.self.layer_norm.weight"} loc("Segformer":0:0), %arg232: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.attention.self.layer_norm.bias"} loc("Segformer":0:0), %arg233: tensor<160x160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.attention.self.key.weight"} loc("Segformer":0:0), %arg234: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.attention.self.key.bias"} loc("Segformer":0:0), %arg235: tensor<160x160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.attention.self.value.weight"} loc("Segformer":0:0), %arg236: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.attention.self.value.bias"} loc("Segformer":0:0), %arg237: tensor<160x160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.attention.output.dense.weight"} loc("Segformer":0:0), %arg238: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.attention.output.dense.bias"} loc("Segformer":0:0), %arg239: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.layer_norm_2.weight"} loc("Segformer":0:0), %arg240: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.layer_norm_2.bias"} loc("Segformer":0:0), %arg241: tensor<160x640xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.mlp.dense1.weight"} loc("Segformer":0:0), %arg242: tensor<640xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.mlp.dense1.bias"} loc("Segformer":0:0), %arg243: tensor<640x1x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.mlp.dwconv.dwconv.weight"} loc("Segformer":0:0), %arg244: tensor<1x1x1x640xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.mlp.dwconv.dwconv.bias"} loc("Segformer":0:0), %arg245: tensor<640x160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.mlp.dense2.weight"} loc("Segformer":0:0), %arg246: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.2.1.mlp.dense2.bias"} loc("Segformer":0:0), %arg247: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.layer_norm.2.weight"} loc("Segformer":0:0), %arg248: tensor<160xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.layer_norm.2.bias"} loc("Segformer":0:0), %arg249: tensor<256x160x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.patch_embeddings.3.proj.weight"} loc("Segformer":0:0), %arg250: tensor<1x1x1x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.patch_embeddings.3.proj.bias"} loc("Segformer":0:0), %arg251: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.patch_embeddings.3.layer_norm.weight"} loc("Segformer":0:0), %arg252: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.patch_embeddings.3.layer_norm.bias"} loc("Segformer":0:0), %arg253: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.0.layer_norm_1.weight"} loc("Segformer":0:0), %arg254: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.0.layer_norm_1.bias"} loc("Segformer":0:0), %arg255: tensor<256x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.0.attention.self.query.weight"} loc("Segformer":0:0), %arg256: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.0.attention.self.query.bias"} loc("Segformer":0:0), %arg257: tensor<256x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.0.attention.self.key.weight"} loc("Segformer":0:0), %arg258: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.0.attention.self.key.bias"} loc("Segformer":0:0), %arg259: tensor<256x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.0.attention.self.value.weight"} loc("Segformer":0:0), %arg260: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.0.attention.self.value.bias"} loc("Segformer":0:0), %arg261: tensor<256x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.0.attention.output.dense.weight"} loc("Segformer":0:0), %arg262: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.0.attention.output.dense.bias"} loc("Segformer":0:0), %arg263: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.0.layer_norm_2.weight"} loc("Segformer":0:0), %arg264: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.0.layer_norm_2.bias"} loc("Segformer":0:0), %arg265: tensor<256x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.0.mlp.dense1.weight"} loc("Segformer":0:0), %arg266: tensor<1024xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.0.mlp.dense1.bias"} loc("Segformer":0:0), %arg267: tensor<1024x1x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.0.mlp.dwconv.dwconv.weight"} loc("Segformer":0:0), %arg268: tensor<1x1x1x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.0.mlp.dwconv.dwconv.bias"} loc("Segformer":0:0), %arg269: tensor<1024x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.0.mlp.dense2.weight"} loc("Segformer":0:0), %arg270: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.0.mlp.dense2.bias"} loc("Segformer":0:0), %arg271: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.1.layer_norm_1.weight"} loc("Segformer":0:0), %arg272: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.1.layer_norm_1.bias"} loc("Segformer":0:0), %arg273: tensor<256x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.1.attention.self.query.weight"} loc("Segformer":0:0), %arg274: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.1.attention.self.query.bias"} loc("Segformer":0:0), %arg275: tensor<256x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.1.attention.self.key.weight"} loc("Segformer":0:0), %arg276: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.1.attention.self.key.bias"} loc("Segformer":0:0), %arg277: tensor<256x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.1.attention.self.value.weight"} loc("Segformer":0:0), %arg278: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.1.attention.self.value.bias"} loc("Segformer":0:0), %arg279: tensor<256x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.1.attention.output.dense.weight"} loc("Segformer":0:0), %arg280: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.1.attention.output.dense.bias"} loc("Segformer":0:0), %arg281: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.1.layer_norm_2.weight"} loc("Segformer":0:0), %arg282: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.1.layer_norm_2.bias"} loc("Segformer":0:0), %arg283: tensor<256x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.1.mlp.dense1.weight"} loc("Segformer":0:0), %arg284: tensor<1024xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.1.mlp.dense1.bias"} loc("Segformer":0:0), %arg285: tensor<1024x1x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.1.mlp.dwconv.dwconv.weight"} loc("Segformer":0:0), %arg286: tensor<1x1x1x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.1.mlp.dwconv.dwconv.bias"} loc("Segformer":0:0), %arg287: tensor<1024x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.1.mlp.dense2.weight"} loc("Segformer":0:0), %arg288: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.block.3.1.mlp.dense2.bias"} loc("Segformer":0:0), %arg289: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.layer_norm.3.weight"} loc("Segformer":0:0), %arg290: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "segformer.encoder.layer_norm.3.bias"} loc("Segformer":0:0), %arg291: tensor<256x1000xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "classifier.weight"} loc("Segformer":0:0), %arg292: tensor<1000xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "classifier.bias"} loc("Segformer":0:0)) -> (tensor<1x1000xbf16> {ttir.name = "Segformer.output_add_590"}) {
    %0 = "ttir.transpose"(%arg0) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x3x512x512xbf16>) -> tensor<1x512x3x512xbf16>
    %1 = "ttir.transpose"(%0) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x512x3x512xbf16>) -> tensor<1x512x512x3xbf16>
    %2 = "ttir.conv2d"(%1, %arg99, %arg100) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 3, 3, 3, 3>, stride = array<i32: 4, 4>}> {channel_last = 1 : si32} : (tensor<1x512x512x3xbf16>, tensor<32x3x7x7xbf16>, tensor<1x1x1x32xbf16>) -> tensor<1x128x128x32xbf16>
    %3 = "ttir.transpose"(%2) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x128x128x32xbf16>) -> tensor<1x128x32x128xbf16>
    %4 = "ttir.transpose"(%3) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x128x32x128xbf16>) -> tensor<1x32x128x128xbf16>
    %5 = "ttir.reshape"(%4) <{shape = [1 : i32, 32 : i32, 16384 : i32, 1 : i32]}> : (tensor<1x32x128x128xbf16>) -> tensor<1x32x16384x1xbf16>
    %6 = "ttir.squeeze"(%5) <{dim = -1 : si32}> : (tensor<1x32x16384x1xbf16>) -> tensor<1x32x16384xbf16>
    %7 = "ttir.transpose"(%6) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x16384xbf16>) -> tensor<1x16384x32xbf16>
    %8 = "ttir.sum"(%7) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x16384x32xbf16>) -> tensor<1x16384x1xbf16>
    %9 = "ttir.multiply"(%arg1, %8) : (tensor<1x16384x32xf32>, tensor<1x16384x1xbf16>) -> tensor<1x16384x32xbf16>
    %10 = "ttir.subtract"(%7, %9) : (tensor<1x16384x32xbf16>, tensor<1x16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %11 = "ttir.multiply"(%10, %10) : (tensor<1x16384x32xbf16>, tensor<1x16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %12 = "ttir.sum"(%11) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x16384x32xbf16>) -> tensor<1x16384x1xbf16>
    %13 = "ttir.multiply"(%arg2, %12) : (tensor<1x16384x1xf32>, tensor<1x16384x1xbf16>) -> tensor<1x16384x1xbf16>
    %14 = "ttir.add"(%13, %arg3) : (tensor<1x16384x1xbf16>, tensor<1x16384x1xf32>) -> tensor<1x16384x1xbf16>
    %15 = "ttir.sqrt"(%14) : (tensor<1x16384x1xbf16>) -> tensor<1x16384x1xbf16>
    %16 = "ttir.reciprocal"(%15) : (tensor<1x16384x1xbf16>) -> tensor<1x16384x1xbf16>
    %17 = "ttir.multiply"(%10, %16) : (tensor<1x16384x32xbf16>, tensor<1x16384x1xbf16>) -> tensor<1x16384x32xbf16>
    %18 = "ttir.multiply"(%17, %arg101) : (tensor<1x16384x32xbf16>, tensor<32xbf16>) -> tensor<1x16384x32xbf16>
    %19 = "ttir.add"(%18, %arg102) : (tensor<1x16384x32xbf16>, tensor<32xbf16>) -> tensor<1x16384x32xbf16>
    %20 = "ttir.sum"(%19) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x16384x32xbf16>) -> tensor<1x16384x1xbf16>
    %21 = "ttir.multiply"(%arg4, %20) : (tensor<1x16384x32xf32>, tensor<1x16384x1xbf16>) -> tensor<1x16384x32xbf16>
    %22 = "ttir.subtract"(%19, %21) : (tensor<1x16384x32xbf16>, tensor<1x16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %23 = "ttir.multiply"(%22, %22) : (tensor<1x16384x32xbf16>, tensor<1x16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %24 = "ttir.sum"(%23) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x16384x32xbf16>) -> tensor<1x16384x1xbf16>
    %25 = "ttir.multiply"(%arg5, %24) : (tensor<1x16384x1xf32>, tensor<1x16384x1xbf16>) -> tensor<1x16384x1xbf16>
    %26 = "ttir.add"(%25, %arg6) : (tensor<1x16384x1xbf16>, tensor<1x16384x1xf32>) -> tensor<1x16384x1xbf16>
    %27 = "ttir.sqrt"(%26) : (tensor<1x16384x1xbf16>) -> tensor<1x16384x1xbf16>
    %28 = "ttir.reciprocal"(%27) : (tensor<1x16384x1xbf16>) -> tensor<1x16384x1xbf16>
    %29 = "ttir.multiply"(%22, %28) : (tensor<1x16384x32xbf16>, tensor<1x16384x1xbf16>) -> tensor<1x16384x32xbf16>
    %30 = "ttir.multiply"(%29, %arg103) : (tensor<1x16384x32xbf16>, tensor<32xbf16>) -> tensor<1x16384x32xbf16>
    %31 = "ttir.add"(%30, %arg104) : (tensor<1x16384x32xbf16>, tensor<32xbf16>) -> tensor<1x16384x32xbf16>
    %32 = "ttir.matmul"(%31, %arg105) <{transpose_a = false, transpose_b = false}> : (tensor<1x16384x32xbf16>, tensor<32x32xbf16>) -> tensor<1x16384x32xbf16>
    %33 = "ttir.add"(%32, %arg106) : (tensor<1x16384x32xbf16>, tensor<32xbf16>) -> tensor<1x16384x32xbf16>
    %34 = "ttir.reshape"(%33) <{shape = [1 : i32, 16384 : i32, 1 : i32, 32 : i32]}> : (tensor<1x16384x32xbf16>) -> tensor<1x16384x1x32xbf16>
    %35 = "ttir.squeeze"(%34) <{dim = -2 : si32}> : (tensor<1x16384x1x32xbf16>) -> tensor<1x16384x32xbf16>
    %36 = "ttir.transpose"(%31) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x16384x32xbf16>) -> tensor<1x32x16384xbf16>
    %37 = "ttir.reshape"(%36) <{shape = [1 : i32, 32 : i32, 128 : i32, 128 : i32]}> : (tensor<1x32x16384xbf16>) -> tensor<1x32x128x128xbf16>
    %38 = "ttir.transpose"(%37) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x128x128xbf16>) -> tensor<1x128x32x128xbf16>
    %39 = "ttir.transpose"(%38) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x128x32x128xbf16>) -> tensor<1x128x128x32xbf16>
    %40 = "ttir.conv2d"(%39, %arg107, %arg108) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 8, 8>}> {channel_last = 1 : si32} : (tensor<1x128x128x32xbf16>, tensor<32x32x8x8xbf16>, tensor<1x1x1x32xbf16>) -> tensor<1x16x16x32xbf16>
    %41 = "ttir.transpose"(%40) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x16x16x32xbf16>) -> tensor<1x16x32x16xbf16>
    %42 = "ttir.transpose"(%41) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x16x32x16xbf16>) -> tensor<1x32x16x16xbf16>
    %43 = "ttir.reshape"(%42) <{shape = [1 : i32, 32 : i32, 256 : i32]}> : (tensor<1x32x16x16xbf16>) -> tensor<1x32x256xbf16>
    %44 = "ttir.transpose"(%43) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x256xbf16>) -> tensor<1x256x32xbf16>
    %45 = "ttir.sum"(%44) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x32xbf16>) -> tensor<1x256x1xbf16>
    %46 = "ttir.multiply"(%arg7, %45) : (tensor<1x256x32xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x32xbf16>
    %47 = "ttir.subtract"(%44, %46) : (tensor<1x256x32xbf16>, tensor<1x256x32xbf16>) -> tensor<1x256x32xbf16>
    %48 = "ttir.multiply"(%47, %47) : (tensor<1x256x32xbf16>, tensor<1x256x32xbf16>) -> tensor<1x256x32xbf16>
    %49 = "ttir.sum"(%48) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x32xbf16>) -> tensor<1x256x1xbf16>
    %50 = "ttir.multiply"(%arg8, %49) : (tensor<1x256x1xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %51 = "ttir.add"(%50, %arg9) : (tensor<1x256x1xbf16>, tensor<1x256x1xf32>) -> tensor<1x256x1xbf16>
    %52 = "ttir.sqrt"(%51) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %53 = "ttir.reciprocal"(%52) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %54 = "ttir.multiply"(%47, %53) : (tensor<1x256x32xbf16>, tensor<1x256x1xbf16>) -> tensor<1x256x32xbf16>
    %55 = "ttir.multiply"(%54, %arg109) : (tensor<1x256x32xbf16>, tensor<32xbf16>) -> tensor<1x256x32xbf16>
    %56 = "ttir.add"(%55, %arg110) : (tensor<1x256x32xbf16>, tensor<32xbf16>) -> tensor<1x256x32xbf16>
    %57 = "ttir.squeeze"(%56) <{dim = 0 : si32}> : (tensor<1x256x32xbf16>) -> tensor<256x32xbf16>
    %58 = "ttir.matmul"(%57, %arg111) <{transpose_a = false, transpose_b = false}> : (tensor<256x32xbf16>, tensor<32x32xbf16>) -> tensor<256x32xbf16>
    %59 = "ttir.unsqueeze"(%58) <{dim = 0 : si32}> : (tensor<256x32xbf16>) -> tensor<1x256x32xbf16>
    %60 = "ttir.add"(%59, %arg112) : (tensor<1x256x32xbf16>, tensor<32xbf16>) -> tensor<1x256x32xbf16>
    %61 = "ttir.reshape"(%60) <{shape = [1 : i32, 256 : i32, 1 : i32, 32 : i32]}> : (tensor<1x256x32xbf16>) -> tensor<1x256x1x32xbf16>
    %62 = "ttir.squeeze"(%61) <{dim = -2 : si32}> : (tensor<1x256x1x32xbf16>) -> tensor<1x256x32xbf16>
    %63 = "ttir.transpose"(%62) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x256x32xbf16>) -> tensor<1x32x256xbf16>
    %64 = "ttir.matmul"(%35, %63) <{transpose_a = false, transpose_b = false}> : (tensor<1x16384x32xbf16>, tensor<1x32x256xbf16>) -> tensor<1x16384x256xbf16>
    %65 = "ttir.unsqueeze"(%64) <{dim = 0 : si32}> : (tensor<1x16384x256xbf16>) -> tensor<1x1x16384x256xbf16>
    %66 = "ttir.div"(%65, %arg10) : (tensor<1x1x16384x256xbf16>, tensor<1xbf16>) -> tensor<1x1x16384x256xbf16>
    %67 = "ttir.softmax"(%66) <{dimension = -1 : si32}> : (tensor<1x1x16384x256xbf16>) -> tensor<1x1x16384x256xbf16>
    %68 = "ttir.squeeze"(%67) <{dim = 0 : si32}> : (tensor<1x1x16384x256xbf16>) -> tensor<1x16384x256xbf16>
    %69 = "ttir.matmul"(%57, %arg113) <{transpose_a = false, transpose_b = false}> : (tensor<256x32xbf16>, tensor<32x32xbf16>) -> tensor<256x32xbf16>
    %70 = "ttir.unsqueeze"(%69) <{dim = 0 : si32}> : (tensor<256x32xbf16>) -> tensor<1x256x32xbf16>
    %71 = "ttir.add"(%70, %arg114) : (tensor<1x256x32xbf16>, tensor<32xbf16>) -> tensor<1x256x32xbf16>
    %72 = "ttir.reshape"(%71) <{shape = [1 : i32, 256 : i32, 1 : i32, 32 : i32]}> : (tensor<1x256x32xbf16>) -> tensor<1x256x1x32xbf16>
    %73 = "ttir.transpose"(%72) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x256x1x32xbf16>) -> tensor<1x1x256x32xbf16>
    %74 = "ttir.transpose"(%73) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x1x256x32xbf16>) -> tensor<1x1x32x256xbf16>
    %75 = "ttir.squeeze"(%74) <{dim = 0 : si32}> : (tensor<1x1x32x256xbf16>) -> tensor<1x32x256xbf16>
    %76 = "ttir.transpose"(%75) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x256xbf16>) -> tensor<1x256x32xbf16>
    %77 = "ttir.matmul"(%68, %76) <{transpose_a = false, transpose_b = false}> : (tensor<1x16384x256xbf16>, tensor<1x256x32xbf16>) -> tensor<1x16384x32xbf16>
    %78 = "ttir.matmul"(%77, %arg115) <{transpose_a = false, transpose_b = false}> : (tensor<1x16384x32xbf16>, tensor<32x32xbf16>) -> tensor<1x16384x32xbf16>
    %79 = "ttir.add"(%78, %arg116) : (tensor<1x16384x32xbf16>, tensor<32xbf16>) -> tensor<1x16384x32xbf16>
    %80 = "ttir.add"(%79, %19) : (tensor<1x16384x32xbf16>, tensor<1x16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %81 = "ttir.sum"(%80) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x16384x32xbf16>) -> tensor<1x16384x1xbf16>
    %82 = "ttir.multiply"(%arg11, %81) : (tensor<1x16384x32xf32>, tensor<1x16384x1xbf16>) -> tensor<1x16384x32xbf16>
    %83 = "ttir.subtract"(%80, %82) : (tensor<1x16384x32xbf16>, tensor<1x16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %84 = "ttir.multiply"(%83, %83) : (tensor<1x16384x32xbf16>, tensor<1x16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %85 = "ttir.sum"(%84) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x16384x32xbf16>) -> tensor<1x16384x1xbf16>
    %86 = "ttir.multiply"(%arg12, %85) : (tensor<1x16384x1xf32>, tensor<1x16384x1xbf16>) -> tensor<1x16384x1xbf16>
    %87 = "ttir.add"(%86, %arg13) : (tensor<1x16384x1xbf16>, tensor<1x16384x1xf32>) -> tensor<1x16384x1xbf16>
    %88 = "ttir.sqrt"(%87) : (tensor<1x16384x1xbf16>) -> tensor<1x16384x1xbf16>
    %89 = "ttir.reciprocal"(%88) : (tensor<1x16384x1xbf16>) -> tensor<1x16384x1xbf16>
    %90 = "ttir.multiply"(%83, %89) : (tensor<1x16384x32xbf16>, tensor<1x16384x1xbf16>) -> tensor<1x16384x32xbf16>
    %91 = "ttir.multiply"(%90, %arg117) : (tensor<1x16384x32xbf16>, tensor<32xbf16>) -> tensor<1x16384x32xbf16>
    %92 = "ttir.add"(%91, %arg118) : (tensor<1x16384x32xbf16>, tensor<32xbf16>) -> tensor<1x16384x32xbf16>
    %93 = "ttir.matmul"(%92, %arg119) <{transpose_a = false, transpose_b = false}> : (tensor<1x16384x32xbf16>, tensor<32x128xbf16>) -> tensor<1x16384x128xbf16>
    %94 = "ttir.add"(%93, %arg120) : (tensor<1x16384x128xbf16>, tensor<128xbf16>) -> tensor<1x16384x128xbf16>
    %95 = "ttir.transpose"(%94) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x16384x128xbf16>) -> tensor<1x128x16384xbf16>
    %96 = "ttir.reshape"(%95) <{shape = [1 : i32, 128 : i32, 128 : i32, 128 : i32]}> : (tensor<1x128x16384xbf16>) -> tensor<1x128x128x128xbf16>
    %97 = "ttir.transpose"(%96) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x128x128x128xbf16>) -> tensor<1x128x128x128xbf16>
    %98 = "ttir.transpose"(%97) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x128x128x128xbf16>) -> tensor<1x128x128x128xbf16>
    %99 = "ttir.conv2d"(%98, %arg121, %arg122) <{dilation = array<i32: 1, 1>, groups = 128 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x128x128x128xbf16>, tensor<128x1x3x3xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x128x128x128xbf16>
    %100 = "ttir.transpose"(%99) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x128x128x128xbf16>) -> tensor<1x128x128x128xbf16>
    %101 = "ttir.transpose"(%100) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x128x128x128xbf16>) -> tensor<1x128x128x128xbf16>
    %102 = "ttir.reshape"(%101) <{shape = [1 : i32, 128 : i32, 16384 : i32, 1 : i32]}> : (tensor<1x128x128x128xbf16>) -> tensor<1x128x16384x1xbf16>
    %103 = "ttir.squeeze"(%102) <{dim = -1 : si32}> : (tensor<1x128x16384x1xbf16>) -> tensor<1x128x16384xbf16>
    %104 = "ttir.transpose"(%103) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x128x16384xbf16>) -> tensor<1x16384x128xbf16>
    %105 = "ttir.gelu"(%104) : (tensor<1x16384x128xbf16>) -> tensor<1x16384x128xbf16>
    %106 = "ttir.matmul"(%105, %arg123) <{transpose_a = false, transpose_b = false}> : (tensor<1x16384x128xbf16>, tensor<128x32xbf16>) -> tensor<1x16384x32xbf16>
    %107 = "ttir.add"(%106, %arg124) : (tensor<1x16384x32xbf16>, tensor<32xbf16>) -> tensor<1x16384x32xbf16>
    %108 = "ttir.add"(%107, %80) : (tensor<1x16384x32xbf16>, tensor<1x16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %109 = "ttir.sum"(%108) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x16384x32xbf16>) -> tensor<1x16384x1xbf16>
    %110 = "ttir.multiply"(%arg14, %109) : (tensor<1x16384x32xf32>, tensor<1x16384x1xbf16>) -> tensor<1x16384x32xbf16>
    %111 = "ttir.subtract"(%108, %110) : (tensor<1x16384x32xbf16>, tensor<1x16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %112 = "ttir.multiply"(%111, %111) : (tensor<1x16384x32xbf16>, tensor<1x16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %113 = "ttir.sum"(%112) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x16384x32xbf16>) -> tensor<1x16384x1xbf16>
    %114 = "ttir.multiply"(%arg15, %113) : (tensor<1x16384x1xf32>, tensor<1x16384x1xbf16>) -> tensor<1x16384x1xbf16>
    %115 = "ttir.add"(%114, %arg16) : (tensor<1x16384x1xbf16>, tensor<1x16384x1xf32>) -> tensor<1x16384x1xbf16>
    %116 = "ttir.sqrt"(%115) : (tensor<1x16384x1xbf16>) -> tensor<1x16384x1xbf16>
    %117 = "ttir.reciprocal"(%116) : (tensor<1x16384x1xbf16>) -> tensor<1x16384x1xbf16>
    %118 = "ttir.multiply"(%111, %117) : (tensor<1x16384x32xbf16>, tensor<1x16384x1xbf16>) -> tensor<1x16384x32xbf16>
    %119 = "ttir.multiply"(%118, %arg125) : (tensor<1x16384x32xbf16>, tensor<32xbf16>) -> tensor<1x16384x32xbf16>
    %120 = "ttir.add"(%119, %arg126) : (tensor<1x16384x32xbf16>, tensor<32xbf16>) -> tensor<1x16384x32xbf16>
    %121 = "ttir.matmul"(%120, %arg127) <{transpose_a = false, transpose_b = false}> : (tensor<1x16384x32xbf16>, tensor<32x32xbf16>) -> tensor<1x16384x32xbf16>
    %122 = "ttir.add"(%121, %arg128) : (tensor<1x16384x32xbf16>, tensor<32xbf16>) -> tensor<1x16384x32xbf16>
    %123 = "ttir.reshape"(%122) <{shape = [1 : i32, 16384 : i32, 1 : i32, 32 : i32]}> : (tensor<1x16384x32xbf16>) -> tensor<1x16384x1x32xbf16>
    %124 = "ttir.squeeze"(%123) <{dim = -2 : si32}> : (tensor<1x16384x1x32xbf16>) -> tensor<1x16384x32xbf16>
    %125 = "ttir.transpose"(%120) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x16384x32xbf16>) -> tensor<1x32x16384xbf16>
    %126 = "ttir.reshape"(%125) <{shape = [1 : i32, 32 : i32, 128 : i32, 128 : i32]}> : (tensor<1x32x16384xbf16>) -> tensor<1x32x128x128xbf16>
    %127 = "ttir.transpose"(%126) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x128x128xbf16>) -> tensor<1x128x32x128xbf16>
    %128 = "ttir.transpose"(%127) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x128x32x128xbf16>) -> tensor<1x128x128x32xbf16>
    %129 = "ttir.conv2d"(%128, %arg129, %arg130) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 8, 8>}> {channel_last = 1 : si32} : (tensor<1x128x128x32xbf16>, tensor<32x32x8x8xbf16>, tensor<1x1x1x32xbf16>) -> tensor<1x16x16x32xbf16>
    %130 = "ttir.transpose"(%129) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x16x16x32xbf16>) -> tensor<1x16x32x16xbf16>
    %131 = "ttir.transpose"(%130) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x16x32x16xbf16>) -> tensor<1x32x16x16xbf16>
    %132 = "ttir.reshape"(%131) <{shape = [1 : i32, 32 : i32, 256 : i32]}> : (tensor<1x32x16x16xbf16>) -> tensor<1x32x256xbf16>
    %133 = "ttir.transpose"(%132) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x256xbf16>) -> tensor<1x256x32xbf16>
    %134 = "ttir.sum"(%133) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x32xbf16>) -> tensor<1x256x1xbf16>
    %135 = "ttir.multiply"(%arg17, %134) : (tensor<1x256x32xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x32xbf16>
    %136 = "ttir.subtract"(%133, %135) : (tensor<1x256x32xbf16>, tensor<1x256x32xbf16>) -> tensor<1x256x32xbf16>
    %137 = "ttir.multiply"(%136, %136) : (tensor<1x256x32xbf16>, tensor<1x256x32xbf16>) -> tensor<1x256x32xbf16>
    %138 = "ttir.sum"(%137) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x32xbf16>) -> tensor<1x256x1xbf16>
    %139 = "ttir.multiply"(%arg18, %138) : (tensor<1x256x1xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %140 = "ttir.add"(%139, %arg19) : (tensor<1x256x1xbf16>, tensor<1x256x1xf32>) -> tensor<1x256x1xbf16>
    %141 = "ttir.sqrt"(%140) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %142 = "ttir.reciprocal"(%141) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %143 = "ttir.multiply"(%136, %142) : (tensor<1x256x32xbf16>, tensor<1x256x1xbf16>) -> tensor<1x256x32xbf16>
    %144 = "ttir.multiply"(%143, %arg131) : (tensor<1x256x32xbf16>, tensor<32xbf16>) -> tensor<1x256x32xbf16>
    %145 = "ttir.add"(%144, %arg132) : (tensor<1x256x32xbf16>, tensor<32xbf16>) -> tensor<1x256x32xbf16>
    %146 = "ttir.squeeze"(%145) <{dim = 0 : si32}> : (tensor<1x256x32xbf16>) -> tensor<256x32xbf16>
    %147 = "ttir.matmul"(%146, %arg133) <{transpose_a = false, transpose_b = false}> : (tensor<256x32xbf16>, tensor<32x32xbf16>) -> tensor<256x32xbf16>
    %148 = "ttir.unsqueeze"(%147) <{dim = 0 : si32}> : (tensor<256x32xbf16>) -> tensor<1x256x32xbf16>
    %149 = "ttir.add"(%148, %arg134) : (tensor<1x256x32xbf16>, tensor<32xbf16>) -> tensor<1x256x32xbf16>
    %150 = "ttir.reshape"(%149) <{shape = [1 : i32, 256 : i32, 1 : i32, 32 : i32]}> : (tensor<1x256x32xbf16>) -> tensor<1x256x1x32xbf16>
    %151 = "ttir.squeeze"(%150) <{dim = -2 : si32}> : (tensor<1x256x1x32xbf16>) -> tensor<1x256x32xbf16>
    %152 = "ttir.transpose"(%151) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x256x32xbf16>) -> tensor<1x32x256xbf16>
    %153 = "ttir.matmul"(%124, %152) <{transpose_a = false, transpose_b = false}> : (tensor<1x16384x32xbf16>, tensor<1x32x256xbf16>) -> tensor<1x16384x256xbf16>
    %154 = "ttir.unsqueeze"(%153) <{dim = 0 : si32}> : (tensor<1x16384x256xbf16>) -> tensor<1x1x16384x256xbf16>
    %155 = "ttir.div"(%154, %arg20) : (tensor<1x1x16384x256xbf16>, tensor<1xbf16>) -> tensor<1x1x16384x256xbf16>
    %156 = "ttir.softmax"(%155) <{dimension = -1 : si32}> : (tensor<1x1x16384x256xbf16>) -> tensor<1x1x16384x256xbf16>
    %157 = "ttir.squeeze"(%156) <{dim = 0 : si32}> : (tensor<1x1x16384x256xbf16>) -> tensor<1x16384x256xbf16>
    %158 = "ttir.matmul"(%146, %arg135) <{transpose_a = false, transpose_b = false}> : (tensor<256x32xbf16>, tensor<32x32xbf16>) -> tensor<256x32xbf16>
    %159 = "ttir.unsqueeze"(%158) <{dim = 0 : si32}> : (tensor<256x32xbf16>) -> tensor<1x256x32xbf16>
    %160 = "ttir.add"(%159, %arg136) : (tensor<1x256x32xbf16>, tensor<32xbf16>) -> tensor<1x256x32xbf16>
    %161 = "ttir.reshape"(%160) <{shape = [1 : i32, 256 : i32, 1 : i32, 32 : i32]}> : (tensor<1x256x32xbf16>) -> tensor<1x256x1x32xbf16>
    %162 = "ttir.transpose"(%161) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x256x1x32xbf16>) -> tensor<1x1x256x32xbf16>
    %163 = "ttir.transpose"(%162) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x1x256x32xbf16>) -> tensor<1x1x32x256xbf16>
    %164 = "ttir.squeeze"(%163) <{dim = 0 : si32}> : (tensor<1x1x32x256xbf16>) -> tensor<1x32x256xbf16>
    %165 = "ttir.transpose"(%164) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x256xbf16>) -> tensor<1x256x32xbf16>
    %166 = "ttir.matmul"(%157, %165) <{transpose_a = false, transpose_b = false}> : (tensor<1x16384x256xbf16>, tensor<1x256x32xbf16>) -> tensor<1x16384x32xbf16>
    %167 = "ttir.matmul"(%166, %arg137) <{transpose_a = false, transpose_b = false}> : (tensor<1x16384x32xbf16>, tensor<32x32xbf16>) -> tensor<1x16384x32xbf16>
    %168 = "ttir.add"(%167, %arg138) : (tensor<1x16384x32xbf16>, tensor<32xbf16>) -> tensor<1x16384x32xbf16>
    %169 = "ttir.add"(%168, %108) : (tensor<1x16384x32xbf16>, tensor<1x16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %170 = "ttir.sum"(%169) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x16384x32xbf16>) -> tensor<1x16384x1xbf16>
    %171 = "ttir.multiply"(%arg21, %170) : (tensor<1x16384x32xf32>, tensor<1x16384x1xbf16>) -> tensor<1x16384x32xbf16>
    %172 = "ttir.subtract"(%169, %171) : (tensor<1x16384x32xbf16>, tensor<1x16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %173 = "ttir.multiply"(%172, %172) : (tensor<1x16384x32xbf16>, tensor<1x16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %174 = "ttir.sum"(%173) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x16384x32xbf16>) -> tensor<1x16384x1xbf16>
    %175 = "ttir.multiply"(%arg22, %174) : (tensor<1x16384x1xf32>, tensor<1x16384x1xbf16>) -> tensor<1x16384x1xbf16>
    %176 = "ttir.add"(%175, %arg23) : (tensor<1x16384x1xbf16>, tensor<1x16384x1xf32>) -> tensor<1x16384x1xbf16>
    %177 = "ttir.sqrt"(%176) : (tensor<1x16384x1xbf16>) -> tensor<1x16384x1xbf16>
    %178 = "ttir.reciprocal"(%177) : (tensor<1x16384x1xbf16>) -> tensor<1x16384x1xbf16>
    %179 = "ttir.multiply"(%172, %178) : (tensor<1x16384x32xbf16>, tensor<1x16384x1xbf16>) -> tensor<1x16384x32xbf16>
    %180 = "ttir.multiply"(%179, %arg139) : (tensor<1x16384x32xbf16>, tensor<32xbf16>) -> tensor<1x16384x32xbf16>
    %181 = "ttir.add"(%180, %arg140) : (tensor<1x16384x32xbf16>, tensor<32xbf16>) -> tensor<1x16384x32xbf16>
    %182 = "ttir.matmul"(%181, %arg141) <{transpose_a = false, transpose_b = false}> : (tensor<1x16384x32xbf16>, tensor<32x128xbf16>) -> tensor<1x16384x128xbf16>
    %183 = "ttir.add"(%182, %arg142) : (tensor<1x16384x128xbf16>, tensor<128xbf16>) -> tensor<1x16384x128xbf16>
    %184 = "ttir.transpose"(%183) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x16384x128xbf16>) -> tensor<1x128x16384xbf16>
    %185 = "ttir.reshape"(%184) <{shape = [1 : i32, 128 : i32, 128 : i32, 128 : i32]}> : (tensor<1x128x16384xbf16>) -> tensor<1x128x128x128xbf16>
    %186 = "ttir.transpose"(%185) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x128x128x128xbf16>) -> tensor<1x128x128x128xbf16>
    %187 = "ttir.transpose"(%186) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x128x128x128xbf16>) -> tensor<1x128x128x128xbf16>
    %188 = "ttir.conv2d"(%187, %arg143, %arg144) <{dilation = array<i32: 1, 1>, groups = 128 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x128x128x128xbf16>, tensor<128x1x3x3xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x128x128x128xbf16>
    %189 = "ttir.transpose"(%188) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x128x128x128xbf16>) -> tensor<1x128x128x128xbf16>
    %190 = "ttir.transpose"(%189) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x128x128x128xbf16>) -> tensor<1x128x128x128xbf16>
    %191 = "ttir.reshape"(%190) <{shape = [1 : i32, 128 : i32, 16384 : i32, 1 : i32]}> : (tensor<1x128x128x128xbf16>) -> tensor<1x128x16384x1xbf16>
    %192 = "ttir.squeeze"(%191) <{dim = -1 : si32}> : (tensor<1x128x16384x1xbf16>) -> tensor<1x128x16384xbf16>
    %193 = "ttir.transpose"(%192) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x128x16384xbf16>) -> tensor<1x16384x128xbf16>
    %194 = "ttir.gelu"(%193) : (tensor<1x16384x128xbf16>) -> tensor<1x16384x128xbf16>
    %195 = "ttir.matmul"(%194, %arg145) <{transpose_a = false, transpose_b = false}> : (tensor<1x16384x128xbf16>, tensor<128x32xbf16>) -> tensor<1x16384x32xbf16>
    %196 = "ttir.add"(%195, %arg146) : (tensor<1x16384x32xbf16>, tensor<32xbf16>) -> tensor<1x16384x32xbf16>
    %197 = "ttir.add"(%196, %169) : (tensor<1x16384x32xbf16>, tensor<1x16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %198 = "ttir.sum"(%197) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x16384x32xbf16>) -> tensor<1x16384x1xbf16>
    %199 = "ttir.multiply"(%arg24, %198) : (tensor<1x16384x32xf32>, tensor<1x16384x1xbf16>) -> tensor<1x16384x32xbf16>
    %200 = "ttir.subtract"(%197, %199) : (tensor<1x16384x32xbf16>, tensor<1x16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %201 = "ttir.multiply"(%200, %200) : (tensor<1x16384x32xbf16>, tensor<1x16384x32xbf16>) -> tensor<1x16384x32xbf16>
    %202 = "ttir.sum"(%201) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x16384x32xbf16>) -> tensor<1x16384x1xbf16>
    %203 = "ttir.multiply"(%arg25, %202) : (tensor<1x16384x1xf32>, tensor<1x16384x1xbf16>) -> tensor<1x16384x1xbf16>
    %204 = "ttir.add"(%203, %arg26) : (tensor<1x16384x1xbf16>, tensor<1x16384x1xf32>) -> tensor<1x16384x1xbf16>
    %205 = "ttir.sqrt"(%204) : (tensor<1x16384x1xbf16>) -> tensor<1x16384x1xbf16>
    %206 = "ttir.reciprocal"(%205) : (tensor<1x16384x1xbf16>) -> tensor<1x16384x1xbf16>
    %207 = "ttir.multiply"(%200, %206) : (tensor<1x16384x32xbf16>, tensor<1x16384x1xbf16>) -> tensor<1x16384x32xbf16>
    %208 = "ttir.multiply"(%207, %arg147) : (tensor<1x16384x32xbf16>, tensor<32xbf16>) -> tensor<1x16384x32xbf16>
    %209 = "ttir.add"(%208, %arg148) : (tensor<1x16384x32xbf16>, tensor<32xbf16>) -> tensor<1x16384x32xbf16>
    %210 = "ttir.reshape"(%209) <{shape = [1 : i32, 128 : i32, 128 : i32, 32 : i32]}> : (tensor<1x16384x32xbf16>) -> tensor<1x128x128x32xbf16>
    %211 = "ttir.transpose"(%210) <{dim0 = -3 : si32, dim1 = -1 : si32}> : (tensor<1x128x128x32xbf16>) -> tensor<1x32x128x128xbf16>
    %212 = "ttir.transpose"(%211) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %213 = "ttir.transpose"(%212) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x128x128xbf16>) -> tensor<1x128x32x128xbf16>
    %214 = "ttir.transpose"(%213) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x128x32x128xbf16>) -> tensor<1x128x128x32xbf16>
    %215 = "ttir.conv2d"(%214, %arg149, %arg150) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x128x128x32xbf16>, tensor<64x32x3x3xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x64x64x64xbf16>
    %216 = "ttir.transpose"(%215) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x64x64x64xbf16>) -> tensor<1x64x64x64xbf16>
    %217 = "ttir.transpose"(%216) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x64x64x64xbf16>) -> tensor<1x64x64x64xbf16>
    %218 = "ttir.reshape"(%217) <{shape = [1 : i32, 64 : i32, 4096 : i32, 1 : i32]}> : (tensor<1x64x64x64xbf16>) -> tensor<1x64x4096x1xbf16>
    %219 = "ttir.squeeze"(%218) <{dim = -1 : si32}> : (tensor<1x64x4096x1xbf16>) -> tensor<1x64x4096xbf16>
    %220 = "ttir.transpose"(%219) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x64x4096xbf16>) -> tensor<1x4096x64xbf16>
    %221 = "ttir.sum"(%220) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x4096x64xbf16>) -> tensor<1x4096x1xbf16>
    %222 = "ttir.multiply"(%arg27, %221) : (tensor<1x4096x64xf32>, tensor<1x4096x1xbf16>) -> tensor<1x4096x64xbf16>
    %223 = "ttir.subtract"(%220, %222) : (tensor<1x4096x64xbf16>, tensor<1x4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %224 = "ttir.multiply"(%223, %223) : (tensor<1x4096x64xbf16>, tensor<1x4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %225 = "ttir.sum"(%224) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x4096x64xbf16>) -> tensor<1x4096x1xbf16>
    %226 = "ttir.multiply"(%arg28, %225) : (tensor<1x4096x1xf32>, tensor<1x4096x1xbf16>) -> tensor<1x4096x1xbf16>
    %227 = "ttir.add"(%226, %arg29) : (tensor<1x4096x1xbf16>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xbf16>
    %228 = "ttir.sqrt"(%227) : (tensor<1x4096x1xbf16>) -> tensor<1x4096x1xbf16>
    %229 = "ttir.reciprocal"(%228) : (tensor<1x4096x1xbf16>) -> tensor<1x4096x1xbf16>
    %230 = "ttir.multiply"(%223, %229) : (tensor<1x4096x64xbf16>, tensor<1x4096x1xbf16>) -> tensor<1x4096x64xbf16>
    %231 = "ttir.multiply"(%230, %arg151) : (tensor<1x4096x64xbf16>, tensor<64xbf16>) -> tensor<1x4096x64xbf16>
    %232 = "ttir.add"(%231, %arg152) : (tensor<1x4096x64xbf16>, tensor<64xbf16>) -> tensor<1x4096x64xbf16>
    %233 = "ttir.sum"(%232) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x4096x64xbf16>) -> tensor<1x4096x1xbf16>
    %234 = "ttir.multiply"(%arg30, %233) : (tensor<1x4096x64xf32>, tensor<1x4096x1xbf16>) -> tensor<1x4096x64xbf16>
    %235 = "ttir.subtract"(%232, %234) : (tensor<1x4096x64xbf16>, tensor<1x4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %236 = "ttir.multiply"(%235, %235) : (tensor<1x4096x64xbf16>, tensor<1x4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %237 = "ttir.sum"(%236) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x4096x64xbf16>) -> tensor<1x4096x1xbf16>
    %238 = "ttir.multiply"(%arg31, %237) : (tensor<1x4096x1xf32>, tensor<1x4096x1xbf16>) -> tensor<1x4096x1xbf16>
    %239 = "ttir.add"(%238, %arg32) : (tensor<1x4096x1xbf16>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xbf16>
    %240 = "ttir.sqrt"(%239) : (tensor<1x4096x1xbf16>) -> tensor<1x4096x1xbf16>
    %241 = "ttir.reciprocal"(%240) : (tensor<1x4096x1xbf16>) -> tensor<1x4096x1xbf16>
    %242 = "ttir.multiply"(%235, %241) : (tensor<1x4096x64xbf16>, tensor<1x4096x1xbf16>) -> tensor<1x4096x64xbf16>
    %243 = "ttir.multiply"(%242, %arg153) : (tensor<1x4096x64xbf16>, tensor<64xbf16>) -> tensor<1x4096x64xbf16>
    %244 = "ttir.add"(%243, %arg154) : (tensor<1x4096x64xbf16>, tensor<64xbf16>) -> tensor<1x4096x64xbf16>
    %245 = "ttir.matmul"(%244, %arg155) <{transpose_a = false, transpose_b = false}> : (tensor<1x4096x64xbf16>, tensor<64x64xbf16>) -> tensor<1x4096x64xbf16>
    %246 = "ttir.add"(%245, %arg156) : (tensor<1x4096x64xbf16>, tensor<64xbf16>) -> tensor<1x4096x64xbf16>
    %247 = "ttir.reshape"(%246) <{shape = [1 : i32, 4096 : i32, 2 : i32, 32 : i32]}> : (tensor<1x4096x64xbf16>) -> tensor<1x4096x2x32xbf16>
    %248 = "ttir.transpose"(%247) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x4096x2x32xbf16>) -> tensor<1x2x4096x32xbf16>
    %249 = "ttir.squeeze"(%248) <{dim = 0 : si32}> : (tensor<1x2x4096x32xbf16>) -> tensor<2x4096x32xbf16>
    %250 = "ttir.transpose"(%244) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x4096x64xbf16>) -> tensor<1x64x4096xbf16>
    %251 = "ttir.reshape"(%250) <{shape = [1 : i32, 64 : i32, 64 : i32, 64 : i32]}> : (tensor<1x64x4096xbf16>) -> tensor<1x64x64x64xbf16>
    %252 = "ttir.transpose"(%251) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x64x64x64xbf16>) -> tensor<1x64x64x64xbf16>
    %253 = "ttir.transpose"(%252) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x64x64x64xbf16>) -> tensor<1x64x64x64xbf16>
    %254 = "ttir.conv2d"(%253, %arg157, %arg158) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 4, 4>}> {channel_last = 1 : si32} : (tensor<1x64x64x64xbf16>, tensor<64x64x4x4xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x16x16x64xbf16>
    %255 = "ttir.transpose"(%254) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x16x16x64xbf16>) -> tensor<1x16x64x16xbf16>
    %256 = "ttir.transpose"(%255) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x16x64x16xbf16>) -> tensor<1x64x16x16xbf16>
    %257 = "ttir.reshape"(%256) <{shape = [1 : i32, 64 : i32, 256 : i32]}> : (tensor<1x64x16x16xbf16>) -> tensor<1x64x256xbf16>
    %258 = "ttir.transpose"(%257) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x64x256xbf16>) -> tensor<1x256x64xbf16>
    %259 = "ttir.sum"(%258) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x64xbf16>) -> tensor<1x256x1xbf16>
    %260 = "ttir.multiply"(%arg33, %259) : (tensor<1x256x64xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x64xbf16>
    %261 = "ttir.subtract"(%258, %260) : (tensor<1x256x64xbf16>, tensor<1x256x64xbf16>) -> tensor<1x256x64xbf16>
    %262 = "ttir.multiply"(%261, %261) : (tensor<1x256x64xbf16>, tensor<1x256x64xbf16>) -> tensor<1x256x64xbf16>
    %263 = "ttir.sum"(%262) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x64xbf16>) -> tensor<1x256x1xbf16>
    %264 = "ttir.multiply"(%arg34, %263) : (tensor<1x256x1xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %265 = "ttir.add"(%264, %arg35) : (tensor<1x256x1xbf16>, tensor<1x256x1xf32>) -> tensor<1x256x1xbf16>
    %266 = "ttir.sqrt"(%265) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %267 = "ttir.reciprocal"(%266) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %268 = "ttir.multiply"(%261, %267) : (tensor<1x256x64xbf16>, tensor<1x256x1xbf16>) -> tensor<1x256x64xbf16>
    %269 = "ttir.multiply"(%268, %arg159) : (tensor<1x256x64xbf16>, tensor<64xbf16>) -> tensor<1x256x64xbf16>
    %270 = "ttir.add"(%269, %arg160) : (tensor<1x256x64xbf16>, tensor<64xbf16>) -> tensor<1x256x64xbf16>
    %271 = "ttir.squeeze"(%270) <{dim = 0 : si32}> : (tensor<1x256x64xbf16>) -> tensor<256x64xbf16>
    %272 = "ttir.matmul"(%271, %arg161) <{transpose_a = false, transpose_b = false}> : (tensor<256x64xbf16>, tensor<64x64xbf16>) -> tensor<256x64xbf16>
    %273 = "ttir.unsqueeze"(%272) <{dim = 0 : si32}> : (tensor<256x64xbf16>) -> tensor<1x256x64xbf16>
    %274 = "ttir.add"(%273, %arg162) : (tensor<1x256x64xbf16>, tensor<64xbf16>) -> tensor<1x256x64xbf16>
    %275 = "ttir.reshape"(%274) <{shape = [1 : i32, 256 : i32, 2 : i32, 32 : i32]}> : (tensor<1x256x64xbf16>) -> tensor<1x256x2x32xbf16>
    %276 = "ttir.transpose"(%275) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x256x2x32xbf16>) -> tensor<1x2x256x32xbf16>
    %277 = "ttir.squeeze"(%276) <{dim = 0 : si32}> : (tensor<1x2x256x32xbf16>) -> tensor<2x256x32xbf16>
    %278 = "ttir.transpose"(%277) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<2x256x32xbf16>) -> tensor<2x32x256xbf16>
    %279 = "ttir.matmul"(%249, %278) <{transpose_a = false, transpose_b = false}> : (tensor<2x4096x32xbf16>, tensor<2x32x256xbf16>) -> tensor<2x4096x256xbf16>
    %280 = "ttir.unsqueeze"(%279) <{dim = 0 : si32}> : (tensor<2x4096x256xbf16>) -> tensor<1x2x4096x256xbf16>
    %281 = "ttir.div"(%280, %arg36) : (tensor<1x2x4096x256xbf16>, tensor<1xbf16>) -> tensor<1x2x4096x256xbf16>
    %282 = "ttir.softmax"(%281) <{dimension = -1 : si32}> : (tensor<1x2x4096x256xbf16>) -> tensor<1x2x4096x256xbf16>
    %283 = "ttir.squeeze"(%282) <{dim = 0 : si32}> : (tensor<1x2x4096x256xbf16>) -> tensor<2x4096x256xbf16>
    %284 = "ttir.matmul"(%271, %arg163) <{transpose_a = false, transpose_b = false}> : (tensor<256x64xbf16>, tensor<64x64xbf16>) -> tensor<256x64xbf16>
    %285 = "ttir.unsqueeze"(%284) <{dim = 0 : si32}> : (tensor<256x64xbf16>) -> tensor<1x256x64xbf16>
    %286 = "ttir.add"(%285, %arg164) : (tensor<1x256x64xbf16>, tensor<64xbf16>) -> tensor<1x256x64xbf16>
    %287 = "ttir.reshape"(%286) <{shape = [1 : i32, 256 : i32, 2 : i32, 32 : i32]}> : (tensor<1x256x64xbf16>) -> tensor<1x256x2x32xbf16>
    %288 = "ttir.transpose"(%287) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x256x2x32xbf16>) -> tensor<1x2x256x32xbf16>
    %289 = "ttir.transpose"(%288) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x2x256x32xbf16>) -> tensor<1x2x32x256xbf16>
    %290 = "ttir.squeeze"(%289) <{dim = 0 : si32}> : (tensor<1x2x32x256xbf16>) -> tensor<2x32x256xbf16>
    %291 = "ttir.transpose"(%290) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<2x32x256xbf16>) -> tensor<2x256x32xbf16>
    %292 = "ttir.matmul"(%283, %291) <{transpose_a = false, transpose_b = false}> : (tensor<2x4096x256xbf16>, tensor<2x256x32xbf16>) -> tensor<2x4096x32xbf16>
    %293 = "ttir.unsqueeze"(%292) <{dim = 0 : si32}> : (tensor<2x4096x32xbf16>) -> tensor<1x2x4096x32xbf16>
    %294 = "ttir.transpose"(%293) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x2x4096x32xbf16>) -> tensor<1x4096x2x32xbf16>
    %295 = "ttir.reshape"(%294) <{shape = [4096 : i32, 64 : i32]}> : (tensor<1x4096x2x32xbf16>) -> tensor<4096x64xbf16>
    %296 = "ttir.matmul"(%295, %arg165) <{transpose_a = false, transpose_b = false}> : (tensor<4096x64xbf16>, tensor<64x64xbf16>) -> tensor<4096x64xbf16>
    %297 = "ttir.unsqueeze"(%296) <{dim = 0 : si32}> : (tensor<4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %298 = "ttir.add"(%297, %arg166) : (tensor<1x4096x64xbf16>, tensor<64xbf16>) -> tensor<1x4096x64xbf16>
    %299 = "ttir.add"(%298, %232) : (tensor<1x4096x64xbf16>, tensor<1x4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %300 = "ttir.sum"(%299) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x4096x64xbf16>) -> tensor<1x4096x1xbf16>
    %301 = "ttir.multiply"(%arg37, %300) : (tensor<1x4096x64xf32>, tensor<1x4096x1xbf16>) -> tensor<1x4096x64xbf16>
    %302 = "ttir.subtract"(%299, %301) : (tensor<1x4096x64xbf16>, tensor<1x4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %303 = "ttir.multiply"(%302, %302) : (tensor<1x4096x64xbf16>, tensor<1x4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %304 = "ttir.sum"(%303) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x4096x64xbf16>) -> tensor<1x4096x1xbf16>
    %305 = "ttir.multiply"(%arg38, %304) : (tensor<1x4096x1xf32>, tensor<1x4096x1xbf16>) -> tensor<1x4096x1xbf16>
    %306 = "ttir.add"(%305, %arg39) : (tensor<1x4096x1xbf16>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xbf16>
    %307 = "ttir.sqrt"(%306) : (tensor<1x4096x1xbf16>) -> tensor<1x4096x1xbf16>
    %308 = "ttir.reciprocal"(%307) : (tensor<1x4096x1xbf16>) -> tensor<1x4096x1xbf16>
    %309 = "ttir.multiply"(%302, %308) : (tensor<1x4096x64xbf16>, tensor<1x4096x1xbf16>) -> tensor<1x4096x64xbf16>
    %310 = "ttir.multiply"(%309, %arg167) : (tensor<1x4096x64xbf16>, tensor<64xbf16>) -> tensor<1x4096x64xbf16>
    %311 = "ttir.add"(%310, %arg168) : (tensor<1x4096x64xbf16>, tensor<64xbf16>) -> tensor<1x4096x64xbf16>
    %312 = "ttir.matmul"(%311, %arg169) <{transpose_a = false, transpose_b = false}> : (tensor<1x4096x64xbf16>, tensor<64x256xbf16>) -> tensor<1x4096x256xbf16>
    %313 = "ttir.add"(%312, %arg170) : (tensor<1x4096x256xbf16>, tensor<256xbf16>) -> tensor<1x4096x256xbf16>
    %314 = "ttir.transpose"(%313) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x4096x256xbf16>) -> tensor<1x256x4096xbf16>
    %315 = "ttir.reshape"(%314) <{shape = [1 : i32, 256 : i32, 64 : i32, 64 : i32]}> : (tensor<1x256x4096xbf16>) -> tensor<1x256x64x64xbf16>
    %316 = "ttir.transpose"(%315) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x256x64x64xbf16>) -> tensor<1x64x256x64xbf16>
    %317 = "ttir.transpose"(%316) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x64x256x64xbf16>) -> tensor<1x64x64x256xbf16>
    %318 = "ttir.conv2d"(%317, %arg171, %arg172) <{dilation = array<i32: 1, 1>, groups = 256 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x64x64x256xbf16>, tensor<256x1x3x3xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x64x64x256xbf16>
    %319 = "ttir.transpose"(%318) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x64x64x256xbf16>) -> tensor<1x64x256x64xbf16>
    %320 = "ttir.transpose"(%319) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x64x256x64xbf16>) -> tensor<1x256x64x64xbf16>
    %321 = "ttir.reshape"(%320) <{shape = [1 : i32, 256 : i32, 4096 : i32, 1 : i32]}> : (tensor<1x256x64x64xbf16>) -> tensor<1x256x4096x1xbf16>
    %322 = "ttir.squeeze"(%321) <{dim = -1 : si32}> : (tensor<1x256x4096x1xbf16>) -> tensor<1x256x4096xbf16>
    %323 = "ttir.transpose"(%322) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x256x4096xbf16>) -> tensor<1x4096x256xbf16>
    %324 = "ttir.gelu"(%323) : (tensor<1x4096x256xbf16>) -> tensor<1x4096x256xbf16>
    %325 = "ttir.matmul"(%324, %arg173) <{transpose_a = false, transpose_b = false}> : (tensor<1x4096x256xbf16>, tensor<256x64xbf16>) -> tensor<1x4096x64xbf16>
    %326 = "ttir.add"(%325, %arg174) : (tensor<1x4096x64xbf16>, tensor<64xbf16>) -> tensor<1x4096x64xbf16>
    %327 = "ttir.add"(%326, %299) : (tensor<1x4096x64xbf16>, tensor<1x4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %328 = "ttir.sum"(%327) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x4096x64xbf16>) -> tensor<1x4096x1xbf16>
    %329 = "ttir.multiply"(%arg40, %328) : (tensor<1x4096x64xf32>, tensor<1x4096x1xbf16>) -> tensor<1x4096x64xbf16>
    %330 = "ttir.subtract"(%327, %329) : (tensor<1x4096x64xbf16>, tensor<1x4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %331 = "ttir.multiply"(%330, %330) : (tensor<1x4096x64xbf16>, tensor<1x4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %332 = "ttir.sum"(%331) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x4096x64xbf16>) -> tensor<1x4096x1xbf16>
    %333 = "ttir.multiply"(%arg41, %332) : (tensor<1x4096x1xf32>, tensor<1x4096x1xbf16>) -> tensor<1x4096x1xbf16>
    %334 = "ttir.add"(%333, %arg42) : (tensor<1x4096x1xbf16>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xbf16>
    %335 = "ttir.sqrt"(%334) : (tensor<1x4096x1xbf16>) -> tensor<1x4096x1xbf16>
    %336 = "ttir.reciprocal"(%335) : (tensor<1x4096x1xbf16>) -> tensor<1x4096x1xbf16>
    %337 = "ttir.multiply"(%330, %336) : (tensor<1x4096x64xbf16>, tensor<1x4096x1xbf16>) -> tensor<1x4096x64xbf16>
    %338 = "ttir.multiply"(%337, %arg175) : (tensor<1x4096x64xbf16>, tensor<64xbf16>) -> tensor<1x4096x64xbf16>
    %339 = "ttir.add"(%338, %arg176) : (tensor<1x4096x64xbf16>, tensor<64xbf16>) -> tensor<1x4096x64xbf16>
    %340 = "ttir.matmul"(%339, %arg177) <{transpose_a = false, transpose_b = false}> : (tensor<1x4096x64xbf16>, tensor<64x64xbf16>) -> tensor<1x4096x64xbf16>
    %341 = "ttir.add"(%340, %arg178) : (tensor<1x4096x64xbf16>, tensor<64xbf16>) -> tensor<1x4096x64xbf16>
    %342 = "ttir.reshape"(%341) <{shape = [1 : i32, 4096 : i32, 2 : i32, 32 : i32]}> : (tensor<1x4096x64xbf16>) -> tensor<1x4096x2x32xbf16>
    %343 = "ttir.transpose"(%342) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x4096x2x32xbf16>) -> tensor<1x2x4096x32xbf16>
    %344 = "ttir.squeeze"(%343) <{dim = 0 : si32}> : (tensor<1x2x4096x32xbf16>) -> tensor<2x4096x32xbf16>
    %345 = "ttir.transpose"(%339) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x4096x64xbf16>) -> tensor<1x64x4096xbf16>
    %346 = "ttir.reshape"(%345) <{shape = [1 : i32, 64 : i32, 64 : i32, 64 : i32]}> : (tensor<1x64x4096xbf16>) -> tensor<1x64x64x64xbf16>
    %347 = "ttir.transpose"(%346) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x64x64x64xbf16>) -> tensor<1x64x64x64xbf16>
    %348 = "ttir.transpose"(%347) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x64x64x64xbf16>) -> tensor<1x64x64x64xbf16>
    %349 = "ttir.conv2d"(%348, %arg179, %arg180) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 4, 4>}> {channel_last = 1 : si32} : (tensor<1x64x64x64xbf16>, tensor<64x64x4x4xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x16x16x64xbf16>
    %350 = "ttir.transpose"(%349) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x16x16x64xbf16>) -> tensor<1x16x64x16xbf16>
    %351 = "ttir.transpose"(%350) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x16x64x16xbf16>) -> tensor<1x64x16x16xbf16>
    %352 = "ttir.reshape"(%351) <{shape = [1 : i32, 64 : i32, 256 : i32]}> : (tensor<1x64x16x16xbf16>) -> tensor<1x64x256xbf16>
    %353 = "ttir.transpose"(%352) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x64x256xbf16>) -> tensor<1x256x64xbf16>
    %354 = "ttir.sum"(%353) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x64xbf16>) -> tensor<1x256x1xbf16>
    %355 = "ttir.multiply"(%arg43, %354) : (tensor<1x256x64xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x64xbf16>
    %356 = "ttir.subtract"(%353, %355) : (tensor<1x256x64xbf16>, tensor<1x256x64xbf16>) -> tensor<1x256x64xbf16>
    %357 = "ttir.multiply"(%356, %356) : (tensor<1x256x64xbf16>, tensor<1x256x64xbf16>) -> tensor<1x256x64xbf16>
    %358 = "ttir.sum"(%357) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x64xbf16>) -> tensor<1x256x1xbf16>
    %359 = "ttir.multiply"(%arg44, %358) : (tensor<1x256x1xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %360 = "ttir.add"(%359, %arg45) : (tensor<1x256x1xbf16>, tensor<1x256x1xf32>) -> tensor<1x256x1xbf16>
    %361 = "ttir.sqrt"(%360) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %362 = "ttir.reciprocal"(%361) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %363 = "ttir.multiply"(%356, %362) : (tensor<1x256x64xbf16>, tensor<1x256x1xbf16>) -> tensor<1x256x64xbf16>
    %364 = "ttir.multiply"(%363, %arg181) : (tensor<1x256x64xbf16>, tensor<64xbf16>) -> tensor<1x256x64xbf16>
    %365 = "ttir.add"(%364, %arg182) : (tensor<1x256x64xbf16>, tensor<64xbf16>) -> tensor<1x256x64xbf16>
    %366 = "ttir.squeeze"(%365) <{dim = 0 : si32}> : (tensor<1x256x64xbf16>) -> tensor<256x64xbf16>
    %367 = "ttir.matmul"(%366, %arg183) <{transpose_a = false, transpose_b = false}> : (tensor<256x64xbf16>, tensor<64x64xbf16>) -> tensor<256x64xbf16>
    %368 = "ttir.unsqueeze"(%367) <{dim = 0 : si32}> : (tensor<256x64xbf16>) -> tensor<1x256x64xbf16>
    %369 = "ttir.add"(%368, %arg184) : (tensor<1x256x64xbf16>, tensor<64xbf16>) -> tensor<1x256x64xbf16>
    %370 = "ttir.reshape"(%369) <{shape = [1 : i32, 256 : i32, 2 : i32, 32 : i32]}> : (tensor<1x256x64xbf16>) -> tensor<1x256x2x32xbf16>
    %371 = "ttir.transpose"(%370) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x256x2x32xbf16>) -> tensor<1x2x256x32xbf16>
    %372 = "ttir.squeeze"(%371) <{dim = 0 : si32}> : (tensor<1x2x256x32xbf16>) -> tensor<2x256x32xbf16>
    %373 = "ttir.transpose"(%372) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<2x256x32xbf16>) -> tensor<2x32x256xbf16>
    %374 = "ttir.matmul"(%344, %373) <{transpose_a = false, transpose_b = false}> : (tensor<2x4096x32xbf16>, tensor<2x32x256xbf16>) -> tensor<2x4096x256xbf16>
    %375 = "ttir.unsqueeze"(%374) <{dim = 0 : si32}> : (tensor<2x4096x256xbf16>) -> tensor<1x2x4096x256xbf16>
    %376 = "ttir.div"(%375, %arg46) : (tensor<1x2x4096x256xbf16>, tensor<1xbf16>) -> tensor<1x2x4096x256xbf16>
    %377 = "ttir.softmax"(%376) <{dimension = -1 : si32}> : (tensor<1x2x4096x256xbf16>) -> tensor<1x2x4096x256xbf16>
    %378 = "ttir.squeeze"(%377) <{dim = 0 : si32}> : (tensor<1x2x4096x256xbf16>) -> tensor<2x4096x256xbf16>
    %379 = "ttir.matmul"(%366, %arg185) <{transpose_a = false, transpose_b = false}> : (tensor<256x64xbf16>, tensor<64x64xbf16>) -> tensor<256x64xbf16>
    %380 = "ttir.unsqueeze"(%379) <{dim = 0 : si32}> : (tensor<256x64xbf16>) -> tensor<1x256x64xbf16>
    %381 = "ttir.add"(%380, %arg186) : (tensor<1x256x64xbf16>, tensor<64xbf16>) -> tensor<1x256x64xbf16>
    %382 = "ttir.reshape"(%381) <{shape = [1 : i32, 256 : i32, 2 : i32, 32 : i32]}> : (tensor<1x256x64xbf16>) -> tensor<1x256x2x32xbf16>
    %383 = "ttir.transpose"(%382) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x256x2x32xbf16>) -> tensor<1x2x256x32xbf16>
    %384 = "ttir.transpose"(%383) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x2x256x32xbf16>) -> tensor<1x2x32x256xbf16>
    %385 = "ttir.squeeze"(%384) <{dim = 0 : si32}> : (tensor<1x2x32x256xbf16>) -> tensor<2x32x256xbf16>
    %386 = "ttir.transpose"(%385) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<2x32x256xbf16>) -> tensor<2x256x32xbf16>
    %387 = "ttir.matmul"(%378, %386) <{transpose_a = false, transpose_b = false}> : (tensor<2x4096x256xbf16>, tensor<2x256x32xbf16>) -> tensor<2x4096x32xbf16>
    %388 = "ttir.unsqueeze"(%387) <{dim = 0 : si32}> : (tensor<2x4096x32xbf16>) -> tensor<1x2x4096x32xbf16>
    %389 = "ttir.transpose"(%388) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x2x4096x32xbf16>) -> tensor<1x4096x2x32xbf16>
    %390 = "ttir.reshape"(%389) <{shape = [4096 : i32, 64 : i32]}> : (tensor<1x4096x2x32xbf16>) -> tensor<4096x64xbf16>
    %391 = "ttir.matmul"(%390, %arg187) <{transpose_a = false, transpose_b = false}> : (tensor<4096x64xbf16>, tensor<64x64xbf16>) -> tensor<4096x64xbf16>
    %392 = "ttir.unsqueeze"(%391) <{dim = 0 : si32}> : (tensor<4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %393 = "ttir.add"(%392, %arg188) : (tensor<1x4096x64xbf16>, tensor<64xbf16>) -> tensor<1x4096x64xbf16>
    %394 = "ttir.add"(%393, %327) : (tensor<1x4096x64xbf16>, tensor<1x4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %395 = "ttir.sum"(%394) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x4096x64xbf16>) -> tensor<1x4096x1xbf16>
    %396 = "ttir.multiply"(%arg47, %395) : (tensor<1x4096x64xf32>, tensor<1x4096x1xbf16>) -> tensor<1x4096x64xbf16>
    %397 = "ttir.subtract"(%394, %396) : (tensor<1x4096x64xbf16>, tensor<1x4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %398 = "ttir.multiply"(%397, %397) : (tensor<1x4096x64xbf16>, tensor<1x4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %399 = "ttir.sum"(%398) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x4096x64xbf16>) -> tensor<1x4096x1xbf16>
    %400 = "ttir.multiply"(%arg48, %399) : (tensor<1x4096x1xf32>, tensor<1x4096x1xbf16>) -> tensor<1x4096x1xbf16>
    %401 = "ttir.add"(%400, %arg49) : (tensor<1x4096x1xbf16>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xbf16>
    %402 = "ttir.sqrt"(%401) : (tensor<1x4096x1xbf16>) -> tensor<1x4096x1xbf16>
    %403 = "ttir.reciprocal"(%402) : (tensor<1x4096x1xbf16>) -> tensor<1x4096x1xbf16>
    %404 = "ttir.multiply"(%397, %403) : (tensor<1x4096x64xbf16>, tensor<1x4096x1xbf16>) -> tensor<1x4096x64xbf16>
    %405 = "ttir.multiply"(%404, %arg189) : (tensor<1x4096x64xbf16>, tensor<64xbf16>) -> tensor<1x4096x64xbf16>
    %406 = "ttir.add"(%405, %arg190) : (tensor<1x4096x64xbf16>, tensor<64xbf16>) -> tensor<1x4096x64xbf16>
    %407 = "ttir.matmul"(%406, %arg191) <{transpose_a = false, transpose_b = false}> : (tensor<1x4096x64xbf16>, tensor<64x256xbf16>) -> tensor<1x4096x256xbf16>
    %408 = "ttir.add"(%407, %arg192) : (tensor<1x4096x256xbf16>, tensor<256xbf16>) -> tensor<1x4096x256xbf16>
    %409 = "ttir.transpose"(%408) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x4096x256xbf16>) -> tensor<1x256x4096xbf16>
    %410 = "ttir.reshape"(%409) <{shape = [1 : i32, 256 : i32, 64 : i32, 64 : i32]}> : (tensor<1x256x4096xbf16>) -> tensor<1x256x64x64xbf16>
    %411 = "ttir.transpose"(%410) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x256x64x64xbf16>) -> tensor<1x64x256x64xbf16>
    %412 = "ttir.transpose"(%411) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x64x256x64xbf16>) -> tensor<1x64x64x256xbf16>
    %413 = "ttir.conv2d"(%412, %arg193, %arg194) <{dilation = array<i32: 1, 1>, groups = 256 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x64x64x256xbf16>, tensor<256x1x3x3xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x64x64x256xbf16>
    %414 = "ttir.transpose"(%413) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x64x64x256xbf16>) -> tensor<1x64x256x64xbf16>
    %415 = "ttir.transpose"(%414) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x64x256x64xbf16>) -> tensor<1x256x64x64xbf16>
    %416 = "ttir.reshape"(%415) <{shape = [1 : i32, 256 : i32, 4096 : i32, 1 : i32]}> : (tensor<1x256x64x64xbf16>) -> tensor<1x256x4096x1xbf16>
    %417 = "ttir.squeeze"(%416) <{dim = -1 : si32}> : (tensor<1x256x4096x1xbf16>) -> tensor<1x256x4096xbf16>
    %418 = "ttir.transpose"(%417) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x256x4096xbf16>) -> tensor<1x4096x256xbf16>
    %419 = "ttir.gelu"(%418) : (tensor<1x4096x256xbf16>) -> tensor<1x4096x256xbf16>
    %420 = "ttir.matmul"(%419, %arg195) <{transpose_a = false, transpose_b = false}> : (tensor<1x4096x256xbf16>, tensor<256x64xbf16>) -> tensor<1x4096x64xbf16>
    %421 = "ttir.add"(%420, %arg196) : (tensor<1x4096x64xbf16>, tensor<64xbf16>) -> tensor<1x4096x64xbf16>
    %422 = "ttir.add"(%421, %394) : (tensor<1x4096x64xbf16>, tensor<1x4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %423 = "ttir.sum"(%422) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x4096x64xbf16>) -> tensor<1x4096x1xbf16>
    %424 = "ttir.multiply"(%arg50, %423) : (tensor<1x4096x64xf32>, tensor<1x4096x1xbf16>) -> tensor<1x4096x64xbf16>
    %425 = "ttir.subtract"(%422, %424) : (tensor<1x4096x64xbf16>, tensor<1x4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %426 = "ttir.multiply"(%425, %425) : (tensor<1x4096x64xbf16>, tensor<1x4096x64xbf16>) -> tensor<1x4096x64xbf16>
    %427 = "ttir.sum"(%426) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x4096x64xbf16>) -> tensor<1x4096x1xbf16>
    %428 = "ttir.multiply"(%arg51, %427) : (tensor<1x4096x1xf32>, tensor<1x4096x1xbf16>) -> tensor<1x4096x1xbf16>
    %429 = "ttir.add"(%428, %arg52) : (tensor<1x4096x1xbf16>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xbf16>
    %430 = "ttir.sqrt"(%429) : (tensor<1x4096x1xbf16>) -> tensor<1x4096x1xbf16>
    %431 = "ttir.reciprocal"(%430) : (tensor<1x4096x1xbf16>) -> tensor<1x4096x1xbf16>
    %432 = "ttir.multiply"(%425, %431) : (tensor<1x4096x64xbf16>, tensor<1x4096x1xbf16>) -> tensor<1x4096x64xbf16>
    %433 = "ttir.multiply"(%432, %arg197) : (tensor<1x4096x64xbf16>, tensor<64xbf16>) -> tensor<1x4096x64xbf16>
    %434 = "ttir.add"(%433, %arg198) : (tensor<1x4096x64xbf16>, tensor<64xbf16>) -> tensor<1x4096x64xbf16>
    %435 = "ttir.reshape"(%434) <{shape = [1 : i32, 64 : i32, 64 : i32, 64 : i32]}> : (tensor<1x4096x64xbf16>) -> tensor<1x64x64x64xbf16>
    %436 = "ttir.transpose"(%435) <{dim0 = -3 : si32, dim1 = -1 : si32}> : (tensor<1x64x64x64xbf16>) -> tensor<1x64x64x64xbf16>
    %437 = "ttir.transpose"(%436) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x64x64x64xbf16>) -> tensor<1x64x64x64xbf16>
    %438 = "ttir.transpose"(%437) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x64x64x64xbf16>) -> tensor<1x64x64x64xbf16>
    %439 = "ttir.transpose"(%438) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x64x64x64xbf16>) -> tensor<1x64x64x64xbf16>
    %440 = "ttir.conv2d"(%439, %arg199, %arg200) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x64x64x64xbf16>, tensor<160x64x3x3xbf16>, tensor<1x1x1x160xbf16>) -> tensor<1x32x32x160xbf16>
    %441 = "ttir.transpose"(%440) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x160xbf16>) -> tensor<1x32x160x32xbf16>
    %442 = "ttir.transpose"(%441) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x160x32xbf16>) -> tensor<1x160x32x32xbf16>
    %443 = "ttir.reshape"(%442) <{shape = [1 : i32, 160 : i32, 1024 : i32, 1 : i32]}> : (tensor<1x160x32x32xbf16>) -> tensor<1x160x1024x1xbf16>
    %444 = "ttir.squeeze"(%443) <{dim = -1 : si32}> : (tensor<1x160x1024x1xbf16>) -> tensor<1x160x1024xbf16>
    %445 = "ttir.transpose"(%444) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x1024xbf16>) -> tensor<1x1024x160xbf16>
    %446 = "ttir.sum"(%445) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x1024x160xbf16>) -> tensor<1x1024x1xbf16>
    %447 = "ttir.multiply"(%arg53, %446) : (tensor<1x1024x160xf32>, tensor<1x1024x1xbf16>) -> tensor<1x1024x160xbf16>
    %448 = "ttir.subtract"(%445, %447) : (tensor<1x1024x160xbf16>, tensor<1x1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %449 = "ttir.multiply"(%448, %448) : (tensor<1x1024x160xbf16>, tensor<1x1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %450 = "ttir.sum"(%449) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x1024x160xbf16>) -> tensor<1x1024x1xbf16>
    %451 = "ttir.multiply"(%arg54, %450) : (tensor<1x1024x1xf32>, tensor<1x1024x1xbf16>) -> tensor<1x1024x1xbf16>
    %452 = "ttir.add"(%451, %arg55) : (tensor<1x1024x1xbf16>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xbf16>
    %453 = "ttir.sqrt"(%452) : (tensor<1x1024x1xbf16>) -> tensor<1x1024x1xbf16>
    %454 = "ttir.reciprocal"(%453) : (tensor<1x1024x1xbf16>) -> tensor<1x1024x1xbf16>
    %455 = "ttir.multiply"(%448, %454) : (tensor<1x1024x160xbf16>, tensor<1x1024x1xbf16>) -> tensor<1x1024x160xbf16>
    %456 = "ttir.multiply"(%455, %arg201) : (tensor<1x1024x160xbf16>, tensor<160xbf16>) -> tensor<1x1024x160xbf16>
    %457 = "ttir.add"(%456, %arg202) : (tensor<1x1024x160xbf16>, tensor<160xbf16>) -> tensor<1x1024x160xbf16>
    %458 = "ttir.sum"(%457) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x1024x160xbf16>) -> tensor<1x1024x1xbf16>
    %459 = "ttir.multiply"(%arg56, %458) : (tensor<1x1024x160xf32>, tensor<1x1024x1xbf16>) -> tensor<1x1024x160xbf16>
    %460 = "ttir.subtract"(%457, %459) : (tensor<1x1024x160xbf16>, tensor<1x1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %461 = "ttir.multiply"(%460, %460) : (tensor<1x1024x160xbf16>, tensor<1x1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %462 = "ttir.sum"(%461) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x1024x160xbf16>) -> tensor<1x1024x1xbf16>
    %463 = "ttir.multiply"(%arg57, %462) : (tensor<1x1024x1xf32>, tensor<1x1024x1xbf16>) -> tensor<1x1024x1xbf16>
    %464 = "ttir.add"(%463, %arg58) : (tensor<1x1024x1xbf16>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xbf16>
    %465 = "ttir.sqrt"(%464) : (tensor<1x1024x1xbf16>) -> tensor<1x1024x1xbf16>
    %466 = "ttir.reciprocal"(%465) : (tensor<1x1024x1xbf16>) -> tensor<1x1024x1xbf16>
    %467 = "ttir.multiply"(%460, %466) : (tensor<1x1024x160xbf16>, tensor<1x1024x1xbf16>) -> tensor<1x1024x160xbf16>
    %468 = "ttir.multiply"(%467, %arg203) : (tensor<1x1024x160xbf16>, tensor<160xbf16>) -> tensor<1x1024x160xbf16>
    %469 = "ttir.add"(%468, %arg204) : (tensor<1x1024x160xbf16>, tensor<160xbf16>) -> tensor<1x1024x160xbf16>
    %470 = "ttir.matmul"(%469, %arg205) <{transpose_a = false, transpose_b = false}> : (tensor<1x1024x160xbf16>, tensor<160x160xbf16>) -> tensor<1x1024x160xbf16>
    %471 = "ttir.add"(%470, %arg206) : (tensor<1x1024x160xbf16>, tensor<160xbf16>) -> tensor<1x1024x160xbf16>
    %472 = "ttir.reshape"(%471) <{shape = [1 : i32, 1024 : i32, 5 : i32, 32 : i32]}> : (tensor<1x1024x160xbf16>) -> tensor<1x1024x5x32xbf16>
    %473 = "ttir.transpose"(%472) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x1024x5x32xbf16>) -> tensor<1x5x1024x32xbf16>
    %474 = "ttir.squeeze"(%473) <{dim = 0 : si32}> : (tensor<1x5x1024x32xbf16>) -> tensor<5x1024x32xbf16>
    %475 = "ttir.transpose"(%469) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x1024x160xbf16>) -> tensor<1x160x1024xbf16>
    %476 = "ttir.reshape"(%475) <{shape = [1 : i32, 160 : i32, 32 : i32, 32 : i32]}> : (tensor<1x160x1024xbf16>) -> tensor<1x160x32x32xbf16>
    %477 = "ttir.transpose"(%476) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x32x32xbf16>) -> tensor<1x32x160x32xbf16>
    %478 = "ttir.transpose"(%477) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x160x32xbf16>) -> tensor<1x32x32x160xbf16>
    %479 = "ttir.conv2d"(%478, %arg207, %arg208) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x32x32x160xbf16>, tensor<160x160x2x2xbf16>, tensor<1x1x1x160xbf16>) -> tensor<1x16x16x160xbf16>
    %480 = "ttir.transpose"(%479) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x16x16x160xbf16>) -> tensor<1x16x160x16xbf16>
    %481 = "ttir.transpose"(%480) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x16x160x16xbf16>) -> tensor<1x160x16x16xbf16>
    %482 = "ttir.reshape"(%481) <{shape = [1 : i32, 160 : i32, 256 : i32]}> : (tensor<1x160x16x16xbf16>) -> tensor<1x160x256xbf16>
    %483 = "ttir.transpose"(%482) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x256xbf16>) -> tensor<1x256x160xbf16>
    %484 = "ttir.sum"(%483) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x160xbf16>) -> tensor<1x256x1xbf16>
    %485 = "ttir.multiply"(%arg59, %484) : (tensor<1x256x160xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x160xbf16>
    %486 = "ttir.subtract"(%483, %485) : (tensor<1x256x160xbf16>, tensor<1x256x160xbf16>) -> tensor<1x256x160xbf16>
    %487 = "ttir.multiply"(%486, %486) : (tensor<1x256x160xbf16>, tensor<1x256x160xbf16>) -> tensor<1x256x160xbf16>
    %488 = "ttir.sum"(%487) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x160xbf16>) -> tensor<1x256x1xbf16>
    %489 = "ttir.multiply"(%arg60, %488) : (tensor<1x256x1xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %490 = "ttir.add"(%489, %arg61) : (tensor<1x256x1xbf16>, tensor<1x256x1xf32>) -> tensor<1x256x1xbf16>
    %491 = "ttir.sqrt"(%490) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %492 = "ttir.reciprocal"(%491) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %493 = "ttir.multiply"(%486, %492) : (tensor<1x256x160xbf16>, tensor<1x256x1xbf16>) -> tensor<1x256x160xbf16>
    %494 = "ttir.multiply"(%493, %arg209) : (tensor<1x256x160xbf16>, tensor<160xbf16>) -> tensor<1x256x160xbf16>
    %495 = "ttir.add"(%494, %arg210) : (tensor<1x256x160xbf16>, tensor<160xbf16>) -> tensor<1x256x160xbf16>
    %496 = "ttir.squeeze"(%495) <{dim = 0 : si32}> : (tensor<1x256x160xbf16>) -> tensor<256x160xbf16>
    %497 = "ttir.matmul"(%496, %arg211) <{transpose_a = false, transpose_b = false}> : (tensor<256x160xbf16>, tensor<160x160xbf16>) -> tensor<256x160xbf16>
    %498 = "ttir.unsqueeze"(%497) <{dim = 0 : si32}> : (tensor<256x160xbf16>) -> tensor<1x256x160xbf16>
    %499 = "ttir.add"(%498, %arg212) : (tensor<1x256x160xbf16>, tensor<160xbf16>) -> tensor<1x256x160xbf16>
    %500 = "ttir.reshape"(%499) <{shape = [1 : i32, 256 : i32, 5 : i32, 32 : i32]}> : (tensor<1x256x160xbf16>) -> tensor<1x256x5x32xbf16>
    %501 = "ttir.transpose"(%500) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x256x5x32xbf16>) -> tensor<1x5x256x32xbf16>
    %502 = "ttir.squeeze"(%501) <{dim = 0 : si32}> : (tensor<1x5x256x32xbf16>) -> tensor<5x256x32xbf16>
    %503 = "ttir.transpose"(%502) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<5x256x32xbf16>) -> tensor<5x32x256xbf16>
    %504 = "ttir.matmul"(%474, %503) <{transpose_a = false, transpose_b = false}> : (tensor<5x1024x32xbf16>, tensor<5x32x256xbf16>) -> tensor<5x1024x256xbf16>
    %505 = "ttir.unsqueeze"(%504) <{dim = 0 : si32}> : (tensor<5x1024x256xbf16>) -> tensor<1x5x1024x256xbf16>
    %506 = "ttir.div"(%505, %arg62) : (tensor<1x5x1024x256xbf16>, tensor<1xbf16>) -> tensor<1x5x1024x256xbf16>
    %507 = "ttir.softmax"(%506) <{dimension = -1 : si32}> : (tensor<1x5x1024x256xbf16>) -> tensor<1x5x1024x256xbf16>
    %508 = "ttir.squeeze"(%507) <{dim = 0 : si32}> : (tensor<1x5x1024x256xbf16>) -> tensor<5x1024x256xbf16>
    %509 = "ttir.matmul"(%496, %arg213) <{transpose_a = false, transpose_b = false}> : (tensor<256x160xbf16>, tensor<160x160xbf16>) -> tensor<256x160xbf16>
    %510 = "ttir.unsqueeze"(%509) <{dim = 0 : si32}> : (tensor<256x160xbf16>) -> tensor<1x256x160xbf16>
    %511 = "ttir.add"(%510, %arg214) : (tensor<1x256x160xbf16>, tensor<160xbf16>) -> tensor<1x256x160xbf16>
    %512 = "ttir.reshape"(%511) <{shape = [1 : i32, 256 : i32, 5 : i32, 32 : i32]}> : (tensor<1x256x160xbf16>) -> tensor<1x256x5x32xbf16>
    %513 = "ttir.transpose"(%512) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x256x5x32xbf16>) -> tensor<1x5x256x32xbf16>
    %514 = "ttir.transpose"(%513) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x5x256x32xbf16>) -> tensor<1x5x32x256xbf16>
    %515 = "ttir.squeeze"(%514) <{dim = 0 : si32}> : (tensor<1x5x32x256xbf16>) -> tensor<5x32x256xbf16>
    %516 = "ttir.transpose"(%515) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<5x32x256xbf16>) -> tensor<5x256x32xbf16>
    %517 = "ttir.matmul"(%508, %516) <{transpose_a = false, transpose_b = false}> : (tensor<5x1024x256xbf16>, tensor<5x256x32xbf16>) -> tensor<5x1024x32xbf16>
    %518 = "ttir.unsqueeze"(%517) <{dim = 0 : si32}> : (tensor<5x1024x32xbf16>) -> tensor<1x5x1024x32xbf16>
    %519 = "ttir.transpose"(%518) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x5x1024x32xbf16>) -> tensor<1x1024x5x32xbf16>
    %520 = "ttir.reshape"(%519) <{shape = [1024 : i32, 160 : i32]}> : (tensor<1x1024x5x32xbf16>) -> tensor<1024x160xbf16>
    %521 = "ttir.matmul"(%520, %arg215) <{transpose_a = false, transpose_b = false}> : (tensor<1024x160xbf16>, tensor<160x160xbf16>) -> tensor<1024x160xbf16>
    %522 = "ttir.unsqueeze"(%521) <{dim = 0 : si32}> : (tensor<1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %523 = "ttir.add"(%522, %arg216) : (tensor<1x1024x160xbf16>, tensor<160xbf16>) -> tensor<1x1024x160xbf16>
    %524 = "ttir.add"(%523, %457) : (tensor<1x1024x160xbf16>, tensor<1x1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %525 = "ttir.sum"(%524) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x1024x160xbf16>) -> tensor<1x1024x1xbf16>
    %526 = "ttir.multiply"(%arg63, %525) : (tensor<1x1024x160xf32>, tensor<1x1024x1xbf16>) -> tensor<1x1024x160xbf16>
    %527 = "ttir.subtract"(%524, %526) : (tensor<1x1024x160xbf16>, tensor<1x1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %528 = "ttir.multiply"(%527, %527) : (tensor<1x1024x160xbf16>, tensor<1x1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %529 = "ttir.sum"(%528) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x1024x160xbf16>) -> tensor<1x1024x1xbf16>
    %530 = "ttir.multiply"(%arg64, %529) : (tensor<1x1024x1xf32>, tensor<1x1024x1xbf16>) -> tensor<1x1024x1xbf16>
    %531 = "ttir.add"(%530, %arg65) : (tensor<1x1024x1xbf16>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xbf16>
    %532 = "ttir.sqrt"(%531) : (tensor<1x1024x1xbf16>) -> tensor<1x1024x1xbf16>
    %533 = "ttir.reciprocal"(%532) : (tensor<1x1024x1xbf16>) -> tensor<1x1024x1xbf16>
    %534 = "ttir.multiply"(%527, %533) : (tensor<1x1024x160xbf16>, tensor<1x1024x1xbf16>) -> tensor<1x1024x160xbf16>
    %535 = "ttir.multiply"(%534, %arg217) : (tensor<1x1024x160xbf16>, tensor<160xbf16>) -> tensor<1x1024x160xbf16>
    %536 = "ttir.add"(%535, %arg218) : (tensor<1x1024x160xbf16>, tensor<160xbf16>) -> tensor<1x1024x160xbf16>
    %537 = "ttir.matmul"(%536, %arg219) <{transpose_a = false, transpose_b = false}> : (tensor<1x1024x160xbf16>, tensor<160x640xbf16>) -> tensor<1x1024x640xbf16>
    %538 = "ttir.add"(%537, %arg220) : (tensor<1x1024x640xbf16>, tensor<640xbf16>) -> tensor<1x1024x640xbf16>
    %539 = "ttir.transpose"(%538) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x1024x640xbf16>) -> tensor<1x640x1024xbf16>
    %540 = "ttir.reshape"(%539) <{shape = [1 : i32, 640 : i32, 32 : i32, 32 : i32]}> : (tensor<1x640x1024xbf16>) -> tensor<1x640x32x32xbf16>
    %541 = "ttir.transpose"(%540) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x640x32x32xbf16>) -> tensor<1x32x640x32xbf16>
    %542 = "ttir.transpose"(%541) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x640x32xbf16>) -> tensor<1x32x32x640xbf16>
    %543 = "ttir.conv2d"(%542, %arg221, %arg222) <{dilation = array<i32: 1, 1>, groups = 640 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x32x32x640xbf16>, tensor<640x1x3x3xbf16>, tensor<1x1x1x640xbf16>) -> tensor<1x32x32x640xbf16>
    %544 = "ttir.transpose"(%543) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x640xbf16>) -> tensor<1x32x640x32xbf16>
    %545 = "ttir.transpose"(%544) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x640x32xbf16>) -> tensor<1x640x32x32xbf16>
    %546 = "ttir.reshape"(%545) <{shape = [1 : i32, 640 : i32, 1024 : i32, 1 : i32]}> : (tensor<1x640x32x32xbf16>) -> tensor<1x640x1024x1xbf16>
    %547 = "ttir.squeeze"(%546) <{dim = -1 : si32}> : (tensor<1x640x1024x1xbf16>) -> tensor<1x640x1024xbf16>
    %548 = "ttir.transpose"(%547) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x640x1024xbf16>) -> tensor<1x1024x640xbf16>
    %549 = "ttir.gelu"(%548) : (tensor<1x1024x640xbf16>) -> tensor<1x1024x640xbf16>
    %550 = "ttir.matmul"(%549, %arg223) <{transpose_a = false, transpose_b = false}> : (tensor<1x1024x640xbf16>, tensor<640x160xbf16>) -> tensor<1x1024x160xbf16>
    %551 = "ttir.add"(%550, %arg224) : (tensor<1x1024x160xbf16>, tensor<160xbf16>) -> tensor<1x1024x160xbf16>
    %552 = "ttir.add"(%551, %524) : (tensor<1x1024x160xbf16>, tensor<1x1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %553 = "ttir.sum"(%552) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x1024x160xbf16>) -> tensor<1x1024x1xbf16>
    %554 = "ttir.multiply"(%arg66, %553) : (tensor<1x1024x160xf32>, tensor<1x1024x1xbf16>) -> tensor<1x1024x160xbf16>
    %555 = "ttir.subtract"(%552, %554) : (tensor<1x1024x160xbf16>, tensor<1x1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %556 = "ttir.multiply"(%555, %555) : (tensor<1x1024x160xbf16>, tensor<1x1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %557 = "ttir.sum"(%556) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x1024x160xbf16>) -> tensor<1x1024x1xbf16>
    %558 = "ttir.multiply"(%arg67, %557) : (tensor<1x1024x1xf32>, tensor<1x1024x1xbf16>) -> tensor<1x1024x1xbf16>
    %559 = "ttir.add"(%558, %arg68) : (tensor<1x1024x1xbf16>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xbf16>
    %560 = "ttir.sqrt"(%559) : (tensor<1x1024x1xbf16>) -> tensor<1x1024x1xbf16>
    %561 = "ttir.reciprocal"(%560) : (tensor<1x1024x1xbf16>) -> tensor<1x1024x1xbf16>
    %562 = "ttir.multiply"(%555, %561) : (tensor<1x1024x160xbf16>, tensor<1x1024x1xbf16>) -> tensor<1x1024x160xbf16>
    %563 = "ttir.multiply"(%562, %arg225) : (tensor<1x1024x160xbf16>, tensor<160xbf16>) -> tensor<1x1024x160xbf16>
    %564 = "ttir.add"(%563, %arg226) : (tensor<1x1024x160xbf16>, tensor<160xbf16>) -> tensor<1x1024x160xbf16>
    %565 = "ttir.matmul"(%564, %arg227) <{transpose_a = false, transpose_b = false}> : (tensor<1x1024x160xbf16>, tensor<160x160xbf16>) -> tensor<1x1024x160xbf16>
    %566 = "ttir.add"(%565, %arg228) : (tensor<1x1024x160xbf16>, tensor<160xbf16>) -> tensor<1x1024x160xbf16>
    %567 = "ttir.reshape"(%566) <{shape = [1 : i32, 1024 : i32, 5 : i32, 32 : i32]}> : (tensor<1x1024x160xbf16>) -> tensor<1x1024x5x32xbf16>
    %568 = "ttir.transpose"(%567) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x1024x5x32xbf16>) -> tensor<1x5x1024x32xbf16>
    %569 = "ttir.squeeze"(%568) <{dim = 0 : si32}> : (tensor<1x5x1024x32xbf16>) -> tensor<5x1024x32xbf16>
    %570 = "ttir.transpose"(%564) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x1024x160xbf16>) -> tensor<1x160x1024xbf16>
    %571 = "ttir.reshape"(%570) <{shape = [1 : i32, 160 : i32, 32 : i32, 32 : i32]}> : (tensor<1x160x1024xbf16>) -> tensor<1x160x32x32xbf16>
    %572 = "ttir.transpose"(%571) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x32x32xbf16>) -> tensor<1x32x160x32xbf16>
    %573 = "ttir.transpose"(%572) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x160x32xbf16>) -> tensor<1x32x32x160xbf16>
    %574 = "ttir.conv2d"(%573, %arg229, %arg230) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x32x32x160xbf16>, tensor<160x160x2x2xbf16>, tensor<1x1x1x160xbf16>) -> tensor<1x16x16x160xbf16>
    %575 = "ttir.transpose"(%574) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x16x16x160xbf16>) -> tensor<1x16x160x16xbf16>
    %576 = "ttir.transpose"(%575) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x16x160x16xbf16>) -> tensor<1x160x16x16xbf16>
    %577 = "ttir.reshape"(%576) <{shape = [1 : i32, 160 : i32, 256 : i32]}> : (tensor<1x160x16x16xbf16>) -> tensor<1x160x256xbf16>
    %578 = "ttir.transpose"(%577) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x256xbf16>) -> tensor<1x256x160xbf16>
    %579 = "ttir.sum"(%578) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x160xbf16>) -> tensor<1x256x1xbf16>
    %580 = "ttir.multiply"(%arg69, %579) : (tensor<1x256x160xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x160xbf16>
    %581 = "ttir.subtract"(%578, %580) : (tensor<1x256x160xbf16>, tensor<1x256x160xbf16>) -> tensor<1x256x160xbf16>
    %582 = "ttir.multiply"(%581, %581) : (tensor<1x256x160xbf16>, tensor<1x256x160xbf16>) -> tensor<1x256x160xbf16>
    %583 = "ttir.sum"(%582) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x160xbf16>) -> tensor<1x256x1xbf16>
    %584 = "ttir.multiply"(%arg70, %583) : (tensor<1x256x1xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %585 = "ttir.add"(%584, %arg71) : (tensor<1x256x1xbf16>, tensor<1x256x1xf32>) -> tensor<1x256x1xbf16>
    %586 = "ttir.sqrt"(%585) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %587 = "ttir.reciprocal"(%586) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %588 = "ttir.multiply"(%581, %587) : (tensor<1x256x160xbf16>, tensor<1x256x1xbf16>) -> tensor<1x256x160xbf16>
    %589 = "ttir.multiply"(%588, %arg231) : (tensor<1x256x160xbf16>, tensor<160xbf16>) -> tensor<1x256x160xbf16>
    %590 = "ttir.add"(%589, %arg232) : (tensor<1x256x160xbf16>, tensor<160xbf16>) -> tensor<1x256x160xbf16>
    %591 = "ttir.squeeze"(%590) <{dim = 0 : si32}> : (tensor<1x256x160xbf16>) -> tensor<256x160xbf16>
    %592 = "ttir.matmul"(%591, %arg233) <{transpose_a = false, transpose_b = false}> : (tensor<256x160xbf16>, tensor<160x160xbf16>) -> tensor<256x160xbf16>
    %593 = "ttir.unsqueeze"(%592) <{dim = 0 : si32}> : (tensor<256x160xbf16>) -> tensor<1x256x160xbf16>
    %594 = "ttir.add"(%593, %arg234) : (tensor<1x256x160xbf16>, tensor<160xbf16>) -> tensor<1x256x160xbf16>
    %595 = "ttir.reshape"(%594) <{shape = [1 : i32, 256 : i32, 5 : i32, 32 : i32]}> : (tensor<1x256x160xbf16>) -> tensor<1x256x5x32xbf16>
    %596 = "ttir.transpose"(%595) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x256x5x32xbf16>) -> tensor<1x5x256x32xbf16>
    %597 = "ttir.squeeze"(%596) <{dim = 0 : si32}> : (tensor<1x5x256x32xbf16>) -> tensor<5x256x32xbf16>
    %598 = "ttir.transpose"(%597) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<5x256x32xbf16>) -> tensor<5x32x256xbf16>
    %599 = "ttir.matmul"(%569, %598) <{transpose_a = false, transpose_b = false}> : (tensor<5x1024x32xbf16>, tensor<5x32x256xbf16>) -> tensor<5x1024x256xbf16>
    %600 = "ttir.unsqueeze"(%599) <{dim = 0 : si32}> : (tensor<5x1024x256xbf16>) -> tensor<1x5x1024x256xbf16>
    %601 = "ttir.div"(%600, %arg72) : (tensor<1x5x1024x256xbf16>, tensor<1xbf16>) -> tensor<1x5x1024x256xbf16>
    %602 = "ttir.softmax"(%601) <{dimension = -1 : si32}> : (tensor<1x5x1024x256xbf16>) -> tensor<1x5x1024x256xbf16>
    %603 = "ttir.squeeze"(%602) <{dim = 0 : si32}> : (tensor<1x5x1024x256xbf16>) -> tensor<5x1024x256xbf16>
    %604 = "ttir.matmul"(%591, %arg235) <{transpose_a = false, transpose_b = false}> : (tensor<256x160xbf16>, tensor<160x160xbf16>) -> tensor<256x160xbf16>
    %605 = "ttir.unsqueeze"(%604) <{dim = 0 : si32}> : (tensor<256x160xbf16>) -> tensor<1x256x160xbf16>
    %606 = "ttir.add"(%605, %arg236) : (tensor<1x256x160xbf16>, tensor<160xbf16>) -> tensor<1x256x160xbf16>
    %607 = "ttir.reshape"(%606) <{shape = [1 : i32, 256 : i32, 5 : i32, 32 : i32]}> : (tensor<1x256x160xbf16>) -> tensor<1x256x5x32xbf16>
    %608 = "ttir.transpose"(%607) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x256x5x32xbf16>) -> tensor<1x5x256x32xbf16>
    %609 = "ttir.transpose"(%608) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x5x256x32xbf16>) -> tensor<1x5x32x256xbf16>
    %610 = "ttir.squeeze"(%609) <{dim = 0 : si32}> : (tensor<1x5x32x256xbf16>) -> tensor<5x32x256xbf16>
    %611 = "ttir.transpose"(%610) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<5x32x256xbf16>) -> tensor<5x256x32xbf16>
    %612 = "ttir.matmul"(%603, %611) <{transpose_a = false, transpose_b = false}> : (tensor<5x1024x256xbf16>, tensor<5x256x32xbf16>) -> tensor<5x1024x32xbf16>
    %613 = "ttir.unsqueeze"(%612) <{dim = 0 : si32}> : (tensor<5x1024x32xbf16>) -> tensor<1x5x1024x32xbf16>
    %614 = "ttir.transpose"(%613) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x5x1024x32xbf16>) -> tensor<1x1024x5x32xbf16>
    %615 = "ttir.reshape"(%614) <{shape = [1024 : i32, 160 : i32]}> : (tensor<1x1024x5x32xbf16>) -> tensor<1024x160xbf16>
    %616 = "ttir.matmul"(%615, %arg237) <{transpose_a = false, transpose_b = false}> : (tensor<1024x160xbf16>, tensor<160x160xbf16>) -> tensor<1024x160xbf16>
    %617 = "ttir.unsqueeze"(%616) <{dim = 0 : si32}> : (tensor<1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %618 = "ttir.add"(%617, %arg238) : (tensor<1x1024x160xbf16>, tensor<160xbf16>) -> tensor<1x1024x160xbf16>
    %619 = "ttir.add"(%618, %552) : (tensor<1x1024x160xbf16>, tensor<1x1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %620 = "ttir.sum"(%619) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x1024x160xbf16>) -> tensor<1x1024x1xbf16>
    %621 = "ttir.multiply"(%arg73, %620) : (tensor<1x1024x160xf32>, tensor<1x1024x1xbf16>) -> tensor<1x1024x160xbf16>
    %622 = "ttir.subtract"(%619, %621) : (tensor<1x1024x160xbf16>, tensor<1x1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %623 = "ttir.multiply"(%622, %622) : (tensor<1x1024x160xbf16>, tensor<1x1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %624 = "ttir.sum"(%623) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x1024x160xbf16>) -> tensor<1x1024x1xbf16>
    %625 = "ttir.multiply"(%arg74, %624) : (tensor<1x1024x1xf32>, tensor<1x1024x1xbf16>) -> tensor<1x1024x1xbf16>
    %626 = "ttir.add"(%625, %arg75) : (tensor<1x1024x1xbf16>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xbf16>
    %627 = "ttir.sqrt"(%626) : (tensor<1x1024x1xbf16>) -> tensor<1x1024x1xbf16>
    %628 = "ttir.reciprocal"(%627) : (tensor<1x1024x1xbf16>) -> tensor<1x1024x1xbf16>
    %629 = "ttir.multiply"(%622, %628) : (tensor<1x1024x160xbf16>, tensor<1x1024x1xbf16>) -> tensor<1x1024x160xbf16>
    %630 = "ttir.multiply"(%629, %arg239) : (tensor<1x1024x160xbf16>, tensor<160xbf16>) -> tensor<1x1024x160xbf16>
    %631 = "ttir.add"(%630, %arg240) : (tensor<1x1024x160xbf16>, tensor<160xbf16>) -> tensor<1x1024x160xbf16>
    %632 = "ttir.matmul"(%631, %arg241) <{transpose_a = false, transpose_b = false}> : (tensor<1x1024x160xbf16>, tensor<160x640xbf16>) -> tensor<1x1024x640xbf16>
    %633 = "ttir.add"(%632, %arg242) : (tensor<1x1024x640xbf16>, tensor<640xbf16>) -> tensor<1x1024x640xbf16>
    %634 = "ttir.transpose"(%633) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x1024x640xbf16>) -> tensor<1x640x1024xbf16>
    %635 = "ttir.reshape"(%634) <{shape = [1 : i32, 640 : i32, 32 : i32, 32 : i32]}> : (tensor<1x640x1024xbf16>) -> tensor<1x640x32x32xbf16>
    %636 = "ttir.transpose"(%635) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x640x32x32xbf16>) -> tensor<1x32x640x32xbf16>
    %637 = "ttir.transpose"(%636) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x640x32xbf16>) -> tensor<1x32x32x640xbf16>
    %638 = "ttir.conv2d"(%637, %arg243, %arg244) <{dilation = array<i32: 1, 1>, groups = 640 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x32x32x640xbf16>, tensor<640x1x3x3xbf16>, tensor<1x1x1x640xbf16>) -> tensor<1x32x32x640xbf16>
    %639 = "ttir.transpose"(%638) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x640xbf16>) -> tensor<1x32x640x32xbf16>
    %640 = "ttir.transpose"(%639) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x640x32xbf16>) -> tensor<1x640x32x32xbf16>
    %641 = "ttir.reshape"(%640) <{shape = [1 : i32, 640 : i32, 1024 : i32, 1 : i32]}> : (tensor<1x640x32x32xbf16>) -> tensor<1x640x1024x1xbf16>
    %642 = "ttir.squeeze"(%641) <{dim = -1 : si32}> : (tensor<1x640x1024x1xbf16>) -> tensor<1x640x1024xbf16>
    %643 = "ttir.transpose"(%642) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x640x1024xbf16>) -> tensor<1x1024x640xbf16>
    %644 = "ttir.gelu"(%643) : (tensor<1x1024x640xbf16>) -> tensor<1x1024x640xbf16>
    %645 = "ttir.matmul"(%644, %arg245) <{transpose_a = false, transpose_b = false}> : (tensor<1x1024x640xbf16>, tensor<640x160xbf16>) -> tensor<1x1024x160xbf16>
    %646 = "ttir.add"(%645, %arg246) : (tensor<1x1024x160xbf16>, tensor<160xbf16>) -> tensor<1x1024x160xbf16>
    %647 = "ttir.add"(%646, %619) : (tensor<1x1024x160xbf16>, tensor<1x1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %648 = "ttir.sum"(%647) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x1024x160xbf16>) -> tensor<1x1024x1xbf16>
    %649 = "ttir.multiply"(%arg76, %648) : (tensor<1x1024x160xf32>, tensor<1x1024x1xbf16>) -> tensor<1x1024x160xbf16>
    %650 = "ttir.subtract"(%647, %649) : (tensor<1x1024x160xbf16>, tensor<1x1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %651 = "ttir.multiply"(%650, %650) : (tensor<1x1024x160xbf16>, tensor<1x1024x160xbf16>) -> tensor<1x1024x160xbf16>
    %652 = "ttir.sum"(%651) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x1024x160xbf16>) -> tensor<1x1024x1xbf16>
    %653 = "ttir.multiply"(%arg77, %652) : (tensor<1x1024x1xf32>, tensor<1x1024x1xbf16>) -> tensor<1x1024x1xbf16>
    %654 = "ttir.add"(%653, %arg78) : (tensor<1x1024x1xbf16>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xbf16>
    %655 = "ttir.sqrt"(%654) : (tensor<1x1024x1xbf16>) -> tensor<1x1024x1xbf16>
    %656 = "ttir.reciprocal"(%655) : (tensor<1x1024x1xbf16>) -> tensor<1x1024x1xbf16>
    %657 = "ttir.multiply"(%650, %656) : (tensor<1x1024x160xbf16>, tensor<1x1024x1xbf16>) -> tensor<1x1024x160xbf16>
    %658 = "ttir.multiply"(%657, %arg247) : (tensor<1x1024x160xbf16>, tensor<160xbf16>) -> tensor<1x1024x160xbf16>
    %659 = "ttir.add"(%658, %arg248) : (tensor<1x1024x160xbf16>, tensor<160xbf16>) -> tensor<1x1024x160xbf16>
    %660 = "ttir.reshape"(%659) <{shape = [1 : i32, 32 : i32, 32 : i32, 160 : i32]}> : (tensor<1x1024x160xbf16>) -> tensor<1x32x32x160xbf16>
    %661 = "ttir.transpose"(%660) <{dim0 = -3 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x160xbf16>) -> tensor<1x160x32x32xbf16>
    %662 = "ttir.transpose"(%661) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x32x32xbf16>) -> tensor<1x160x32x32xbf16>
    %663 = "ttir.transpose"(%662) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x32x32xbf16>) -> tensor<1x32x160x32xbf16>
    %664 = "ttir.transpose"(%663) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x160x32xbf16>) -> tensor<1x32x32x160xbf16>
    %665 = "ttir.conv2d"(%664, %arg249, %arg250) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x32x32x160xbf16>, tensor<256x160x3x3xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x16x16x256xbf16>
    %666 = "ttir.transpose"(%665) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x16x16x256xbf16>) -> tensor<1x16x256x16xbf16>
    %667 = "ttir.transpose"(%666) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x16x256x16xbf16>) -> tensor<1x256x16x16xbf16>
    %668 = "ttir.reshape"(%667) <{shape = [1 : i32, 256 : i32, 256 : i32, 1 : i32]}> : (tensor<1x256x16x16xbf16>) -> tensor<1x256x256x1xbf16>
    %669 = "ttir.squeeze"(%668) <{dim = -1 : si32}> : (tensor<1x256x256x1xbf16>) -> tensor<1x256x256xbf16>
    %670 = "ttir.transpose"(%669) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %671 = "ttir.sum"(%670) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x256xbf16>) -> tensor<1x256x1xbf16>
    %672 = "ttir.multiply"(%arg79, %671) : (tensor<1x256x256xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x256xbf16>
    %673 = "ttir.subtract"(%670, %672) : (tensor<1x256x256xbf16>, tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %674 = "ttir.multiply"(%673, %673) : (tensor<1x256x256xbf16>, tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %675 = "ttir.sum"(%674) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x256xbf16>) -> tensor<1x256x1xbf16>
    %676 = "ttir.multiply"(%arg80, %675) : (tensor<1x256x1xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %677 = "ttir.add"(%676, %arg81) : (tensor<1x256x1xbf16>, tensor<1x256x1xf32>) -> tensor<1x256x1xbf16>
    %678 = "ttir.sqrt"(%677) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %679 = "ttir.reciprocal"(%678) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %680 = "ttir.multiply"(%673, %679) : (tensor<1x256x256xbf16>, tensor<1x256x1xbf16>) -> tensor<1x256x256xbf16>
    %681 = "ttir.multiply"(%680, %arg251) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %682 = "ttir.add"(%681, %arg252) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %683 = "ttir.sum"(%682) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x256xbf16>) -> tensor<1x256x1xbf16>
    %684 = "ttir.multiply"(%arg82, %683) : (tensor<1x256x256xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x256xbf16>
    %685 = "ttir.subtract"(%682, %684) : (tensor<1x256x256xbf16>, tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %686 = "ttir.multiply"(%685, %685) : (tensor<1x256x256xbf16>, tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %687 = "ttir.sum"(%686) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x256xbf16>) -> tensor<1x256x1xbf16>
    %688 = "ttir.multiply"(%arg83, %687) : (tensor<1x256x1xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %689 = "ttir.add"(%688, %arg84) : (tensor<1x256x1xbf16>, tensor<1x256x1xf32>) -> tensor<1x256x1xbf16>
    %690 = "ttir.sqrt"(%689) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %691 = "ttir.reciprocal"(%690) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %692 = "ttir.multiply"(%685, %691) : (tensor<1x256x256xbf16>, tensor<1x256x1xbf16>) -> tensor<1x256x256xbf16>
    %693 = "ttir.multiply"(%692, %arg253) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %694 = "ttir.add"(%693, %arg254) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %695 = "ttir.squeeze"(%694) <{dim = 0 : si32}> : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %696 = "ttir.matmul"(%695, %arg255) <{transpose_a = false, transpose_b = false}> : (tensor<256x256xbf16>, tensor<256x256xbf16>) -> tensor<256x256xbf16>
    %697 = "ttir.unsqueeze"(%696) <{dim = 0 : si32}> : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %698 = "ttir.add"(%697, %arg256) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %699 = "ttir.reshape"(%698) <{shape = [1 : i32, 256 : i32, 8 : i32, 32 : i32]}> : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %700 = "ttir.transpose"(%699) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %701 = "ttir.squeeze"(%700) <{dim = 0 : si32}> : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %702 = "ttir.matmul"(%695, %arg257) <{transpose_a = false, transpose_b = false}> : (tensor<256x256xbf16>, tensor<256x256xbf16>) -> tensor<256x256xbf16>
    %703 = "ttir.unsqueeze"(%702) <{dim = 0 : si32}> : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %704 = "ttir.add"(%703, %arg258) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %705 = "ttir.reshape"(%704) <{shape = [1 : i32, 256 : i32, 8 : i32, 32 : i32]}> : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %706 = "ttir.transpose"(%705) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %707 = "ttir.squeeze"(%706) <{dim = 0 : si32}> : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %708 = "ttir.transpose"(%707) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<8x256x32xbf16>) -> tensor<8x32x256xbf16>
    %709 = "ttir.matmul"(%701, %708) <{transpose_a = false, transpose_b = false}> : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %710 = "ttir.unsqueeze"(%709) <{dim = 0 : si32}> : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %711 = "ttir.div"(%710, %arg85) : (tensor<1x8x256x256xbf16>, tensor<1xbf16>) -> tensor<1x8x256x256xbf16>
    %712 = "ttir.softmax"(%711) <{dimension = -1 : si32}> : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %713 = "ttir.squeeze"(%712) <{dim = 0 : si32}> : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %714 = "ttir.matmul"(%695, %arg259) <{transpose_a = false, transpose_b = false}> : (tensor<256x256xbf16>, tensor<256x256xbf16>) -> tensor<256x256xbf16>
    %715 = "ttir.unsqueeze"(%714) <{dim = 0 : si32}> : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %716 = "ttir.add"(%715, %arg260) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %717 = "ttir.reshape"(%716) <{shape = [1 : i32, 256 : i32, 8 : i32, 32 : i32]}> : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %718 = "ttir.transpose"(%717) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %719 = "ttir.transpose"(%718) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %720 = "ttir.squeeze"(%719) <{dim = 0 : si32}> : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %721 = "ttir.transpose"(%720) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<8x32x256xbf16>) -> tensor<8x256x32xbf16>
    %722 = "ttir.matmul"(%713, %721) <{transpose_a = false, transpose_b = false}> : (tensor<8x256x256xbf16>, tensor<8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %723 = "ttir.unsqueeze"(%722) <{dim = 0 : si32}> : (tensor<8x256x32xbf16>) -> tensor<1x8x256x32xbf16>
    %724 = "ttir.transpose"(%723) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x8x256x32xbf16>) -> tensor<1x256x8x32xbf16>
    %725 = "ttir.reshape"(%724) <{shape = [256 : i32, 256 : i32]}> : (tensor<1x256x8x32xbf16>) -> tensor<256x256xbf16>
    %726 = "ttir.matmul"(%725, %arg261) <{transpose_a = false, transpose_b = false}> : (tensor<256x256xbf16>, tensor<256x256xbf16>) -> tensor<256x256xbf16>
    %727 = "ttir.unsqueeze"(%726) <{dim = 0 : si32}> : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %728 = "ttir.add"(%727, %arg262) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %729 = "ttir.add"(%728, %682) : (tensor<1x256x256xbf16>, tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %730 = "ttir.sum"(%729) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x256xbf16>) -> tensor<1x256x1xbf16>
    %731 = "ttir.multiply"(%arg86, %730) : (tensor<1x256x256xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x256xbf16>
    %732 = "ttir.subtract"(%729, %731) : (tensor<1x256x256xbf16>, tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %733 = "ttir.multiply"(%732, %732) : (tensor<1x256x256xbf16>, tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %734 = "ttir.sum"(%733) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x256xbf16>) -> tensor<1x256x1xbf16>
    %735 = "ttir.multiply"(%arg87, %734) : (tensor<1x256x1xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %736 = "ttir.add"(%735, %arg88) : (tensor<1x256x1xbf16>, tensor<1x256x1xf32>) -> tensor<1x256x1xbf16>
    %737 = "ttir.sqrt"(%736) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %738 = "ttir.reciprocal"(%737) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %739 = "ttir.multiply"(%732, %738) : (tensor<1x256x256xbf16>, tensor<1x256x1xbf16>) -> tensor<1x256x256xbf16>
    %740 = "ttir.multiply"(%739, %arg263) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %741 = "ttir.add"(%740, %arg264) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %742 = "ttir.matmul"(%741, %arg265) <{transpose_a = false, transpose_b = false}> : (tensor<1x256x256xbf16>, tensor<256x1024xbf16>) -> tensor<1x256x1024xbf16>
    %743 = "ttir.add"(%742, %arg266) : (tensor<1x256x1024xbf16>, tensor<1024xbf16>) -> tensor<1x256x1024xbf16>
    %744 = "ttir.transpose"(%743) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x256x1024xbf16>) -> tensor<1x1024x256xbf16>
    %745 = "ttir.reshape"(%744) <{shape = [1 : i32, 1024 : i32, 16 : i32, 16 : i32]}> : (tensor<1x1024x256xbf16>) -> tensor<1x1024x16x16xbf16>
    %746 = "ttir.transpose"(%745) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x1024x16x16xbf16>) -> tensor<1x16x1024x16xbf16>
    %747 = "ttir.transpose"(%746) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x16x1024x16xbf16>) -> tensor<1x16x16x1024xbf16>
    %748 = "ttir.conv2d"(%747, %arg267, %arg268) <{dilation = array<i32: 1, 1>, groups = 1024 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x16x16x1024xbf16>, tensor<1024x1x3x3xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<1x16x16x1024xbf16>
    %749 = "ttir.transpose"(%748) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x16x16x1024xbf16>) -> tensor<1x16x1024x16xbf16>
    %750 = "ttir.transpose"(%749) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x16x1024x16xbf16>) -> tensor<1x1024x16x16xbf16>
    %751 = "ttir.reshape"(%750) <{shape = [1 : i32, 1024 : i32, 256 : i32, 1 : i32]}> : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x256x1xbf16>
    %752 = "ttir.squeeze"(%751) <{dim = -1 : si32}> : (tensor<1x1024x256x1xbf16>) -> tensor<1x1024x256xbf16>
    %753 = "ttir.transpose"(%752) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x1024x256xbf16>) -> tensor<1x256x1024xbf16>
    %754 = "ttir.gelu"(%753) : (tensor<1x256x1024xbf16>) -> tensor<1x256x1024xbf16>
    %755 = "ttir.matmul"(%754, %arg269) <{transpose_a = false, transpose_b = false}> : (tensor<1x256x1024xbf16>, tensor<1024x256xbf16>) -> tensor<1x256x256xbf16>
    %756 = "ttir.add"(%755, %arg270) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %757 = "ttir.add"(%756, %729) : (tensor<1x256x256xbf16>, tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %758 = "ttir.sum"(%757) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x256xbf16>) -> tensor<1x256x1xbf16>
    %759 = "ttir.multiply"(%arg89, %758) : (tensor<1x256x256xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x256xbf16>
    %760 = "ttir.subtract"(%757, %759) : (tensor<1x256x256xbf16>, tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %761 = "ttir.multiply"(%760, %760) : (tensor<1x256x256xbf16>, tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %762 = "ttir.sum"(%761) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x256xbf16>) -> tensor<1x256x1xbf16>
    %763 = "ttir.multiply"(%arg90, %762) : (tensor<1x256x1xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %764 = "ttir.add"(%763, %arg91) : (tensor<1x256x1xbf16>, tensor<1x256x1xf32>) -> tensor<1x256x1xbf16>
    %765 = "ttir.sqrt"(%764) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %766 = "ttir.reciprocal"(%765) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %767 = "ttir.multiply"(%760, %766) : (tensor<1x256x256xbf16>, tensor<1x256x1xbf16>) -> tensor<1x256x256xbf16>
    %768 = "ttir.multiply"(%767, %arg271) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %769 = "ttir.add"(%768, %arg272) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %770 = "ttir.squeeze"(%769) <{dim = 0 : si32}> : (tensor<1x256x256xbf16>) -> tensor<256x256xbf16>
    %771 = "ttir.matmul"(%770, %arg273) <{transpose_a = false, transpose_b = false}> : (tensor<256x256xbf16>, tensor<256x256xbf16>) -> tensor<256x256xbf16>
    %772 = "ttir.unsqueeze"(%771) <{dim = 0 : si32}> : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %773 = "ttir.add"(%772, %arg274) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %774 = "ttir.reshape"(%773) <{shape = [1 : i32, 256 : i32, 8 : i32, 32 : i32]}> : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %775 = "ttir.transpose"(%774) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %776 = "ttir.squeeze"(%775) <{dim = 0 : si32}> : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %777 = "ttir.matmul"(%770, %arg275) <{transpose_a = false, transpose_b = false}> : (tensor<256x256xbf16>, tensor<256x256xbf16>) -> tensor<256x256xbf16>
    %778 = "ttir.unsqueeze"(%777) <{dim = 0 : si32}> : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %779 = "ttir.add"(%778, %arg276) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %780 = "ttir.reshape"(%779) <{shape = [1 : i32, 256 : i32, 8 : i32, 32 : i32]}> : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %781 = "ttir.transpose"(%780) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %782 = "ttir.squeeze"(%781) <{dim = 0 : si32}> : (tensor<1x8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %783 = "ttir.transpose"(%782) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<8x256x32xbf16>) -> tensor<8x32x256xbf16>
    %784 = "ttir.matmul"(%776, %783) <{transpose_a = false, transpose_b = false}> : (tensor<8x256x32xbf16>, tensor<8x32x256xbf16>) -> tensor<8x256x256xbf16>
    %785 = "ttir.unsqueeze"(%784) <{dim = 0 : si32}> : (tensor<8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %786 = "ttir.div"(%785, %arg92) : (tensor<1x8x256x256xbf16>, tensor<1xbf16>) -> tensor<1x8x256x256xbf16>
    %787 = "ttir.softmax"(%786) <{dimension = -1 : si32}> : (tensor<1x8x256x256xbf16>) -> tensor<1x8x256x256xbf16>
    %788 = "ttir.squeeze"(%787) <{dim = 0 : si32}> : (tensor<1x8x256x256xbf16>) -> tensor<8x256x256xbf16>
    %789 = "ttir.matmul"(%770, %arg277) <{transpose_a = false, transpose_b = false}> : (tensor<256x256xbf16>, tensor<256x256xbf16>) -> tensor<256x256xbf16>
    %790 = "ttir.unsqueeze"(%789) <{dim = 0 : si32}> : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %791 = "ttir.add"(%790, %arg278) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %792 = "ttir.reshape"(%791) <{shape = [1 : i32, 256 : i32, 8 : i32, 32 : i32]}> : (tensor<1x256x256xbf16>) -> tensor<1x256x8x32xbf16>
    %793 = "ttir.transpose"(%792) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x256x8x32xbf16>) -> tensor<1x8x256x32xbf16>
    %794 = "ttir.transpose"(%793) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x256x32xbf16>) -> tensor<1x8x32x256xbf16>
    %795 = "ttir.squeeze"(%794) <{dim = 0 : si32}> : (tensor<1x8x32x256xbf16>) -> tensor<8x32x256xbf16>
    %796 = "ttir.transpose"(%795) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<8x32x256xbf16>) -> tensor<8x256x32xbf16>
    %797 = "ttir.matmul"(%788, %796) <{transpose_a = false, transpose_b = false}> : (tensor<8x256x256xbf16>, tensor<8x256x32xbf16>) -> tensor<8x256x32xbf16>
    %798 = "ttir.unsqueeze"(%797) <{dim = 0 : si32}> : (tensor<8x256x32xbf16>) -> tensor<1x8x256x32xbf16>
    %799 = "ttir.transpose"(%798) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x8x256x32xbf16>) -> tensor<1x256x8x32xbf16>
    %800 = "ttir.reshape"(%799) <{shape = [256 : i32, 256 : i32]}> : (tensor<1x256x8x32xbf16>) -> tensor<256x256xbf16>
    %801 = "ttir.matmul"(%800, %arg279) <{transpose_a = false, transpose_b = false}> : (tensor<256x256xbf16>, tensor<256x256xbf16>) -> tensor<256x256xbf16>
    %802 = "ttir.unsqueeze"(%801) <{dim = 0 : si32}> : (tensor<256x256xbf16>) -> tensor<1x256x256xbf16>
    %803 = "ttir.add"(%802, %arg280) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %804 = "ttir.add"(%803, %757) : (tensor<1x256x256xbf16>, tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %805 = "ttir.sum"(%804) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x256xbf16>) -> tensor<1x256x1xbf16>
    %806 = "ttir.multiply"(%arg93, %805) : (tensor<1x256x256xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x256xbf16>
    %807 = "ttir.subtract"(%804, %806) : (tensor<1x256x256xbf16>, tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %808 = "ttir.multiply"(%807, %807) : (tensor<1x256x256xbf16>, tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %809 = "ttir.sum"(%808) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x256xbf16>) -> tensor<1x256x1xbf16>
    %810 = "ttir.multiply"(%arg94, %809) : (tensor<1x256x1xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %811 = "ttir.add"(%810, %arg95) : (tensor<1x256x1xbf16>, tensor<1x256x1xf32>) -> tensor<1x256x1xbf16>
    %812 = "ttir.sqrt"(%811) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %813 = "ttir.reciprocal"(%812) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %814 = "ttir.multiply"(%807, %813) : (tensor<1x256x256xbf16>, tensor<1x256x1xbf16>) -> tensor<1x256x256xbf16>
    %815 = "ttir.multiply"(%814, %arg281) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %816 = "ttir.add"(%815, %arg282) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %817 = "ttir.matmul"(%816, %arg283) <{transpose_a = false, transpose_b = false}> : (tensor<1x256x256xbf16>, tensor<256x1024xbf16>) -> tensor<1x256x1024xbf16>
    %818 = "ttir.add"(%817, %arg284) : (tensor<1x256x1024xbf16>, tensor<1024xbf16>) -> tensor<1x256x1024xbf16>
    %819 = "ttir.transpose"(%818) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x256x1024xbf16>) -> tensor<1x1024x256xbf16>
    %820 = "ttir.reshape"(%819) <{shape = [1 : i32, 1024 : i32, 16 : i32, 16 : i32]}> : (tensor<1x1024x256xbf16>) -> tensor<1x1024x16x16xbf16>
    %821 = "ttir.transpose"(%820) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x1024x16x16xbf16>) -> tensor<1x16x1024x16xbf16>
    %822 = "ttir.transpose"(%821) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x16x1024x16xbf16>) -> tensor<1x16x16x1024xbf16>
    %823 = "ttir.conv2d"(%822, %arg285, %arg286) <{dilation = array<i32: 1, 1>, groups = 1024 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x16x16x1024xbf16>, tensor<1024x1x3x3xbf16>, tensor<1x1x1x1024xbf16>) -> tensor<1x16x16x1024xbf16>
    %824 = "ttir.transpose"(%823) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x16x16x1024xbf16>) -> tensor<1x16x1024x16xbf16>
    %825 = "ttir.transpose"(%824) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x16x1024x16xbf16>) -> tensor<1x1024x16x16xbf16>
    %826 = "ttir.reshape"(%825) <{shape = [1 : i32, 1024 : i32, 256 : i32, 1 : i32]}> : (tensor<1x1024x16x16xbf16>) -> tensor<1x1024x256x1xbf16>
    %827 = "ttir.squeeze"(%826) <{dim = -1 : si32}> : (tensor<1x1024x256x1xbf16>) -> tensor<1x1024x256xbf16>
    %828 = "ttir.transpose"(%827) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x1024x256xbf16>) -> tensor<1x256x1024xbf16>
    %829 = "ttir.gelu"(%828) : (tensor<1x256x1024xbf16>) -> tensor<1x256x1024xbf16>
    %830 = "ttir.matmul"(%829, %arg287) <{transpose_a = false, transpose_b = false}> : (tensor<1x256x1024xbf16>, tensor<1024x256xbf16>) -> tensor<1x256x256xbf16>
    %831 = "ttir.add"(%830, %arg288) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %832 = "ttir.add"(%831, %804) : (tensor<1x256x256xbf16>, tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %833 = "ttir.sum"(%832) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x256xbf16>) -> tensor<1x256x1xbf16>
    %834 = "ttir.multiply"(%arg96, %833) : (tensor<1x256x256xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x256xbf16>
    %835 = "ttir.subtract"(%832, %834) : (tensor<1x256x256xbf16>, tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %836 = "ttir.multiply"(%835, %835) : (tensor<1x256x256xbf16>, tensor<1x256x256xbf16>) -> tensor<1x256x256xbf16>
    %837 = "ttir.sum"(%836) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x256x256xbf16>) -> tensor<1x256x1xbf16>
    %838 = "ttir.multiply"(%arg97, %837) : (tensor<1x256x1xf32>, tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %839 = "ttir.add"(%838, %arg98) : (tensor<1x256x1xbf16>, tensor<1x256x1xf32>) -> tensor<1x256x1xbf16>
    %840 = "ttir.sqrt"(%839) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %841 = "ttir.reciprocal"(%840) : (tensor<1x256x1xbf16>) -> tensor<1x256x1xbf16>
    %842 = "ttir.multiply"(%835, %841) : (tensor<1x256x256xbf16>, tensor<1x256x1xbf16>) -> tensor<1x256x256xbf16>
    %843 = "ttir.multiply"(%842, %arg289) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %844 = "ttir.add"(%843, %arg290) : (tensor<1x256x256xbf16>, tensor<256xbf16>) -> tensor<1x256x256xbf16>
    %845 = "ttir.mean"(%844) <{dim_arg = [-2 : i32], keep_dim = true}> : (tensor<1x256x256xbf16>) -> tensor<1x1x256xbf16>
    %846 = "ttir.squeeze"(%845) <{dim = 1 : si32}> : (tensor<1x1x256xbf16>) -> tensor<1x256xbf16>
    %847 = "ttir.matmul"(%846, %arg291) <{transpose_a = false, transpose_b = false}> : (tensor<1x256xbf16>, tensor<256x1000xbf16>) -> tensor<1x1000xbf16>
    %848 = "ttir.add"(%847, %arg292) : (tensor<1x1000xbf16>, tensor<1000xbf16>) -> tensor<1x1000xbf16>
    return %848 : tensor<1x1000xbf16> loc(#loc71)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("transformers.models.segformer.modeling_segformer.SegformerForImageClassification::")
#loc2 = loc("add_11")
#loc3 = loc("add_27")
#loc4 = loc("transpose_30")
#loc5 = loc("add_40")
#loc6 = loc("transpose_45")
#loc7 = loc("add_49")
#loc8 = loc("add_55")
#loc9 = loc("add_70")
#loc10 = loc("add_76")
#loc11 = loc("add_92")
#loc12 = loc("transpose_95")
#loc13 = loc("add_105")
#loc14 = loc("transpose_110")
#loc15 = loc("add_114")
#loc16 = loc("add_120")
#loc17 = loc("add_135")
#loc18 = loc("transpose_140")
#loc19 = loc("add_153")
#loc20 = loc("add_170")
#loc21 = loc("transpose_174")
#loc22 = loc("add_184")
#loc23 = loc("transpose_189")
#loc24 = loc("add_197")
#loc25 = loc("add_203")
#loc26 = loc("add_218")
#loc27 = loc("add_224")
#loc28 = loc("add_241")
#loc29 = loc("transpose_245")
#loc30 = loc("add_255")
#loc31 = loc("transpose_260")
#loc32 = loc("add_268")
#loc33 = loc("add_274")
#loc34 = loc("add_289")
#loc35 = loc("transpose_294")
#loc36 = loc("add_307")
#loc37 = loc("add_324")
#loc38 = loc("transpose_328")
#loc39 = loc("add_338")
#loc40 = loc("transpose_343")
#loc41 = loc("add_351")
#loc42 = loc("add_357")
#loc43 = loc("add_372")
#loc44 = loc("add_378")
#loc45 = loc("add_395")
#loc46 = loc("transpose_399")
#loc47 = loc("add_409")
#loc48 = loc("transpose_414")
#loc49 = loc("add_422")
#loc50 = loc("add_428")
#loc51 = loc("add_443")
#loc52 = loc("transpose_448")
#loc53 = loc("add_463")
#loc54 = loc("add_470")
#loc55 = loc("transpose_474")
#loc56 = loc("add_484")
#loc57 = loc("transpose_489")
#loc58 = loc("add_497")
#loc59 = loc("add_503")
#loc60 = loc("add_518")
#loc61 = loc("add_526")
#loc62 = loc("add_533")
#loc63 = loc("transpose_537")
#loc64 = loc("add_547")
#loc65 = loc("transpose_552")
#loc66 = loc("add_560")
#loc67 = loc("add_566")
#loc68 = loc("add_581")
#loc69 = loc("reduce_avg_586")
#loc70 = loc("add_590")
#loc71 = loc(unknown)
#loc72 = loc("transformers.models.segformer.modeling_segformer.SegformerModel::segformer"(#loc1))
#loc73 = loc("squeeze_587"(#loc1))
#loc74 = loc("torch.nn.modules.linear.Linear::classifier"(#loc1))
#loc75 = loc("transformers.models.segformer.modeling_segformer.SegformerEncoder::encoder"(#loc72))
#loc76 = loc("matmul_589"(#loc74))
#loc77 = loc("transformers.models.segformer.modeling_segformer.SegformerOverlapPatchEmbeddings::patch_embeddings.0"(#loc75))
#loc78 = loc("transformers.models.segformer.modeling_segformer.SegformerLayer::0"(#loc75))
#loc79 = loc("transformers.models.segformer.modeling_segformer.SegformerLayer::1"(#loc75))
#loc80 = loc("torch.nn.modules.normalization.LayerNorm::0"(#loc75))
#loc81 = loc("reshape_139"(#loc75))
#loc82 = loc("transpose_141"(#loc75))
#loc83 = loc("transformers.models.segformer.modeling_segformer.SegformerOverlapPatchEmbeddings::patch_embeddings.1"(#loc75))
#loc84 = loc("torch.nn.modules.normalization.LayerNorm::1"(#loc75))
#loc85 = loc("reshape_293"(#loc75))
#loc86 = loc("transpose_295"(#loc75))
#loc87 = loc("transformers.models.segformer.modeling_segformer.SegformerOverlapPatchEmbeddings::patch_embeddings.2"(#loc75))
#loc88 = loc("torch.nn.modules.normalization.LayerNorm::2"(#loc75))
#loc89 = loc("reshape_447"(#loc75))
#loc90 = loc("transpose_449"(#loc75))
#loc91 = loc("transformers.models.segformer.modeling_segformer.SegformerOverlapPatchEmbeddings::patch_embeddings.3"(#loc75))
#loc92 = loc("torch.nn.modules.normalization.LayerNorm::3"(#loc75))
#loc93 = loc("torch.nn.modules.conv.Conv2d::proj"(#loc77))
#loc94 = loc("reshape_4"(#loc77))
#loc95 = loc("squeeze_5"(#loc77))
#loc96 = loc("transpose_6"(#loc77))
#loc97 = loc("torch.nn.modules.normalization.LayerNorm::layer_norm"(#loc77))
#loc98 = loc("torch.nn.modules.normalization.LayerNorm::layer_norm_1"(#loc78))
#loc99 = loc("transformers.models.segformer.modeling_segformer.SegformerAttention::attention"(#loc78))
#loc100 = loc("add_51"(#loc78))
#loc101 = loc("torch.nn.modules.normalization.LayerNorm::layer_norm_2"(#loc78))
#loc102 = loc("transformers.models.segformer.modeling_segformer.SegformerMixFFN::mlp"(#loc78))
#loc103 = loc("add_72"(#loc78))
#loc104 = loc("torch.nn.modules.normalization.LayerNorm::layer_norm_1"(#loc79))
#loc105 = loc("transformers.models.segformer.modeling_segformer.SegformerAttention::attention"(#loc79))
#loc106 = loc("add_116"(#loc79))
#loc107 = loc("torch.nn.modules.normalization.LayerNorm::layer_norm_2"(#loc79))
#loc108 = loc("transformers.models.segformer.modeling_segformer.SegformerMixFFN::mlp"(#loc79))
#loc109 = loc("add_137"(#loc79))
#loc110 = loc("layernorm_138.dc.reduce_sum.0"(#loc80))
#loc111 = loc("layernorm_138.dc.multiply.2"(#loc80))
#loc112 = loc("layernorm_138.dc.subtract.3"(#loc80))
#loc113 = loc("layernorm_138.dc.multiply.4"(#loc80))
#loc114 = loc("layernorm_138.dc.reduce_sum.5"(#loc80))
#loc115 = loc("layernorm_138.dc.multiply.7"(#loc80))
#loc116 = loc("layernorm_138.dc.add.9"(#loc80))
#loc117 = loc("layernorm_138.dc.sqrt.10"(#loc80))
#loc118 = loc("layernorm_138.dc.reciprocal.11"(#loc80))
#loc119 = loc("layernorm_138.dc.multiply.12"(#loc80))
#loc120 = loc("layernorm_138.dc.multiply.13"(#loc80))
#loc121 = loc("layernorm_138.dc.add.14"(#loc80))
#loc122 = loc("torch.nn.modules.conv.Conv2d::proj"(#loc83))
#loc123 = loc("reshape_146"(#loc83))
#loc124 = loc("squeeze_147"(#loc83))
#loc125 = loc("transpose_148"(#loc83))
#loc126 = loc("torch.nn.modules.normalization.LayerNorm::layer_norm"(#loc83))
#loc127 = loc("add_199"(#loc78))
#loc128 = loc("add_220"(#loc78))
#loc129 = loc("add_270"(#loc79))
#loc130 = loc("add_291"(#loc79))
#loc131 = loc("layernorm_292.dc.reduce_sum.0"(#loc84))
#loc132 = loc("layernorm_292.dc.multiply.2"(#loc84))
#loc133 = loc("layernorm_292.dc.subtract.3"(#loc84))
#loc134 = loc("layernorm_292.dc.multiply.4"(#loc84))
#loc135 = loc("layernorm_292.dc.reduce_sum.5"(#loc84))
#loc136 = loc("layernorm_292.dc.multiply.7"(#loc84))
#loc137 = loc("layernorm_292.dc.add.9"(#loc84))
#loc138 = loc("layernorm_292.dc.sqrt.10"(#loc84))
#loc139 = loc("layernorm_292.dc.reciprocal.11"(#loc84))
#loc140 = loc("layernorm_292.dc.multiply.12"(#loc84))
#loc141 = loc("layernorm_292.dc.multiply.13"(#loc84))
#loc142 = loc("layernorm_292.dc.add.14"(#loc84))
#loc143 = loc("torch.nn.modules.conv.Conv2d::proj"(#loc87))
#loc144 = loc("reshape_300"(#loc87))
#loc145 = loc("squeeze_301"(#loc87))
#loc146 = loc("transpose_302"(#loc87))
#loc147 = loc("torch.nn.modules.normalization.LayerNorm::layer_norm"(#loc87))
#loc148 = loc("add_353"(#loc78))
#loc149 = loc("add_374"(#loc78))
#loc150 = loc("add_424"(#loc79))
#loc151 = loc("add_445"(#loc79))
#loc152 = loc("layernorm_446.dc.reduce_sum.0"(#loc88))
#loc153 = loc("layernorm_446.dc.multiply.2"(#loc88))
#loc154 = loc("layernorm_446.dc.subtract.3"(#loc88))
#loc155 = loc("layernorm_446.dc.multiply.4"(#loc88))
#loc156 = loc("layernorm_446.dc.reduce_sum.5"(#loc88))
#loc157 = loc("layernorm_446.dc.multiply.7"(#loc88))
#loc158 = loc("layernorm_446.dc.add.9"(#loc88))
#loc159 = loc("layernorm_446.dc.sqrt.10"(#loc88))
#loc160 = loc("layernorm_446.dc.reciprocal.11"(#loc88))
#loc161 = loc("layernorm_446.dc.multiply.12"(#loc88))
#loc162 = loc("layernorm_446.dc.multiply.13"(#loc88))
#loc163 = loc("layernorm_446.dc.add.14"(#loc88))
#loc164 = loc("torch.nn.modules.conv.Conv2d::proj"(#loc91))
#loc165 = loc("reshape_454"(#loc91))
#loc166 = loc("squeeze_455"(#loc91))
#loc167 = loc("transpose_456"(#loc91))
#loc168 = loc("torch.nn.modules.normalization.LayerNorm::layer_norm"(#loc91))
#loc169 = loc("add_499"(#loc78))
#loc170 = loc("add_520"(#loc78))
#loc171 = loc("add_562"(#loc79))
#loc172 = loc("add_583"(#loc79))
#loc173 = loc("layernorm_584.dc.reduce_sum.0"(#loc92))
#loc174 = loc("layernorm_584.dc.multiply.2"(#loc92))
#loc175 = loc("layernorm_584.dc.subtract.3"(#loc92))
#loc176 = loc("layernorm_584.dc.multiply.4"(#loc92))
#loc177 = loc("layernorm_584.dc.reduce_sum.5"(#loc92))
#loc178 = loc("layernorm_584.dc.multiply.7"(#loc92))
#loc179 = loc("layernorm_584.dc.add.9"(#loc92))
#loc180 = loc("layernorm_584.dc.sqrt.10"(#loc92))
#loc181 = loc("layernorm_584.dc.reciprocal.11"(#loc92))
#loc182 = loc("layernorm_584.dc.multiply.12"(#loc92))
#loc183 = loc("layernorm_584.dc.multiply.13"(#loc92))
#loc184 = loc("layernorm_584.dc.add.14"(#loc92))
#loc185 = loc("conv2d_0.dc.transpose.0"(#loc93))
#loc186 = loc("conv2d_0.dc.transpose.1"(#loc93))
#loc187 = loc("conv2d_0.dc.conv2d.4"(#loc93))
#loc188 = loc("conv2d_0.dc.transpose.5"(#loc93))
#loc189 = loc("conv2d_0.dc.transpose.6"(#loc93))
#loc190 = loc("layernorm_7.dc.reduce_sum.0"(#loc97))
#loc191 = loc("layernorm_7.dc.multiply.2"(#loc97))
#loc192 = loc("layernorm_7.dc.subtract.3"(#loc97))
#loc193 = loc("layernorm_7.dc.multiply.4"(#loc97))
#loc194 = loc("layernorm_7.dc.reduce_sum.5"(#loc97))
#loc195 = loc("layernorm_7.dc.multiply.7"(#loc97))
#loc196 = loc("layernorm_7.dc.add.9"(#loc97))
#loc197 = loc("layernorm_7.dc.sqrt.10"(#loc97))
#loc198 = loc("layernorm_7.dc.reciprocal.11"(#loc97))
#loc199 = loc("layernorm_7.dc.multiply.12"(#loc97))
#loc200 = loc("layernorm_7.dc.multiply.13"(#loc97))
#loc201 = loc("layernorm_7.dc.add.14"(#loc97))
#loc202 = loc("layernorm_8.dc.reduce_sum.0"(#loc98))
#loc203 = loc("layernorm_8.dc.multiply.2"(#loc98))
#loc204 = loc("layernorm_8.dc.subtract.3"(#loc98))
#loc205 = loc("layernorm_8.dc.multiply.4"(#loc98))
#loc206 = loc("layernorm_8.dc.reduce_sum.5"(#loc98))
#loc207 = loc("layernorm_8.dc.multiply.7"(#loc98))
#loc208 = loc("layernorm_8.dc.add.9"(#loc98))
#loc209 = loc("layernorm_8.dc.sqrt.10"(#loc98))
#loc210 = loc("layernorm_8.dc.reciprocal.11"(#loc98))
#loc211 = loc("layernorm_8.dc.multiply.12"(#loc98))
#loc212 = loc("layernorm_8.dc.multiply.13"(#loc98))
#loc213 = loc("layernorm_8.dc.add.14"(#loc98))
#loc214 = loc("transformers.models.segformer.modeling_segformer.SegformerEfficientSelfAttention::self"(#loc99))
#loc215 = loc("transformers.models.segformer.modeling_segformer.SegformerSelfOutput::output"(#loc99))
#loc216 = loc("layernorm_52.dc.reduce_sum.0"(#loc101))
#loc217 = loc("layernorm_52.dc.multiply.2"(#loc101))
#loc218 = loc("layernorm_52.dc.subtract.3"(#loc101))
#loc219 = loc("layernorm_52.dc.multiply.4"(#loc101))
#loc220 = loc("layernorm_52.dc.reduce_sum.5"(#loc101))
#loc221 = loc("layernorm_52.dc.multiply.7"(#loc101))
#loc222 = loc("layernorm_52.dc.add.9"(#loc101))
#loc223 = loc("layernorm_52.dc.sqrt.10"(#loc101))
#loc224 = loc("layernorm_52.dc.reciprocal.11"(#loc101))
#loc225 = loc("layernorm_52.dc.multiply.12"(#loc101))
#loc226 = loc("layernorm_52.dc.multiply.13"(#loc101))
#loc227 = loc("layernorm_52.dc.add.14"(#loc101))
#loc228 = loc("torch.nn.modules.linear.Linear::dense1"(#loc102))
#loc229 = loc("transformers.models.segformer.modeling_segformer.SegformerDWConv::dwconv"(#loc102))
#loc230 = loc("transformers.activations.GELUActivation::intermediate_act_fn"(#loc102))
#loc231 = loc("torch.nn.modules.linear.Linear::dense2"(#loc102))
#loc232 = loc("layernorm_73.dc.reduce_sum.0"(#loc104))
#loc233 = loc("layernorm_73.dc.multiply.2"(#loc104))
#loc234 = loc("layernorm_73.dc.subtract.3"(#loc104))
#loc235 = loc("layernorm_73.dc.multiply.4"(#loc104))
#loc236 = loc("layernorm_73.dc.reduce_sum.5"(#loc104))
#loc237 = loc("layernorm_73.dc.multiply.7"(#loc104))
#loc238 = loc("layernorm_73.dc.add.9"(#loc104))
#loc239 = loc("layernorm_73.dc.sqrt.10"(#loc104))
#loc240 = loc("layernorm_73.dc.reciprocal.11"(#loc104))
#loc241 = loc("layernorm_73.dc.multiply.12"(#loc104))
#loc242 = loc("layernorm_73.dc.multiply.13"(#loc104))
#loc243 = loc("layernorm_73.dc.add.14"(#loc104))
#loc244 = loc("transformers.models.segformer.modeling_segformer.SegformerEfficientSelfAttention::self"(#loc105))
#loc245 = loc("transformers.models.segformer.modeling_segformer.SegformerSelfOutput::output"(#loc105))
#loc246 = loc("layernorm_117.dc.reduce_sum.0"(#loc107))
#loc247 = loc("layernorm_117.dc.multiply.2"(#loc107))
#loc248 = loc("layernorm_117.dc.subtract.3"(#loc107))
#loc249 = loc("layernorm_117.dc.multiply.4"(#loc107))
#loc250 = loc("layernorm_117.dc.reduce_sum.5"(#loc107))
#loc251 = loc("layernorm_117.dc.multiply.7"(#loc107))
#loc252 = loc("layernorm_117.dc.add.9"(#loc107))
#loc253 = loc("layernorm_117.dc.sqrt.10"(#loc107))
#loc254 = loc("layernorm_117.dc.reciprocal.11"(#loc107))
#loc255 = loc("layernorm_117.dc.multiply.12"(#loc107))
#loc256 = loc("layernorm_117.dc.multiply.13"(#loc107))
#loc257 = loc("layernorm_117.dc.add.14"(#loc107))
#loc258 = loc("torch.nn.modules.linear.Linear::dense1"(#loc108))
#loc259 = loc("transformers.models.segformer.modeling_segformer.SegformerDWConv::dwconv"(#loc108))
#loc260 = loc("transformers.activations.GELUActivation::intermediate_act_fn"(#loc108))
#loc261 = loc("torch.nn.modules.linear.Linear::dense2"(#loc108))
#loc262 = loc("conv2d_142.dc.transpose.0"(#loc122))
#loc263 = loc("conv2d_142.dc.transpose.1"(#loc122))
#loc264 = loc("conv2d_142.dc.conv2d.4"(#loc122))
#loc265 = loc("conv2d_142.dc.transpose.5"(#loc122))
#loc266 = loc("conv2d_142.dc.transpose.6"(#loc122))
#loc267 = loc("layernorm_149.dc.reduce_sum.0"(#loc126))
#loc268 = loc("layernorm_149.dc.multiply.2"(#loc126))
#loc269 = loc("layernorm_149.dc.subtract.3"(#loc126))
#loc270 = loc("layernorm_149.dc.multiply.4"(#loc126))
#loc271 = loc("layernorm_149.dc.reduce_sum.5"(#loc126))
#loc272 = loc("layernorm_149.dc.multiply.7"(#loc126))
#loc273 = loc("layernorm_149.dc.add.9"(#loc126))
#loc274 = loc("layernorm_149.dc.sqrt.10"(#loc126))
#loc275 = loc("layernorm_149.dc.reciprocal.11"(#loc126))
#loc276 = loc("layernorm_149.dc.multiply.12"(#loc126))
#loc277 = loc("layernorm_149.dc.multiply.13"(#loc126))
#loc278 = loc("layernorm_149.dc.add.14"(#loc126))
#loc279 = loc("layernorm_150.dc.reduce_sum.0"(#loc98))
#loc280 = loc("layernorm_150.dc.multiply.2"(#loc98))
#loc281 = loc("layernorm_150.dc.subtract.3"(#loc98))
#loc282 = loc("layernorm_150.dc.multiply.4"(#loc98))
#loc283 = loc("layernorm_150.dc.reduce_sum.5"(#loc98))
#loc284 = loc("layernorm_150.dc.multiply.7"(#loc98))
#loc285 = loc("layernorm_150.dc.add.9"(#loc98))
#loc286 = loc("layernorm_150.dc.sqrt.10"(#loc98))
#loc287 = loc("layernorm_150.dc.reciprocal.11"(#loc98))
#loc288 = loc("layernorm_150.dc.multiply.12"(#loc98))
#loc289 = loc("layernorm_150.dc.multiply.13"(#loc98))
#loc290 = loc("layernorm_150.dc.add.14"(#loc98))
#loc291 = loc("layernorm_200.dc.reduce_sum.0"(#loc101))
#loc292 = loc("layernorm_200.dc.multiply.2"(#loc101))
#loc293 = loc("layernorm_200.dc.subtract.3"(#loc101))
#loc294 = loc("layernorm_200.dc.multiply.4"(#loc101))
#loc295 = loc("layernorm_200.dc.reduce_sum.5"(#loc101))
#loc296 = loc("layernorm_200.dc.multiply.7"(#loc101))
#loc297 = loc("layernorm_200.dc.add.9"(#loc101))
#loc298 = loc("layernorm_200.dc.sqrt.10"(#loc101))
#loc299 = loc("layernorm_200.dc.reciprocal.11"(#loc101))
#loc300 = loc("layernorm_200.dc.multiply.12"(#loc101))
#loc301 = loc("layernorm_200.dc.multiply.13"(#loc101))
#loc302 = loc("layernorm_200.dc.add.14"(#loc101))
#loc303 = loc("layernorm_221.dc.reduce_sum.0"(#loc104))
#loc304 = loc("layernorm_221.dc.multiply.2"(#loc104))
#loc305 = loc("layernorm_221.dc.subtract.3"(#loc104))
#loc306 = loc("layernorm_221.dc.multiply.4"(#loc104))
#loc307 = loc("layernorm_221.dc.reduce_sum.5"(#loc104))
#loc308 = loc("layernorm_221.dc.multiply.7"(#loc104))
#loc309 = loc("layernorm_221.dc.add.9"(#loc104))
#loc310 = loc("layernorm_221.dc.sqrt.10"(#loc104))
#loc311 = loc("layernorm_221.dc.reciprocal.11"(#loc104))
#loc312 = loc("layernorm_221.dc.multiply.12"(#loc104))
#loc313 = loc("layernorm_221.dc.multiply.13"(#loc104))
#loc314 = loc("layernorm_221.dc.add.14"(#loc104))
#loc315 = loc("layernorm_271.dc.reduce_sum.0"(#loc107))
#loc316 = loc("layernorm_271.dc.multiply.2"(#loc107))
#loc317 = loc("layernorm_271.dc.subtract.3"(#loc107))
#loc318 = loc("layernorm_271.dc.multiply.4"(#loc107))
#loc319 = loc("layernorm_271.dc.reduce_sum.5"(#loc107))
#loc320 = loc("layernorm_271.dc.multiply.7"(#loc107))
#loc321 = loc("layernorm_271.dc.add.9"(#loc107))
#loc322 = loc("layernorm_271.dc.sqrt.10"(#loc107))
#loc323 = loc("layernorm_271.dc.reciprocal.11"(#loc107))
#loc324 = loc("layernorm_271.dc.multiply.12"(#loc107))
#loc325 = loc("layernorm_271.dc.multiply.13"(#loc107))
#loc326 = loc("layernorm_271.dc.add.14"(#loc107))
#loc327 = loc("conv2d_296.dc.transpose.0"(#loc143))
#loc328 = loc("conv2d_296.dc.transpose.1"(#loc143))
#loc329 = loc("conv2d_296.dc.conv2d.4"(#loc143))
#loc330 = loc("conv2d_296.dc.transpose.5"(#loc143))
#loc331 = loc("conv2d_296.dc.transpose.6"(#loc143))
#loc332 = loc("layernorm_303.dc.reduce_sum.0"(#loc147))
#loc333 = loc("layernorm_303.dc.multiply.2"(#loc147))
#loc334 = loc("layernorm_303.dc.subtract.3"(#loc147))
#loc335 = loc("layernorm_303.dc.multiply.4"(#loc147))
#loc336 = loc("layernorm_303.dc.reduce_sum.5"(#loc147))
#loc337 = loc("layernorm_303.dc.multiply.7"(#loc147))
#loc338 = loc("layernorm_303.dc.add.9"(#loc147))
#loc339 = loc("layernorm_303.dc.sqrt.10"(#loc147))
#loc340 = loc("layernorm_303.dc.reciprocal.11"(#loc147))
#loc341 = loc("layernorm_303.dc.multiply.12"(#loc147))
#loc342 = loc("layernorm_303.dc.multiply.13"(#loc147))
#loc343 = loc("layernorm_303.dc.add.14"(#loc147))
#loc344 = loc("layernorm_304.dc.reduce_sum.0"(#loc98))
#loc345 = loc("layernorm_304.dc.multiply.2"(#loc98))
#loc346 = loc("layernorm_304.dc.subtract.3"(#loc98))
#loc347 = loc("layernorm_304.dc.multiply.4"(#loc98))
#loc348 = loc("layernorm_304.dc.reduce_sum.5"(#loc98))
#loc349 = loc("layernorm_304.dc.multiply.7"(#loc98))
#loc350 = loc("layernorm_304.dc.add.9"(#loc98))
#loc351 = loc("layernorm_304.dc.sqrt.10"(#loc98))
#loc352 = loc("layernorm_304.dc.reciprocal.11"(#loc98))
#loc353 = loc("layernorm_304.dc.multiply.12"(#loc98))
#loc354 = loc("layernorm_304.dc.multiply.13"(#loc98))
#loc355 = loc("layernorm_304.dc.add.14"(#loc98))
#loc356 = loc("layernorm_354.dc.reduce_sum.0"(#loc101))
#loc357 = loc("layernorm_354.dc.multiply.2"(#loc101))
#loc358 = loc("layernorm_354.dc.subtract.3"(#loc101))
#loc359 = loc("layernorm_354.dc.multiply.4"(#loc101))
#loc360 = loc("layernorm_354.dc.reduce_sum.5"(#loc101))
#loc361 = loc("layernorm_354.dc.multiply.7"(#loc101))
#loc362 = loc("layernorm_354.dc.add.9"(#loc101))
#loc363 = loc("layernorm_354.dc.sqrt.10"(#loc101))
#loc364 = loc("layernorm_354.dc.reciprocal.11"(#loc101))
#loc365 = loc("layernorm_354.dc.multiply.12"(#loc101))
#loc366 = loc("layernorm_354.dc.multiply.13"(#loc101))
#loc367 = loc("layernorm_354.dc.add.14"(#loc101))
#loc368 = loc("layernorm_375.dc.reduce_sum.0"(#loc104))
#loc369 = loc("layernorm_375.dc.multiply.2"(#loc104))
#loc370 = loc("layernorm_375.dc.subtract.3"(#loc104))
#loc371 = loc("layernorm_375.dc.multiply.4"(#loc104))
#loc372 = loc("layernorm_375.dc.reduce_sum.5"(#loc104))
#loc373 = loc("layernorm_375.dc.multiply.7"(#loc104))
#loc374 = loc("layernorm_375.dc.add.9"(#loc104))
#loc375 = loc("layernorm_375.dc.sqrt.10"(#loc104))
#loc376 = loc("layernorm_375.dc.reciprocal.11"(#loc104))
#loc377 = loc("layernorm_375.dc.multiply.12"(#loc104))
#loc378 = loc("layernorm_375.dc.multiply.13"(#loc104))
#loc379 = loc("layernorm_375.dc.add.14"(#loc104))
#loc380 = loc("layernorm_425.dc.reduce_sum.0"(#loc107))
#loc381 = loc("layernorm_425.dc.multiply.2"(#loc107))
#loc382 = loc("layernorm_425.dc.subtract.3"(#loc107))
#loc383 = loc("layernorm_425.dc.multiply.4"(#loc107))
#loc384 = loc("layernorm_425.dc.reduce_sum.5"(#loc107))
#loc385 = loc("layernorm_425.dc.multiply.7"(#loc107))
#loc386 = loc("layernorm_425.dc.add.9"(#loc107))
#loc387 = loc("layernorm_425.dc.sqrt.10"(#loc107))
#loc388 = loc("layernorm_425.dc.reciprocal.11"(#loc107))
#loc389 = loc("layernorm_425.dc.multiply.12"(#loc107))
#loc390 = loc("layernorm_425.dc.multiply.13"(#loc107))
#loc391 = loc("layernorm_425.dc.add.14"(#loc107))
#loc392 = loc("conv2d_450.dc.transpose.0"(#loc164))
#loc393 = loc("conv2d_450.dc.transpose.1"(#loc164))
#loc394 = loc("conv2d_450.dc.conv2d.4"(#loc164))
#loc395 = loc("conv2d_450.dc.transpose.5"(#loc164))
#loc396 = loc("conv2d_450.dc.transpose.6"(#loc164))
#loc397 = loc("layernorm_457.dc.reduce_sum.0"(#loc168))
#loc398 = loc("layernorm_457.dc.multiply.2"(#loc168))
#loc399 = loc("layernorm_457.dc.subtract.3"(#loc168))
#loc400 = loc("layernorm_457.dc.multiply.4"(#loc168))
#loc401 = loc("layernorm_457.dc.reduce_sum.5"(#loc168))
#loc402 = loc("layernorm_457.dc.multiply.7"(#loc168))
#loc403 = loc("layernorm_457.dc.add.9"(#loc168))
#loc404 = loc("layernorm_457.dc.sqrt.10"(#loc168))
#loc405 = loc("layernorm_457.dc.reciprocal.11"(#loc168))
#loc406 = loc("layernorm_457.dc.multiply.12"(#loc168))
#loc407 = loc("layernorm_457.dc.multiply.13"(#loc168))
#loc408 = loc("layernorm_457.dc.add.14"(#loc168))
#loc409 = loc("layernorm_458.dc.reduce_sum.0"(#loc98))
#loc410 = loc("layernorm_458.dc.multiply.2"(#loc98))
#loc411 = loc("layernorm_458.dc.subtract.3"(#loc98))
#loc412 = loc("layernorm_458.dc.multiply.4"(#loc98))
#loc413 = loc("layernorm_458.dc.reduce_sum.5"(#loc98))
#loc414 = loc("layernorm_458.dc.multiply.7"(#loc98))
#loc415 = loc("layernorm_458.dc.add.9"(#loc98))
#loc416 = loc("layernorm_458.dc.sqrt.10"(#loc98))
#loc417 = loc("layernorm_458.dc.reciprocal.11"(#loc98))
#loc418 = loc("layernorm_458.dc.multiply.12"(#loc98))
#loc419 = loc("layernorm_458.dc.multiply.13"(#loc98))
#loc420 = loc("layernorm_458.dc.add.14"(#loc98))
#loc421 = loc("layernorm_500.dc.reduce_sum.0"(#loc101))
#loc422 = loc("layernorm_500.dc.multiply.2"(#loc101))
#loc423 = loc("layernorm_500.dc.subtract.3"(#loc101))
#loc424 = loc("layernorm_500.dc.multiply.4"(#loc101))
#loc425 = loc("layernorm_500.dc.reduce_sum.5"(#loc101))
#loc426 = loc("layernorm_500.dc.multiply.7"(#loc101))
#loc427 = loc("layernorm_500.dc.add.9"(#loc101))
#loc428 = loc("layernorm_500.dc.sqrt.10"(#loc101))
#loc429 = loc("layernorm_500.dc.reciprocal.11"(#loc101))
#loc430 = loc("layernorm_500.dc.multiply.12"(#loc101))
#loc431 = loc("layernorm_500.dc.multiply.13"(#loc101))
#loc432 = loc("layernorm_500.dc.add.14"(#loc101))
#loc433 = loc("layernorm_521.dc.reduce_sum.0"(#loc104))
#loc434 = loc("layernorm_521.dc.multiply.2"(#loc104))
#loc435 = loc("layernorm_521.dc.subtract.3"(#loc104))
#loc436 = loc("layernorm_521.dc.multiply.4"(#loc104))
#loc437 = loc("layernorm_521.dc.reduce_sum.5"(#loc104))
#loc438 = loc("layernorm_521.dc.multiply.7"(#loc104))
#loc439 = loc("layernorm_521.dc.add.9"(#loc104))
#loc440 = loc("layernorm_521.dc.sqrt.10"(#loc104))
#loc441 = loc("layernorm_521.dc.reciprocal.11"(#loc104))
#loc442 = loc("layernorm_521.dc.multiply.12"(#loc104))
#loc443 = loc("layernorm_521.dc.multiply.13"(#loc104))
#loc444 = loc("layernorm_521.dc.add.14"(#loc104))
#loc445 = loc("layernorm_563.dc.reduce_sum.0"(#loc107))
#loc446 = loc("layernorm_563.dc.multiply.2"(#loc107))
#loc447 = loc("layernorm_563.dc.subtract.3"(#loc107))
#loc448 = loc("layernorm_563.dc.multiply.4"(#loc107))
#loc449 = loc("layernorm_563.dc.reduce_sum.5"(#loc107))
#loc450 = loc("layernorm_563.dc.multiply.7"(#loc107))
#loc451 = loc("layernorm_563.dc.add.9"(#loc107))
#loc452 = loc("layernorm_563.dc.sqrt.10"(#loc107))
#loc453 = loc("layernorm_563.dc.reciprocal.11"(#loc107))
#loc454 = loc("layernorm_563.dc.multiply.12"(#loc107))
#loc455 = loc("layernorm_563.dc.multiply.13"(#loc107))
#loc456 = loc("layernorm_563.dc.add.14"(#loc107))
#loc457 = loc("torch.nn.modules.linear.Linear::query"(#loc214))
#loc458 = loc("reshape_12"(#loc214))
#loc459 = loc("squeeze_13"(#loc214))
#loc460 = loc("transpose_14"(#loc214))
#loc461 = loc("reshape_15"(#loc214))
#loc462 = loc("torch.nn.modules.conv.Conv2d::sr"(#loc214))
#loc463 = loc("reshape_20"(#loc214))
#loc464 = loc("transpose_21"(#loc214))
#loc465 = loc("torch.nn.modules.normalization.LayerNorm::layer_norm"(#loc214))
#loc466 = loc("torch.nn.modules.linear.Linear::key"(#loc214))
#loc467 = loc("reshape_28"(#loc214))
#loc468 = loc("squeeze_29"(#loc214))
#loc469 = loc("matmul_31"(#loc214))
#loc470 = loc("reshape_32.dc.unsqueeze.0"(#loc214))
#loc471 = loc("divide_33"(#loc214))
#loc472 = loc("softmax_34"(#loc214))
#loc473 = loc("reshape_36.dc.squeeze.0"(#loc214))
#loc474 = loc("torch.nn.modules.linear.Linear::value"(#loc214))
#loc475 = loc("reshape_41"(#loc214))
#loc476 = loc("transpose_42"(#loc214))
#loc477 = loc("transpose_43"(#loc214))
#loc478 = loc("reshape_44.dc.squeeze.0"(#loc214))
#loc479 = loc("matmul_46"(#loc214))
#loc480 = loc("torch.nn.modules.linear.Linear::dense"(#loc215))
#loc481 = loc("matmul_54"(#loc228))
#loc482 = loc("transpose_56"(#loc229))
#loc483 = loc("reshape_57"(#loc229))
#loc484 = loc("torch.nn.modules.conv.Conv2d::dwconv"(#loc229))
#loc485 = loc("reshape_63"(#loc229))
#loc486 = loc("squeeze_64"(#loc229))
#loc487 = loc("transpose_65"(#loc229))
#loc488 = loc("gelu_66"(#loc230))
#loc489 = loc("matmul_69"(#loc231))
#loc490 = loc("torch.nn.modules.linear.Linear::query"(#loc244))
#loc491 = loc("reshape_77"(#loc244))
#loc492 = loc("squeeze_78"(#loc244))
#loc493 = loc("transpose_79"(#loc244))
#loc494 = loc("reshape_80"(#loc244))
#loc495 = loc("torch.nn.modules.conv.Conv2d::sr"(#loc244))
#loc496 = loc("reshape_85"(#loc244))
#loc497 = loc("transpose_86"(#loc244))
#loc498 = loc("torch.nn.modules.normalization.LayerNorm::layer_norm"(#loc244))
#loc499 = loc("torch.nn.modules.linear.Linear::key"(#loc244))
#loc500 = loc("reshape_93"(#loc244))
#loc501 = loc("squeeze_94"(#loc244))
#loc502 = loc("matmul_96"(#loc244))
#loc503 = loc("reshape_97.dc.unsqueeze.0"(#loc244))
#loc504 = loc("divide_98"(#loc244))
#loc505 = loc("softmax_99"(#loc244))
#loc506 = loc("reshape_101.dc.squeeze.0"(#loc244))
#loc507 = loc("torch.nn.modules.linear.Linear::value"(#loc244))
#loc508 = loc("reshape_106"(#loc244))
#loc509 = loc("transpose_107"(#loc244))
#loc510 = loc("transpose_108"(#loc244))
#loc511 = loc("reshape_109.dc.squeeze.0"(#loc244))
#loc512 = loc("matmul_111"(#loc244))
#loc513 = loc("torch.nn.modules.linear.Linear::dense"(#loc245))
#loc514 = loc("matmul_119"(#loc258))
#loc515 = loc("transpose_121"(#loc259))
#loc516 = loc("reshape_122"(#loc259))
#loc517 = loc("torch.nn.modules.conv.Conv2d::dwconv"(#loc259))
#loc518 = loc("reshape_128"(#loc259))
#loc519 = loc("squeeze_129"(#loc259))
#loc520 = loc("transpose_130"(#loc259))
#loc521 = loc("gelu_131"(#loc260))
#loc522 = loc("matmul_134"(#loc261))
#loc523 = loc("reshape_154"(#loc214))
#loc524 = loc("transpose_155"(#loc214))
#loc525 = loc("reshape_156.dc.squeeze.0"(#loc214))
#loc526 = loc("transpose_157"(#loc214))
#loc527 = loc("reshape_158"(#loc214))
#loc528 = loc("reshape_163"(#loc214))
#loc529 = loc("transpose_164"(#loc214))
#loc530 = loc("reshape_171"(#loc214))
#loc531 = loc("transpose_172"(#loc214))
#loc532 = loc("reshape_173.dc.squeeze.0"(#loc214))
#loc533 = loc("matmul_175"(#loc214))
#loc534 = loc("reshape_176.dc.unsqueeze.0"(#loc214))
#loc535 = loc("divide_177"(#loc214))
#loc536 = loc("softmax_178"(#loc214))
#loc537 = loc("reshape_180.dc.squeeze.0"(#loc214))
#loc538 = loc("reshape_185"(#loc214))
#loc539 = loc("transpose_186"(#loc214))
#loc540 = loc("transpose_187"(#loc214))
#loc541 = loc("reshape_188.dc.squeeze.0"(#loc214))
#loc542 = loc("matmul_190"(#loc214))
#loc543 = loc("reshape_191.dc.unsqueeze.0"(#loc214))
#loc544 = loc("transpose_192"(#loc214))
#loc545 = loc("matmul_202"(#loc228))
#loc546 = loc("transpose_204"(#loc229))
#loc547 = loc("reshape_205"(#loc229))
#loc548 = loc("reshape_211"(#loc229))
#loc549 = loc("squeeze_212"(#loc229))
#loc550 = loc("transpose_213"(#loc229))
#loc551 = loc("gelu_214"(#loc230))
#loc552 = loc("matmul_217"(#loc231))
#loc553 = loc("reshape_225"(#loc244))
#loc554 = loc("transpose_226"(#loc244))
#loc555 = loc("reshape_227.dc.squeeze.0"(#loc244))
#loc556 = loc("transpose_228"(#loc244))
#loc557 = loc("reshape_229"(#loc244))
#loc558 = loc("reshape_234"(#loc244))
#loc559 = loc("transpose_235"(#loc244))
#loc560 = loc("reshape_242"(#loc244))
#loc561 = loc("transpose_243"(#loc244))
#loc562 = loc("reshape_244.dc.squeeze.0"(#loc244))
#loc563 = loc("matmul_246"(#loc244))
#loc564 = loc("reshape_247.dc.unsqueeze.0"(#loc244))
#loc565 = loc("divide_248"(#loc244))
#loc566 = loc("softmax_249"(#loc244))
#loc567 = loc("reshape_251.dc.squeeze.0"(#loc244))
#loc568 = loc("reshape_256"(#loc244))
#loc569 = loc("transpose_257"(#loc244))
#loc570 = loc("transpose_258"(#loc244))
#loc571 = loc("reshape_259.dc.squeeze.0"(#loc244))
#loc572 = loc("matmul_261"(#loc244))
#loc573 = loc("reshape_262.dc.unsqueeze.0"(#loc244))
#loc574 = loc("transpose_263"(#loc244))
#loc575 = loc("matmul_273"(#loc258))
#loc576 = loc("transpose_275"(#loc259))
#loc577 = loc("reshape_276"(#loc259))
#loc578 = loc("reshape_282"(#loc259))
#loc579 = loc("squeeze_283"(#loc259))
#loc580 = loc("transpose_284"(#loc259))
#loc581 = loc("gelu_285"(#loc260))
#loc582 = loc("matmul_288"(#loc261))
#loc583 = loc("reshape_308"(#loc214))
#loc584 = loc("transpose_309"(#loc214))
#loc585 = loc("reshape_310.dc.squeeze.0"(#loc214))
#loc586 = loc("transpose_311"(#loc214))
#loc587 = loc("reshape_312"(#loc214))
#loc588 = loc("reshape_317"(#loc214))
#loc589 = loc("transpose_318"(#loc214))
#loc590 = loc("reshape_325"(#loc214))
#loc591 = loc("transpose_326"(#loc214))
#loc592 = loc("reshape_327.dc.squeeze.0"(#loc214))
#loc593 = loc("matmul_329"(#loc214))
#loc594 = loc("reshape_330.dc.unsqueeze.0"(#loc214))
#loc595 = loc("divide_331"(#loc214))
#loc596 = loc("softmax_332"(#loc214))
#loc597 = loc("reshape_334.dc.squeeze.0"(#loc214))
#loc598 = loc("reshape_339"(#loc214))
#loc599 = loc("transpose_340"(#loc214))
#loc600 = loc("transpose_341"(#loc214))
#loc601 = loc("reshape_342.dc.squeeze.0"(#loc214))
#loc602 = loc("matmul_344"(#loc214))
#loc603 = loc("reshape_345.dc.unsqueeze.0"(#loc214))
#loc604 = loc("transpose_346"(#loc214))
#loc605 = loc("matmul_356"(#loc228))
#loc606 = loc("transpose_358"(#loc229))
#loc607 = loc("reshape_359"(#loc229))
#loc608 = loc("reshape_365"(#loc229))
#loc609 = loc("squeeze_366"(#loc229))
#loc610 = loc("transpose_367"(#loc229))
#loc611 = loc("gelu_368"(#loc230))
#loc612 = loc("matmul_371"(#loc231))
#loc613 = loc("reshape_379"(#loc244))
#loc614 = loc("transpose_380"(#loc244))
#loc615 = loc("reshape_381.dc.squeeze.0"(#loc244))
#loc616 = loc("transpose_382"(#loc244))
#loc617 = loc("reshape_383"(#loc244))
#loc618 = loc("reshape_388"(#loc244))
#loc619 = loc("transpose_389"(#loc244))
#loc620 = loc("reshape_396"(#loc244))
#loc621 = loc("transpose_397"(#loc244))
#loc622 = loc("reshape_398.dc.squeeze.0"(#loc244))
#loc623 = loc("matmul_400"(#loc244))
#loc624 = loc("reshape_401.dc.unsqueeze.0"(#loc244))
#loc625 = loc("divide_402"(#loc244))
#loc626 = loc("softmax_403"(#loc244))
#loc627 = loc("reshape_405.dc.squeeze.0"(#loc244))
#loc628 = loc("reshape_410"(#loc244))
#loc629 = loc("transpose_411"(#loc244))
#loc630 = loc("transpose_412"(#loc244))
#loc631 = loc("reshape_413.dc.squeeze.0"(#loc244))
#loc632 = loc("matmul_415"(#loc244))
#loc633 = loc("reshape_416.dc.unsqueeze.0"(#loc244))
#loc634 = loc("transpose_417"(#loc244))
#loc635 = loc("matmul_427"(#loc258))
#loc636 = loc("transpose_429"(#loc259))
#loc637 = loc("reshape_430"(#loc259))
#loc638 = loc("reshape_436"(#loc259))
#loc639 = loc("squeeze_437"(#loc259))
#loc640 = loc("transpose_438"(#loc259))
#loc641 = loc("gelu_439"(#loc260))
#loc642 = loc("matmul_442"(#loc261))
#loc643 = loc("reshape_464"(#loc214))
#loc644 = loc("transpose_465"(#loc214))
#loc645 = loc("reshape_466.dc.squeeze.0"(#loc214))
#loc646 = loc("reshape_471"(#loc214))
#loc647 = loc("transpose_472"(#loc214))
#loc648 = loc("reshape_473.dc.squeeze.0"(#loc214))
#loc649 = loc("matmul_475"(#loc214))
#loc650 = loc("reshape_476.dc.unsqueeze.0"(#loc214))
#loc651 = loc("divide_477"(#loc214))
#loc652 = loc("softmax_478"(#loc214))
#loc653 = loc("reshape_480.dc.squeeze.0"(#loc214))
#loc654 = loc("reshape_485"(#loc214))
#loc655 = loc("transpose_486"(#loc214))
#loc656 = loc("transpose_487"(#loc214))
#loc657 = loc("reshape_488.dc.squeeze.0"(#loc214))
#loc658 = loc("matmul_490"(#loc214))
#loc659 = loc("reshape_491.dc.unsqueeze.0"(#loc214))
#loc660 = loc("transpose_492"(#loc214))
#loc661 = loc("matmul_502"(#loc228))
#loc662 = loc("transpose_504"(#loc229))
#loc663 = loc("reshape_505"(#loc229))
#loc664 = loc("reshape_511"(#loc229))
#loc665 = loc("squeeze_512"(#loc229))
#loc666 = loc("transpose_513"(#loc229))
#loc667 = loc("gelu_514"(#loc230))
#loc668 = loc("matmul_517"(#loc231))
#loc669 = loc("reshape_527"(#loc244))
#loc670 = loc("transpose_528"(#loc244))
#loc671 = loc("reshape_529.dc.squeeze.0"(#loc244))
#loc672 = loc("reshape_534"(#loc244))
#loc673 = loc("transpose_535"(#loc244))
#loc674 = loc("reshape_536.dc.squeeze.0"(#loc244))
#loc675 = loc("matmul_538"(#loc244))
#loc676 = loc("reshape_539.dc.unsqueeze.0"(#loc244))
#loc677 = loc("divide_540"(#loc244))
#loc678 = loc("softmax_541"(#loc244))
#loc679 = loc("reshape_543.dc.squeeze.0"(#loc244))
#loc680 = loc("reshape_548"(#loc244))
#loc681 = loc("transpose_549"(#loc244))
#loc682 = loc("transpose_550"(#loc244))
#loc683 = loc("reshape_551.dc.squeeze.0"(#loc244))
#loc684 = loc("matmul_553"(#loc244))
#loc685 = loc("reshape_554.dc.unsqueeze.0"(#loc244))
#loc686 = loc("transpose_555"(#loc244))
#loc687 = loc("matmul_565"(#loc258))
#loc688 = loc("transpose_567"(#loc259))
#loc689 = loc("reshape_568"(#loc259))
#loc690 = loc("reshape_574"(#loc259))
#loc691 = loc("squeeze_575"(#loc259))
#loc692 = loc("transpose_576"(#loc259))
#loc693 = loc("gelu_577"(#loc260))
#loc694 = loc("matmul_580"(#loc261))
#loc695 = loc("matmul_10"(#loc457))
#loc696 = loc("conv2d_16.dc.transpose.0"(#loc462))
#loc697 = loc("conv2d_16.dc.transpose.1"(#loc462))
#loc698 = loc("conv2d_16.dc.conv2d.4"(#loc462))
#loc699 = loc("conv2d_16.dc.transpose.5"(#loc462))
#loc700 = loc("conv2d_16.dc.transpose.6"(#loc462))
#loc701 = loc("layernorm_22.dc.reduce_sum.0"(#loc465))
#loc702 = loc("layernorm_22.dc.multiply.2"(#loc465))
#loc703 = loc("layernorm_22.dc.subtract.3"(#loc465))
#loc704 = loc("layernorm_22.dc.multiply.4"(#loc465))
#loc705 = loc("layernorm_22.dc.reduce_sum.5"(#loc465))
#loc706 = loc("layernorm_22.dc.multiply.7"(#loc465))
#loc707 = loc("layernorm_22.dc.add.9"(#loc465))
#loc708 = loc("layernorm_22.dc.sqrt.10"(#loc465))
#loc709 = loc("layernorm_22.dc.reciprocal.11"(#loc465))
#loc710 = loc("layernorm_22.dc.multiply.12"(#loc465))
#loc711 = loc("layernorm_22.dc.multiply.13"(#loc465))
#loc712 = loc("layernorm_22.dc.add.14"(#loc465))
#loc713 = loc("reshape_23.dc.squeeze.0"(#loc466))
#loc714 = loc("matmul_25"(#loc466))
#loc715 = loc("reshape_26.dc.unsqueeze.0"(#loc466))
#loc716 = loc("matmul_38"(#loc474))
#loc717 = loc("reshape_39.dc.unsqueeze.0"(#loc474))
#loc718 = loc("matmul_48"(#loc480))
#loc719 = loc("conv2d_59.dc.transpose.0"(#loc484))
#loc720 = loc("conv2d_59.dc.transpose.1"(#loc484))
#loc721 = loc("conv2d_59.dc.conv2d.4"(#loc484))
#loc722 = loc("conv2d_59.dc.transpose.5"(#loc484))
#loc723 = loc("conv2d_59.dc.transpose.6"(#loc484))
#loc724 = loc("matmul_75"(#loc490))
#loc725 = loc("conv2d_81.dc.transpose.0"(#loc495))
#loc726 = loc("conv2d_81.dc.transpose.1"(#loc495))
#loc727 = loc("conv2d_81.dc.conv2d.4"(#loc495))
#loc728 = loc("conv2d_81.dc.transpose.5"(#loc495))
#loc729 = loc("conv2d_81.dc.transpose.6"(#loc495))
#loc730 = loc("layernorm_87.dc.reduce_sum.0"(#loc498))
#loc731 = loc("layernorm_87.dc.multiply.2"(#loc498))
#loc732 = loc("layernorm_87.dc.subtract.3"(#loc498))
#loc733 = loc("layernorm_87.dc.multiply.4"(#loc498))
#loc734 = loc("layernorm_87.dc.reduce_sum.5"(#loc498))
#loc735 = loc("layernorm_87.dc.multiply.7"(#loc498))
#loc736 = loc("layernorm_87.dc.add.9"(#loc498))
#loc737 = loc("layernorm_87.dc.sqrt.10"(#loc498))
#loc738 = loc("layernorm_87.dc.reciprocal.11"(#loc498))
#loc739 = loc("layernorm_87.dc.multiply.12"(#loc498))
#loc740 = loc("layernorm_87.dc.multiply.13"(#loc498))
#loc741 = loc("layernorm_87.dc.add.14"(#loc498))
#loc742 = loc("reshape_88.dc.squeeze.0"(#loc499))
#loc743 = loc("matmul_90"(#loc499))
#loc744 = loc("reshape_91.dc.unsqueeze.0"(#loc499))
#loc745 = loc("matmul_103"(#loc507))
#loc746 = loc("reshape_104.dc.unsqueeze.0"(#loc507))
#loc747 = loc("matmul_113"(#loc513))
#loc748 = loc("conv2d_124.dc.transpose.0"(#loc517))
#loc749 = loc("conv2d_124.dc.transpose.1"(#loc517))
#loc750 = loc("conv2d_124.dc.conv2d.4"(#loc517))
#loc751 = loc("conv2d_124.dc.transpose.5"(#loc517))
#loc752 = loc("conv2d_124.dc.transpose.6"(#loc517))
#loc753 = loc("matmul_152"(#loc457))
#loc754 = loc("conv2d_159.dc.transpose.0"(#loc462))
#loc755 = loc("conv2d_159.dc.transpose.1"(#loc462))
#loc756 = loc("conv2d_159.dc.conv2d.4"(#loc462))
#loc757 = loc("conv2d_159.dc.transpose.5"(#loc462))
#loc758 = loc("conv2d_159.dc.transpose.6"(#loc462))
#loc759 = loc("layernorm_165.dc.reduce_sum.0"(#loc465))
#loc760 = loc("layernorm_165.dc.multiply.2"(#loc465))
#loc761 = loc("layernorm_165.dc.subtract.3"(#loc465))
#loc762 = loc("layernorm_165.dc.multiply.4"(#loc465))
#loc763 = loc("layernorm_165.dc.reduce_sum.5"(#loc465))
#loc764 = loc("layernorm_165.dc.multiply.7"(#loc465))
#loc765 = loc("layernorm_165.dc.add.9"(#loc465))
#loc766 = loc("layernorm_165.dc.sqrt.10"(#loc465))
#loc767 = loc("layernorm_165.dc.reciprocal.11"(#loc465))
#loc768 = loc("layernorm_165.dc.multiply.12"(#loc465))
#loc769 = loc("layernorm_165.dc.multiply.13"(#loc465))
#loc770 = loc("layernorm_165.dc.add.14"(#loc465))
#loc771 = loc("reshape_166.dc.squeeze.0"(#loc466))
#loc772 = loc("matmul_168"(#loc466))
#loc773 = loc("reshape_169.dc.unsqueeze.0"(#loc466))
#loc774 = loc("matmul_182"(#loc474))
#loc775 = loc("reshape_183.dc.unsqueeze.0"(#loc474))
#loc776 = loc("reshape_193"(#loc480))
#loc777 = loc("matmul_195"(#loc480))
#loc778 = loc("reshape_196.dc.unsqueeze.0"(#loc480))
#loc779 = loc("conv2d_207.dc.transpose.0"(#loc484))
#loc780 = loc("conv2d_207.dc.transpose.1"(#loc484))
#loc781 = loc("conv2d_207.dc.conv2d.4"(#loc484))
#loc782 = loc("conv2d_207.dc.transpose.5"(#loc484))
#loc783 = loc("conv2d_207.dc.transpose.6"(#loc484))
#loc784 = loc("matmul_223"(#loc490))
#loc785 = loc("conv2d_230.dc.transpose.0"(#loc495))
#loc786 = loc("conv2d_230.dc.transpose.1"(#loc495))
#loc787 = loc("conv2d_230.dc.conv2d.4"(#loc495))
#loc788 = loc("conv2d_230.dc.transpose.5"(#loc495))
#loc789 = loc("conv2d_230.dc.transpose.6"(#loc495))
#loc790 = loc("layernorm_236.dc.reduce_sum.0"(#loc498))
#loc791 = loc("layernorm_236.dc.multiply.2"(#loc498))
#loc792 = loc("layernorm_236.dc.subtract.3"(#loc498))
#loc793 = loc("layernorm_236.dc.multiply.4"(#loc498))
#loc794 = loc("layernorm_236.dc.reduce_sum.5"(#loc498))
#loc795 = loc("layernorm_236.dc.multiply.7"(#loc498))
#loc796 = loc("layernorm_236.dc.add.9"(#loc498))
#loc797 = loc("layernorm_236.dc.sqrt.10"(#loc498))
#loc798 = loc("layernorm_236.dc.reciprocal.11"(#loc498))
#loc799 = loc("layernorm_236.dc.multiply.12"(#loc498))
#loc800 = loc("layernorm_236.dc.multiply.13"(#loc498))
#loc801 = loc("layernorm_236.dc.add.14"(#loc498))
#loc802 = loc("reshape_237.dc.squeeze.0"(#loc499))
#loc803 = loc("matmul_239"(#loc499))
#loc804 = loc("reshape_240.dc.unsqueeze.0"(#loc499))
#loc805 = loc("matmul_253"(#loc507))
#loc806 = loc("reshape_254.dc.unsqueeze.0"(#loc507))
#loc807 = loc("reshape_264"(#loc513))
#loc808 = loc("matmul_266"(#loc513))
#loc809 = loc("reshape_267.dc.unsqueeze.0"(#loc513))
#loc810 = loc("conv2d_278.dc.transpose.0"(#loc517))
#loc811 = loc("conv2d_278.dc.transpose.1"(#loc517))
#loc812 = loc("conv2d_278.dc.conv2d.4"(#loc517))
#loc813 = loc("conv2d_278.dc.transpose.5"(#loc517))
#loc814 = loc("conv2d_278.dc.transpose.6"(#loc517))
#loc815 = loc("matmul_306"(#loc457))
#loc816 = loc("conv2d_313.dc.transpose.0"(#loc462))
#loc817 = loc("conv2d_313.dc.transpose.1"(#loc462))
#loc818 = loc("conv2d_313.dc.conv2d.4"(#loc462))
#loc819 = loc("conv2d_313.dc.transpose.5"(#loc462))
#loc820 = loc("conv2d_313.dc.transpose.6"(#loc462))
#loc821 = loc("layernorm_319.dc.reduce_sum.0"(#loc465))
#loc822 = loc("layernorm_319.dc.multiply.2"(#loc465))
#loc823 = loc("layernorm_319.dc.subtract.3"(#loc465))
#loc824 = loc("layernorm_319.dc.multiply.4"(#loc465))
#loc825 = loc("layernorm_319.dc.reduce_sum.5"(#loc465))
#loc826 = loc("layernorm_319.dc.multiply.7"(#loc465))
#loc827 = loc("layernorm_319.dc.add.9"(#loc465))
#loc828 = loc("layernorm_319.dc.sqrt.10"(#loc465))
#loc829 = loc("layernorm_319.dc.reciprocal.11"(#loc465))
#loc830 = loc("layernorm_319.dc.multiply.12"(#loc465))
#loc831 = loc("layernorm_319.dc.multiply.13"(#loc465))
#loc832 = loc("layernorm_319.dc.add.14"(#loc465))
#loc833 = loc("reshape_320.dc.squeeze.0"(#loc466))
#loc834 = loc("matmul_322"(#loc466))
#loc835 = loc("reshape_323.dc.unsqueeze.0"(#loc466))
#loc836 = loc("matmul_336"(#loc474))
#loc837 = loc("reshape_337.dc.unsqueeze.0"(#loc474))
#loc838 = loc("reshape_347"(#loc480))
#loc839 = loc("matmul_349"(#loc480))
#loc840 = loc("reshape_350.dc.unsqueeze.0"(#loc480))
#loc841 = loc("conv2d_361.dc.transpose.0"(#loc484))
#loc842 = loc("conv2d_361.dc.transpose.1"(#loc484))
#loc843 = loc("conv2d_361.dc.conv2d.4"(#loc484))
#loc844 = loc("conv2d_361.dc.transpose.5"(#loc484))
#loc845 = loc("conv2d_361.dc.transpose.6"(#loc484))
#loc846 = loc("matmul_377"(#loc490))
#loc847 = loc("conv2d_384.dc.transpose.0"(#loc495))
#loc848 = loc("conv2d_384.dc.transpose.1"(#loc495))
#loc849 = loc("conv2d_384.dc.conv2d.4"(#loc495))
#loc850 = loc("conv2d_384.dc.transpose.5"(#loc495))
#loc851 = loc("conv2d_384.dc.transpose.6"(#loc495))
#loc852 = loc("layernorm_390.dc.reduce_sum.0"(#loc498))
#loc853 = loc("layernorm_390.dc.multiply.2"(#loc498))
#loc854 = loc("layernorm_390.dc.subtract.3"(#loc498))
#loc855 = loc("layernorm_390.dc.multiply.4"(#loc498))
#loc856 = loc("layernorm_390.dc.reduce_sum.5"(#loc498))
#loc857 = loc("layernorm_390.dc.multiply.7"(#loc498))
#loc858 = loc("layernorm_390.dc.add.9"(#loc498))
#loc859 = loc("layernorm_390.dc.sqrt.10"(#loc498))
#loc860 = loc("layernorm_390.dc.reciprocal.11"(#loc498))
#loc861 = loc("layernorm_390.dc.multiply.12"(#loc498))
#loc862 = loc("layernorm_390.dc.multiply.13"(#loc498))
#loc863 = loc("layernorm_390.dc.add.14"(#loc498))
#loc864 = loc("reshape_391.dc.squeeze.0"(#loc499))
#loc865 = loc("matmul_393"(#loc499))
#loc866 = loc("reshape_394.dc.unsqueeze.0"(#loc499))
#loc867 = loc("matmul_407"(#loc507))
#loc868 = loc("reshape_408.dc.unsqueeze.0"(#loc507))
#loc869 = loc("reshape_418"(#loc513))
#loc870 = loc("matmul_420"(#loc513))
#loc871 = loc("reshape_421.dc.unsqueeze.0"(#loc513))
#loc872 = loc("conv2d_432.dc.transpose.0"(#loc517))
#loc873 = loc("conv2d_432.dc.transpose.1"(#loc517))
#loc874 = loc("conv2d_432.dc.conv2d.4"(#loc517))
#loc875 = loc("conv2d_432.dc.transpose.5"(#loc517))
#loc876 = loc("conv2d_432.dc.transpose.6"(#loc517))
#loc877 = loc("reshape_459.dc.squeeze.0"(#loc457))
#loc878 = loc("matmul_461"(#loc457))
#loc879 = loc("reshape_462.dc.unsqueeze.0"(#loc457))
#loc880 = loc("matmul_468"(#loc466))
#loc881 = loc("reshape_469.dc.unsqueeze.0"(#loc466))
#loc882 = loc("matmul_482"(#loc474))
#loc883 = loc("reshape_483.dc.unsqueeze.0"(#loc474))
#loc884 = loc("reshape_493"(#loc480))
#loc885 = loc("matmul_495"(#loc480))
#loc886 = loc("reshape_496.dc.unsqueeze.0"(#loc480))
#loc887 = loc("conv2d_507.dc.transpose.0"(#loc484))
#loc888 = loc("conv2d_507.dc.transpose.1"(#loc484))
#loc889 = loc("conv2d_507.dc.conv2d.4"(#loc484))
#loc890 = loc("conv2d_507.dc.transpose.5"(#loc484))
#loc891 = loc("conv2d_507.dc.transpose.6"(#loc484))
#loc892 = loc("reshape_522.dc.squeeze.0"(#loc490))
#loc893 = loc("matmul_524"(#loc490))
#loc894 = loc("reshape_525.dc.unsqueeze.0"(#loc490))
#loc895 = loc("matmul_531"(#loc499))
#loc896 = loc("reshape_532.dc.unsqueeze.0"(#loc499))
#loc897 = loc("matmul_545"(#loc507))
#loc898 = loc("reshape_546.dc.unsqueeze.0"(#loc507))
#loc899 = loc("reshape_556"(#loc513))
#loc900 = loc("matmul_558"(#loc513))
#loc901 = loc("reshape_559.dc.unsqueeze.0"(#loc513))
#loc902 = loc("conv2d_570.dc.transpose.0"(#loc517))
#loc903 = loc("conv2d_570.dc.transpose.1"(#loc517))
#loc904 = loc("conv2d_570.dc.conv2d.4"(#loc517))
#loc905 = loc("conv2d_570.dc.transpose.5"(#loc517))
#loc906 = loc("conv2d_570.dc.transpose.6"(#loc517))
