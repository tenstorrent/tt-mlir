// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-tuplify-tensors --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

// https://huggingface.co/meta-llama/Llama-3.2-1B
module @LLama_3.2_1B attributes {} {
  func.func @forward(%arg0: tensor<1x11xi32> {ttir.name = "input_1"}, %arg1: tensor<1xf32> {ttir.name = "input_1_add_152"}, %arg2: tensor<1x11x32xf32> {ttir.name = "input_0_unsqueeze_162"}, %arg3: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_172.2"}, %arg4: tensor<1xf32> {ttir.name = "input_1_multiply_173"}, %arg5: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_174.2"}, %arg6: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_186.2"}, %arg7: tensor<1xf32> {ttir.name = "input_1_multiply_187"}, %arg8: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_188.2"}, %arg9: tensor<1xf32> {ttir.name = "input_1_multiply_199"}, %arg10: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_200"}, %arg11: tensor<1xf32> {ttir.name = "input_1_add_225"}, %arg12: tensor<1xf32> {ttir.name = "input_1_add_245"}, %arg13: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_256.2"}, %arg14: tensor<1xf32> {ttir.name = "input_1_multiply_257"}, %arg15: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_258.2"}, %arg16: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_268.2"}, %arg17: tensor<1xf32> {ttir.name = "input_1_multiply_269"}, %arg18: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_270.2"}, %arg19: tensor<1xf32> {ttir.name = "input_1_multiply_281"}, %arg20: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_282"}, %arg21: tensor<1xf32> {ttir.name = "input_1_add_307"}, %arg22: tensor<1xf32> {ttir.name = "input_1_add_327"}, %arg23: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_338.2"}, %arg24: tensor<1xf32> {ttir.name = "input_1_multiply_339"}, %arg25: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_340.2"}, %arg26: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_350.2"}, %arg27: tensor<1xf32> {ttir.name = "input_1_multiply_351"}, %arg28: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_352.2"}, %arg29: tensor<1xf32> {ttir.name = "input_1_multiply_363"}, %arg30: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_364"}, %arg31: tensor<1xf32> {ttir.name = "input_1_add_389"}, %arg32: tensor<1xf32> {ttir.name = "input_1_add_409"}, %arg33: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_420.2"}, %arg34: tensor<1xf32> {ttir.name = "input_1_multiply_421"}, %arg35: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_422.2"}, %arg36: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_432.2"}, %arg37: tensor<1xf32> {ttir.name = "input_1_multiply_433"}, %arg38: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_434.2"}, %arg39: tensor<1xf32> {ttir.name = "input_1_multiply_445"}, %arg40: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_446"}, %arg41: tensor<1xf32> {ttir.name = "input_1_add_471"}, %arg42: tensor<1xf32> {ttir.name = "input_1_add_491"}, %arg43: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_502.2"}, %arg44: tensor<1xf32> {ttir.name = "input_1_multiply_503"}, %arg45: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_504.2"}, %arg46: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_514.2"}, %arg47: tensor<1xf32> {ttir.name = "input_1_multiply_515"}, %arg48: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_516.2"}, %arg49: tensor<1xf32> {ttir.name = "input_1_multiply_527"}, %arg50: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_528"}, %arg51: tensor<1xf32> {ttir.name = "input_1_add_553"}, %arg52: tensor<1xf32> {ttir.name = "input_1_add_573"}, %arg53: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_584.2"}, %arg54: tensor<1xf32> {ttir.name = "input_1_multiply_585"}, %arg55: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_586.2"}, %arg56: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_596.2"}, %arg57: tensor<1xf32> {ttir.name = "input_1_multiply_597"}, %arg58: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_598.2"}, %arg59: tensor<1xf32> {ttir.name = "input_1_multiply_609"}, %arg60: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_610"}, %arg61: tensor<1xf32> {ttir.name = "input_1_add_635"}, %arg62: tensor<1xf32> {ttir.name = "input_1_add_655"}, %arg63: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_666.2"}, %arg64: tensor<1xf32> {ttir.name = "input_1_multiply_667"}, %arg65: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_668.2"}, %arg66: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_678.2"}, %arg67: tensor<1xf32> {ttir.name = "input_1_multiply_679"}, %arg68: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_680.2"}, %arg69: tensor<1xf32> {ttir.name = "input_1_multiply_691"}, %arg70: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_692"}, %arg71: tensor<1xf32> {ttir.name = "input_1_add_717"}, %arg72: tensor<1xf32> {ttir.name = "input_1_add_737"}, %arg73: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_748.2"}, %arg74: tensor<1xf32> {ttir.name = "input_1_multiply_749"}, %arg75: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_750.2"}, %arg76: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_760.2"}, %arg77: tensor<1xf32> {ttir.name = "input_1_multiply_761"}, %arg78: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_762.2"}, %arg79: tensor<1xf32> {ttir.name = "input_1_multiply_773"}, %arg80: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_774"}, %arg81: tensor<1xf32> {ttir.name = "input_1_add_799"}, %arg82: tensor<1xf32> {ttir.name = "input_1_add_819"}, %arg83: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_830.2"}, %arg84: tensor<1xf32> {ttir.name = "input_1_multiply_831"}, %arg85: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_832.2"}, %arg86: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_842.2"}, %arg87: tensor<1xf32> {ttir.name = "input_1_multiply_843"}, %arg88: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_844.2"}, %arg89: tensor<1xf32> {ttir.name = "input_1_multiply_855"}, %arg90: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_856"}, %arg91: tensor<1xf32> {ttir.name = "input_1_add_881"}, %arg92: tensor<1xf32> {ttir.name = "input_1_add_901"}, %arg93: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_912.2"}, %arg94: tensor<1xf32> {ttir.name = "input_1_multiply_913"}, %arg95: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_914.2"}, %arg96: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_924.2"}, %arg97: tensor<1xf32> {ttir.name = "input_1_multiply_925"}, %arg98: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_926.2"}, %arg99: tensor<1xf32> {ttir.name = "input_1_multiply_937"}, %arg100: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_938"}, %arg101: tensor<1xf32> {ttir.name = "input_1_add_963"}, %arg102: tensor<1xf32> {ttir.name = "input_1_add_983"}, %arg103: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_994.2"}, %arg104: tensor<1xf32> {ttir.name = "input_1_multiply_995"}, %arg105: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_996.2"}, %arg106: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1006.2"}, %arg107: tensor<1xf32> {ttir.name = "input_1_multiply_1007"}, %arg108: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1008.2"}, %arg109: tensor<1xf32> {ttir.name = "input_1_multiply_1019"}, %arg110: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_1020"}, %arg111: tensor<1xf32> {ttir.name = "input_1_add_1045"}, %arg112: tensor<1xf32> {ttir.name = "input_1_add_1065"}, %arg113: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_1076.2"}, %arg114: tensor<1xf32> {ttir.name = "input_1_multiply_1077"}, %arg115: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_1078.2"}, %arg116: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1088.2"}, %arg117: tensor<1xf32> {ttir.name = "input_1_multiply_1089"}, %arg118: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1090.2"}, %arg119: tensor<1xf32> {ttir.name = "input_1_multiply_1101"}, %arg120: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_1102"}, %arg121: tensor<1xf32> {ttir.name = "input_1_add_1127"}, %arg122: tensor<1xf32> {ttir.name = "input_1_add_1147"}, %arg123: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_1158.2"}, %arg124: tensor<1xf32> {ttir.name = "input_1_multiply_1159"}, %arg125: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_1160.2"}, %arg126: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1170.2"}, %arg127: tensor<1xf32> {ttir.name = "input_1_multiply_1171"}, %arg128: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1172.2"}, %arg129: tensor<1xf32> {ttir.name = "input_1_multiply_1183"}, %arg130: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_1184"}, %arg131: tensor<1xf32> {ttir.name = "input_1_add_1209"}, %arg132: tensor<1xf32> {ttir.name = "input_1_add_1229"}, %arg133: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_1240.2"}, %arg134: tensor<1xf32> {ttir.name = "input_1_multiply_1241"}, %arg135: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_1242.2"}, %arg136: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1252.2"}, %arg137: tensor<1xf32> {ttir.name = "input_1_multiply_1253"}, %arg138: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1254.2"}, %arg139: tensor<1xf32> {ttir.name = "input_1_multiply_1265"}, %arg140: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_1266"}, %arg141: tensor<1xf32> {ttir.name = "input_1_add_1291"}, %arg142: tensor<1xf32> {ttir.name = "input_1_add_1311"}, %arg143: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_1322.2"}, %arg144: tensor<1xf32> {ttir.name = "input_1_multiply_1323"}, %arg145: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_1324.2"}, %arg146: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1334.2"}, %arg147: tensor<1xf32> {ttir.name = "input_1_multiply_1335"}, %arg148: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1336.2"}, %arg149: tensor<1xf32> {ttir.name = "input_1_multiply_1347"}, %arg150: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_1348"}, %arg151: tensor<1xf32> {ttir.name = "input_1_add_1373"}, %arg152: tensor<1xf32> {ttir.name = "input_1_add_1393"}, %arg153: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_1404.2"}, %arg154: tensor<1xf32> {ttir.name = "input_1_multiply_1405"}, %arg155: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_1406.2"}, %arg156: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1416.2"}, %arg157: tensor<1xf32> {ttir.name = "input_1_multiply_1417"}, %arg158: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1418.2"}, %arg159: tensor<1xf32> {ttir.name = "input_1_multiply_1429"}, %arg160: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_1430"}, %arg161: tensor<1xf32> {ttir.name = "input_1_add_1455"}, %arg162: tensor<1xf32> {ttir.name = "input_1_add_1475"}, %arg163: tensor<2048xf32> {ttir.name = "norm.weight"}, %arg164: tensor<128256x2048xbf16> {ttir.name = "embed_tokens.weight"}, %arg165: tensor<2048xf32> {ttir.name = "layers.0.input_layernorm.weight"}, %arg166: tensor<2048x2048xf32> {ttir.name = "layers.0.self_attn.q_proj.weight"}, %arg167: tensor<2048x512xf32> {ttir.name = "layers.0.self_attn.k_proj.weight"}, %arg168: tensor<2048x512xf32> {ttir.name = "layers.0.self_attn.v_proj.weight"}, %arg169: tensor<2048x2048xf32> {ttir.name = "layers.0.self_attn.o_proj.weight"}, %arg170: tensor<2048xf32> {ttir.name = "layers.0.post_attention_layernorm.weight"}, %arg171: tensor<2048x8192xf32> {ttir.name = "layers.0.mlp.gate_proj.weight"}, %arg172: tensor<2048x8192xf32> {ttir.name = "layers.0.mlp.up_proj.weight"}, %arg173: tensor<8192x2048xf32> {ttir.name = "layers.0.mlp.down_proj.weight"}, %arg174: tensor<2048xf32> {ttir.name = "layers.1.input_layernorm.weight"}, %arg175: tensor<2048x2048xf32> {ttir.name = "layers.1.self_attn.q_proj.weight"}, %arg176: tensor<2048x512xf32> {ttir.name = "layers.1.self_attn.k_proj.weight"}, %arg177: tensor<2048x512xf32> {ttir.name = "layers.1.self_attn.v_proj.weight"}, %arg178: tensor<2048x2048xf32> {ttir.name = "layers.1.self_attn.o_proj.weight"}, %arg179: tensor<2048xf32> {ttir.name = "layers.1.post_attention_layernorm.weight"}, %arg180: tensor<2048x8192xf32> {ttir.name = "layers.1.mlp.gate_proj.weight"}, %arg181: tensor<2048x8192xf32> {ttir.name = "layers.1.mlp.up_proj.weight"}, %arg182: tensor<8192x2048xf32> {ttir.name = "layers.1.mlp.down_proj.weight"}, %arg183: tensor<2048xf32> {ttir.name = "layers.2.input_layernorm.weight"}, %arg184: tensor<2048x2048xf32> {ttir.name = "layers.2.self_attn.q_proj.weight"}, %arg185: tensor<2048x512xf32> {ttir.name = "layers.2.self_attn.k_proj.weight"}, %arg186: tensor<2048x512xf32> {ttir.name = "layers.2.self_attn.v_proj.weight"}, %arg187: tensor<2048x2048xf32> {ttir.name = "layers.2.self_attn.o_proj.weight"}, %arg188: tensor<2048xf32> {ttir.name = "layers.2.post_attention_layernorm.weight"}, %arg189: tensor<2048x8192xf32> {ttir.name = "layers.2.mlp.gate_proj.weight"}, %arg190: tensor<2048x8192xf32> {ttir.name = "layers.2.mlp.up_proj.weight"}, %arg191: tensor<8192x2048xf32> {ttir.name = "layers.2.mlp.down_proj.weight"}, %arg192: tensor<2048xf32> {ttir.name = "layers.3.input_layernorm.weight"}, %arg193: tensor<2048x2048xf32> {ttir.name = "layers.3.self_attn.q_proj.weight"}, %arg194: tensor<2048x512xf32> {ttir.name = "layers.3.self_attn.k_proj.weight"}, %arg195: tensor<2048x512xf32> {ttir.name = "layers.3.self_attn.v_proj.weight"}, %arg196: tensor<2048x2048xf32> {ttir.name = "layers.3.self_attn.o_proj.weight"}, %arg197: tensor<2048xf32> {ttir.name = "layers.3.post_attention_layernorm.weight"}, %arg198: tensor<2048x8192xf32> {ttir.name = "layers.3.mlp.gate_proj.weight"}, %arg199: tensor<2048x8192xf32> {ttir.name = "layers.3.mlp.up_proj.weight"}, %arg200: tensor<8192x2048xf32> {ttir.name = "layers.3.mlp.down_proj.weight"}, %arg201: tensor<2048xf32> {ttir.name = "layers.4.input_layernorm.weight"}, %arg202: tensor<2048x2048xf32> {ttir.name = "layers.4.self_attn.q_proj.weight"}, %arg203: tensor<2048x512xf32> {ttir.name = "layers.4.self_attn.k_proj.weight"}, %arg204: tensor<2048x512xf32> {ttir.name = "layers.4.self_attn.v_proj.weight"}, %arg205: tensor<2048x2048xf32> {ttir.name = "layers.4.self_attn.o_proj.weight"}, %arg206: tensor<2048xf32> {ttir.name = "layers.4.post_attention_layernorm.weight"}, %arg207: tensor<2048x8192xf32> {ttir.name = "layers.4.mlp.gate_proj.weight"}, %arg208: tensor<2048x8192xf32> {ttir.name = "layers.4.mlp.up_proj.weight"}, %arg209: tensor<8192x2048xf32> {ttir.name = "layers.4.mlp.down_proj.weight"}, %arg210: tensor<2048xf32> {ttir.name = "layers.5.input_layernorm.weight"}, %arg211: tensor<2048x2048xf32> {ttir.name = "layers.5.self_attn.q_proj.weight"}, %arg212: tensor<2048x512xf32> {ttir.name = "layers.5.self_attn.k_proj.weight"}, %arg213: tensor<2048x512xf32> {ttir.name = "layers.5.self_attn.v_proj.weight"}, %arg214: tensor<2048x2048xf32> {ttir.name = "layers.5.self_attn.o_proj.weight"}, %arg215: tensor<2048xf32> {ttir.name = "layers.5.post_attention_layernorm.weight"}, %arg216: tensor<2048x8192xf32> {ttir.name = "layers.5.mlp.gate_proj.weight"}, %arg217: tensor<2048x8192xf32> {ttir.name = "layers.5.mlp.up_proj.weight"}, %arg218: tensor<8192x2048xf32> {ttir.name = "layers.5.mlp.down_proj.weight"}, %arg219: tensor<2048xf32> {ttir.name = "layers.6.input_layernorm.weight"}, %arg220: tensor<2048x2048xf32> {ttir.name = "layers.6.self_attn.q_proj.weight"}, %arg221: tensor<2048x512xf32> {ttir.name = "layers.6.self_attn.k_proj.weight"}, %arg222: tensor<2048x512xf32> {ttir.name = "layers.6.self_attn.v_proj.weight"}, %arg223: tensor<2048x2048xf32> {ttir.name = "layers.6.self_attn.o_proj.weight"}, %arg224: tensor<2048xf32> {ttir.name = "layers.6.post_attention_layernorm.weight"}, %arg225: tensor<2048x8192xf32> {ttir.name = "layers.6.mlp.gate_proj.weight"}, %arg226: tensor<2048x8192xf32> {ttir.name = "layers.6.mlp.up_proj.weight"}, %arg227: tensor<8192x2048xf32> {ttir.name = "layers.6.mlp.down_proj.weight"}, %arg228: tensor<2048xf32> {ttir.name = "layers.7.input_layernorm.weight"}, %arg229: tensor<2048x2048xf32> {ttir.name = "layers.7.self_attn.q_proj.weight"}, %arg230: tensor<2048x512xf32> {ttir.name = "layers.7.self_attn.k_proj.weight"}, %arg231: tensor<2048x512xf32> {ttir.name = "layers.7.self_attn.v_proj.weight"}, %arg232: tensor<2048x2048xf32> {ttir.name = "layers.7.self_attn.o_proj.weight"}, %arg233: tensor<2048xf32> {ttir.name = "layers.7.post_attention_layernorm.weight"}, %arg234: tensor<2048x8192xf32> {ttir.name = "layers.7.mlp.gate_proj.weight"}, %arg235: tensor<2048x8192xf32> {ttir.name = "layers.7.mlp.up_proj.weight"}, %arg236: tensor<8192x2048xf32> {ttir.name = "layers.7.mlp.down_proj.weight"}, %arg237: tensor<2048xf32> {ttir.name = "layers.8.input_layernorm.weight"}, %arg238: tensor<2048x2048xf32> {ttir.name = "layers.8.self_attn.q_proj.weight"}, %arg239: tensor<2048x512xf32> {ttir.name = "layers.8.self_attn.k_proj.weight"}, %arg240: tensor<2048x512xf32> {ttir.name = "layers.8.self_attn.v_proj.weight"}, %arg241: tensor<2048x2048xf32> {ttir.name = "layers.8.self_attn.o_proj.weight"}, %arg242: tensor<2048xf32> {ttir.name = "layers.8.post_attention_layernorm.weight"}, %arg243: tensor<2048x8192xf32> {ttir.name = "layers.8.mlp.gate_proj.weight"}, %arg244: tensor<2048x8192xf32> {ttir.name = "layers.8.mlp.up_proj.weight"}, %arg245: tensor<8192x2048xf32> {ttir.name = "layers.8.mlp.down_proj.weight"}, %arg246: tensor<2048xf32> {ttir.name = "layers.9.input_layernorm.weight"}, %arg247: tensor<2048x2048xf32> {ttir.name = "layers.9.self_attn.q_proj.weight"}, %arg248: tensor<2048x512xf32> {ttir.name = "layers.9.self_attn.k_proj.weight"}, %arg249: tensor<2048x512xf32> {ttir.name = "layers.9.self_attn.v_proj.weight"}, %arg250: tensor<2048x2048xf32> {ttir.name = "layers.9.self_attn.o_proj.weight"}, %arg251: tensor<2048xf32> {ttir.name = "layers.9.post_attention_layernorm.weight"}, %arg252: tensor<2048x8192xf32> {ttir.name = "layers.9.mlp.gate_proj.weight"}, %arg253: tensor<2048x8192xf32> {ttir.name = "layers.9.mlp.up_proj.weight"}, %arg254: tensor<8192x2048xf32> {ttir.name = "layers.9.mlp.down_proj.weight"}, %arg255: tensor<2048xf32> {ttir.name = "layers.10.input_layernorm.weight"}, %arg256: tensor<2048x2048xf32> {ttir.name = "layers.10.self_attn.q_proj.weight"}, %arg257: tensor<2048x512xf32> {ttir.name = "layers.10.self_attn.k_proj.weight"}, %arg258: tensor<2048x512xf32> {ttir.name = "layers.10.self_attn.v_proj.weight"}, %arg259: tensor<2048x2048xf32> {ttir.name = "layers.10.self_attn.o_proj.weight"}, %arg260: tensor<2048xf32> {ttir.name = "layers.10.post_attention_layernorm.weight"}, %arg261: tensor<2048x8192xf32> {ttir.name = "layers.10.mlp.gate_proj.weight"}, %arg262: tensor<2048x8192xf32> {ttir.name = "layers.10.mlp.up_proj.weight"}, %arg263: tensor<8192x2048xf32> {ttir.name = "layers.10.mlp.down_proj.weight"}, %arg264: tensor<2048xf32> {ttir.name = "layers.11.input_layernorm.weight"}, %arg265: tensor<2048x2048xf32> {ttir.name = "layers.11.self_attn.q_proj.weight"}, %arg266: tensor<2048x512xf32> {ttir.name = "layers.11.self_attn.k_proj.weight"}, %arg267: tensor<2048x512xf32> {ttir.name = "layers.11.self_attn.v_proj.weight"}, %arg268: tensor<2048x2048xf32> {ttir.name = "layers.11.self_attn.o_proj.weight"}, %arg269: tensor<2048xf32> {ttir.name = "layers.11.post_attention_layernorm.weight"}, %arg270: tensor<2048x8192xf32> {ttir.name = "layers.11.mlp.gate_proj.weight"}, %arg271: tensor<2048x8192xf32> {ttir.name = "layers.11.mlp.up_proj.weight"}, %arg272: tensor<8192x2048xf32> {ttir.name = "layers.11.mlp.down_proj.weight"}, %arg273: tensor<2048xf32> {ttir.name = "layers.12.input_layernorm.weight"}, %arg274: tensor<2048x2048xf32> {ttir.name = "layers.12.self_attn.q_proj.weight"}, %arg275: tensor<2048x512xf32> {ttir.name = "layers.12.self_attn.k_proj.weight"}, %arg276: tensor<2048x512xf32> {ttir.name = "layers.12.self_attn.v_proj.weight"}, %arg277: tensor<2048x2048xf32> {ttir.name = "layers.12.self_attn.o_proj.weight"}, %arg278: tensor<2048xf32> {ttir.name = "layers.12.post_attention_layernorm.weight"}, %arg279: tensor<2048x8192xf32> {ttir.name = "layers.12.mlp.gate_proj.weight"}, %arg280: tensor<2048x8192xf32> {ttir.name = "layers.12.mlp.up_proj.weight"}, %arg281: tensor<8192x2048xf32> {ttir.name = "layers.12.mlp.down_proj.weight"}, %arg282: tensor<2048xf32> {ttir.name = "layers.13.input_layernorm.weight"}, %arg283: tensor<2048x2048xf32> {ttir.name = "layers.13.self_attn.q_proj.weight"}, %arg284: tensor<2048x512xf32> {ttir.name = "layers.13.self_attn.k_proj.weight"}, %arg285: tensor<2048x512xf32> {ttir.name = "layers.13.self_attn.v_proj.weight"}, %arg286: tensor<2048x2048xf32> {ttir.name = "layers.13.self_attn.o_proj.weight"}, %arg287: tensor<2048xf32> {ttir.name = "layers.13.post_attention_layernorm.weight"}, %arg288: tensor<2048x8192xf32> {ttir.name = "layers.13.mlp.gate_proj.weight"}, %arg289: tensor<2048x8192xf32> {ttir.name = "layers.13.mlp.up_proj.weight"}, %arg290: tensor<8192x2048xf32> {ttir.name = "layers.13.mlp.down_proj.weight"}, %arg291: tensor<2048xf32> {ttir.name = "layers.14.input_layernorm.weight"}, %arg292: tensor<2048x2048xf32> {ttir.name = "layers.14.self_attn.q_proj.weight"}, %arg293: tensor<2048x512xf32> {ttir.name = "layers.14.self_attn.k_proj.weight"}, %arg294: tensor<2048x512xf32> {ttir.name = "layers.14.self_attn.v_proj.weight"}, %arg295: tensor<2048x2048xf32> {ttir.name = "layers.14.self_attn.o_proj.weight"}, %arg296: tensor<2048xf32> {ttir.name = "layers.14.post_attention_layernorm.weight"}, %arg297: tensor<2048x8192xf32> {ttir.name = "layers.14.mlp.gate_proj.weight"}, %arg298: tensor<2048x8192xf32> {ttir.name = "layers.14.mlp.up_proj.weight"}, %arg299: tensor<8192x2048xf32> {ttir.name = "layers.14.mlp.down_proj.weight"}, %arg300: tensor<2048xf32> {ttir.name = "layers.15.input_layernorm.weight"}, %arg301: tensor<2048x2048xf32> {ttir.name = "layers.15.self_attn.q_proj.weight"}, %arg302: tensor<2048x512xf32> {ttir.name = "layers.15.self_attn.k_proj.weight"}, %arg303: tensor<2048x512xf32> {ttir.name = "layers.15.self_attn.v_proj.weight"}, %arg304: tensor<2048x2048xf32> {ttir.name = "layers.15.self_attn.o_proj.weight"}, %arg305: tensor<2048xf32> {ttir.name = "layers.15.post_attention_layernorm.weight"}, %arg306: tensor<2048x8192xf32> {ttir.name = "layers.15.mlp.gate_proj.weight"}, %arg307: tensor<2048x8192xf32> {ttir.name = "layers.15.mlp.up_proj.weight"}, %arg308: tensor<8192x2048xf32> {ttir.name = "layers.15.mlp.down_proj.weight"}) -> (tensor<1x32x11x11xf32> {ttir.name = "LlamaModel.output_multiply_1479"}) {
    %0 = ttir.empty() : tensor<1x11x2048xbf16>
    %1 = "ttir.embedding"(%arg0, %arg164, %0) : (tensor<1x11xi32>, tensor<128256x2048xbf16>, tensor<1x11x2048xbf16>) -> tensor<1x11x2048xbf16>
    %2 = ttir.empty() : tensor<1x11x2048xf32>
    %3 = "ttir.typecast"(%1, %2) {dtype = "Float32"} : (tensor<1x11x2048xbf16>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %4 = ttir.empty() : tensor<1x11x2048xf32>
    %5 = "ttir.typecast"(%1, %4) {dtype = "Float32"} : (tensor<1x11x2048xbf16>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %6 = ttir.empty() : tensor<1x11x2048xf32>
    %7 = "ttir.typecast"(%1, %6) {dtype = "Float32"} : (tensor<1x11x2048xbf16>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %8 = ttir.empty() : tensor<1x11x2048xf32>
    %9 = "ttir.multiply"(%7, %7, %8) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %10 = ttir.empty() : tensor<1x11x1xf32>
    %11 = "ttir.mean"(%9, %10) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %12 = ttir.empty() : tensor<1x11x1xf32>
    %13 = "ttir.add"(%11, %arg1, %12) : (tensor<1x11x1xf32>, tensor<1xf32>, tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %14 = ttir.empty() : tensor<1x11x1xf32>
    %15 = "ttir.sqrt"(%13, %14) : (tensor<1x11x1xf32>, tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %16 = ttir.empty() : tensor<1x11x1xf32>
    %17 = "ttir.reciprocal"(%15, %16) : (tensor<1x11x1xf32>, tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %18 = ttir.empty() : tensor<1x11x2048xf32>
    %19 = "ttir.multiply"(%5, %17, %18) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %20 = ttir.empty() : tensor<1x11x2048xf32>
    %21 = "ttir.multiply"(%arg165, %19, %20) : (tensor<2048xf32>, tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %22 = ttir.empty() : tensor<11x2048xf32>
    %23 = "ttir.squeeze"(%21, %22) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>, tensor<11x2048xf32>) -> tensor<11x2048xf32>
    %24 = ttir.empty() : tensor<11x2048xf32>
    %25 = "ttir.matmul"(%23, %arg166, %24) : (tensor<11x2048xf32>, tensor<2048x2048xf32>, tensor<11x2048xf32>) -> tensor<11x2048xf32>
    %26 = ttir.empty() : tensor<1x11x32x64xf32>
    %27 = "ttir.reshape"(%25, %26) <{shape = [1 : i32, 11 : i32, 32 : i32, 64 : i32]}> : (tensor<11x2048xf32>, tensor<1x11x32x64xf32>) -> tensor<1x11x32x64xf32>
    %28 = ttir.empty() : tensor<1x32x11x64xf32>
    %29 = "ttir.transpose"(%27, %28) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x32x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %30 = ttir.empty() : tensor<1x11x64xf32>
    %31 = "ttir.concat"(%arg2, %arg2, %30) <{dim = -1 : si32}> : (tensor<1x11x32xf32>, tensor<1x11x32xf32>, tensor<1x11x64xf32>) -> tensor<1x11x64xf32>
    %32 = ttir.empty() : tensor<1x11x64xf32>
    %33 = "ttir.cos"(%31, %32) : (tensor<1x11x64xf32>, tensor<1x11x64xf32>) -> tensor<1x11x64xf32>
    %34 = ttir.empty() : tensor<1x1x11x64xf32>
    %35 = "ttir.unsqueeze"(%33, %34) <{dim = 1 : si32}> : (tensor<1x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x1x11x64xf32>
    %36 = ttir.empty() : tensor<1x32x11x64xf32>
    %37 = "ttir.multiply"(%29, %35, %36) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %38 = ttir.empty() : tensor<1x32x64x11xf32>
    %39 = "ttir.transpose"(%29, %38) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x64x11xf32>
    %40 = ttir.empty() : tensor<1x32x32x11xf32>
    %41 = "ttir.matmul"(%arg3, %39, %40) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>, tensor<1x32x32x11xf32>) -> tensor<1x32x32x11xf32>
    %42 = ttir.empty() : tensor<1x32x11x32xf32>
    %43 = "ttir.transpose"(%41, %42) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x32xf32>
    %44 = ttir.empty() : tensor<1x32x11x32xf32>
    %45 = "ttir.multiply"(%43, %arg4, %44) : (tensor<1x32x11x32xf32>, tensor<1xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x32xf32>
    %46 = ttir.empty() : tensor<1x32x64x11xf32>
    %47 = "ttir.transpose"(%29, %46) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x64x11xf32>
    %48 = ttir.empty() : tensor<1x32x32x11xf32>
    %49 = "ttir.matmul"(%arg5, %47, %48) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>, tensor<1x32x32x11xf32>) -> tensor<1x32x32x11xf32>
    %50 = ttir.empty() : tensor<1x32x11x32xf32>
    %51 = "ttir.transpose"(%49, %50) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x32xf32>
    %52 = ttir.empty() : tensor<1x32x11x64xf32>
    %53 = "ttir.concat"(%45, %51, %52) <{dim = -1 : si32}> : (tensor<1x32x11x32xf32>, tensor<1x32x11x32xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %54 = ttir.empty() : tensor<1x11x64xf32>
    %55 = "ttir.sin"(%31, %54) : (tensor<1x11x64xf32>, tensor<1x11x64xf32>) -> tensor<1x11x64xf32>
    %56 = ttir.empty() : tensor<1x1x11x64xf32>
    %57 = "ttir.unsqueeze"(%55, %56) <{dim = 1 : si32}> : (tensor<1x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x1x11x64xf32>
    %58 = ttir.empty() : tensor<1x32x11x64xf32>
    %59 = "ttir.multiply"(%53, %57, %58) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %60 = ttir.empty() : tensor<1x32x11x64xf32>
    %61 = "ttir.add"(%37, %59, %60) : (tensor<1x32x11x64xf32>, tensor<1x32x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %62 = ttir.empty() : tensor<32x11x64xf32>
    %63 = "ttir.squeeze"(%61, %62) <{dim = 0 : si32}> : (tensor<1x32x11x64xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %64 = ttir.empty() : tensor<11x512xf32>
    %65 = "ttir.matmul"(%23, %arg167, %64) : (tensor<11x2048xf32>, tensor<2048x512xf32>, tensor<11x512xf32>) -> tensor<11x512xf32>
    %66 = ttir.empty() : tensor<1x11x8x64xf32>
    %67 = "ttir.reshape"(%65, %66) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>, tensor<1x11x8x64xf32>) -> tensor<1x11x8x64xf32>
    %68 = ttir.empty() : tensor<1x8x11x64xf32>
    %69 = "ttir.transpose"(%67, %68) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %70 = ttir.empty() : tensor<1x8x11x64xf32>
    %71 = "ttir.multiply"(%69, %35, %70) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %72 = ttir.empty() : tensor<1x8x64x11xf32>
    %73 = "ttir.transpose"(%69, %72) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x64x11xf32>
    %74 = ttir.empty() : tensor<1x8x32x11xf32>
    %75 = "ttir.matmul"(%arg6, %73, %74) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>, tensor<1x8x32x11xf32>) -> tensor<1x8x32x11xf32>
    %76 = ttir.empty() : tensor<1x8x11x32xf32>
    %77 = "ttir.transpose"(%75, %76) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x32xf32>
    %78 = ttir.empty() : tensor<1x8x11x32xf32>
    %79 = "ttir.multiply"(%77, %arg7, %78) : (tensor<1x8x11x32xf32>, tensor<1xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x32xf32>
    %80 = ttir.empty() : tensor<1x8x64x11xf32>
    %81 = "ttir.transpose"(%69, %80) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x64x11xf32>
    %82 = ttir.empty() : tensor<1x8x32x11xf32>
    %83 = "ttir.matmul"(%arg8, %81, %82) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>, tensor<1x8x32x11xf32>) -> tensor<1x8x32x11xf32>
    %84 = ttir.empty() : tensor<1x8x11x32xf32>
    %85 = "ttir.transpose"(%83, %84) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x32xf32>
    %86 = ttir.empty() : tensor<1x8x11x64xf32>
    %87 = "ttir.concat"(%79, %85, %86) <{dim = -1 : si32}> : (tensor<1x8x11x32xf32>, tensor<1x8x11x32xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %88 = ttir.empty() : tensor<1x8x11x64xf32>
    %89 = "ttir.multiply"(%87, %57, %88) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %90 = ttir.empty() : tensor<1x8x11x64xf32>
    %91 = "ttir.add"(%71, %89, %90) : (tensor<1x8x11x64xf32>, tensor<1x8x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %92 = ttir.empty() : tensor<1x8x1x11x64xf32>
    %93 = "ttir.unsqueeze"(%91, %92) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>, tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %94 = ttir.empty() : tensor<1x8x1x11x64xf32>
    %95 = "ttir.repeat_interleave"(%93, %94) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>, tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %96 = ttir.empty() : tensor<1x8x4x11x64xf32>
    %97 = "ttir.repeat_interleave"(%95, %96) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>, tensor<1x8x4x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %98 = ttir.empty() : tensor<32x11x64xf32>
    %99 = "ttir.reshape"(%97, %98) <{shape = [32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %100 = ttir.empty() : tensor<32x64x11xf32>
    %101 = "ttir.transpose"(%99, %100) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x11x64xf32>, tensor<32x64x11xf32>) -> tensor<32x64x11xf32>
    %102 = ttir.empty() : tensor<32x11x11xf32>
    %103 = "ttir.matmul"(%63, %101, %102) : (tensor<32x11x64xf32>, tensor<32x64x11xf32>, tensor<32x11x11xf32>) -> tensor<32x11x11xf32>
    %104 = ttir.empty() : tensor<1x32x11x11xf32>
    %105 = "ttir.unsqueeze"(%103, %104) <{dim = 0 : si32}> : (tensor<32x11x11xf32>, tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %106 = ttir.empty() : tensor<1x32x11x11xf32>
    %107 = "ttir.multiply"(%105, %arg9, %106) : (tensor<1x32x11x11xf32>, tensor<1xf32>, tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %108 = ttir.empty() : tensor<1x32x11x11xf32>
    %109 = "ttir.add"(%107, %arg10, %108) : (tensor<1x32x11x11xf32>, tensor<1x1x11x11xf32>, tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %110 = ttir.empty() : tensor<1x32x11x11xf32>
    %111 = "ttir.softmax"(%109, %110) <{dimension = -1 : si32}> : (tensor<1x32x11x11xf32>, tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %112 = ttir.empty() : tensor<32x11x11xf32>
    %113 = "ttir.squeeze"(%111, %112) <{dim = 0 : si32}> : (tensor<1x32x11x11xf32>, tensor<32x11x11xf32>) -> tensor<32x11x11xf32>
    %114 = ttir.empty() : tensor<11x512xf32>
    %115 = "ttir.matmul"(%23, %arg168, %114) : (tensor<11x2048xf32>, tensor<2048x512xf32>, tensor<11x512xf32>) -> tensor<11x512xf32>
    %116 = ttir.empty() : tensor<1x11x8x64xf32>
    %117 = "ttir.reshape"(%115, %116) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>, tensor<1x11x8x64xf32>) -> tensor<1x11x8x64xf32>
    %118 = ttir.empty() : tensor<1x8x11x64xf32>
    %119 = "ttir.transpose"(%117, %118) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %120 = ttir.empty() : tensor<1x8x1x11x64xf32>
    %121 = "ttir.unsqueeze"(%119, %120) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>, tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %122 = ttir.empty() : tensor<1x8x1x11x64xf32>
    %123 = "ttir.repeat_interleave"(%121, %122) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>, tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %124 = ttir.empty() : tensor<1x8x4x11x64xf32>
    %125 = "ttir.repeat_interleave"(%123, %124) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>, tensor<1x8x4x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %126 = ttir.empty() : tensor<1x32x11x64xf32>
    %127 = "ttir.reshape"(%125, %126) <{shape = [1 : i32, 32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %128 = ttir.empty() : tensor<1x32x64x11xf32>
    %129 = "ttir.transpose"(%127, %128) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x64x11xf32>
    %130 = ttir.empty() : tensor<32x64x11xf32>
    %131 = "ttir.squeeze"(%129, %130) <{dim = 0 : si32}> : (tensor<1x32x64x11xf32>, tensor<32x64x11xf32>) -> tensor<32x64x11xf32>
    %132 = ttir.empty() : tensor<32x11x64xf32>
    %133 = "ttir.transpose"(%131, %132) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x64x11xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %134 = ttir.empty() : tensor<32x11x64xf32>
    %135 = "ttir.matmul"(%113, %133, %134) : (tensor<32x11x11xf32>, tensor<32x11x64xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %136 = ttir.empty() : tensor<1x32x11x64xf32>
    %137 = "ttir.unsqueeze"(%135, %136) <{dim = 0 : si32}> : (tensor<32x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %138 = ttir.empty() : tensor<1x11x32x64xf32>
    %139 = "ttir.transpose"(%137, %138) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x11x64xf32>, tensor<1x11x32x64xf32>) -> tensor<1x11x32x64xf32>
    %140 = ttir.empty() : tensor<11x2048xf32>
    %141 = "ttir.reshape"(%139, %140) <{shape = [11 : i32, 2048 : i32]}> : (tensor<1x11x32x64xf32>, tensor<11x2048xf32>) -> tensor<11x2048xf32>
    %142 = ttir.empty() : tensor<11x2048xf32>
    %143 = "ttir.matmul"(%141, %arg169, %142) : (tensor<11x2048xf32>, tensor<2048x2048xf32>, tensor<11x2048xf32>) -> tensor<11x2048xf32>
    %144 = ttir.empty() : tensor<1x11x2048xf32>
    %145 = "ttir.unsqueeze"(%143, %144) <{dim = 0 : si32}> : (tensor<11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %146 = ttir.empty() : tensor<1x11x2048xf32>
    %147 = "ttir.add"(%3, %145, %146) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %148 = ttir.empty() : tensor<1x11x2048xf32>
    %149 = "ttir.multiply"(%147, %147, %148) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %150 = ttir.empty() : tensor<1x11x1xf32>
    %151 = "ttir.mean"(%149, %150) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %152 = ttir.empty() : tensor<1x11x1xf32>
    %153 = "ttir.add"(%151, %arg11, %152) : (tensor<1x11x1xf32>, tensor<1xf32>, tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %154 = ttir.empty() : tensor<1x11x1xf32>
    %155 = "ttir.sqrt"(%153, %154) : (tensor<1x11x1xf32>, tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %156 = ttir.empty() : tensor<1x11x1xf32>
    %157 = "ttir.reciprocal"(%155, %156) : (tensor<1x11x1xf32>, tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %158 = ttir.empty() : tensor<1x11x2048xf32>
    %159 = "ttir.multiply"(%147, %157, %158) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %160 = ttir.empty() : tensor<1x11x2048xf32>
    %161 = "ttir.multiply"(%arg170, %159, %160) : (tensor<2048xf32>, tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %162 = ttir.empty() : tensor<11x2048xf32>
    %163 = "ttir.squeeze"(%161, %162) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>, tensor<11x2048xf32>) -> tensor<11x2048xf32>
    %164 = ttir.empty() : tensor<11x8192xf32>
    %165 = "ttir.matmul"(%163, %arg171, %164) : (tensor<11x2048xf32>, tensor<2048x8192xf32>, tensor<11x8192xf32>) -> tensor<11x8192xf32>
    %166 = ttir.empty() : tensor<1x11x8192xf32>
    %167 = "ttir.unsqueeze"(%165, %166) <{dim = 0 : si32}> : (tensor<11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %168 = ttir.empty() : tensor<1x11x8192xf32>
    %169 = "ttir.sigmoid"(%167, %168) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %170 = ttir.empty() : tensor<1x11x8192xf32>
    %171 = "ttir.multiply"(%167, %169, %170) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %172 = ttir.empty() : tensor<11x8192xf32>
    %173 = "ttir.matmul"(%163, %arg172, %172) : (tensor<11x2048xf32>, tensor<2048x8192xf32>, tensor<11x8192xf32>) -> tensor<11x8192xf32>
    %174 = ttir.empty() : tensor<1x11x8192xf32>
    %175 = "ttir.unsqueeze"(%173, %174) <{dim = 0 : si32}> : (tensor<11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %176 = ttir.empty() : tensor<1x11x8192xf32>
    %177 = "ttir.multiply"(%171, %175, %176) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %178 = ttir.empty() : tensor<1x11x2048xf32>
    %179 = "ttir.matmul"(%177, %arg173, %178) : (tensor<1x11x8192xf32>, tensor<8192x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %180 = ttir.empty() : tensor<1x11x2048xf32>
    %181 = "ttir.add"(%147, %179, %180) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %182 = ttir.empty() : tensor<1x11x2048xf32>
    %183 = "ttir.multiply"(%181, %181, %182) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %184 = ttir.empty() : tensor<1x11x1xf32>
    %185 = "ttir.mean"(%183, %184) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %186 = ttir.empty() : tensor<1x11x1xf32>
    %187 = "ttir.add"(%185, %arg12, %186) : (tensor<1x11x1xf32>, tensor<1xf32>, tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %188 = ttir.empty() : tensor<1x11x1xf32>
    %189 = "ttir.sqrt"(%187, %188) : (tensor<1x11x1xf32>, tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %190 = ttir.empty() : tensor<1x11x1xf32>
    %191 = "ttir.reciprocal"(%189, %190) : (tensor<1x11x1xf32>, tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %192 = ttir.empty() : tensor<1x11x2048xf32>
    %193 = "ttir.multiply"(%181, %191, %192) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %194 = ttir.empty() : tensor<1x11x2048xf32>
    %195 = "ttir.multiply"(%arg174, %193, %194) : (tensor<2048xf32>, tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %196 = ttir.empty() : tensor<11x2048xf32>
    %197 = "ttir.squeeze"(%195, %196) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>, tensor<11x2048xf32>) -> tensor<11x2048xf32>
    %198 = ttir.empty() : tensor<11x2048xf32>
    %199 = "ttir.matmul"(%197, %arg175, %198) : (tensor<11x2048xf32>, tensor<2048x2048xf32>, tensor<11x2048xf32>) -> tensor<11x2048xf32>
    %200 = ttir.empty() : tensor<1x11x32x64xf32>
    %201 = "ttir.reshape"(%199, %200) <{shape = [1 : i32, 11 : i32, 32 : i32, 64 : i32]}> : (tensor<11x2048xf32>, tensor<1x11x32x64xf32>) -> tensor<1x11x32x64xf32>
    %202 = ttir.empty() : tensor<1x32x11x64xf32>
    %203 = "ttir.transpose"(%201, %202) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x32x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %204 = ttir.empty() : tensor<1x32x11x64xf32>
    %205 = "ttir.multiply"(%203, %35, %204) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %206 = ttir.empty() : tensor<1x32x64x11xf32>
    %207 = "ttir.transpose"(%203, %206) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x64x11xf32>
    %208 = ttir.empty() : tensor<1x32x32x11xf32>
    %209 = "ttir.matmul"(%arg13, %207, %208) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>, tensor<1x32x32x11xf32>) -> tensor<1x32x32x11xf32>
    %210 = ttir.empty() : tensor<1x32x11x32xf32>
    %211 = "ttir.transpose"(%209, %210) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x32xf32>
    %212 = ttir.empty() : tensor<1x32x11x32xf32>
    %213 = "ttir.multiply"(%211, %arg14, %212) : (tensor<1x32x11x32xf32>, tensor<1xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x32xf32>
    %214 = ttir.empty() : tensor<1x32x64x11xf32>
    %215 = "ttir.transpose"(%203, %214) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x64x11xf32>
    %216 = ttir.empty() : tensor<1x32x32x11xf32>
    %217 = "ttir.matmul"(%arg15, %215, %216) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>, tensor<1x32x32x11xf32>) -> tensor<1x32x32x11xf32>
    %218 = ttir.empty() : tensor<1x32x11x32xf32>
    %219 = "ttir.transpose"(%217, %218) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x32xf32>
    %220 = ttir.empty() : tensor<1x32x11x64xf32>
    %221 = "ttir.concat"(%213, %219, %220) <{dim = -1 : si32}> : (tensor<1x32x11x32xf32>, tensor<1x32x11x32xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %222 = ttir.empty() : tensor<1x32x11x64xf32>
    %223 = "ttir.multiply"(%221, %57, %222) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %224 = ttir.empty() : tensor<1x32x11x64xf32>
    %225 = "ttir.add"(%205, %223, %224) : (tensor<1x32x11x64xf32>, tensor<1x32x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %226 = ttir.empty() : tensor<32x11x64xf32>
    %227 = "ttir.squeeze"(%225, %226) <{dim = 0 : si32}> : (tensor<1x32x11x64xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %228 = ttir.empty() : tensor<11x512xf32>
    %229 = "ttir.matmul"(%197, %arg176, %228) : (tensor<11x2048xf32>, tensor<2048x512xf32>, tensor<11x512xf32>) -> tensor<11x512xf32>
    %230 = ttir.empty() : tensor<1x11x8x64xf32>
    %231 = "ttir.reshape"(%229, %230) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>, tensor<1x11x8x64xf32>) -> tensor<1x11x8x64xf32>
    %232 = ttir.empty() : tensor<1x8x11x64xf32>
    %233 = "ttir.transpose"(%231, %232) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %234 = ttir.empty() : tensor<1x8x11x64xf32>
    %235 = "ttir.multiply"(%233, %35, %234) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %236 = ttir.empty() : tensor<1x8x64x11xf32>
    %237 = "ttir.transpose"(%233, %236) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x64x11xf32>
    %238 = ttir.empty() : tensor<1x8x32x11xf32>
    %239 = "ttir.matmul"(%arg16, %237, %238) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>, tensor<1x8x32x11xf32>) -> tensor<1x8x32x11xf32>
    %240 = ttir.empty() : tensor<1x8x11x32xf32>
    %241 = "ttir.transpose"(%239, %240) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x32xf32>
    %242 = ttir.empty() : tensor<1x8x11x32xf32>
    %243 = "ttir.multiply"(%241, %arg17, %242) : (tensor<1x8x11x32xf32>, tensor<1xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x32xf32>
    %244 = ttir.empty() : tensor<1x8x64x11xf32>
    %245 = "ttir.transpose"(%233, %244) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x64x11xf32>
    %246 = ttir.empty() : tensor<1x8x32x11xf32>
    %247 = "ttir.matmul"(%arg18, %245, %246) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>, tensor<1x8x32x11xf32>) -> tensor<1x8x32x11xf32>
    %248 = ttir.empty() : tensor<1x8x11x32xf32>
    %249 = "ttir.transpose"(%247, %248) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x32xf32>
    %250 = ttir.empty() : tensor<1x8x11x64xf32>
    %251 = "ttir.concat"(%243, %249, %250) <{dim = -1 : si32}> : (tensor<1x8x11x32xf32>, tensor<1x8x11x32xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %252 = ttir.empty() : tensor<1x8x11x64xf32>
    %253 = "ttir.multiply"(%251, %57, %252) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %254 = ttir.empty() : tensor<1x8x11x64xf32>
    %255 = "ttir.add"(%235, %253, %254) : (tensor<1x8x11x64xf32>, tensor<1x8x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %256 = ttir.empty() : tensor<1x8x1x11x64xf32>
    %257 = "ttir.unsqueeze"(%255, %256) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>, tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %258 = ttir.empty() : tensor<1x8x1x11x64xf32>
    %259 = "ttir.repeat_interleave"(%257, %258) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>, tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %260 = ttir.empty() : tensor<1x8x4x11x64xf32>
    %261 = "ttir.repeat_interleave"(%259, %260) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>, tensor<1x8x4x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %262 = ttir.empty() : tensor<32x11x64xf32>
    %263 = "ttir.reshape"(%261, %262) <{shape = [32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %264 = ttir.empty() : tensor<32x64x11xf32>
    %265 = "ttir.transpose"(%263, %264) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x11x64xf32>, tensor<32x64x11xf32>) -> tensor<32x64x11xf32>
    %266 = ttir.empty() : tensor<32x11x11xf32>
    %267 = "ttir.matmul"(%227, %265, %266) : (tensor<32x11x64xf32>, tensor<32x64x11xf32>, tensor<32x11x11xf32>) -> tensor<32x11x11xf32>
    %268 = ttir.empty() : tensor<1x32x11x11xf32>
    %269 = "ttir.unsqueeze"(%267, %268) <{dim = 0 : si32}> : (tensor<32x11x11xf32>, tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %270 = ttir.empty() : tensor<1x32x11x11xf32>
    %271 = "ttir.multiply"(%269, %arg19, %270) : (tensor<1x32x11x11xf32>, tensor<1xf32>, tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %272 = ttir.empty() : tensor<1x32x11x11xf32>
    %273 = "ttir.add"(%271, %arg20, %272) : (tensor<1x32x11x11xf32>, tensor<1x1x11x11xf32>, tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %274 = ttir.empty() : tensor<1x32x11x11xf32>
    %275 = "ttir.softmax"(%273, %274) <{dimension = -1 : si32}> : (tensor<1x32x11x11xf32>, tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    return %275 : tensor<1x32x11x11xf32>
  }
}
