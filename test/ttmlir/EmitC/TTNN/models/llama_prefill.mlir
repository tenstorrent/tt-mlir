// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-flatbuffer-pipeline -o %t_fb.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_fb.mlir
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

// https://huggingface.co/meta-llama/Llama-3.2-1B
module @LLama_3.2_1B attributes {} {
  func.func @forward(%arg0: tensor<1x11xi32> {ttir.name = "input_1"}, %arg1: tensor<1xf32> {ttir.name = "input_1_add_152"}, %arg2: tensor<1x11x32xf32> {ttir.name = "input_0_unsqueeze_162"}, %arg3: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_172.2"}, %arg4: tensor<1xf32> {ttir.name = "input_1_multiply_173"}, %arg5: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_174.2"}, %arg6: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_186.2"}, %arg7: tensor<1xf32> {ttir.name = "input_1_multiply_187"}, %arg8: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_188.2"}, %arg9: tensor<1xf32> {ttir.name = "input_1_multiply_199"}, %arg10: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_200"}, %arg11: tensor<1xf32> {ttir.name = "input_1_add_225"}, %arg12: tensor<1xf32> {ttir.name = "input_1_add_245"}, %arg13: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_256.2"}, %arg14: tensor<1xf32> {ttir.name = "input_1_multiply_257"}, %arg15: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_258.2"}, %arg16: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_268.2"}, %arg17: tensor<1xf32> {ttir.name = "input_1_multiply_269"}, %arg18: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_270.2"}, %arg19: tensor<1xf32> {ttir.name = "input_1_multiply_281"}, %arg20: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_282"}, %arg21: tensor<1xf32> {ttir.name = "input_1_add_307"}, %arg22: tensor<1xf32> {ttir.name = "input_1_add_327"}, %arg23: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_338.2"}, %arg24: tensor<1xf32> {ttir.name = "input_1_multiply_339"}, %arg25: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_340.2"}, %arg26: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_350.2"}, %arg27: tensor<1xf32> {ttir.name = "input_1_multiply_351"}, %arg28: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_352.2"}, %arg29: tensor<1xf32> {ttir.name = "input_1_multiply_363"}, %arg30: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_364"}, %arg31: tensor<1xf32> {ttir.name = "input_1_add_389"}, %arg32: tensor<1xf32> {ttir.name = "input_1_add_409"}, %arg33: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_420.2"}, %arg34: tensor<1xf32> {ttir.name = "input_1_multiply_421"}, %arg35: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_422.2"}, %arg36: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_432.2"}, %arg37: tensor<1xf32> {ttir.name = "input_1_multiply_433"}, %arg38: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_434.2"}, %arg39: tensor<1xf32> {ttir.name = "input_1_multiply_445"}, %arg40: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_446"}, %arg41: tensor<1xf32> {ttir.name = "input_1_add_471"}, %arg42: tensor<1xf32> {ttir.name = "input_1_add_491"}, %arg43: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_502.2"}, %arg44: tensor<1xf32> {ttir.name = "input_1_multiply_503"}, %arg45: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_504.2"}, %arg46: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_514.2"}, %arg47: tensor<1xf32> {ttir.name = "input_1_multiply_515"}, %arg48: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_516.2"}, %arg49: tensor<1xf32> {ttir.name = "input_1_multiply_527"}, %arg50: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_528"}, %arg51: tensor<1xf32> {ttir.name = "input_1_add_553"}, %arg52: tensor<1xf32> {ttir.name = "input_1_add_573"}, %arg53: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_584.2"}, %arg54: tensor<1xf32> {ttir.name = "input_1_multiply_585"}, %arg55: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_586.2"}, %arg56: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_596.2"}, %arg57: tensor<1xf32> {ttir.name = "input_1_multiply_597"}, %arg58: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_598.2"}, %arg59: tensor<1xf32> {ttir.name = "input_1_multiply_609"}, %arg60: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_610"}, %arg61: tensor<1xf32> {ttir.name = "input_1_add_635"}, %arg62: tensor<1xf32> {ttir.name = "input_1_add_655"}, %arg63: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_666.2"}, %arg64: tensor<1xf32> {ttir.name = "input_1_multiply_667"}, %arg65: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_668.2"}, %arg66: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_678.2"}, %arg67: tensor<1xf32> {ttir.name = "input_1_multiply_679"}, %arg68: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_680.2"}, %arg69: tensor<1xf32> {ttir.name = "input_1_multiply_691"}, %arg70: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_692"}, %arg71: tensor<1xf32> {ttir.name = "input_1_add_717"}, %arg72: tensor<1xf32> {ttir.name = "input_1_add_737"}, %arg73: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_748.2"}, %arg74: tensor<1xf32> {ttir.name = "input_1_multiply_749"}, %arg75: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_750.2"}, %arg76: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_760.2"}, %arg77: tensor<1xf32> {ttir.name = "input_1_multiply_761"}, %arg78: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_762.2"}, %arg79: tensor<1xf32> {ttir.name = "input_1_multiply_773"}, %arg80: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_774"}, %arg81: tensor<1xf32> {ttir.name = "input_1_add_799"}, %arg82: tensor<1xf32> {ttir.name = "input_1_add_819"}, %arg83: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_830.2"}, %arg84: tensor<1xf32> {ttir.name = "input_1_multiply_831"}, %arg85: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_832.2"}, %arg86: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_842.2"}, %arg87: tensor<1xf32> {ttir.name = "input_1_multiply_843"}, %arg88: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_844.2"}, %arg89: tensor<1xf32> {ttir.name = "input_1_multiply_855"}, %arg90: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_856"}, %arg91: tensor<1xf32> {ttir.name = "input_1_add_881"}, %arg92: tensor<1xf32> {ttir.name = "input_1_add_901"}, %arg93: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_912.2"}, %arg94: tensor<1xf32> {ttir.name = "input_1_multiply_913"}, %arg95: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_914.2"}, %arg96: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_924.2"}, %arg97: tensor<1xf32> {ttir.name = "input_1_multiply_925"}, %arg98: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_926.2"}, %arg99: tensor<1xf32> {ttir.name = "input_1_multiply_937"}, %arg100: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_938"}, %arg101: tensor<1xf32> {ttir.name = "input_1_add_963"}, %arg102: tensor<1xf32> {ttir.name = "input_1_add_983"}, %arg103: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_994.2"}, %arg104: tensor<1xf32> {ttir.name = "input_1_multiply_995"}, %arg105: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_996.2"}, %arg106: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1006.2"}, %arg107: tensor<1xf32> {ttir.name = "input_1_multiply_1007"}, %arg108: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1008.2"}, %arg109: tensor<1xf32> {ttir.name = "input_1_multiply_1019"}, %arg110: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_1020"}, %arg111: tensor<1xf32> {ttir.name = "input_1_add_1045"}, %arg112: tensor<1xf32> {ttir.name = "input_1_add_1065"}, %arg113: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_1076.2"}, %arg114: tensor<1xf32> {ttir.name = "input_1_multiply_1077"}, %arg115: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_1078.2"}, %arg116: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1088.2"}, %arg117: tensor<1xf32> {ttir.name = "input_1_multiply_1089"}, %arg118: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1090.2"}, %arg119: tensor<1xf32> {ttir.name = "input_1_multiply_1101"}, %arg120: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_1102"}, %arg121: tensor<1xf32> {ttir.name = "input_1_add_1127"}, %arg122: tensor<1xf32> {ttir.name = "input_1_add_1147"}, %arg123: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_1158.2"}, %arg124: tensor<1xf32> {ttir.name = "input_1_multiply_1159"}, %arg125: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_1160.2"}, %arg126: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1170.2"}, %arg127: tensor<1xf32> {ttir.name = "input_1_multiply_1171"}, %arg128: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1172.2"}, %arg129: tensor<1xf32> {ttir.name = "input_1_multiply_1183"}, %arg130: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_1184"}, %arg131: tensor<1xf32> {ttir.name = "input_1_add_1209"}, %arg132: tensor<1xf32> {ttir.name = "input_1_add_1229"}, %arg133: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_1240.2"}, %arg134: tensor<1xf32> {ttir.name = "input_1_multiply_1241"}, %arg135: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_1242.2"}, %arg136: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1252.2"}, %arg137: tensor<1xf32> {ttir.name = "input_1_multiply_1253"}, %arg138: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1254.2"}, %arg139: tensor<1xf32> {ttir.name = "input_1_multiply_1265"}, %arg140: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_1266"}, %arg141: tensor<1xf32> {ttir.name = "input_1_add_1291"}, %arg142: tensor<1xf32> {ttir.name = "input_1_add_1311"}, %arg143: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_1322.2"}, %arg144: tensor<1xf32> {ttir.name = "input_1_multiply_1323"}, %arg145: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_1324.2"}, %arg146: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1334.2"}, %arg147: tensor<1xf32> {ttir.name = "input_1_multiply_1335"}, %arg148: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1336.2"}, %arg149: tensor<1xf32> {ttir.name = "input_1_multiply_1347"}, %arg150: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_1348"}, %arg151: tensor<1xf32> {ttir.name = "input_1_add_1373"}, %arg152: tensor<1xf32> {ttir.name = "input_1_add_1393"}, %arg153: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_1404.2"}, %arg154: tensor<1xf32> {ttir.name = "input_1_multiply_1405"}, %arg155: tensor<1x32x32x64xf32> {ttir.name = "dc.input_tensor.index_1406.2"}, %arg156: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1416.2"}, %arg157: tensor<1xf32> {ttir.name = "input_1_multiply_1417"}, %arg158: tensor<1x8x32x64xf32> {ttir.name = "dc.input_tensor.index_1418.2"}, %arg159: tensor<1xf32> {ttir.name = "input_1_multiply_1429"}, %arg160: tensor<1x1x11x11xf32> {ttir.name = "input_1_add_1430"}, %arg161: tensor<1xf32> {ttir.name = "input_1_add_1455"}, %arg162: tensor<1xf32> {ttir.name = "input_1_add_1475"}, %arg163: tensor<2048xf32> {ttir.name = "norm.weight"}, %arg164: tensor<128256x2048xbf16> {ttir.name = "embed_tokens.weight"}, %arg165: tensor<2048xf32> {ttir.name = "layers.0.input_layernorm.weight"}, %arg166: tensor<2048x2048xf32> {ttir.name = "layers.0.self_attn.q_proj.weight"}, %arg167: tensor<2048x512xf32> {ttir.name = "layers.0.self_attn.k_proj.weight"}, %arg168: tensor<2048x512xf32> {ttir.name = "layers.0.self_attn.v_proj.weight"}, %arg169: tensor<2048x2048xf32> {ttir.name = "layers.0.self_attn.o_proj.weight"}, %arg170: tensor<2048xf32> {ttir.name = "layers.0.post_attention_layernorm.weight"}, %arg171: tensor<2048x8192xf32> {ttir.name = "layers.0.mlp.gate_proj.weight"}, %arg172: tensor<2048x8192xf32> {ttir.name = "layers.0.mlp.up_proj.weight"}, %arg173: tensor<8192x2048xf32> {ttir.name = "layers.0.mlp.down_proj.weight"}, %arg174: tensor<2048xf32> {ttir.name = "layers.1.input_layernorm.weight"}, %arg175: tensor<2048x2048xf32> {ttir.name = "layers.1.self_attn.q_proj.weight"}, %arg176: tensor<2048x512xf32> {ttir.name = "layers.1.self_attn.k_proj.weight"}, %arg177: tensor<2048x512xf32> {ttir.name = "layers.1.self_attn.v_proj.weight"}, %arg178: tensor<2048x2048xf32> {ttir.name = "layers.1.self_attn.o_proj.weight"}, %arg179: tensor<2048xf32> {ttir.name = "layers.1.post_attention_layernorm.weight"}, %arg180: tensor<2048x8192xf32> {ttir.name = "layers.1.mlp.gate_proj.weight"}, %arg181: tensor<2048x8192xf32> {ttir.name = "layers.1.mlp.up_proj.weight"}, %arg182: tensor<8192x2048xf32> {ttir.name = "layers.1.mlp.down_proj.weight"}, %arg183: tensor<2048xf32> {ttir.name = "layers.2.input_layernorm.weight"}, %arg184: tensor<2048x2048xf32> {ttir.name = "layers.2.self_attn.q_proj.weight"}, %arg185: tensor<2048x512xf32> {ttir.name = "layers.2.self_attn.k_proj.weight"}, %arg186: tensor<2048x512xf32> {ttir.name = "layers.2.self_attn.v_proj.weight"}, %arg187: tensor<2048x2048xf32> {ttir.name = "layers.2.self_attn.o_proj.weight"}, %arg188: tensor<2048xf32> {ttir.name = "layers.2.post_attention_layernorm.weight"}, %arg189: tensor<2048x8192xf32> {ttir.name = "layers.2.mlp.gate_proj.weight"}, %arg190: tensor<2048x8192xf32> {ttir.name = "layers.2.mlp.up_proj.weight"}, %arg191: tensor<8192x2048xf32> {ttir.name = "layers.2.mlp.down_proj.weight"}, %arg192: tensor<2048xf32> {ttir.name = "layers.3.input_layernorm.weight"}, %arg193: tensor<2048x2048xf32> {ttir.name = "layers.3.self_attn.q_proj.weight"}, %arg194: tensor<2048x512xf32> {ttir.name = "layers.3.self_attn.k_proj.weight"}, %arg195: tensor<2048x512xf32> {ttir.name = "layers.3.self_attn.v_proj.weight"}, %arg196: tensor<2048x2048xf32> {ttir.name = "layers.3.self_attn.o_proj.weight"}, %arg197: tensor<2048xf32> {ttir.name = "layers.3.post_attention_layernorm.weight"}, %arg198: tensor<2048x8192xf32> {ttir.name = "layers.3.mlp.gate_proj.weight"}, %arg199: tensor<2048x8192xf32> {ttir.name = "layers.3.mlp.up_proj.weight"}, %arg200: tensor<8192x2048xf32> {ttir.name = "layers.3.mlp.down_proj.weight"}, %arg201: tensor<2048xf32> {ttir.name = "layers.4.input_layernorm.weight"}, %arg202: tensor<2048x2048xf32> {ttir.name = "layers.4.self_attn.q_proj.weight"}, %arg203: tensor<2048x512xf32> {ttir.name = "layers.4.self_attn.k_proj.weight"}, %arg204: tensor<2048x512xf32> {ttir.name = "layers.4.self_attn.v_proj.weight"}, %arg205: tensor<2048x2048xf32> {ttir.name = "layers.4.self_attn.o_proj.weight"}, %arg206: tensor<2048xf32> {ttir.name = "layers.4.post_attention_layernorm.weight"}, %arg207: tensor<2048x8192xf32> {ttir.name = "layers.4.mlp.gate_proj.weight"}, %arg208: tensor<2048x8192xf32> {ttir.name = "layers.4.mlp.up_proj.weight"}, %arg209: tensor<8192x2048xf32> {ttir.name = "layers.4.mlp.down_proj.weight"}, %arg210: tensor<2048xf32> {ttir.name = "layers.5.input_layernorm.weight"}, %arg211: tensor<2048x2048xf32> {ttir.name = "layers.5.self_attn.q_proj.weight"}, %arg212: tensor<2048x512xf32> {ttir.name = "layers.5.self_attn.k_proj.weight"}, %arg213: tensor<2048x512xf32> {ttir.name = "layers.5.self_attn.v_proj.weight"}, %arg214: tensor<2048x2048xf32> {ttir.name = "layers.5.self_attn.o_proj.weight"}, %arg215: tensor<2048xf32> {ttir.name = "layers.5.post_attention_layernorm.weight"}, %arg216: tensor<2048x8192xf32> {ttir.name = "layers.5.mlp.gate_proj.weight"}, %arg217: tensor<2048x8192xf32> {ttir.name = "layers.5.mlp.up_proj.weight"}, %arg218: tensor<8192x2048xf32> {ttir.name = "layers.5.mlp.down_proj.weight"}, %arg219: tensor<2048xf32> {ttir.name = "layers.6.input_layernorm.weight"}, %arg220: tensor<2048x2048xf32> {ttir.name = "layers.6.self_attn.q_proj.weight"}, %arg221: tensor<2048x512xf32> {ttir.name = "layers.6.self_attn.k_proj.weight"}, %arg222: tensor<2048x512xf32> {ttir.name = "layers.6.self_attn.v_proj.weight"}, %arg223: tensor<2048x2048xf32> {ttir.name = "layers.6.self_attn.o_proj.weight"}, %arg224: tensor<2048xf32> {ttir.name = "layers.6.post_attention_layernorm.weight"}, %arg225: tensor<2048x8192xf32> {ttir.name = "layers.6.mlp.gate_proj.weight"}, %arg226: tensor<2048x8192xf32> {ttir.name = "layers.6.mlp.up_proj.weight"}, %arg227: tensor<8192x2048xf32> {ttir.name = "layers.6.mlp.down_proj.weight"}, %arg228: tensor<2048xf32> {ttir.name = "layers.7.input_layernorm.weight"}, %arg229: tensor<2048x2048xf32> {ttir.name = "layers.7.self_attn.q_proj.weight"}, %arg230: tensor<2048x512xf32> {ttir.name = "layers.7.self_attn.k_proj.weight"}, %arg231: tensor<2048x512xf32> {ttir.name = "layers.7.self_attn.v_proj.weight"}, %arg232: tensor<2048x2048xf32> {ttir.name = "layers.7.self_attn.o_proj.weight"}, %arg233: tensor<2048xf32> {ttir.name = "layers.7.post_attention_layernorm.weight"}, %arg234: tensor<2048x8192xf32> {ttir.name = "layers.7.mlp.gate_proj.weight"}, %arg235: tensor<2048x8192xf32> {ttir.name = "layers.7.mlp.up_proj.weight"}, %arg236: tensor<8192x2048xf32> {ttir.name = "layers.7.mlp.down_proj.weight"}, %arg237: tensor<2048xf32> {ttir.name = "layers.8.input_layernorm.weight"}, %arg238: tensor<2048x2048xf32> {ttir.name = "layers.8.self_attn.q_proj.weight"}, %arg239: tensor<2048x512xf32> {ttir.name = "layers.8.self_attn.k_proj.weight"}, %arg240: tensor<2048x512xf32> {ttir.name = "layers.8.self_attn.v_proj.weight"}, %arg241: tensor<2048x2048xf32> {ttir.name = "layers.8.self_attn.o_proj.weight"}, %arg242: tensor<2048xf32> {ttir.name = "layers.8.post_attention_layernorm.weight"}, %arg243: tensor<2048x8192xf32> {ttir.name = "layers.8.mlp.gate_proj.weight"}, %arg244: tensor<2048x8192xf32> {ttir.name = "layers.8.mlp.up_proj.weight"}, %arg245: tensor<8192x2048xf32> {ttir.name = "layers.8.mlp.down_proj.weight"}, %arg246: tensor<2048xf32> {ttir.name = "layers.9.input_layernorm.weight"}, %arg247: tensor<2048x2048xf32> {ttir.name = "layers.9.self_attn.q_proj.weight"}, %arg248: tensor<2048x512xf32> {ttir.name = "layers.9.self_attn.k_proj.weight"}, %arg249: tensor<2048x512xf32> {ttir.name = "layers.9.self_attn.v_proj.weight"}, %arg250: tensor<2048x2048xf32> {ttir.name = "layers.9.self_attn.o_proj.weight"}, %arg251: tensor<2048xf32> {ttir.name = "layers.9.post_attention_layernorm.weight"}, %arg252: tensor<2048x8192xf32> {ttir.name = "layers.9.mlp.gate_proj.weight"}, %arg253: tensor<2048x8192xf32> {ttir.name = "layers.9.mlp.up_proj.weight"}, %arg254: tensor<8192x2048xf32> {ttir.name = "layers.9.mlp.down_proj.weight"}, %arg255: tensor<2048xf32> {ttir.name = "layers.10.input_layernorm.weight"}, %arg256: tensor<2048x2048xf32> {ttir.name = "layers.10.self_attn.q_proj.weight"}, %arg257: tensor<2048x512xf32> {ttir.name = "layers.10.self_attn.k_proj.weight"}, %arg258: tensor<2048x512xf32> {ttir.name = "layers.10.self_attn.v_proj.weight"}, %arg259: tensor<2048x2048xf32> {ttir.name = "layers.10.self_attn.o_proj.weight"}, %arg260: tensor<2048xf32> {ttir.name = "layers.10.post_attention_layernorm.weight"}, %arg261: tensor<2048x8192xf32> {ttir.name = "layers.10.mlp.gate_proj.weight"}, %arg262: tensor<2048x8192xf32> {ttir.name = "layers.10.mlp.up_proj.weight"}, %arg263: tensor<8192x2048xf32> {ttir.name = "layers.10.mlp.down_proj.weight"}, %arg264: tensor<2048xf32> {ttir.name = "layers.11.input_layernorm.weight"}, %arg265: tensor<2048x2048xf32> {ttir.name = "layers.11.self_attn.q_proj.weight"}, %arg266: tensor<2048x512xf32> {ttir.name = "layers.11.self_attn.k_proj.weight"}, %arg267: tensor<2048x512xf32> {ttir.name = "layers.11.self_attn.v_proj.weight"}, %arg268: tensor<2048x2048xf32> {ttir.name = "layers.11.self_attn.o_proj.weight"}, %arg269: tensor<2048xf32> {ttir.name = "layers.11.post_attention_layernorm.weight"}, %arg270: tensor<2048x8192xf32> {ttir.name = "layers.11.mlp.gate_proj.weight"}, %arg271: tensor<2048x8192xf32> {ttir.name = "layers.11.mlp.up_proj.weight"}, %arg272: tensor<8192x2048xf32> {ttir.name = "layers.11.mlp.down_proj.weight"}, %arg273: tensor<2048xf32> {ttir.name = "layers.12.input_layernorm.weight"}, %arg274: tensor<2048x2048xf32> {ttir.name = "layers.12.self_attn.q_proj.weight"}, %arg275: tensor<2048x512xf32> {ttir.name = "layers.12.self_attn.k_proj.weight"}, %arg276: tensor<2048x512xf32> {ttir.name = "layers.12.self_attn.v_proj.weight"}, %arg277: tensor<2048x2048xf32> {ttir.name = "layers.12.self_attn.o_proj.weight"}, %arg278: tensor<2048xf32> {ttir.name = "layers.12.post_attention_layernorm.weight"}, %arg279: tensor<2048x8192xf32> {ttir.name = "layers.12.mlp.gate_proj.weight"}, %arg280: tensor<2048x8192xf32> {ttir.name = "layers.12.mlp.up_proj.weight"}, %arg281: tensor<8192x2048xf32> {ttir.name = "layers.12.mlp.down_proj.weight"}, %arg282: tensor<2048xf32> {ttir.name = "layers.13.input_layernorm.weight"}, %arg283: tensor<2048x2048xf32> {ttir.name = "layers.13.self_attn.q_proj.weight"}, %arg284: tensor<2048x512xf32> {ttir.name = "layers.13.self_attn.k_proj.weight"}, %arg285: tensor<2048x512xf32> {ttir.name = "layers.13.self_attn.v_proj.weight"}, %arg286: tensor<2048x2048xf32> {ttir.name = "layers.13.self_attn.o_proj.weight"}, %arg287: tensor<2048xf32> {ttir.name = "layers.13.post_attention_layernorm.weight"}, %arg288: tensor<2048x8192xf32> {ttir.name = "layers.13.mlp.gate_proj.weight"}, %arg289: tensor<2048x8192xf32> {ttir.name = "layers.13.mlp.up_proj.weight"}, %arg290: tensor<8192x2048xf32> {ttir.name = "layers.13.mlp.down_proj.weight"}, %arg291: tensor<2048xf32> {ttir.name = "layers.14.input_layernorm.weight"}, %arg292: tensor<2048x2048xf32> {ttir.name = "layers.14.self_attn.q_proj.weight"}, %arg293: tensor<2048x512xf32> {ttir.name = "layers.14.self_attn.k_proj.weight"}, %arg294: tensor<2048x512xf32> {ttir.name = "layers.14.self_attn.v_proj.weight"}, %arg295: tensor<2048x2048xf32> {ttir.name = "layers.14.self_attn.o_proj.weight"}, %arg296: tensor<2048xf32> {ttir.name = "layers.14.post_attention_layernorm.weight"}, %arg297: tensor<2048x8192xf32> {ttir.name = "layers.14.mlp.gate_proj.weight"}, %arg298: tensor<2048x8192xf32> {ttir.name = "layers.14.mlp.up_proj.weight"}, %arg299: tensor<8192x2048xf32> {ttir.name = "layers.14.mlp.down_proj.weight"}, %arg300: tensor<2048xf32> {ttir.name = "layers.15.input_layernorm.weight"}, %arg301: tensor<2048x2048xf32> {ttir.name = "layers.15.self_attn.q_proj.weight"}, %arg302: tensor<2048x512xf32> {ttir.name = "layers.15.self_attn.k_proj.weight"}, %arg303: tensor<2048x512xf32> {ttir.name = "layers.15.self_attn.v_proj.weight"}, %arg304: tensor<2048x2048xf32> {ttir.name = "layers.15.self_attn.o_proj.weight"}, %arg305: tensor<2048xf32> {ttir.name = "layers.15.post_attention_layernorm.weight"}, %arg306: tensor<2048x8192xf32> {ttir.name = "layers.15.mlp.gate_proj.weight"}, %arg307: tensor<2048x8192xf32> {ttir.name = "layers.15.mlp.up_proj.weight"}, %arg308: tensor<8192x2048xf32> {ttir.name = "layers.15.mlp.down_proj.weight"}) -> (tensor<1x11x2048xf32> {ttir.name = "LlamaModel.output_multiply_1479"}) {
    %0 = "ttir.embedding"(%arg0, %arg164) : (tensor<1x11xi32>, tensor<128256x2048xbf16>) -> tensor<1x11x2048xbf16>
    %1 = "ttir.typecast"(%0) {dtype = "Float32"} : (tensor<1x11x2048xbf16>) -> tensor<1x11x2048xf32>
    %2 = "ttir.typecast"(%0) {dtype = "Float32"} : (tensor<1x11x2048xbf16>) -> tensor<1x11x2048xf32>
    %3 = "ttir.typecast"(%0) {dtype = "Float32"} : (tensor<1x11x2048xbf16>) -> tensor<1x11x2048xf32>
    %4 = "ttir.multiply"(%3, %3) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %5 = "ttir.mean"(%4) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %6 = "ttir.add"(%5, %arg1) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %7 = "ttir.sqrt"(%6) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %8 = "ttir.reciprocal"(%7) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %9 = "ttir.multiply"(%2, %8) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %10 = "ttir.multiply"(%arg165, %9) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %11 = "ttir.squeeze"(%10) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %12 = "ttir.matmul"(%11, %arg166) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %13 = "ttir.reshape"(%12) <{shape = [1 : i32, 11 : i32, 32 : i32, 64 : i32]}> : (tensor<11x2048xf32>) -> tensor<1x11x32x64xf32>
    %14 = "ttir.transpose"(%13) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x32x64xf32>) -> tensor<1x32x11x64xf32>
    %15 = "ttir.concat"(%arg2, %arg2) <{dim = -1 : si32}> : (tensor<1x11x32xf32>, tensor<1x11x32xf32>) -> tensor<1x11x64xf32>
    %16 = "ttir.cos"(%15) : (tensor<1x11x64xf32>) -> tensor<1x11x64xf32>
    %17 = "ttir.unsqueeze"(%16) <{dim = 1 : si32}> : (tensor<1x11x64xf32>) -> tensor<1x1x11x64xf32>
    %18 = "ttir.multiply"(%14, %17) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %19 = "ttir.transpose"(%14) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %20 = "ttir.matmul"(%arg3, %19) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %21 = "ttir.transpose"(%20) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %22 = "ttir.multiply"(%21, %arg4) : (tensor<1x32x11x32xf32>, tensor<1xf32>) -> tensor<1x32x11x32xf32>
    %23 = "ttir.transpose"(%14) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %24 = "ttir.matmul"(%arg5, %23) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %25 = "ttir.transpose"(%24) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %26 = "ttir.concat"(%22, %25) <{dim = -1 : si32}> : (tensor<1x32x11x32xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x64xf32>
    %27 = "ttir.sin"(%15) : (tensor<1x11x64xf32>) -> tensor<1x11x64xf32>
    %28 = "ttir.unsqueeze"(%27) <{dim = 1 : si32}> : (tensor<1x11x64xf32>) -> tensor<1x1x11x64xf32>
    %29 = "ttir.multiply"(%26, %28) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %30 = "ttir.add"(%18, %29) : (tensor<1x32x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %31 = "ttir.squeeze"(%30) <{dim = 0 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<32x11x64xf32>
    %32 = "ttir.matmul"(%11, %arg167) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %33 = "ttir.reshape"(%32) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %34 = "ttir.transpose"(%33) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %35 = "ttir.multiply"(%34, %17) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %36 = "ttir.transpose"(%34) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %37 = "ttir.matmul"(%arg6, %36) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %38 = "ttir.transpose"(%37) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %39 = "ttir.multiply"(%38, %arg7) : (tensor<1x8x11x32xf32>, tensor<1xf32>) -> tensor<1x8x11x32xf32>
    %40 = "ttir.transpose"(%34) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %41 = "ttir.matmul"(%arg8, %40) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %42 = "ttir.transpose"(%41) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %43 = "ttir.concat"(%39, %42) <{dim = -1 : si32}> : (tensor<1x8x11x32xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x64xf32>
    %44 = "ttir.multiply"(%43, %28) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %45 = "ttir.add"(%35, %44) : (tensor<1x8x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %46 = "ttir.unsqueeze"(%45) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %47 = "ttir.repeat_interleave"(%46) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %48 = "ttir.repeat_interleave"(%47) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %49 = "ttir.reshape"(%48) <{shape = [32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<32x11x64xf32>
    %50 = "ttir.transpose"(%49) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x11x64xf32>) -> tensor<32x64x11xf32>
    %51 = "ttir.matmul"(%31, %50) : (tensor<32x11x64xf32>, tensor<32x64x11xf32>) -> tensor<32x11x11xf32>
    %52 = "ttir.unsqueeze"(%51) <{dim = 0 : si32}> : (tensor<32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %53 = "ttir.multiply"(%52, %arg9) : (tensor<1x32x11x11xf32>, tensor<1xf32>) -> tensor<1x32x11x11xf32>
    %54 = "ttir.add"(%53, %arg10) : (tensor<1x32x11x11xf32>, tensor<1x1x11x11xf32>) -> tensor<1x32x11x11xf32>
    %55 = "ttir.softmax"(%54) <{dimension = -1 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %56 = "ttir.squeeze"(%55) <{dim = 0 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<32x11x11xf32>
    %57 = "ttir.matmul"(%11, %arg168) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %58 = "ttir.reshape"(%57) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %59 = "ttir.transpose"(%58) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %60 = "ttir.unsqueeze"(%59) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %61 = "ttir.repeat_interleave"(%60) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %62 = "ttir.repeat_interleave"(%61) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %63 = "ttir.reshape"(%62) <{shape = [1 : i32, 32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<1x32x11x64xf32>
    %64 = "ttir.transpose"(%63) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %65 = "ttir.squeeze"(%64) <{dim = 0 : si32}> : (tensor<1x32x64x11xf32>) -> tensor<32x64x11xf32>
    %66 = "ttir.transpose"(%65) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x64x11xf32>) -> tensor<32x11x64xf32>
    %67 = "ttir.matmul"(%56, %66) : (tensor<32x11x11xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %68 = "ttir.unsqueeze"(%67) <{dim = 0 : si32}> : (tensor<32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %69 = "ttir.transpose"(%68) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x11x32x64xf32>
    %70 = "ttir.reshape"(%69) <{shape = [11 : i32, 2048 : i32]}> : (tensor<1x11x32x64xf32>) -> tensor<11x2048xf32>
    %71 = "ttir.matmul"(%70, %arg169) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %72 = "ttir.unsqueeze"(%71) <{dim = 0 : si32}> : (tensor<11x2048xf32>) -> tensor<1x11x2048xf32>
    %73 = "ttir.add"(%1, %72) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %74 = "ttir.multiply"(%73, %73) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %75 = "ttir.mean"(%74) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %76 = "ttir.add"(%75, %arg11) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %77 = "ttir.sqrt"(%76) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %78 = "ttir.reciprocal"(%77) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %79 = "ttir.multiply"(%73, %78) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %80 = "ttir.multiply"(%arg170, %79) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %81 = "ttir.squeeze"(%80) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %82 = "ttir.matmul"(%81, %arg171) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %83 = "ttir.unsqueeze"(%82) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %84 = "ttir.sigmoid"(%83) : (tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %85 = "ttir.multiply"(%83, %84) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %86 = "ttir.matmul"(%81, %arg172) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %87 = "ttir.unsqueeze"(%86) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %88 = "ttir.multiply"(%85, %87) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %89 = "ttir.matmul"(%88, %arg173) : (tensor<1x11x8192xf32>, tensor<8192x2048xf32>) -> tensor<1x11x2048xf32>
    %90 = "ttir.add"(%73, %89) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %91 = "ttir.multiply"(%90, %90) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %92 = "ttir.mean"(%91) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %93 = "ttir.add"(%92, %arg12) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %94 = "ttir.sqrt"(%93) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %95 = "ttir.reciprocal"(%94) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %96 = "ttir.multiply"(%90, %95) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %97 = "ttir.multiply"(%arg174, %96) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %98 = "ttir.squeeze"(%97) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %99 = "ttir.matmul"(%98, %arg175) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %100 = "ttir.reshape"(%99) <{shape = [1 : i32, 11 : i32, 32 : i32, 64 : i32]}> : (tensor<11x2048xf32>) -> tensor<1x11x32x64xf32>
    %101 = "ttir.transpose"(%100) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x32x64xf32>) -> tensor<1x32x11x64xf32>
    %102 = "ttir.multiply"(%101, %17) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %103 = "ttir.transpose"(%101) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %104 = "ttir.matmul"(%arg13, %103) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %105 = "ttir.transpose"(%104) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %106 = "ttir.multiply"(%105, %arg14) : (tensor<1x32x11x32xf32>, tensor<1xf32>) -> tensor<1x32x11x32xf32>
    %107 = "ttir.transpose"(%101) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %108 = "ttir.matmul"(%arg15, %107) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %109 = "ttir.transpose"(%108) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %110 = "ttir.concat"(%106, %109) <{dim = -1 : si32}> : (tensor<1x32x11x32xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x64xf32>
    %111 = "ttir.multiply"(%110, %28) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %112 = "ttir.add"(%102, %111) : (tensor<1x32x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %113 = "ttir.squeeze"(%112) <{dim = 0 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<32x11x64xf32>
    %114 = "ttir.matmul"(%98, %arg176) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %115 = "ttir.reshape"(%114) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %116 = "ttir.transpose"(%115) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %117 = "ttir.multiply"(%116, %17) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %118 = "ttir.transpose"(%116) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %119 = "ttir.matmul"(%arg16, %118) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %120 = "ttir.transpose"(%119) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %121 = "ttir.multiply"(%120, %arg17) : (tensor<1x8x11x32xf32>, tensor<1xf32>) -> tensor<1x8x11x32xf32>
    %122 = "ttir.transpose"(%116) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %123 = "ttir.matmul"(%arg18, %122) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %124 = "ttir.transpose"(%123) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %125 = "ttir.concat"(%121, %124) <{dim = -1 : si32}> : (tensor<1x8x11x32xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x64xf32>
    %126 = "ttir.multiply"(%125, %28) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %127 = "ttir.add"(%117, %126) : (tensor<1x8x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %128 = "ttir.unsqueeze"(%127) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %129 = "ttir.repeat_interleave"(%128) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %130 = "ttir.repeat_interleave"(%129) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %131 = "ttir.reshape"(%130) <{shape = [32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<32x11x64xf32>
    %132 = "ttir.transpose"(%131) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x11x64xf32>) -> tensor<32x64x11xf32>
    %133 = "ttir.matmul"(%113, %132) : (tensor<32x11x64xf32>, tensor<32x64x11xf32>) -> tensor<32x11x11xf32>
    %134 = "ttir.unsqueeze"(%133) <{dim = 0 : si32}> : (tensor<32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %135 = "ttir.multiply"(%134, %arg19) : (tensor<1x32x11x11xf32>, tensor<1xf32>) -> tensor<1x32x11x11xf32>
    %136 = "ttir.add"(%135, %arg20) : (tensor<1x32x11x11xf32>, tensor<1x1x11x11xf32>) -> tensor<1x32x11x11xf32>
    %137 = "ttir.softmax"(%136) <{dimension = -1 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %138 = "ttir.squeeze"(%137) <{dim = 0 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<32x11x11xf32>
    %139 = "ttir.matmul"(%98, %arg177) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %140 = "ttir.reshape"(%139) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %141 = "ttir.transpose"(%140) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %142 = "ttir.unsqueeze"(%141) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %143 = "ttir.repeat_interleave"(%142) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %144 = "ttir.repeat_interleave"(%143) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %145 = "ttir.reshape"(%144) <{shape = [1 : i32, 32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<1x32x11x64xf32>
    %146 = "ttir.transpose"(%145) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %147 = "ttir.squeeze"(%146) <{dim = 0 : si32}> : (tensor<1x32x64x11xf32>) -> tensor<32x64x11xf32>
    %148 = "ttir.transpose"(%147) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x64x11xf32>) -> tensor<32x11x64xf32>
    %149 = "ttir.matmul"(%138, %148) : (tensor<32x11x11xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %150 = "ttir.unsqueeze"(%149) <{dim = 0 : si32}> : (tensor<32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %151 = "ttir.transpose"(%150) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x11x32x64xf32>
    %152 = "ttir.reshape"(%151) <{shape = [11 : i32, 2048 : i32]}> : (tensor<1x11x32x64xf32>) -> tensor<11x2048xf32>
    %153 = "ttir.matmul"(%152, %arg178) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %154 = "ttir.unsqueeze"(%153) <{dim = 0 : si32}> : (tensor<11x2048xf32>) -> tensor<1x11x2048xf32>
    %155 = "ttir.add"(%90, %154) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %156 = "ttir.multiply"(%155, %155) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %157 = "ttir.mean"(%156) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %158 = "ttir.add"(%157, %arg21) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %159 = "ttir.sqrt"(%158) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %160 = "ttir.reciprocal"(%159) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %161 = "ttir.multiply"(%155, %160) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %162 = "ttir.multiply"(%arg179, %161) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %163 = "ttir.squeeze"(%162) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %164 = "ttir.matmul"(%163, %arg180) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %165 = "ttir.unsqueeze"(%164) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %166 = "ttir.sigmoid"(%165) : (tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %167 = "ttir.multiply"(%165, %166) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %168 = "ttir.matmul"(%163, %arg181) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %169 = "ttir.unsqueeze"(%168) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %170 = "ttir.multiply"(%167, %169) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %171 = "ttir.matmul"(%170, %arg182) : (tensor<1x11x8192xf32>, tensor<8192x2048xf32>) -> tensor<1x11x2048xf32>
    %172 = "ttir.add"(%155, %171) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %173 = "ttir.multiply"(%172, %172) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %174 = "ttir.mean"(%173) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %175 = "ttir.add"(%174, %arg22) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %176 = "ttir.sqrt"(%175) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %177 = "ttir.reciprocal"(%176) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %178 = "ttir.multiply"(%172, %177) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %179 = "ttir.multiply"(%arg183, %178) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %180 = "ttir.squeeze"(%179) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %181 = "ttir.matmul"(%180, %arg184) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %182 = "ttir.reshape"(%181) <{shape = [1 : i32, 11 : i32, 32 : i32, 64 : i32]}> : (tensor<11x2048xf32>) -> tensor<1x11x32x64xf32>
    %183 = "ttir.transpose"(%182) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x32x64xf32>) -> tensor<1x32x11x64xf32>
    %184 = "ttir.multiply"(%183, %17) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %185 = "ttir.transpose"(%183) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %186 = "ttir.matmul"(%arg23, %185) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %187 = "ttir.transpose"(%186) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %188 = "ttir.multiply"(%187, %arg24) : (tensor<1x32x11x32xf32>, tensor<1xf32>) -> tensor<1x32x11x32xf32>
    %189 = "ttir.transpose"(%183) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %190 = "ttir.matmul"(%arg25, %189) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %191 = "ttir.transpose"(%190) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %192 = "ttir.concat"(%188, %191) <{dim = -1 : si32}> : (tensor<1x32x11x32xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x64xf32>
    %193 = "ttir.multiply"(%192, %28) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %194 = "ttir.add"(%184, %193) : (tensor<1x32x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %195 = "ttir.squeeze"(%194) <{dim = 0 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<32x11x64xf32>
    %196 = "ttir.matmul"(%180, %arg185) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %197 = "ttir.reshape"(%196) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %198 = "ttir.transpose"(%197) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %199 = "ttir.multiply"(%198, %17) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %200 = "ttir.transpose"(%198) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %201 = "ttir.matmul"(%arg26, %200) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %202 = "ttir.transpose"(%201) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %203 = "ttir.multiply"(%202, %arg27) : (tensor<1x8x11x32xf32>, tensor<1xf32>) -> tensor<1x8x11x32xf32>
    %204 = "ttir.transpose"(%198) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %205 = "ttir.matmul"(%arg28, %204) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %206 = "ttir.transpose"(%205) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %207 = "ttir.concat"(%203, %206) <{dim = -1 : si32}> : (tensor<1x8x11x32xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x64xf32>
    %208 = "ttir.multiply"(%207, %28) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %209 = "ttir.add"(%199, %208) : (tensor<1x8x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %210 = "ttir.unsqueeze"(%209) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %211 = "ttir.repeat_interleave"(%210) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %212 = "ttir.repeat_interleave"(%211) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %213 = "ttir.reshape"(%212) <{shape = [32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<32x11x64xf32>
    %214 = "ttir.transpose"(%213) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x11x64xf32>) -> tensor<32x64x11xf32>
    %215 = "ttir.matmul"(%195, %214) : (tensor<32x11x64xf32>, tensor<32x64x11xf32>) -> tensor<32x11x11xf32>
    %216 = "ttir.unsqueeze"(%215) <{dim = 0 : si32}> : (tensor<32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %217 = "ttir.multiply"(%216, %arg29) : (tensor<1x32x11x11xf32>, tensor<1xf32>) -> tensor<1x32x11x11xf32>
    %218 = "ttir.add"(%217, %arg30) : (tensor<1x32x11x11xf32>, tensor<1x1x11x11xf32>) -> tensor<1x32x11x11xf32>
    %219 = "ttir.softmax"(%218) <{dimension = -1 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %220 = "ttir.squeeze"(%219) <{dim = 0 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<32x11x11xf32>
    %221 = "ttir.matmul"(%180, %arg186) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %222 = "ttir.reshape"(%221) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %223 = "ttir.transpose"(%222) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %224 = "ttir.unsqueeze"(%223) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %225 = "ttir.repeat_interleave"(%224) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %226 = "ttir.repeat_interleave"(%225) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %227 = "ttir.reshape"(%226) <{shape = [1 : i32, 32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<1x32x11x64xf32>
    %228 = "ttir.transpose"(%227) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %229 = "ttir.squeeze"(%228) <{dim = 0 : si32}> : (tensor<1x32x64x11xf32>) -> tensor<32x64x11xf32>
    %230 = "ttir.transpose"(%229) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x64x11xf32>) -> tensor<32x11x64xf32>
    %231 = "ttir.matmul"(%220, %230) : (tensor<32x11x11xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %232 = "ttir.unsqueeze"(%231) <{dim = 0 : si32}> : (tensor<32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %233 = "ttir.transpose"(%232) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x11x32x64xf32>
    %234 = "ttir.reshape"(%233) <{shape = [11 : i32, 2048 : i32]}> : (tensor<1x11x32x64xf32>) -> tensor<11x2048xf32>
    %235 = "ttir.matmul"(%234, %arg187) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %236 = "ttir.unsqueeze"(%235) <{dim = 0 : si32}> : (tensor<11x2048xf32>) -> tensor<1x11x2048xf32>
    %237 = "ttir.add"(%172, %236) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %238 = "ttir.multiply"(%237, %237) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %239 = "ttir.mean"(%238) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %240 = "ttir.add"(%239, %arg31) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %241 = "ttir.sqrt"(%240) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %242 = "ttir.reciprocal"(%241) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %243 = "ttir.multiply"(%237, %242) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %244 = "ttir.multiply"(%arg188, %243) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %245 = "ttir.squeeze"(%244) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %246 = "ttir.matmul"(%245, %arg189) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %247 = "ttir.unsqueeze"(%246) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %248 = "ttir.sigmoid"(%247) : (tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %249 = "ttir.multiply"(%247, %248) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %250 = "ttir.matmul"(%245, %arg190) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %251 = "ttir.unsqueeze"(%250) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %252 = "ttir.multiply"(%249, %251) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %253 = "ttir.matmul"(%252, %arg191) : (tensor<1x11x8192xf32>, tensor<8192x2048xf32>) -> tensor<1x11x2048xf32>
    %254 = "ttir.add"(%237, %253) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %255 = "ttir.multiply"(%254, %254) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %256 = "ttir.mean"(%255) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %257 = "ttir.add"(%256, %arg32) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %258 = "ttir.sqrt"(%257) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %259 = "ttir.reciprocal"(%258) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %260 = "ttir.multiply"(%254, %259) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %261 = "ttir.multiply"(%arg192, %260) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %262 = "ttir.squeeze"(%261) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %263 = "ttir.matmul"(%262, %arg193) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %264 = "ttir.reshape"(%263) <{shape = [1 : i32, 11 : i32, 32 : i32, 64 : i32]}> : (tensor<11x2048xf32>) -> tensor<1x11x32x64xf32>
    %265 = "ttir.transpose"(%264) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x32x64xf32>) -> tensor<1x32x11x64xf32>
    %266 = "ttir.multiply"(%265, %17) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %267 = "ttir.transpose"(%265) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %268 = "ttir.matmul"(%arg33, %267) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %269 = "ttir.transpose"(%268) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %270 = "ttir.multiply"(%269, %arg34) : (tensor<1x32x11x32xf32>, tensor<1xf32>) -> tensor<1x32x11x32xf32>
    %271 = "ttir.transpose"(%265) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %272 = "ttir.matmul"(%arg35, %271) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %273 = "ttir.transpose"(%272) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %274 = "ttir.concat"(%270, %273) <{dim = -1 : si32}> : (tensor<1x32x11x32xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x64xf32>
    %275 = "ttir.multiply"(%274, %28) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %276 = "ttir.add"(%266, %275) : (tensor<1x32x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %277 = "ttir.squeeze"(%276) <{dim = 0 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<32x11x64xf32>
    %278 = "ttir.matmul"(%262, %arg194) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %279 = "ttir.reshape"(%278) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %280 = "ttir.transpose"(%279) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %281 = "ttir.multiply"(%280, %17) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %282 = "ttir.transpose"(%280) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %283 = "ttir.matmul"(%arg36, %282) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %284 = "ttir.transpose"(%283) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %285 = "ttir.multiply"(%284, %arg37) : (tensor<1x8x11x32xf32>, tensor<1xf32>) -> tensor<1x8x11x32xf32>
    %286 = "ttir.transpose"(%280) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %287 = "ttir.matmul"(%arg38, %286) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %288 = "ttir.transpose"(%287) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %289 = "ttir.concat"(%285, %288) <{dim = -1 : si32}> : (tensor<1x8x11x32xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x64xf32>
    %290 = "ttir.multiply"(%289, %28) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %291 = "ttir.add"(%281, %290) : (tensor<1x8x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %292 = "ttir.unsqueeze"(%291) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %293 = "ttir.repeat_interleave"(%292) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %294 = "ttir.repeat_interleave"(%293) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %295 = "ttir.reshape"(%294) <{shape = [32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<32x11x64xf32>
    %296 = "ttir.transpose"(%295) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x11x64xf32>) -> tensor<32x64x11xf32>
    %297 = "ttir.matmul"(%277, %296) : (tensor<32x11x64xf32>, tensor<32x64x11xf32>) -> tensor<32x11x11xf32>
    %298 = "ttir.unsqueeze"(%297) <{dim = 0 : si32}> : (tensor<32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %299 = "ttir.multiply"(%298, %arg39) : (tensor<1x32x11x11xf32>, tensor<1xf32>) -> tensor<1x32x11x11xf32>
    %300 = "ttir.add"(%299, %arg40) : (tensor<1x32x11x11xf32>, tensor<1x1x11x11xf32>) -> tensor<1x32x11x11xf32>
    %301 = "ttir.softmax"(%300) <{dimension = -1 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %302 = "ttir.squeeze"(%301) <{dim = 0 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<32x11x11xf32>
    %303 = "ttir.matmul"(%262, %arg195) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %304 = "ttir.reshape"(%303) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %305 = "ttir.transpose"(%304) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %306 = "ttir.unsqueeze"(%305) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %307 = "ttir.repeat_interleave"(%306) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %308 = "ttir.repeat_interleave"(%307) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %309 = "ttir.reshape"(%308) <{shape = [1 : i32, 32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<1x32x11x64xf32>
    %310 = "ttir.transpose"(%309) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %311 = "ttir.squeeze"(%310) <{dim = 0 : si32}> : (tensor<1x32x64x11xf32>) -> tensor<32x64x11xf32>
    %312 = "ttir.transpose"(%311) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x64x11xf32>) -> tensor<32x11x64xf32>
    %313 = "ttir.matmul"(%302, %312) : (tensor<32x11x11xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %314 = "ttir.unsqueeze"(%313) <{dim = 0 : si32}> : (tensor<32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %315 = "ttir.transpose"(%314) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x11x32x64xf32>
    %316 = "ttir.reshape"(%315) <{shape = [11 : i32, 2048 : i32]}> : (tensor<1x11x32x64xf32>) -> tensor<11x2048xf32>
    %317 = "ttir.matmul"(%316, %arg196) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %318 = "ttir.unsqueeze"(%317) <{dim = 0 : si32}> : (tensor<11x2048xf32>) -> tensor<1x11x2048xf32>
    %319 = "ttir.add"(%254, %318) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %320 = "ttir.multiply"(%319, %319) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %321 = "ttir.mean"(%320) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %322 = "ttir.add"(%321, %arg41) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %323 = "ttir.sqrt"(%322) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %324 = "ttir.reciprocal"(%323) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %325 = "ttir.multiply"(%319, %324) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %326 = "ttir.multiply"(%arg197, %325) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %327 = "ttir.squeeze"(%326) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %328 = "ttir.matmul"(%327, %arg198) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %329 = "ttir.unsqueeze"(%328) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %330 = "ttir.sigmoid"(%329) : (tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %331 = "ttir.multiply"(%329, %330) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %332 = "ttir.matmul"(%327, %arg199) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %333 = "ttir.unsqueeze"(%332) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %334 = "ttir.multiply"(%331, %333) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %335 = "ttir.matmul"(%334, %arg200) : (tensor<1x11x8192xf32>, tensor<8192x2048xf32>) -> tensor<1x11x2048xf32>
    %336 = "ttir.add"(%319, %335) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %337 = "ttir.multiply"(%336, %336) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %338 = "ttir.mean"(%337) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %339 = "ttir.add"(%338, %arg42) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %340 = "ttir.sqrt"(%339) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %341 = "ttir.reciprocal"(%340) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %342 = "ttir.multiply"(%336, %341) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %343 = "ttir.multiply"(%arg201, %342) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %344 = "ttir.squeeze"(%343) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %345 = "ttir.matmul"(%344, %arg202) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %346 = "ttir.reshape"(%345) <{shape = [1 : i32, 11 : i32, 32 : i32, 64 : i32]}> : (tensor<11x2048xf32>) -> tensor<1x11x32x64xf32>
    %347 = "ttir.transpose"(%346) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x32x64xf32>) -> tensor<1x32x11x64xf32>
    %348 = "ttir.multiply"(%347, %17) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %349 = "ttir.transpose"(%347) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %350 = "ttir.matmul"(%arg43, %349) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %351 = "ttir.transpose"(%350) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %352 = "ttir.multiply"(%351, %arg44) : (tensor<1x32x11x32xf32>, tensor<1xf32>) -> tensor<1x32x11x32xf32>
    %353 = "ttir.transpose"(%347) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %354 = "ttir.matmul"(%arg45, %353) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %355 = "ttir.transpose"(%354) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %356 = "ttir.concat"(%352, %355) <{dim = -1 : si32}> : (tensor<1x32x11x32xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x64xf32>
    %357 = "ttir.multiply"(%356, %28) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %358 = "ttir.add"(%348, %357) : (tensor<1x32x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %359 = "ttir.squeeze"(%358) <{dim = 0 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<32x11x64xf32>
    %360 = "ttir.matmul"(%344, %arg203) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %361 = "ttir.reshape"(%360) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %362 = "ttir.transpose"(%361) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %363 = "ttir.multiply"(%362, %17) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %364 = "ttir.transpose"(%362) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %365 = "ttir.matmul"(%arg46, %364) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %366 = "ttir.transpose"(%365) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %367 = "ttir.multiply"(%366, %arg47) : (tensor<1x8x11x32xf32>, tensor<1xf32>) -> tensor<1x8x11x32xf32>
    %368 = "ttir.transpose"(%362) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %369 = "ttir.matmul"(%arg48, %368) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %370 = "ttir.transpose"(%369) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %371 = "ttir.concat"(%367, %370) <{dim = -1 : si32}> : (tensor<1x8x11x32xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x64xf32>
    %372 = "ttir.multiply"(%371, %28) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %373 = "ttir.add"(%363, %372) : (tensor<1x8x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %374 = "ttir.unsqueeze"(%373) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %375 = "ttir.repeat_interleave"(%374) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %376 = "ttir.repeat_interleave"(%375) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %377 = "ttir.reshape"(%376) <{shape = [32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<32x11x64xf32>
    %378 = "ttir.transpose"(%377) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x11x64xf32>) -> tensor<32x64x11xf32>
    %379 = "ttir.matmul"(%359, %378) : (tensor<32x11x64xf32>, tensor<32x64x11xf32>) -> tensor<32x11x11xf32>
    %380 = "ttir.unsqueeze"(%379) <{dim = 0 : si32}> : (tensor<32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %381 = "ttir.multiply"(%380, %arg49) : (tensor<1x32x11x11xf32>, tensor<1xf32>) -> tensor<1x32x11x11xf32>
    %382 = "ttir.add"(%381, %arg50) : (tensor<1x32x11x11xf32>, tensor<1x1x11x11xf32>) -> tensor<1x32x11x11xf32>
    %383 = "ttir.softmax"(%382) <{dimension = -1 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %384 = "ttir.squeeze"(%383) <{dim = 0 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<32x11x11xf32>
    %385 = "ttir.matmul"(%344, %arg204) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %386 = "ttir.reshape"(%385) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %387 = "ttir.transpose"(%386) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %388 = "ttir.unsqueeze"(%387) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %389 = "ttir.repeat_interleave"(%388) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %390 = "ttir.repeat_interleave"(%389) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %391 = "ttir.reshape"(%390) <{shape = [1 : i32, 32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<1x32x11x64xf32>
    %392 = "ttir.transpose"(%391) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %393 = "ttir.squeeze"(%392) <{dim = 0 : si32}> : (tensor<1x32x64x11xf32>) -> tensor<32x64x11xf32>
    %394 = "ttir.transpose"(%393) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x64x11xf32>) -> tensor<32x11x64xf32>
    %395 = "ttir.matmul"(%384, %394) : (tensor<32x11x11xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %396 = "ttir.unsqueeze"(%395) <{dim = 0 : si32}> : (tensor<32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %397 = "ttir.transpose"(%396) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x11x32x64xf32>
    %398 = "ttir.reshape"(%397) <{shape = [11 : i32, 2048 : i32]}> : (tensor<1x11x32x64xf32>) -> tensor<11x2048xf32>
    %399 = "ttir.matmul"(%398, %arg205) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %400 = "ttir.unsqueeze"(%399) <{dim = 0 : si32}> : (tensor<11x2048xf32>) -> tensor<1x11x2048xf32>
    %401 = "ttir.add"(%336, %400) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %402 = "ttir.multiply"(%401, %401) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %403 = "ttir.mean"(%402) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %404 = "ttir.add"(%403, %arg51) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %405 = "ttir.sqrt"(%404) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %406 = "ttir.reciprocal"(%405) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %407 = "ttir.multiply"(%401, %406) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %408 = "ttir.multiply"(%arg206, %407) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %409 = "ttir.squeeze"(%408) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %410 = "ttir.matmul"(%409, %arg207) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %411 = "ttir.unsqueeze"(%410) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %412 = "ttir.sigmoid"(%411) : (tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %413 = "ttir.multiply"(%411, %412) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %414 = "ttir.matmul"(%409, %arg208) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %415 = "ttir.unsqueeze"(%414) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %416 = "ttir.multiply"(%413, %415) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %417 = "ttir.matmul"(%416, %arg209) : (tensor<1x11x8192xf32>, tensor<8192x2048xf32>) -> tensor<1x11x2048xf32>
    %418 = "ttir.add"(%401, %417) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %419 = "ttir.multiply"(%418, %418) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %420 = "ttir.mean"(%419) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %421 = "ttir.add"(%420, %arg52) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %422 = "ttir.sqrt"(%421) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %423 = "ttir.reciprocal"(%422) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %424 = "ttir.multiply"(%418, %423) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %425 = "ttir.multiply"(%arg210, %424) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %426 = "ttir.squeeze"(%425) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %427 = "ttir.matmul"(%426, %arg211) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %428 = "ttir.reshape"(%427) <{shape = [1 : i32, 11 : i32, 32 : i32, 64 : i32]}> : (tensor<11x2048xf32>) -> tensor<1x11x32x64xf32>
    %429 = "ttir.transpose"(%428) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x32x64xf32>) -> tensor<1x32x11x64xf32>
    %430 = "ttir.multiply"(%429, %17) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %431 = "ttir.transpose"(%429) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %432 = "ttir.matmul"(%arg53, %431) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %433 = "ttir.transpose"(%432) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %434 = "ttir.multiply"(%433, %arg54) : (tensor<1x32x11x32xf32>, tensor<1xf32>) -> tensor<1x32x11x32xf32>
    %435 = "ttir.transpose"(%429) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %436 = "ttir.matmul"(%arg55, %435) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %437 = "ttir.transpose"(%436) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %438 = "ttir.concat"(%434, %437) <{dim = -1 : si32}> : (tensor<1x32x11x32xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x64xf32>
    %439 = "ttir.multiply"(%438, %28) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %440 = "ttir.add"(%430, %439) : (tensor<1x32x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %441 = "ttir.squeeze"(%440) <{dim = 0 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<32x11x64xf32>
    %442 = "ttir.matmul"(%426, %arg212) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %443 = "ttir.reshape"(%442) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %444 = "ttir.transpose"(%443) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %445 = "ttir.multiply"(%444, %17) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %446 = "ttir.transpose"(%444) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %447 = "ttir.matmul"(%arg56, %446) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %448 = "ttir.transpose"(%447) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %449 = "ttir.multiply"(%448, %arg57) : (tensor<1x8x11x32xf32>, tensor<1xf32>) -> tensor<1x8x11x32xf32>
    %450 = "ttir.transpose"(%444) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %451 = "ttir.matmul"(%arg58, %450) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %452 = "ttir.transpose"(%451) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %453 = "ttir.concat"(%449, %452) <{dim = -1 : si32}> : (tensor<1x8x11x32xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x64xf32>
    %454 = "ttir.multiply"(%453, %28) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %455 = "ttir.add"(%445, %454) : (tensor<1x8x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %456 = "ttir.unsqueeze"(%455) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %457 = "ttir.repeat_interleave"(%456) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %458 = "ttir.repeat_interleave"(%457) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %459 = "ttir.reshape"(%458) <{shape = [32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<32x11x64xf32>
    %460 = "ttir.transpose"(%459) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x11x64xf32>) -> tensor<32x64x11xf32>
    %461 = "ttir.matmul"(%441, %460) : (tensor<32x11x64xf32>, tensor<32x64x11xf32>) -> tensor<32x11x11xf32>
    %462 = "ttir.unsqueeze"(%461) <{dim = 0 : si32}> : (tensor<32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %463 = "ttir.multiply"(%462, %arg59) : (tensor<1x32x11x11xf32>, tensor<1xf32>) -> tensor<1x32x11x11xf32>
    %464 = "ttir.add"(%463, %arg60) : (tensor<1x32x11x11xf32>, tensor<1x1x11x11xf32>) -> tensor<1x32x11x11xf32>
    %465 = "ttir.softmax"(%464) <{dimension = -1 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %466 = "ttir.squeeze"(%465) <{dim = 0 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<32x11x11xf32>
    %467 = "ttir.matmul"(%426, %arg213) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %468 = "ttir.reshape"(%467) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %469 = "ttir.transpose"(%468) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %470 = "ttir.unsqueeze"(%469) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %471 = "ttir.repeat_interleave"(%470) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %472 = "ttir.repeat_interleave"(%471) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %473 = "ttir.reshape"(%472) <{shape = [1 : i32, 32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<1x32x11x64xf32>
    %474 = "ttir.transpose"(%473) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %475 = "ttir.squeeze"(%474) <{dim = 0 : si32}> : (tensor<1x32x64x11xf32>) -> tensor<32x64x11xf32>
    %476 = "ttir.transpose"(%475) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x64x11xf32>) -> tensor<32x11x64xf32>
    %477 = "ttir.matmul"(%466, %476) : (tensor<32x11x11xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %478 = "ttir.unsqueeze"(%477) <{dim = 0 : si32}> : (tensor<32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %479 = "ttir.transpose"(%478) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x11x32x64xf32>
    %480 = "ttir.reshape"(%479) <{shape = [11 : i32, 2048 : i32]}> : (tensor<1x11x32x64xf32>) -> tensor<11x2048xf32>
    %481 = "ttir.matmul"(%480, %arg214) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %482 = "ttir.unsqueeze"(%481) <{dim = 0 : si32}> : (tensor<11x2048xf32>) -> tensor<1x11x2048xf32>
    %483 = "ttir.add"(%418, %482) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %484 = "ttir.multiply"(%483, %483) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %485 = "ttir.mean"(%484) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %486 = "ttir.add"(%485, %arg61) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %487 = "ttir.sqrt"(%486) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %488 = "ttir.reciprocal"(%487) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %489 = "ttir.multiply"(%483, %488) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %490 = "ttir.multiply"(%arg215, %489) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %491 = "ttir.squeeze"(%490) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %492 = "ttir.matmul"(%491, %arg216) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %493 = "ttir.unsqueeze"(%492) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %494 = "ttir.sigmoid"(%493) : (tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %495 = "ttir.multiply"(%493, %494) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %496 = "ttir.matmul"(%491, %arg217) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %497 = "ttir.unsqueeze"(%496) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %498 = "ttir.multiply"(%495, %497) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %499 = "ttir.matmul"(%498, %arg218) : (tensor<1x11x8192xf32>, tensor<8192x2048xf32>) -> tensor<1x11x2048xf32>
    %500 = "ttir.add"(%483, %499) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %501 = "ttir.multiply"(%500, %500) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %502 = "ttir.mean"(%501) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %503 = "ttir.add"(%502, %arg62) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %504 = "ttir.sqrt"(%503) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %505 = "ttir.reciprocal"(%504) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %506 = "ttir.multiply"(%500, %505) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %507 = "ttir.multiply"(%arg219, %506) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %508 = "ttir.squeeze"(%507) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %509 = "ttir.matmul"(%508, %arg220) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %510 = "ttir.reshape"(%509) <{shape = [1 : i32, 11 : i32, 32 : i32, 64 : i32]}> : (tensor<11x2048xf32>) -> tensor<1x11x32x64xf32>
    %511 = "ttir.transpose"(%510) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x32x64xf32>) -> tensor<1x32x11x64xf32>
    %512 = "ttir.multiply"(%511, %17) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %513 = "ttir.transpose"(%511) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %514 = "ttir.matmul"(%arg63, %513) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %515 = "ttir.transpose"(%514) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %516 = "ttir.multiply"(%515, %arg64) : (tensor<1x32x11x32xf32>, tensor<1xf32>) -> tensor<1x32x11x32xf32>
    %517 = "ttir.transpose"(%511) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %518 = "ttir.matmul"(%arg65, %517) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %519 = "ttir.transpose"(%518) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %520 = "ttir.concat"(%516, %519) <{dim = -1 : si32}> : (tensor<1x32x11x32xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x64xf32>
    %521 = "ttir.multiply"(%520, %28) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %522 = "ttir.add"(%512, %521) : (tensor<1x32x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %523 = "ttir.squeeze"(%522) <{dim = 0 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<32x11x64xf32>
    %524 = "ttir.matmul"(%508, %arg221) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %525 = "ttir.reshape"(%524) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %526 = "ttir.transpose"(%525) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %527 = "ttir.multiply"(%526, %17) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %528 = "ttir.transpose"(%526) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %529 = "ttir.matmul"(%arg66, %528) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %530 = "ttir.transpose"(%529) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %531 = "ttir.multiply"(%530, %arg67) : (tensor<1x8x11x32xf32>, tensor<1xf32>) -> tensor<1x8x11x32xf32>
    %532 = "ttir.transpose"(%526) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %533 = "ttir.matmul"(%arg68, %532) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %534 = "ttir.transpose"(%533) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %535 = "ttir.concat"(%531, %534) <{dim = -1 : si32}> : (tensor<1x8x11x32xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x64xf32>
    %536 = "ttir.multiply"(%535, %28) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %537 = "ttir.add"(%527, %536) : (tensor<1x8x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %538 = "ttir.unsqueeze"(%537) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %539 = "ttir.repeat_interleave"(%538) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %540 = "ttir.repeat_interleave"(%539) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %541 = "ttir.reshape"(%540) <{shape = [32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<32x11x64xf32>
    %542 = "ttir.transpose"(%541) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x11x64xf32>) -> tensor<32x64x11xf32>
    %543 = "ttir.matmul"(%523, %542) : (tensor<32x11x64xf32>, tensor<32x64x11xf32>) -> tensor<32x11x11xf32>
    %544 = "ttir.unsqueeze"(%543) <{dim = 0 : si32}> : (tensor<32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %545 = "ttir.multiply"(%544, %arg69) : (tensor<1x32x11x11xf32>, tensor<1xf32>) -> tensor<1x32x11x11xf32>
    %546 = "ttir.add"(%545, %arg70) : (tensor<1x32x11x11xf32>, tensor<1x1x11x11xf32>) -> tensor<1x32x11x11xf32>
    %547 = "ttir.softmax"(%546) <{dimension = -1 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %548 = "ttir.squeeze"(%547) <{dim = 0 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<32x11x11xf32>
    %549 = "ttir.matmul"(%508, %arg222) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %550 = "ttir.reshape"(%549) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %551 = "ttir.transpose"(%550) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %552 = "ttir.unsqueeze"(%551) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %553 = "ttir.repeat_interleave"(%552) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %554 = "ttir.repeat_interleave"(%553) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %555 = "ttir.reshape"(%554) <{shape = [1 : i32, 32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<1x32x11x64xf32>
    %556 = "ttir.transpose"(%555) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %557 = "ttir.squeeze"(%556) <{dim = 0 : si32}> : (tensor<1x32x64x11xf32>) -> tensor<32x64x11xf32>
    %558 = "ttir.transpose"(%557) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x64x11xf32>) -> tensor<32x11x64xf32>
    %559 = "ttir.matmul"(%548, %558) : (tensor<32x11x11xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %560 = "ttir.unsqueeze"(%559) <{dim = 0 : si32}> : (tensor<32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %561 = "ttir.transpose"(%560) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x11x32x64xf32>
    %562 = "ttir.reshape"(%561) <{shape = [11 : i32, 2048 : i32]}> : (tensor<1x11x32x64xf32>) -> tensor<11x2048xf32>
    %563 = "ttir.matmul"(%562, %arg223) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %564 = "ttir.unsqueeze"(%563) <{dim = 0 : si32}> : (tensor<11x2048xf32>) -> tensor<1x11x2048xf32>
    %565 = "ttir.add"(%500, %564) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %566 = "ttir.multiply"(%565, %565) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %567 = "ttir.mean"(%566) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %568 = "ttir.add"(%567, %arg71) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %569 = "ttir.sqrt"(%568) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %570 = "ttir.reciprocal"(%569) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %571 = "ttir.multiply"(%565, %570) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %572 = "ttir.multiply"(%arg224, %571) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %573 = "ttir.squeeze"(%572) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %574 = "ttir.matmul"(%573, %arg225) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %575 = "ttir.unsqueeze"(%574) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %576 = "ttir.sigmoid"(%575) : (tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %577 = "ttir.multiply"(%575, %576) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %578 = "ttir.matmul"(%573, %arg226) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %579 = "ttir.unsqueeze"(%578) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %580 = "ttir.multiply"(%577, %579) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %581 = "ttir.matmul"(%580, %arg227) : (tensor<1x11x8192xf32>, tensor<8192x2048xf32>) -> tensor<1x11x2048xf32>
    %582 = "ttir.add"(%565, %581) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %583 = "ttir.multiply"(%582, %582) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %584 = "ttir.mean"(%583) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %585 = "ttir.add"(%584, %arg72) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %586 = "ttir.sqrt"(%585) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %587 = "ttir.reciprocal"(%586) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %588 = "ttir.multiply"(%582, %587) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %589 = "ttir.multiply"(%arg228, %588) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %590 = "ttir.squeeze"(%589) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %591 = "ttir.matmul"(%590, %arg229) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %592 = "ttir.reshape"(%591) <{shape = [1 : i32, 11 : i32, 32 : i32, 64 : i32]}> : (tensor<11x2048xf32>) -> tensor<1x11x32x64xf32>
    %593 = "ttir.transpose"(%592) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x32x64xf32>) -> tensor<1x32x11x64xf32>
    %594 = "ttir.multiply"(%593, %17) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %595 = "ttir.transpose"(%593) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %596 = "ttir.matmul"(%arg73, %595) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %597 = "ttir.transpose"(%596) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %598 = "ttir.multiply"(%597, %arg74) : (tensor<1x32x11x32xf32>, tensor<1xf32>) -> tensor<1x32x11x32xf32>
    %599 = "ttir.transpose"(%593) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %600 = "ttir.matmul"(%arg75, %599) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %601 = "ttir.transpose"(%600) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %602 = "ttir.concat"(%598, %601) <{dim = -1 : si32}> : (tensor<1x32x11x32xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x64xf32>
    %603 = "ttir.multiply"(%602, %28) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %604 = "ttir.add"(%594, %603) : (tensor<1x32x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %605 = "ttir.squeeze"(%604) <{dim = 0 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<32x11x64xf32>
    %606 = "ttir.matmul"(%590, %arg230) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %607 = "ttir.reshape"(%606) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %608 = "ttir.transpose"(%607) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %609 = "ttir.multiply"(%608, %17) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %610 = "ttir.transpose"(%608) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %611 = "ttir.matmul"(%arg76, %610) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %612 = "ttir.transpose"(%611) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %613 = "ttir.multiply"(%612, %arg77) : (tensor<1x8x11x32xf32>, tensor<1xf32>) -> tensor<1x8x11x32xf32>
    %614 = "ttir.transpose"(%608) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %615 = "ttir.matmul"(%arg78, %614) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %616 = "ttir.transpose"(%615) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %617 = "ttir.concat"(%613, %616) <{dim = -1 : si32}> : (tensor<1x8x11x32xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x64xf32>
    %618 = "ttir.multiply"(%617, %28) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %619 = "ttir.add"(%609, %618) : (tensor<1x8x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %620 = "ttir.unsqueeze"(%619) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %621 = "ttir.repeat_interleave"(%620) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %622 = "ttir.repeat_interleave"(%621) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %623 = "ttir.reshape"(%622) <{shape = [32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<32x11x64xf32>
    %624 = "ttir.transpose"(%623) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x11x64xf32>) -> tensor<32x64x11xf32>
    %625 = "ttir.matmul"(%605, %624) : (tensor<32x11x64xf32>, tensor<32x64x11xf32>) -> tensor<32x11x11xf32>
    %626 = "ttir.unsqueeze"(%625) <{dim = 0 : si32}> : (tensor<32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %627 = "ttir.multiply"(%626, %arg79) : (tensor<1x32x11x11xf32>, tensor<1xf32>) -> tensor<1x32x11x11xf32>
    %628 = "ttir.add"(%627, %arg80) : (tensor<1x32x11x11xf32>, tensor<1x1x11x11xf32>) -> tensor<1x32x11x11xf32>
    %629 = "ttir.softmax"(%628) <{dimension = -1 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %630 = "ttir.squeeze"(%629) <{dim = 0 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<32x11x11xf32>
    %631 = "ttir.matmul"(%590, %arg231) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %632 = "ttir.reshape"(%631) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %633 = "ttir.transpose"(%632) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %634 = "ttir.unsqueeze"(%633) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %635 = "ttir.repeat_interleave"(%634) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %636 = "ttir.repeat_interleave"(%635) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %637 = "ttir.reshape"(%636) <{shape = [1 : i32, 32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<1x32x11x64xf32>
    %638 = "ttir.transpose"(%637) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %639 = "ttir.squeeze"(%638) <{dim = 0 : si32}> : (tensor<1x32x64x11xf32>) -> tensor<32x64x11xf32>
    %640 = "ttir.transpose"(%639) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x64x11xf32>) -> tensor<32x11x64xf32>
    %641 = "ttir.matmul"(%630, %640) : (tensor<32x11x11xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %642 = "ttir.unsqueeze"(%641) <{dim = 0 : si32}> : (tensor<32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %643 = "ttir.transpose"(%642) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x11x32x64xf32>
    %644 = "ttir.reshape"(%643) <{shape = [11 : i32, 2048 : i32]}> : (tensor<1x11x32x64xf32>) -> tensor<11x2048xf32>
    %645 = "ttir.matmul"(%644, %arg232) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %646 = "ttir.unsqueeze"(%645) <{dim = 0 : si32}> : (tensor<11x2048xf32>) -> tensor<1x11x2048xf32>
    %647 = "ttir.add"(%582, %646) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %648 = "ttir.multiply"(%647, %647) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %649 = "ttir.mean"(%648) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %650 = "ttir.add"(%649, %arg81) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %651 = "ttir.sqrt"(%650) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %652 = "ttir.reciprocal"(%651) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %653 = "ttir.multiply"(%647, %652) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %654 = "ttir.multiply"(%arg233, %653) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %655 = "ttir.squeeze"(%654) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %656 = "ttir.matmul"(%655, %arg234) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %657 = "ttir.unsqueeze"(%656) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %658 = "ttir.sigmoid"(%657) : (tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %659 = "ttir.multiply"(%657, %658) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %660 = "ttir.matmul"(%655, %arg235) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %661 = "ttir.unsqueeze"(%660) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %662 = "ttir.multiply"(%659, %661) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %663 = "ttir.matmul"(%662, %arg236) : (tensor<1x11x8192xf32>, tensor<8192x2048xf32>) -> tensor<1x11x2048xf32>
    %664 = "ttir.add"(%647, %663) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %665 = "ttir.multiply"(%664, %664) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %666 = "ttir.mean"(%665) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %667 = "ttir.add"(%666, %arg82) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %668 = "ttir.sqrt"(%667) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %669 = "ttir.reciprocal"(%668) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %670 = "ttir.multiply"(%664, %669) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %671 = "ttir.multiply"(%arg237, %670) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %672 = "ttir.squeeze"(%671) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %673 = "ttir.matmul"(%672, %arg238) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %674 = "ttir.reshape"(%673) <{shape = [1 : i32, 11 : i32, 32 : i32, 64 : i32]}> : (tensor<11x2048xf32>) -> tensor<1x11x32x64xf32>
    %675 = "ttir.transpose"(%674) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x32x64xf32>) -> tensor<1x32x11x64xf32>
    %676 = "ttir.multiply"(%675, %17) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %677 = "ttir.transpose"(%675) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %678 = "ttir.matmul"(%arg83, %677) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %679 = "ttir.transpose"(%678) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %680 = "ttir.multiply"(%679, %arg84) : (tensor<1x32x11x32xf32>, tensor<1xf32>) -> tensor<1x32x11x32xf32>
    %681 = "ttir.transpose"(%675) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %682 = "ttir.matmul"(%arg85, %681) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %683 = "ttir.transpose"(%682) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %684 = "ttir.concat"(%680, %683) <{dim = -1 : si32}> : (tensor<1x32x11x32xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x64xf32>
    %685 = "ttir.multiply"(%684, %28) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %686 = "ttir.add"(%676, %685) : (tensor<1x32x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %687 = "ttir.squeeze"(%686) <{dim = 0 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<32x11x64xf32>
    %688 = "ttir.matmul"(%672, %arg239) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %689 = "ttir.reshape"(%688) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %690 = "ttir.transpose"(%689) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %691 = "ttir.multiply"(%690, %17) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %692 = "ttir.transpose"(%690) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %693 = "ttir.matmul"(%arg86, %692) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %694 = "ttir.transpose"(%693) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %695 = "ttir.multiply"(%694, %arg87) : (tensor<1x8x11x32xf32>, tensor<1xf32>) -> tensor<1x8x11x32xf32>
    %696 = "ttir.transpose"(%690) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %697 = "ttir.matmul"(%arg88, %696) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %698 = "ttir.transpose"(%697) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %699 = "ttir.concat"(%695, %698) <{dim = -1 : si32}> : (tensor<1x8x11x32xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x64xf32>
    %700 = "ttir.multiply"(%699, %28) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %701 = "ttir.add"(%691, %700) : (tensor<1x8x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %702 = "ttir.unsqueeze"(%701) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %703 = "ttir.repeat_interleave"(%702) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %704 = "ttir.repeat_interleave"(%703) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %705 = "ttir.reshape"(%704) <{shape = [32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<32x11x64xf32>
    %706 = "ttir.transpose"(%705) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x11x64xf32>) -> tensor<32x64x11xf32>
    %707 = "ttir.matmul"(%687, %706) : (tensor<32x11x64xf32>, tensor<32x64x11xf32>) -> tensor<32x11x11xf32>
    %708 = "ttir.unsqueeze"(%707) <{dim = 0 : si32}> : (tensor<32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %709 = "ttir.multiply"(%708, %arg89) : (tensor<1x32x11x11xf32>, tensor<1xf32>) -> tensor<1x32x11x11xf32>
    %710 = "ttir.add"(%709, %arg90) : (tensor<1x32x11x11xf32>, tensor<1x1x11x11xf32>) -> tensor<1x32x11x11xf32>
    %711 = "ttir.softmax"(%710) <{dimension = -1 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %712 = "ttir.squeeze"(%711) <{dim = 0 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<32x11x11xf32>
    %713 = "ttir.matmul"(%672, %arg240) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %714 = "ttir.reshape"(%713) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %715 = "ttir.transpose"(%714) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %716 = "ttir.unsqueeze"(%715) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %717 = "ttir.repeat_interleave"(%716) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %718 = "ttir.repeat_interleave"(%717) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %719 = "ttir.reshape"(%718) <{shape = [1 : i32, 32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<1x32x11x64xf32>
    %720 = "ttir.transpose"(%719) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %721 = "ttir.squeeze"(%720) <{dim = 0 : si32}> : (tensor<1x32x64x11xf32>) -> tensor<32x64x11xf32>
    %722 = "ttir.transpose"(%721) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x64x11xf32>) -> tensor<32x11x64xf32>
    %723 = "ttir.matmul"(%712, %722) : (tensor<32x11x11xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %724 = "ttir.unsqueeze"(%723) <{dim = 0 : si32}> : (tensor<32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %725 = "ttir.transpose"(%724) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x11x32x64xf32>
    %726 = "ttir.reshape"(%725) <{shape = [11 : i32, 2048 : i32]}> : (tensor<1x11x32x64xf32>) -> tensor<11x2048xf32>
    %727 = "ttir.matmul"(%726, %arg241) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %728 = "ttir.unsqueeze"(%727) <{dim = 0 : si32}> : (tensor<11x2048xf32>) -> tensor<1x11x2048xf32>
    %729 = "ttir.add"(%664, %728) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %730 = "ttir.multiply"(%729, %729) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %731 = "ttir.mean"(%730) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %732 = "ttir.add"(%731, %arg91) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %733 = "ttir.sqrt"(%732) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %734 = "ttir.reciprocal"(%733) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %735 = "ttir.multiply"(%729, %734) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %736 = "ttir.multiply"(%arg242, %735) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %737 = "ttir.squeeze"(%736) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %738 = "ttir.matmul"(%737, %arg243) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %739 = "ttir.unsqueeze"(%738) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %740 = "ttir.sigmoid"(%739) : (tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %741 = "ttir.multiply"(%739, %740) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %742 = "ttir.matmul"(%737, %arg244) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %743 = "ttir.unsqueeze"(%742) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %744 = "ttir.multiply"(%741, %743) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %745 = "ttir.matmul"(%744, %arg245) : (tensor<1x11x8192xf32>, tensor<8192x2048xf32>) -> tensor<1x11x2048xf32>
    %746 = "ttir.add"(%729, %745) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %747 = "ttir.multiply"(%746, %746) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %748 = "ttir.mean"(%747) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %749 = "ttir.add"(%748, %arg92) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %750 = "ttir.sqrt"(%749) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %751 = "ttir.reciprocal"(%750) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %752 = "ttir.multiply"(%746, %751) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %753 = "ttir.multiply"(%arg246, %752) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %754 = "ttir.squeeze"(%753) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %755 = "ttir.matmul"(%754, %arg247) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %756 = "ttir.reshape"(%755) <{shape = [1 : i32, 11 : i32, 32 : i32, 64 : i32]}> : (tensor<11x2048xf32>) -> tensor<1x11x32x64xf32>
    %757 = "ttir.transpose"(%756) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x32x64xf32>) -> tensor<1x32x11x64xf32>
    %758 = "ttir.multiply"(%757, %17) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %759 = "ttir.transpose"(%757) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %760 = "ttir.matmul"(%arg93, %759) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %761 = "ttir.transpose"(%760) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %762 = "ttir.multiply"(%761, %arg94) : (tensor<1x32x11x32xf32>, tensor<1xf32>) -> tensor<1x32x11x32xf32>
    %763 = "ttir.transpose"(%757) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %764 = "ttir.matmul"(%arg95, %763) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %765 = "ttir.transpose"(%764) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %766 = "ttir.concat"(%762, %765) <{dim = -1 : si32}> : (tensor<1x32x11x32xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x64xf32>
    %767 = "ttir.multiply"(%766, %28) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %768 = "ttir.add"(%758, %767) : (tensor<1x32x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %769 = "ttir.squeeze"(%768) <{dim = 0 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<32x11x64xf32>
    %770 = "ttir.matmul"(%754, %arg248) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %771 = "ttir.reshape"(%770) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %772 = "ttir.transpose"(%771) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %773 = "ttir.multiply"(%772, %17) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %774 = "ttir.transpose"(%772) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %775 = "ttir.matmul"(%arg96, %774) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %776 = "ttir.transpose"(%775) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %777 = "ttir.multiply"(%776, %arg97) : (tensor<1x8x11x32xf32>, tensor<1xf32>) -> tensor<1x8x11x32xf32>
    %778 = "ttir.transpose"(%772) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %779 = "ttir.matmul"(%arg98, %778) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %780 = "ttir.transpose"(%779) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %781 = "ttir.concat"(%777, %780) <{dim = -1 : si32}> : (tensor<1x8x11x32xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x64xf32>
    %782 = "ttir.multiply"(%781, %28) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %783 = "ttir.add"(%773, %782) : (tensor<1x8x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %784 = "ttir.unsqueeze"(%783) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %785 = "ttir.repeat_interleave"(%784) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %786 = "ttir.repeat_interleave"(%785) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %787 = "ttir.reshape"(%786) <{shape = [32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<32x11x64xf32>
    %788 = "ttir.transpose"(%787) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x11x64xf32>) -> tensor<32x64x11xf32>
    %789 = "ttir.matmul"(%769, %788) : (tensor<32x11x64xf32>, tensor<32x64x11xf32>) -> tensor<32x11x11xf32>
    %790 = "ttir.unsqueeze"(%789) <{dim = 0 : si32}> : (tensor<32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %791 = "ttir.multiply"(%790, %arg99) : (tensor<1x32x11x11xf32>, tensor<1xf32>) -> tensor<1x32x11x11xf32>
    %792 = "ttir.add"(%791, %arg100) : (tensor<1x32x11x11xf32>, tensor<1x1x11x11xf32>) -> tensor<1x32x11x11xf32>
    %793 = "ttir.softmax"(%792) <{dimension = -1 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %794 = "ttir.squeeze"(%793) <{dim = 0 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<32x11x11xf32>
    %795 = "ttir.matmul"(%754, %arg249) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %796 = "ttir.reshape"(%795) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %797 = "ttir.transpose"(%796) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %798 = "ttir.unsqueeze"(%797) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %799 = "ttir.repeat_interleave"(%798) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %800 = "ttir.repeat_interleave"(%799) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %801 = "ttir.reshape"(%800) <{shape = [1 : i32, 32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<1x32x11x64xf32>
    %802 = "ttir.transpose"(%801) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %803 = "ttir.squeeze"(%802) <{dim = 0 : si32}> : (tensor<1x32x64x11xf32>) -> tensor<32x64x11xf32>
    %804 = "ttir.transpose"(%803) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x64x11xf32>) -> tensor<32x11x64xf32>
    %805 = "ttir.matmul"(%794, %804) : (tensor<32x11x11xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %806 = "ttir.unsqueeze"(%805) <{dim = 0 : si32}> : (tensor<32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %807 = "ttir.transpose"(%806) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x11x32x64xf32>
    %808 = "ttir.reshape"(%807) <{shape = [11 : i32, 2048 : i32]}> : (tensor<1x11x32x64xf32>) -> tensor<11x2048xf32>
    %809 = "ttir.matmul"(%808, %arg250) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %810 = "ttir.unsqueeze"(%809) <{dim = 0 : si32}> : (tensor<11x2048xf32>) -> tensor<1x11x2048xf32>
    %811 = "ttir.add"(%746, %810) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %812 = "ttir.multiply"(%811, %811) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %813 = "ttir.mean"(%812) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %814 = "ttir.add"(%813, %arg101) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %815 = "ttir.sqrt"(%814) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %816 = "ttir.reciprocal"(%815) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %817 = "ttir.multiply"(%811, %816) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %818 = "ttir.multiply"(%arg251, %817) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %819 = "ttir.squeeze"(%818) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %820 = "ttir.matmul"(%819, %arg252) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %821 = "ttir.unsqueeze"(%820) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %822 = "ttir.sigmoid"(%821) : (tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %823 = "ttir.multiply"(%821, %822) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %824 = "ttir.matmul"(%819, %arg253) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %825 = "ttir.unsqueeze"(%824) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %826 = "ttir.multiply"(%823, %825) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %827 = "ttir.matmul"(%826, %arg254) : (tensor<1x11x8192xf32>, tensor<8192x2048xf32>) -> tensor<1x11x2048xf32>
    %828 = "ttir.add"(%811, %827) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %829 = "ttir.multiply"(%828, %828) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %830 = "ttir.mean"(%829) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %831 = "ttir.add"(%830, %arg102) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %832 = "ttir.sqrt"(%831) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %833 = "ttir.reciprocal"(%832) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %834 = "ttir.multiply"(%828, %833) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %835 = "ttir.multiply"(%arg255, %834) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %836 = "ttir.squeeze"(%835) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %837 = "ttir.matmul"(%836, %arg256) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %838 = "ttir.reshape"(%837) <{shape = [1 : i32, 11 : i32, 32 : i32, 64 : i32]}> : (tensor<11x2048xf32>) -> tensor<1x11x32x64xf32>
    %839 = "ttir.transpose"(%838) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x32x64xf32>) -> tensor<1x32x11x64xf32>
    %840 = "ttir.multiply"(%839, %17) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %841 = "ttir.transpose"(%839) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %842 = "ttir.matmul"(%arg103, %841) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %843 = "ttir.transpose"(%842) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %844 = "ttir.multiply"(%843, %arg104) : (tensor<1x32x11x32xf32>, tensor<1xf32>) -> tensor<1x32x11x32xf32>
    %845 = "ttir.transpose"(%839) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %846 = "ttir.matmul"(%arg105, %845) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %847 = "ttir.transpose"(%846) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %848 = "ttir.concat"(%844, %847) <{dim = -1 : si32}> : (tensor<1x32x11x32xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x64xf32>
    %849 = "ttir.multiply"(%848, %28) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %850 = "ttir.add"(%840, %849) : (tensor<1x32x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %851 = "ttir.squeeze"(%850) <{dim = 0 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<32x11x64xf32>
    %852 = "ttir.matmul"(%836, %arg257) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %853 = "ttir.reshape"(%852) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %854 = "ttir.transpose"(%853) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %855 = "ttir.multiply"(%854, %17) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %856 = "ttir.transpose"(%854) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %857 = "ttir.matmul"(%arg106, %856) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %858 = "ttir.transpose"(%857) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %859 = "ttir.multiply"(%858, %arg107) : (tensor<1x8x11x32xf32>, tensor<1xf32>) -> tensor<1x8x11x32xf32>
    %860 = "ttir.transpose"(%854) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %861 = "ttir.matmul"(%arg108, %860) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %862 = "ttir.transpose"(%861) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %863 = "ttir.concat"(%859, %862) <{dim = -1 : si32}> : (tensor<1x8x11x32xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x64xf32>
    %864 = "ttir.multiply"(%863, %28) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %865 = "ttir.add"(%855, %864) : (tensor<1x8x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %866 = "ttir.unsqueeze"(%865) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %867 = "ttir.repeat_interleave"(%866) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %868 = "ttir.repeat_interleave"(%867) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %869 = "ttir.reshape"(%868) <{shape = [32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<32x11x64xf32>
    %870 = "ttir.transpose"(%869) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x11x64xf32>) -> tensor<32x64x11xf32>
    %871 = "ttir.matmul"(%851, %870) : (tensor<32x11x64xf32>, tensor<32x64x11xf32>) -> tensor<32x11x11xf32>
    %872 = "ttir.unsqueeze"(%871) <{dim = 0 : si32}> : (tensor<32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %873 = "ttir.multiply"(%872, %arg109) : (tensor<1x32x11x11xf32>, tensor<1xf32>) -> tensor<1x32x11x11xf32>
    %874 = "ttir.add"(%873, %arg110) : (tensor<1x32x11x11xf32>, tensor<1x1x11x11xf32>) -> tensor<1x32x11x11xf32>
    %875 = "ttir.softmax"(%874) <{dimension = -1 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %876 = "ttir.squeeze"(%875) <{dim = 0 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<32x11x11xf32>
    %877 = "ttir.matmul"(%836, %arg258) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %878 = "ttir.reshape"(%877) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %879 = "ttir.transpose"(%878) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %880 = "ttir.unsqueeze"(%879) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %881 = "ttir.repeat_interleave"(%880) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %882 = "ttir.repeat_interleave"(%881) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %883 = "ttir.reshape"(%882) <{shape = [1 : i32, 32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<1x32x11x64xf32>
    %884 = "ttir.transpose"(%883) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %885 = "ttir.squeeze"(%884) <{dim = 0 : si32}> : (tensor<1x32x64x11xf32>) -> tensor<32x64x11xf32>
    %886 = "ttir.transpose"(%885) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x64x11xf32>) -> tensor<32x11x64xf32>
    %887 = "ttir.matmul"(%876, %886) : (tensor<32x11x11xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %888 = "ttir.unsqueeze"(%887) <{dim = 0 : si32}> : (tensor<32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %889 = "ttir.transpose"(%888) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x11x32x64xf32>
    %890 = "ttir.reshape"(%889) <{shape = [11 : i32, 2048 : i32]}> : (tensor<1x11x32x64xf32>) -> tensor<11x2048xf32>
    %891 = "ttir.matmul"(%890, %arg259) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %892 = "ttir.unsqueeze"(%891) <{dim = 0 : si32}> : (tensor<11x2048xf32>) -> tensor<1x11x2048xf32>
    %893 = "ttir.add"(%828, %892) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %894 = "ttir.multiply"(%893, %893) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %895 = "ttir.mean"(%894) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %896 = "ttir.add"(%895, %arg111) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %897 = "ttir.sqrt"(%896) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %898 = "ttir.reciprocal"(%897) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %899 = "ttir.multiply"(%893, %898) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %900 = "ttir.multiply"(%arg260, %899) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %901 = "ttir.squeeze"(%900) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %902 = "ttir.matmul"(%901, %arg261) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %903 = "ttir.unsqueeze"(%902) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %904 = "ttir.sigmoid"(%903) : (tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %905 = "ttir.multiply"(%903, %904) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %906 = "ttir.matmul"(%901, %arg262) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %907 = "ttir.unsqueeze"(%906) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %908 = "ttir.multiply"(%905, %907) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %909 = "ttir.matmul"(%908, %arg263) : (tensor<1x11x8192xf32>, tensor<8192x2048xf32>) -> tensor<1x11x2048xf32>
    %910 = "ttir.add"(%893, %909) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %911 = "ttir.multiply"(%910, %910) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %912 = "ttir.mean"(%911) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %913 = "ttir.add"(%912, %arg112) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %914 = "ttir.sqrt"(%913) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %915 = "ttir.reciprocal"(%914) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %916 = "ttir.multiply"(%910, %915) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %917 = "ttir.multiply"(%arg264, %916) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %918 = "ttir.squeeze"(%917) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %919 = "ttir.matmul"(%918, %arg265) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %920 = "ttir.reshape"(%919) <{shape = [1 : i32, 11 : i32, 32 : i32, 64 : i32]}> : (tensor<11x2048xf32>) -> tensor<1x11x32x64xf32>
    %921 = "ttir.transpose"(%920) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x32x64xf32>) -> tensor<1x32x11x64xf32>
    %922 = "ttir.multiply"(%921, %17) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %923 = "ttir.transpose"(%921) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %924 = "ttir.matmul"(%arg113, %923) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %925 = "ttir.transpose"(%924) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %926 = "ttir.multiply"(%925, %arg114) : (tensor<1x32x11x32xf32>, tensor<1xf32>) -> tensor<1x32x11x32xf32>
    %927 = "ttir.transpose"(%921) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %928 = "ttir.matmul"(%arg115, %927) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %929 = "ttir.transpose"(%928) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %930 = "ttir.concat"(%926, %929) <{dim = -1 : si32}> : (tensor<1x32x11x32xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x64xf32>
    %931 = "ttir.multiply"(%930, %28) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %932 = "ttir.add"(%922, %931) : (tensor<1x32x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %933 = "ttir.squeeze"(%932) <{dim = 0 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<32x11x64xf32>
    %934 = "ttir.matmul"(%918, %arg266) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %935 = "ttir.reshape"(%934) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %936 = "ttir.transpose"(%935) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %937 = "ttir.multiply"(%936, %17) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %938 = "ttir.transpose"(%936) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %939 = "ttir.matmul"(%arg116, %938) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %940 = "ttir.transpose"(%939) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %941 = "ttir.multiply"(%940, %arg117) : (tensor<1x8x11x32xf32>, tensor<1xf32>) -> tensor<1x8x11x32xf32>
    %942 = "ttir.transpose"(%936) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %943 = "ttir.matmul"(%arg118, %942) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %944 = "ttir.transpose"(%943) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %945 = "ttir.concat"(%941, %944) <{dim = -1 : si32}> : (tensor<1x8x11x32xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x64xf32>
    %946 = "ttir.multiply"(%945, %28) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %947 = "ttir.add"(%937, %946) : (tensor<1x8x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %948 = "ttir.unsqueeze"(%947) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %949 = "ttir.repeat_interleave"(%948) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %950 = "ttir.repeat_interleave"(%949) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %951 = "ttir.reshape"(%950) <{shape = [32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<32x11x64xf32>
    %952 = "ttir.transpose"(%951) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x11x64xf32>) -> tensor<32x64x11xf32>
    %953 = "ttir.matmul"(%933, %952) : (tensor<32x11x64xf32>, tensor<32x64x11xf32>) -> tensor<32x11x11xf32>
    %954 = "ttir.unsqueeze"(%953) <{dim = 0 : si32}> : (tensor<32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %955 = "ttir.multiply"(%954, %arg119) : (tensor<1x32x11x11xf32>, tensor<1xf32>) -> tensor<1x32x11x11xf32>
    %956 = "ttir.add"(%955, %arg120) : (tensor<1x32x11x11xf32>, tensor<1x1x11x11xf32>) -> tensor<1x32x11x11xf32>
    %957 = "ttir.softmax"(%956) <{dimension = -1 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %958 = "ttir.squeeze"(%957) <{dim = 0 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<32x11x11xf32>
    %959 = "ttir.matmul"(%918, %arg267) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %960 = "ttir.reshape"(%959) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %961 = "ttir.transpose"(%960) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %962 = "ttir.unsqueeze"(%961) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %963 = "ttir.repeat_interleave"(%962) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %964 = "ttir.repeat_interleave"(%963) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %965 = "ttir.reshape"(%964) <{shape = [1 : i32, 32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<1x32x11x64xf32>
    %966 = "ttir.transpose"(%965) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %967 = "ttir.squeeze"(%966) <{dim = 0 : si32}> : (tensor<1x32x64x11xf32>) -> tensor<32x64x11xf32>
    %968 = "ttir.transpose"(%967) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x64x11xf32>) -> tensor<32x11x64xf32>
    %969 = "ttir.matmul"(%958, %968) : (tensor<32x11x11xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %970 = "ttir.unsqueeze"(%969) <{dim = 0 : si32}> : (tensor<32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %971 = "ttir.transpose"(%970) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x11x32x64xf32>
    %972 = "ttir.reshape"(%971) <{shape = [11 : i32, 2048 : i32]}> : (tensor<1x11x32x64xf32>) -> tensor<11x2048xf32>
    %973 = "ttir.matmul"(%972, %arg268) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %974 = "ttir.unsqueeze"(%973) <{dim = 0 : si32}> : (tensor<11x2048xf32>) -> tensor<1x11x2048xf32>
    %975 = "ttir.add"(%910, %974) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %976 = "ttir.multiply"(%975, %975) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %977 = "ttir.mean"(%976) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %978 = "ttir.add"(%977, %arg121) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %979 = "ttir.sqrt"(%978) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %980 = "ttir.reciprocal"(%979) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %981 = "ttir.multiply"(%975, %980) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %982 = "ttir.multiply"(%arg269, %981) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %983 = "ttir.squeeze"(%982) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %984 = "ttir.matmul"(%983, %arg270) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %985 = "ttir.unsqueeze"(%984) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %986 = "ttir.sigmoid"(%985) : (tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %987 = "ttir.multiply"(%985, %986) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %988 = "ttir.matmul"(%983, %arg271) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %989 = "ttir.unsqueeze"(%988) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %990 = "ttir.multiply"(%987, %989) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %991 = "ttir.matmul"(%990, %arg272) : (tensor<1x11x8192xf32>, tensor<8192x2048xf32>) -> tensor<1x11x2048xf32>
    %992 = "ttir.add"(%975, %991) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %993 = "ttir.multiply"(%992, %992) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %994 = "ttir.mean"(%993) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %995 = "ttir.add"(%994, %arg122) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %996 = "ttir.sqrt"(%995) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %997 = "ttir.reciprocal"(%996) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %998 = "ttir.multiply"(%992, %997) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %999 = "ttir.multiply"(%arg273, %998) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1000 = "ttir.squeeze"(%999) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %1001 = "ttir.matmul"(%1000, %arg274) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %1002 = "ttir.reshape"(%1001) <{shape = [1 : i32, 11 : i32, 32 : i32, 64 : i32]}> : (tensor<11x2048xf32>) -> tensor<1x11x32x64xf32>
    %1003 = "ttir.transpose"(%1002) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x32x64xf32>) -> tensor<1x32x11x64xf32>
    %1004 = "ttir.multiply"(%1003, %17) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %1005 = "ttir.transpose"(%1003) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %1006 = "ttir.matmul"(%arg123, %1005) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %1007 = "ttir.transpose"(%1006) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %1008 = "ttir.multiply"(%1007, %arg124) : (tensor<1x32x11x32xf32>, tensor<1xf32>) -> tensor<1x32x11x32xf32>
    %1009 = "ttir.transpose"(%1003) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %1010 = "ttir.matmul"(%arg125, %1009) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %1011 = "ttir.transpose"(%1010) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %1012 = "ttir.concat"(%1008, %1011) <{dim = -1 : si32}> : (tensor<1x32x11x32xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x64xf32>
    %1013 = "ttir.multiply"(%1012, %28) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %1014 = "ttir.add"(%1004, %1013) : (tensor<1x32x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %1015 = "ttir.squeeze"(%1014) <{dim = 0 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<32x11x64xf32>
    %1016 = "ttir.matmul"(%1000, %arg275) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %1017 = "ttir.reshape"(%1016) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %1018 = "ttir.transpose"(%1017) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %1019 = "ttir.multiply"(%1018, %17) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %1020 = "ttir.transpose"(%1018) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %1021 = "ttir.matmul"(%arg126, %1020) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %1022 = "ttir.transpose"(%1021) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %1023 = "ttir.multiply"(%1022, %arg127) : (tensor<1x8x11x32xf32>, tensor<1xf32>) -> tensor<1x8x11x32xf32>
    %1024 = "ttir.transpose"(%1018) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %1025 = "ttir.matmul"(%arg128, %1024) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %1026 = "ttir.transpose"(%1025) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %1027 = "ttir.concat"(%1023, %1026) <{dim = -1 : si32}> : (tensor<1x8x11x32xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x64xf32>
    %1028 = "ttir.multiply"(%1027, %28) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %1029 = "ttir.add"(%1019, %1028) : (tensor<1x8x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %1030 = "ttir.unsqueeze"(%1029) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %1031 = "ttir.repeat_interleave"(%1030) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %1032 = "ttir.repeat_interleave"(%1031) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %1033 = "ttir.reshape"(%1032) <{shape = [32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<32x11x64xf32>
    %1034 = "ttir.transpose"(%1033) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x11x64xf32>) -> tensor<32x64x11xf32>
    %1035 = "ttir.matmul"(%1015, %1034) : (tensor<32x11x64xf32>, tensor<32x64x11xf32>) -> tensor<32x11x11xf32>
    %1036 = "ttir.unsqueeze"(%1035) <{dim = 0 : si32}> : (tensor<32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %1037 = "ttir.multiply"(%1036, %arg129) : (tensor<1x32x11x11xf32>, tensor<1xf32>) -> tensor<1x32x11x11xf32>
    %1038 = "ttir.add"(%1037, %arg130) : (tensor<1x32x11x11xf32>, tensor<1x1x11x11xf32>) -> tensor<1x32x11x11xf32>
    %1039 = "ttir.softmax"(%1038) <{dimension = -1 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %1040 = "ttir.squeeze"(%1039) <{dim = 0 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<32x11x11xf32>
    %1041 = "ttir.matmul"(%1000, %arg276) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %1042 = "ttir.reshape"(%1041) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %1043 = "ttir.transpose"(%1042) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %1044 = "ttir.unsqueeze"(%1043) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %1045 = "ttir.repeat_interleave"(%1044) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %1046 = "ttir.repeat_interleave"(%1045) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %1047 = "ttir.reshape"(%1046) <{shape = [1 : i32, 32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<1x32x11x64xf32>
    %1048 = "ttir.transpose"(%1047) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %1049 = "ttir.squeeze"(%1048) <{dim = 0 : si32}> : (tensor<1x32x64x11xf32>) -> tensor<32x64x11xf32>
    %1050 = "ttir.transpose"(%1049) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x64x11xf32>) -> tensor<32x11x64xf32>
    %1051 = "ttir.matmul"(%1040, %1050) : (tensor<32x11x11xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %1052 = "ttir.unsqueeze"(%1051) <{dim = 0 : si32}> : (tensor<32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %1053 = "ttir.transpose"(%1052) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x11x32x64xf32>
    %1054 = "ttir.reshape"(%1053) <{shape = [11 : i32, 2048 : i32]}> : (tensor<1x11x32x64xf32>) -> tensor<11x2048xf32>
    %1055 = "ttir.matmul"(%1054, %arg277) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %1056 = "ttir.unsqueeze"(%1055) <{dim = 0 : si32}> : (tensor<11x2048xf32>) -> tensor<1x11x2048xf32>
    %1057 = "ttir.add"(%992, %1056) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1058 = "ttir.multiply"(%1057, %1057) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1059 = "ttir.mean"(%1058) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %1060 = "ttir.add"(%1059, %arg131) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %1061 = "ttir.sqrt"(%1060) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %1062 = "ttir.reciprocal"(%1061) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %1063 = "ttir.multiply"(%1057, %1062) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %1064 = "ttir.multiply"(%arg278, %1063) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1065 = "ttir.squeeze"(%1064) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %1066 = "ttir.matmul"(%1065, %arg279) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %1067 = "ttir.unsqueeze"(%1066) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %1068 = "ttir.sigmoid"(%1067) : (tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %1069 = "ttir.multiply"(%1067, %1068) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %1070 = "ttir.matmul"(%1065, %arg280) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %1071 = "ttir.unsqueeze"(%1070) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %1072 = "ttir.multiply"(%1069, %1071) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %1073 = "ttir.matmul"(%1072, %arg281) : (tensor<1x11x8192xf32>, tensor<8192x2048xf32>) -> tensor<1x11x2048xf32>
    %1074 = "ttir.add"(%1057, %1073) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1075 = "ttir.multiply"(%1074, %1074) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1076 = "ttir.mean"(%1075) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %1077 = "ttir.add"(%1076, %arg132) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %1078 = "ttir.sqrt"(%1077) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %1079 = "ttir.reciprocal"(%1078) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %1080 = "ttir.multiply"(%1074, %1079) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %1081 = "ttir.multiply"(%arg282, %1080) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1082 = "ttir.squeeze"(%1081) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %1083 = "ttir.matmul"(%1082, %arg283) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %1084 = "ttir.reshape"(%1083) <{shape = [1 : i32, 11 : i32, 32 : i32, 64 : i32]}> : (tensor<11x2048xf32>) -> tensor<1x11x32x64xf32>
    %1085 = "ttir.transpose"(%1084) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x32x64xf32>) -> tensor<1x32x11x64xf32>
    %1086 = "ttir.multiply"(%1085, %17) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %1087 = "ttir.transpose"(%1085) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %1088 = "ttir.matmul"(%arg133, %1087) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %1089 = "ttir.transpose"(%1088) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %1090 = "ttir.multiply"(%1089, %arg134) : (tensor<1x32x11x32xf32>, tensor<1xf32>) -> tensor<1x32x11x32xf32>
    %1091 = "ttir.transpose"(%1085) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %1092 = "ttir.matmul"(%arg135, %1091) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %1093 = "ttir.transpose"(%1092) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %1094 = "ttir.concat"(%1090, %1093) <{dim = -1 : si32}> : (tensor<1x32x11x32xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x64xf32>
    %1095 = "ttir.multiply"(%1094, %28) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %1096 = "ttir.add"(%1086, %1095) : (tensor<1x32x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %1097 = "ttir.squeeze"(%1096) <{dim = 0 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<32x11x64xf32>
    %1098 = "ttir.matmul"(%1082, %arg284) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %1099 = "ttir.reshape"(%1098) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %1100 = "ttir.transpose"(%1099) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %1101 = "ttir.multiply"(%1100, %17) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %1102 = "ttir.transpose"(%1100) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %1103 = "ttir.matmul"(%arg136, %1102) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %1104 = "ttir.transpose"(%1103) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %1105 = "ttir.multiply"(%1104, %arg137) : (tensor<1x8x11x32xf32>, tensor<1xf32>) -> tensor<1x8x11x32xf32>
    %1106 = "ttir.transpose"(%1100) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %1107 = "ttir.matmul"(%arg138, %1106) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %1108 = "ttir.transpose"(%1107) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %1109 = "ttir.concat"(%1105, %1108) <{dim = -1 : si32}> : (tensor<1x8x11x32xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x64xf32>
    %1110 = "ttir.multiply"(%1109, %28) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %1111 = "ttir.add"(%1101, %1110) : (tensor<1x8x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %1112 = "ttir.unsqueeze"(%1111) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %1113 = "ttir.repeat_interleave"(%1112) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %1114 = "ttir.repeat_interleave"(%1113) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %1115 = "ttir.reshape"(%1114) <{shape = [32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<32x11x64xf32>
    %1116 = "ttir.transpose"(%1115) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x11x64xf32>) -> tensor<32x64x11xf32>
    %1117 = "ttir.matmul"(%1097, %1116) : (tensor<32x11x64xf32>, tensor<32x64x11xf32>) -> tensor<32x11x11xf32>
    %1118 = "ttir.unsqueeze"(%1117) <{dim = 0 : si32}> : (tensor<32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %1119 = "ttir.multiply"(%1118, %arg139) : (tensor<1x32x11x11xf32>, tensor<1xf32>) -> tensor<1x32x11x11xf32>
    %1120 = "ttir.add"(%1119, %arg140) : (tensor<1x32x11x11xf32>, tensor<1x1x11x11xf32>) -> tensor<1x32x11x11xf32>
    %1121 = "ttir.softmax"(%1120) <{dimension = -1 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %1122 = "ttir.squeeze"(%1121) <{dim = 0 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<32x11x11xf32>
    %1123 = "ttir.matmul"(%1082, %arg285) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %1124 = "ttir.reshape"(%1123) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %1125 = "ttir.transpose"(%1124) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %1126 = "ttir.unsqueeze"(%1125) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %1127 = "ttir.repeat_interleave"(%1126) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %1128 = "ttir.repeat_interleave"(%1127) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %1129 = "ttir.reshape"(%1128) <{shape = [1 : i32, 32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<1x32x11x64xf32>
    %1130 = "ttir.transpose"(%1129) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %1131 = "ttir.squeeze"(%1130) <{dim = 0 : si32}> : (tensor<1x32x64x11xf32>) -> tensor<32x64x11xf32>
    %1132 = "ttir.transpose"(%1131) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x64x11xf32>) -> tensor<32x11x64xf32>
    %1133 = "ttir.matmul"(%1122, %1132) : (tensor<32x11x11xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %1134 = "ttir.unsqueeze"(%1133) <{dim = 0 : si32}> : (tensor<32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %1135 = "ttir.transpose"(%1134) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x11x32x64xf32>
    %1136 = "ttir.reshape"(%1135) <{shape = [11 : i32, 2048 : i32]}> : (tensor<1x11x32x64xf32>) -> tensor<11x2048xf32>
    %1137 = "ttir.matmul"(%1136, %arg286) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %1138 = "ttir.unsqueeze"(%1137) <{dim = 0 : si32}> : (tensor<11x2048xf32>) -> tensor<1x11x2048xf32>
    %1139 = "ttir.add"(%1074, %1138) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1140 = "ttir.multiply"(%1139, %1139) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1141 = "ttir.mean"(%1140) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %1142 = "ttir.add"(%1141, %arg141) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %1143 = "ttir.sqrt"(%1142) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %1144 = "ttir.reciprocal"(%1143) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %1145 = "ttir.multiply"(%1139, %1144) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %1146 = "ttir.multiply"(%arg287, %1145) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1147 = "ttir.squeeze"(%1146) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %1148 = "ttir.matmul"(%1147, %arg288) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %1149 = "ttir.unsqueeze"(%1148) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %1150 = "ttir.sigmoid"(%1149) : (tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %1151 = "ttir.multiply"(%1149, %1150) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %1152 = "ttir.matmul"(%1147, %arg289) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %1153 = "ttir.unsqueeze"(%1152) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %1154 = "ttir.multiply"(%1151, %1153) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %1155 = "ttir.matmul"(%1154, %arg290) : (tensor<1x11x8192xf32>, tensor<8192x2048xf32>) -> tensor<1x11x2048xf32>
    %1156 = "ttir.add"(%1139, %1155) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1157 = "ttir.multiply"(%1156, %1156) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1158 = "ttir.mean"(%1157) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %1159 = "ttir.add"(%1158, %arg142) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %1160 = "ttir.sqrt"(%1159) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %1161 = "ttir.reciprocal"(%1160) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %1162 = "ttir.multiply"(%1156, %1161) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %1163 = "ttir.multiply"(%arg291, %1162) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1164 = "ttir.squeeze"(%1163) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %1165 = "ttir.matmul"(%1164, %arg292) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %1166 = "ttir.reshape"(%1165) <{shape = [1 : i32, 11 : i32, 32 : i32, 64 : i32]}> : (tensor<11x2048xf32>) -> tensor<1x11x32x64xf32>
    %1167 = "ttir.transpose"(%1166) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x32x64xf32>) -> tensor<1x32x11x64xf32>
    %1168 = "ttir.multiply"(%1167, %17) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %1169 = "ttir.transpose"(%1167) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %1170 = "ttir.matmul"(%arg143, %1169) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %1171 = "ttir.transpose"(%1170) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %1172 = "ttir.multiply"(%1171, %arg144) : (tensor<1x32x11x32xf32>, tensor<1xf32>) -> tensor<1x32x11x32xf32>
    %1173 = "ttir.transpose"(%1167) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %1174 = "ttir.matmul"(%arg145, %1173) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %1175 = "ttir.transpose"(%1174) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %1176 = "ttir.concat"(%1172, %1175) <{dim = -1 : si32}> : (tensor<1x32x11x32xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x64xf32>
    %1177 = "ttir.multiply"(%1176, %28) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %1178 = "ttir.add"(%1168, %1177) : (tensor<1x32x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %1179 = "ttir.squeeze"(%1178) <{dim = 0 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<32x11x64xf32>
    %1180 = "ttir.matmul"(%1164, %arg293) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %1181 = "ttir.reshape"(%1180) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %1182 = "ttir.transpose"(%1181) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %1183 = "ttir.multiply"(%1182, %17) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %1184 = "ttir.transpose"(%1182) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %1185 = "ttir.matmul"(%arg146, %1184) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %1186 = "ttir.transpose"(%1185) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %1187 = "ttir.multiply"(%1186, %arg147) : (tensor<1x8x11x32xf32>, tensor<1xf32>) -> tensor<1x8x11x32xf32>
    %1188 = "ttir.transpose"(%1182) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %1189 = "ttir.matmul"(%arg148, %1188) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %1190 = "ttir.transpose"(%1189) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %1191 = "ttir.concat"(%1187, %1190) <{dim = -1 : si32}> : (tensor<1x8x11x32xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x64xf32>
    %1192 = "ttir.multiply"(%1191, %28) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %1193 = "ttir.add"(%1183, %1192) : (tensor<1x8x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %1194 = "ttir.unsqueeze"(%1193) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %1195 = "ttir.repeat_interleave"(%1194) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %1196 = "ttir.repeat_interleave"(%1195) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %1197 = "ttir.reshape"(%1196) <{shape = [32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<32x11x64xf32>
    %1198 = "ttir.transpose"(%1197) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x11x64xf32>) -> tensor<32x64x11xf32>
    %1199 = "ttir.matmul"(%1179, %1198) : (tensor<32x11x64xf32>, tensor<32x64x11xf32>) -> tensor<32x11x11xf32>
    %1200 = "ttir.unsqueeze"(%1199) <{dim = 0 : si32}> : (tensor<32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %1201 = "ttir.multiply"(%1200, %arg149) : (tensor<1x32x11x11xf32>, tensor<1xf32>) -> tensor<1x32x11x11xf32>
    %1202 = "ttir.add"(%1201, %arg150) : (tensor<1x32x11x11xf32>, tensor<1x1x11x11xf32>) -> tensor<1x32x11x11xf32>
    %1203 = "ttir.softmax"(%1202) <{dimension = -1 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %1204 = "ttir.squeeze"(%1203) <{dim = 0 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<32x11x11xf32>
    %1205 = "ttir.matmul"(%1164, %arg294) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %1206 = "ttir.reshape"(%1205) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %1207 = "ttir.transpose"(%1206) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %1208 = "ttir.unsqueeze"(%1207) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %1209 = "ttir.repeat_interleave"(%1208) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %1210 = "ttir.repeat_interleave"(%1209) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %1211 = "ttir.reshape"(%1210) <{shape = [1 : i32, 32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<1x32x11x64xf32>
    %1212 = "ttir.transpose"(%1211) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %1213 = "ttir.squeeze"(%1212) <{dim = 0 : si32}> : (tensor<1x32x64x11xf32>) -> tensor<32x64x11xf32>
    %1214 = "ttir.transpose"(%1213) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x64x11xf32>) -> tensor<32x11x64xf32>
    %1215 = "ttir.matmul"(%1204, %1214) : (tensor<32x11x11xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %1216 = "ttir.unsqueeze"(%1215) <{dim = 0 : si32}> : (tensor<32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %1217 = "ttir.transpose"(%1216) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x11x32x64xf32>
    %1218 = "ttir.reshape"(%1217) <{shape = [11 : i32, 2048 : i32]}> : (tensor<1x11x32x64xf32>) -> tensor<11x2048xf32>
    %1219 = "ttir.matmul"(%1218, %arg295) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %1220 = "ttir.unsqueeze"(%1219) <{dim = 0 : si32}> : (tensor<11x2048xf32>) -> tensor<1x11x2048xf32>
    %1221 = "ttir.add"(%1156, %1220) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1222 = "ttir.multiply"(%1221, %1221) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1223 = "ttir.mean"(%1222) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %1224 = "ttir.add"(%1223, %arg151) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %1225 = "ttir.sqrt"(%1224) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %1226 = "ttir.reciprocal"(%1225) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %1227 = "ttir.multiply"(%1221, %1226) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %1228 = "ttir.multiply"(%arg296, %1227) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1229 = "ttir.squeeze"(%1228) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %1230 = "ttir.matmul"(%1229, %arg297) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %1231 = "ttir.unsqueeze"(%1230) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %1232 = "ttir.sigmoid"(%1231) : (tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %1233 = "ttir.multiply"(%1231, %1232) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %1234 = "ttir.matmul"(%1229, %arg298) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %1235 = "ttir.unsqueeze"(%1234) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %1236 = "ttir.multiply"(%1233, %1235) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %1237 = "ttir.matmul"(%1236, %arg299) : (tensor<1x11x8192xf32>, tensor<8192x2048xf32>) -> tensor<1x11x2048xf32>
    %1238 = "ttir.add"(%1221, %1237) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1239 = "ttir.multiply"(%1238, %1238) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1240 = "ttir.mean"(%1239) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %1241 = "ttir.add"(%1240, %arg152) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %1242 = "ttir.sqrt"(%1241) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %1243 = "ttir.reciprocal"(%1242) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %1244 = "ttir.multiply"(%1238, %1243) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %1245 = "ttir.multiply"(%arg300, %1244) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1246 = "ttir.squeeze"(%1245) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %1247 = "ttir.matmul"(%1246, %arg301) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %1248 = "ttir.reshape"(%1247) <{shape = [1 : i32, 11 : i32, 32 : i32, 64 : i32]}> : (tensor<11x2048xf32>) -> tensor<1x11x32x64xf32>
    %1249 = "ttir.transpose"(%1248) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x32x64xf32>) -> tensor<1x32x11x64xf32>
    %1250 = "ttir.multiply"(%1249, %17) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %1251 = "ttir.transpose"(%1249) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %1252 = "ttir.matmul"(%arg153, %1251) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %1253 = "ttir.transpose"(%1252) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %1254 = "ttir.multiply"(%1253, %arg154) : (tensor<1x32x11x32xf32>, tensor<1xf32>) -> tensor<1x32x11x32xf32>
    %1255 = "ttir.transpose"(%1249) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %1256 = "ttir.matmul"(%arg155, %1255) : (tensor<1x32x32x64xf32>, tensor<1x32x64x11xf32>) -> tensor<1x32x32x11xf32>
    %1257 = "ttir.transpose"(%1256) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x32x11xf32>) -> tensor<1x32x11x32xf32>
    %1258 = "ttir.concat"(%1254, %1257) <{dim = -1 : si32}> : (tensor<1x32x11x32xf32>, tensor<1x32x11x32xf32>) -> tensor<1x32x11x64xf32>
    %1259 = "ttir.multiply"(%1258, %28) : (tensor<1x32x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x32x11x64xf32>
    %1260 = "ttir.add"(%1250, %1259) : (tensor<1x32x11x64xf32>, tensor<1x32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %1261 = "ttir.squeeze"(%1260) <{dim = 0 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<32x11x64xf32>
    %1262 = "ttir.matmul"(%1246, %arg302) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %1263 = "ttir.reshape"(%1262) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %1264 = "ttir.transpose"(%1263) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %1265 = "ttir.multiply"(%1264, %17) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %1266 = "ttir.transpose"(%1264) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %1267 = "ttir.matmul"(%arg156, %1266) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %1268 = "ttir.transpose"(%1267) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %1269 = "ttir.multiply"(%1268, %arg157) : (tensor<1x8x11x32xf32>, tensor<1xf32>) -> tensor<1x8x11x32xf32>
    %1270 = "ttir.transpose"(%1264) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x64x11xf32>
    %1271 = "ttir.matmul"(%arg158, %1270) : (tensor<1x8x32x64xf32>, tensor<1x8x64x11xf32>) -> tensor<1x8x32x11xf32>
    %1272 = "ttir.transpose"(%1271) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x32x11xf32>) -> tensor<1x8x11x32xf32>
    %1273 = "ttir.concat"(%1269, %1272) <{dim = -1 : si32}> : (tensor<1x8x11x32xf32>, tensor<1x8x11x32xf32>) -> tensor<1x8x11x64xf32>
    %1274 = "ttir.multiply"(%1273, %28) : (tensor<1x8x11x64xf32>, tensor<1x1x11x64xf32>) -> tensor<1x8x11x64xf32>
    %1275 = "ttir.add"(%1265, %1274) : (tensor<1x8x11x64xf32>, tensor<1x8x11x64xf32>) -> tensor<1x8x11x64xf32>
    %1276 = "ttir.unsqueeze"(%1275) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %1277 = "ttir.repeat_interleave"(%1276) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %1278 = "ttir.repeat_interleave"(%1277) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %1279 = "ttir.reshape"(%1278) <{shape = [32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<32x11x64xf32>
    %1280 = "ttir.transpose"(%1279) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x11x64xf32>) -> tensor<32x64x11xf32>
    %1281 = "ttir.matmul"(%1261, %1280) : (tensor<32x11x64xf32>, tensor<32x64x11xf32>) -> tensor<32x11x11xf32>
    %1282 = "ttir.unsqueeze"(%1281) <{dim = 0 : si32}> : (tensor<32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %1283 = "ttir.multiply"(%1282, %arg159) : (tensor<1x32x11x11xf32>, tensor<1xf32>) -> tensor<1x32x11x11xf32>
    %1284 = "ttir.add"(%1283, %arg160) : (tensor<1x32x11x11xf32>, tensor<1x1x11x11xf32>) -> tensor<1x32x11x11xf32>
    %1285 = "ttir.softmax"(%1284) <{dimension = -1 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<1x32x11x11xf32>
    %1286 = "ttir.squeeze"(%1285) <{dim = 0 : si32}> : (tensor<1x32x11x11xf32>) -> tensor<32x11x11xf32>
    %1287 = "ttir.matmul"(%1246, %arg303) : (tensor<11x2048xf32>, tensor<2048x512xf32>) -> tensor<11x512xf32>
    %1288 = "ttir.reshape"(%1287) <{shape = [1 : i32, 11 : i32, 8 : i32, 64 : i32]}> : (tensor<11x512xf32>) -> tensor<1x11x8x64xf32>
    %1289 = "ttir.transpose"(%1288) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x11x8x64xf32>) -> tensor<1x8x11x64xf32>
    %1290 = "ttir.unsqueeze"(%1289) <{dim = 2 : si32}> : (tensor<1x8x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %1291 = "ttir.repeat_interleave"(%1290) <{dim = 0 : si32, repeats = 1 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x1x11x64xf32>
    %1292 = "ttir.repeat_interleave"(%1291) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x1x11x64xf32>) -> tensor<1x8x4x11x64xf32>
    %1293 = "ttir.reshape"(%1292) <{shape = [1 : i32, 32 : i32, 11 : i32, 64 : i32]}> : (tensor<1x8x4x11x64xf32>) -> tensor<1x32x11x64xf32>
    %1294 = "ttir.transpose"(%1293) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x32x64x11xf32>
    %1295 = "ttir.squeeze"(%1294) <{dim = 0 : si32}> : (tensor<1x32x64x11xf32>) -> tensor<32x64x11xf32>
    %1296 = "ttir.transpose"(%1295) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x64x11xf32>) -> tensor<32x11x64xf32>
    %1297 = "ttir.matmul"(%1286, %1296) : (tensor<32x11x11xf32>, tensor<32x11x64xf32>) -> tensor<32x11x64xf32>
    %1298 = "ttir.unsqueeze"(%1297) <{dim = 0 : si32}> : (tensor<32x11x64xf32>) -> tensor<1x32x11x64xf32>
    %1299 = "ttir.transpose"(%1298) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x11x64xf32>) -> tensor<1x11x32x64xf32>
    %1300 = "ttir.reshape"(%1299) <{shape = [11 : i32, 2048 : i32]}> : (tensor<1x11x32x64xf32>) -> tensor<11x2048xf32>
    %1301 = "ttir.matmul"(%1300, %arg304) : (tensor<11x2048xf32>, tensor<2048x2048xf32>) -> tensor<11x2048xf32>
    %1302 = "ttir.unsqueeze"(%1301) <{dim = 0 : si32}> : (tensor<11x2048xf32>) -> tensor<1x11x2048xf32>
    %1303 = "ttir.add"(%1238, %1302) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1304 = "ttir.multiply"(%1303, %1303) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1305 = "ttir.mean"(%1304) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %1306 = "ttir.add"(%1305, %arg161) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %1307 = "ttir.sqrt"(%1306) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %1308 = "ttir.reciprocal"(%1307) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %1309 = "ttir.multiply"(%1303, %1308) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %1310 = "ttir.multiply"(%arg305, %1309) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1311 = "ttir.squeeze"(%1310) <{dim = 0 : si32}> : (tensor<1x11x2048xf32>) -> tensor<11x2048xf32>
    %1312 = "ttir.matmul"(%1311, %arg306) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %1313 = "ttir.unsqueeze"(%1312) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %1314 = "ttir.sigmoid"(%1313) : (tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %1315 = "ttir.multiply"(%1313, %1314) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %1316 = "ttir.matmul"(%1311, %arg307) : (tensor<11x2048xf32>, tensor<2048x8192xf32>) -> tensor<11x8192xf32>
    %1317 = "ttir.unsqueeze"(%1316) <{dim = 0 : si32}> : (tensor<11x8192xf32>) -> tensor<1x11x8192xf32>
    %1318 = "ttir.multiply"(%1315, %1317) : (tensor<1x11x8192xf32>, tensor<1x11x8192xf32>) -> tensor<1x11x8192xf32>
    %1319 = "ttir.matmul"(%1318, %arg308) : (tensor<1x11x8192xf32>, tensor<8192x2048xf32>) -> tensor<1x11x2048xf32>
    %1320 = "ttir.add"(%1303, %1319) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1321 = "ttir.multiply"(%1320, %1320) : (tensor<1x11x2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    %1322 = "ttir.mean"(%1321) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<1x11x2048xf32>) -> tensor<1x11x1xf32>
    %1323 = "ttir.add"(%1322, %arg162) : (tensor<1x11x1xf32>, tensor<1xf32>) -> tensor<1x11x1xf32>
    %1324 = "ttir.sqrt"(%1323) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %1325 = "ttir.reciprocal"(%1324) : (tensor<1x11x1xf32>) -> tensor<1x11x1xf32>
    %1326 = "ttir.multiply"(%1320, %1325) : (tensor<1x11x2048xf32>, tensor<1x11x1xf32>) -> tensor<1x11x2048xf32>
    %1327 = "ttir.multiply"(%arg163, %1326) : (tensor<2048xf32>, tensor<1x11x2048xf32>) -> tensor<1x11x2048xf32>
    return %1327 : tensor<1x11x2048xf32>
  }
}
