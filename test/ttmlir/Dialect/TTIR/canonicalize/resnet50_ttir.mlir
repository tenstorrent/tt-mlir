
#input = #tt.argument_type<input>
#param = #tt.argument_type<parameter>

module {
  func.func @main(%arg0: tensor<1x3x224x224xbf16, #input>, %arg1: tensor<64x3x7x7xbf16>, %arg2: tensor<64x64x1x1xbf16>, %arg3: tensor<64x64x3x3xbf16>, %arg4: tensor<256x64x1x1xbf16>, %arg5: tensor<256x64x1x1xbf16>, %arg6: tensor<64x256x1x1xbf16>, %arg7: tensor<64x64x3x3xbf16>, %arg8: tensor<256x64x1x1xbf16>, %arg9: tensor<64x256x1x1xbf16>, %arg10: tensor<64x64x3x3xbf16>, %arg11: tensor<256x64x1x1xbf16>, %arg12: tensor<128x256x1x1xbf16>, %arg13: tensor<128x128x3x3xbf16>, %arg14: tensor<512x128x1x1xbf16>, %arg15: tensor<512x256x1x1xbf16>, %arg16: tensor<128x512x1x1xbf16>, %arg17: tensor<128x128x3x3xbf16>, %arg18: tensor<512x128x1x1xbf16>, %arg19: tensor<128x512x1x1xbf16>, %arg20: tensor<128x128x3x3xbf16>, %arg21: tensor<512x128x1x1xbf16>, %arg22: tensor<128x512x1x1xbf16>, %arg23: tensor<128x128x3x3xbf16>, %arg24: tensor<512x128x1x1xbf16>, %arg25: tensor<256x512x1x1xbf16>, %arg26: tensor<256x256x3x3xbf16>, %arg27: tensor<1024x256x1x1xbf16>, %arg28: tensor<1024x512x1x1xbf16>, %arg29: tensor<256x1024x1x1xbf16>, %arg30: tensor<256x256x3x3xbf16>, %arg31: tensor<1024x256x1x1xbf16>, %arg32: tensor<256x1024x1x1xbf16>, %arg33: tensor<256x256x3x3xbf16>, %arg34: tensor<1024x256x1x1xbf16>, %arg35: tensor<256x1024x1x1xbf16>, %arg36: tensor<256x256x3x3xbf16>, %arg37: tensor<1024x256x1x1xbf16>, %arg38: tensor<256x1024x1x1xbf16>, %arg39: tensor<256x256x3x3xbf16>, %arg40: tensor<1024x256x1x1xbf16>, %arg41: tensor<256x1024x1x1xbf16>, %arg42: tensor<256x256x3x3xbf16>, %arg43: tensor<1024x256x1x1xbf16>, %arg44: tensor<512x1024x1x1xbf16>, %arg45: tensor<512x512x3x3xbf16>, %arg46: tensor<2048x512x1x1xbf16>, %arg47: tensor<2048x1024x1x1xbf16>, %arg48: tensor<512x2048x1x1xbf16>, %arg49: tensor<512x512x3x3xbf16>, %arg50: tensor<2048x512x1x1xbf16>, %arg51: tensor<512x2048x1x1xbf16>, %arg52: tensor<512x512x3x3xbf16>, %arg53: tensor<2048x512x1x1xbf16>, %arg54: tensor<64x1x1xbf16>, %arg55: tensor<64x1x1xbf16>, %arg56: tensor<64x1x1xbf16>, %arg57: tensor<64x1x1xbf16>, %arg58: tensor<64x1x1xbf16>, %arg59: tensor<64x1x1xbf16>, %arg60: tensor<64x1x1xbf16>, %arg61: tensor<64x1x1xbf16>, %arg62: tensor<64x1x1xbf16>, %arg63: tensor<64x1x1xbf16>, %arg64: tensor<64x1x1xbf16>, %arg65: tensor<64x1x1xbf16>, %arg66: tensor<256x1x1xbf16>, %arg67: tensor<256x1x1xbf16>, %arg68: tensor<256x1x1xbf16>, %arg69: tensor<256x1x1xbf16>, %arg70: tensor<256x1x1xbf16>, %arg71: tensor<256x1x1xbf16>, %arg72: tensor<256x1x1xbf16>, %arg73: tensor<256x1x1xbf16>, %arg74: tensor<64x1x1xbf16>, %arg75: tensor<64x1x1xbf16>, %arg76: tensor<64x1x1xbf16>, %arg77: tensor<64x1x1xbf16>, %arg78: tensor<64x1x1xbf16>, %arg79: tensor<64x1x1xbf16>, %arg80: tensor<64x1x1xbf16>, %arg81: tensor<64x1x1xbf16>, %arg82: tensor<256x1x1xbf16>, %arg83: tensor<256x1x1xbf16>, %arg84: tensor<256x1x1xbf16>, %arg85: tensor<256x1x1xbf16>, %arg86: tensor<64x1x1xbf16>, %arg87: tensor<64x1x1xbf16>, %arg88: tensor<64x1x1xbf16>, %arg89: tensor<64x1x1xbf16>, %arg90: tensor<64x1x1xbf16>, %arg91: tensor<64x1x1xbf16>, %arg92: tensor<64x1x1xbf16>, %arg93: tensor<64x1x1xbf16>, %arg94: tensor<256x1x1xbf16>, %arg95: tensor<256x1x1xbf16>, %arg96: tensor<256x1x1xbf16>, %arg97: tensor<256x1x1xbf16>, %arg98: tensor<128x1x1xbf16>, %arg99: tensor<128x1x1xbf16>, %arg100: tensor<128x1x1xbf16>, %arg101: tensor<128x1x1xbf16>, %arg102: tensor<128x1x1xbf16>, %arg103: tensor<128x1x1xbf16>, %arg104: tensor<128x1x1xbf16>, %arg105: tensor<128x1x1xbf16>, %arg106: tensor<512x1x1xbf16>, %arg107: tensor<512x1x1xbf16>, %arg108: tensor<512x1x1xbf16>, %arg109: tensor<512x1x1xbf16>, %arg110: tensor<512x1x1xbf16>, %arg111: tensor<512x1x1xbf16>, %arg112: tensor<512x1x1xbf16>, %arg113: tensor<512x1x1xbf16>, %arg114: tensor<128x1x1xbf16>, %arg115: tensor<128x1x1xbf16>, %arg116: tensor<128x1x1xbf16>, %arg117: tensor<128x1x1xbf16>, %arg118: tensor<128x1x1xbf16>, %arg119: tensor<128x1x1xbf16>, %arg120: tensor<128x1x1xbf16>, %arg121: tensor<128x1x1xbf16>, %arg122: tensor<512x1x1xbf16>, %arg123: tensor<512x1x1xbf16>, %arg124: tensor<512x1x1xbf16>, %arg125: tensor<512x1x1xbf16>, %arg126: tensor<128x1x1xbf16>, %arg127: tensor<128x1x1xbf16>, %arg128: tensor<128x1x1xbf16>, %arg129: tensor<128x1x1xbf16>, %arg130: tensor<128x1x1xbf16>, %arg131: tensor<128x1x1xbf16>, %arg132: tensor<128x1x1xbf16>, %arg133: tensor<128x1x1xbf16>, %arg134: tensor<512x1x1xbf16>, %arg135: tensor<512x1x1xbf16>, %arg136: tensor<512x1x1xbf16>, %arg137: tensor<512x1x1xbf16>, %arg138: tensor<128x1x1xbf16>, %arg139: tensor<128x1x1xbf16>, %arg140: tensor<128x1x1xbf16>, %arg141: tensor<128x1x1xbf16>, %arg142: tensor<128x1x1xbf16>, %arg143: tensor<128x1x1xbf16>, %arg144: tensor<128x1x1xbf16>, %arg145: tensor<128x1x1xbf16>, %arg146: tensor<512x1x1xbf16>, %arg147: tensor<512x1x1xbf16>, %arg148: tensor<512x1x1xbf16>, %arg149: tensor<512x1x1xbf16>, %arg150: tensor<256x1x1xbf16>, %arg151: tensor<256x1x1xbf16>, %arg152: tensor<256x1x1xbf16>, %arg153: tensor<256x1x1xbf16>, %arg154: tensor<256x1x1xbf16>, %arg155: tensor<256x1x1xbf16>, %arg156: tensor<256x1x1xbf16>, %arg157: tensor<256x1x1xbf16>, %arg158: tensor<1024x1x1xbf16>, %arg159: tensor<1024x1x1xbf16>, %arg160: tensor<1024x1x1xbf16>, %arg161: tensor<1024x1x1xbf16>, %arg162: tensor<1024x1x1xbf16>, %arg163: tensor<1024x1x1xbf16>, %arg164: tensor<1024x1x1xbf16>, %arg165: tensor<1024x1x1xbf16>, %arg166: tensor<256x1x1xbf16>, %arg167: tensor<256x1x1xbf16>, %arg168: tensor<256x1x1xbf16>, %arg169: tensor<256x1x1xbf16>, %arg170: tensor<256x1x1xbf16>, %arg171: tensor<256x1x1xbf16>, %arg172: tensor<256x1x1xbf16>, %arg173: tensor<256x1x1xbf16>, %arg174: tensor<1024x1x1xbf16>, %arg175: tensor<1024x1x1xbf16>, %arg176: tensor<1024x1x1xbf16>, %arg177: tensor<1024x1x1xbf16>, %arg178: tensor<256x1x1xbf16>, %arg179: tensor<256x1x1xbf16>, %arg180: tensor<256x1x1xbf16>, %arg181: tensor<256x1x1xbf16>, %arg182: tensor<256x1x1xbf16>, %arg183: tensor<256x1x1xbf16>, %arg184: tensor<256x1x1xbf16>, %arg185: tensor<256x1x1xbf16>, %arg186: tensor<1024x1x1xbf16>, %arg187: tensor<1024x1x1xbf16>, %arg188: tensor<1024x1x1xbf16>, %arg189: tensor<1024x1x1xbf16>, %arg190: tensor<256x1x1xbf16>, %arg191: tensor<256x1x1xbf16>, %arg192: tensor<256x1x1xbf16>, %arg193: tensor<256x1x1xbf16>, %arg194: tensor<256x1x1xbf16>, %arg195: tensor<256x1x1xbf16>, %arg196: tensor<256x1x1xbf16>, %arg197: tensor<256x1x1xbf16>, %arg198: tensor<1024x1x1xbf16>, %arg199: tensor<1024x1x1xbf16>, %arg200: tensor<1024x1x1xbf16>, %arg201: tensor<1024x1x1xbf16>, %arg202: tensor<256x1x1xbf16>, %arg203: tensor<256x1x1xbf16>, %arg204: tensor<256x1x1xbf16>, %arg205: tensor<256x1x1xbf16>, %arg206: tensor<256x1x1xbf16>, %arg207: tensor<256x1x1xbf16>, %arg208: tensor<256x1x1xbf16>, %arg209: tensor<256x1x1xbf16>, %arg210: tensor<1024x1x1xbf16>, %arg211: tensor<1024x1x1xbf16>, %arg212: tensor<1024x1x1xbf16>, %arg213: tensor<1024x1x1xbf16>, %arg214: tensor<256x1x1xbf16>, %arg215: tensor<256x1x1xbf16>, %arg216: tensor<256x1x1xbf16>, %arg217: tensor<256x1x1xbf16>, %arg218: tensor<256x1x1xbf16>, %arg219: tensor<256x1x1xbf16>, %arg220: tensor<256x1x1xbf16>, %arg221: tensor<256x1x1xbf16>, %arg222: tensor<1024x1x1xbf16>, %arg223: tensor<1024x1x1xbf16>, %arg224: tensor<1024x1x1xbf16>, %arg225: tensor<1024x1x1xbf16>, %arg226: tensor<512x1x1xbf16>, %arg227: tensor<512x1x1xbf16>, %arg228: tensor<512x1x1xbf16>, %arg229: tensor<512x1x1xbf16>, %arg230: tensor<512x1x1xbf16>, %arg231: tensor<512x1x1xbf16>, %arg232: tensor<512x1x1xbf16>, %arg233: tensor<512x1x1xbf16>, %arg234: tensor<2048x1x1xbf16>, %arg235: tensor<2048x1x1xbf16>, %arg236: tensor<2048x1x1xbf16>, %arg237: tensor<2048x1x1xbf16>, %arg238: tensor<2048x1x1xbf16>, %arg239: tensor<2048x1x1xbf16>, %arg240: tensor<2048x1x1xbf16>, %arg241: tensor<2048x1x1xbf16>, %arg242: tensor<512x1x1xbf16>, %arg243: tensor<512x1x1xbf16>, %arg244: tensor<512x1x1xbf16>, %arg245: tensor<512x1x1xbf16>, %arg246: tensor<512x1x1xbf16>, %arg247: tensor<512x1x1xbf16>, %arg248: tensor<512x1x1xbf16>, %arg249: tensor<512x1x1xbf16>, %arg250: tensor<2048x1x1xbf16>, %arg251: tensor<2048x1x1xbf16>, %arg252: tensor<2048x1x1xbf16>, %arg253: tensor<2048x1x1xbf16>, %arg254: tensor<512x1x1xbf16>, %arg255: tensor<512x1x1xbf16>, %arg256: tensor<512x1x1xbf16>, %arg257: tensor<512x1x1xbf16>, %arg258: tensor<512x1x1xbf16>, %arg259: tensor<512x1x1xbf16>, %arg260: tensor<512x1x1xbf16>, %arg261: tensor<512x1x1xbf16>, %arg262: tensor<2048x1x1xbf16>, %arg263: tensor<2048x1x1xbf16>, %arg264: tensor<2048x1x1xbf16>, %arg265: tensor<2048x1x1xbf16>, %arg266: tensor<2048x1000xbf16>, %arg267: tensor<1000xbf16>) -> tensor<1x1000xbf16> attributes {constants = [], inputs = [0 : i32, 1 : i32], parameters = [1 : i32, 268 : i32]} {
    %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x64x112x112xbf16>}> : () -> tensor<1x64x112x112xbf16>
    %1 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x64x56x56xbf16>}> : () -> tensor<1x64x56x56xbf16>
    %2 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x256x56x56xbf16>}> : () -> tensor<1x256x56x56xbf16>
    %3 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x128x56x56xbf16>}> : () -> tensor<1x128x56x56xbf16>
    %4 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x128x28x28xbf16>}> : () -> tensor<1x128x28x28xbf16>
    %5 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x512x28x28xbf16>}> : () -> tensor<1x512x28x28xbf16>
    %6 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x256x28x28xbf16>}> : () -> tensor<1x256x28x28xbf16>
    %7 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x256x14x14xbf16>}> : () -> tensor<1x256x14x14xbf16>
    %8 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x1024x14x14xbf16>}> : () -> tensor<1x1024x14x14xbf16>
    %9 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x512x14x14xbf16>}> : () -> tensor<1x512x14x14xbf16>
    %10 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x512x7x7xbf16>}> : () -> tensor<1x512x7x7xbf16>
    %11 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x2048x7x7xbf16>}> : () -> tensor<1x2048x7x7xbf16>
    %12 = "ttir.constant"() <{value = dense<49> : tensor<1xi32>}> : () -> tensor<1xi32>
    %13 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %14 = tensor.empty() : tensor<1x64x112x112xbf16>
    %15 = "ttir.convolution"(%arg0, %arg1, %14) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 3, 3, 3, 3>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<1x3x224x224xbf16, #input>, tensor<64x3x7x7xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %16 = tensor.empty() : tensor<1x64x112x112xbf16>
    %17 = "ttir.typecast"(%15, %16) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %18 = tensor.empty() : tensor<1x64x112x112xbf16>
    %19 = "ttir.broadcast"(%17, %18) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %20 = tensor.empty() : tensor<1x64x1x1xbf16>
    %21 = "ttir.reshape"(%arg54, %20) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %22 = tensor.empty() : tensor<1x64x112x112xbf16>
    %23 = "ttir.broadcast"(%21, %22) <{broadcast_dimensions = array<i32: 1, 1, 112, 112>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %24 = tensor.empty() : tensor<1x64x112x112xbf16>
    %25 = "ttir.subtract"(%19, %23, %24) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %26 = tensor.empty() : tensor<1x64x112x112xbf16>
    %27 = "ttir.broadcast"(%25, %26) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %28 = tensor.empty() : tensor<1x64x1x1xbf16>
    %29 = "ttir.reshape"(%arg55, %28) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %30 = tensor.empty() : tensor<1x64x112x112xbf16>
    %31 = "ttir.broadcast"(%29, %30) <{broadcast_dimensions = array<i32: 1, 1, 112, 112>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %32 = tensor.empty() : tensor<1x64x112x112xbf16>
    %33 = "ttir.multiply"(%27, %31, %32) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %34 = tensor.empty() : tensor<64x1x1xbf16>
    %35 = "ttir.typecast"(%arg56, %34) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x1x1xbf16>, tensor<64x1x1xbf16>) -> tensor<64x1x1xbf16>
    %36 = tensor.empty() : tensor<1x64x112x112xbf16>
    %37 = "ttir.broadcast"(%33, %36) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %38 = tensor.empty() : tensor<1x64x1x1xbf16>
    %39 = "ttir.reshape"(%35, %38) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %40 = tensor.empty() : tensor<1x64x112x112xbf16>
    %41 = "ttir.broadcast"(%39, %40) <{broadcast_dimensions = array<i32: 1, 1, 112, 112>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %42 = tensor.empty() : tensor<1x64x112x112xbf16>
    %43 = "ttir.multiply"(%37, %41, %42) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %44 = tensor.empty() : tensor<64x1x1xbf16>
    %45 = "ttir.typecast"(%arg57, %44) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x1x1xbf16>, tensor<64x1x1xbf16>) -> tensor<64x1x1xbf16>
    %46 = tensor.empty() : tensor<1x64x112x112xbf16>
    %47 = "ttir.broadcast"(%43, %46) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %48 = tensor.empty() : tensor<1x64x1x1xbf16>
    %49 = "ttir.reshape"(%45, %48) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %50 = tensor.empty() : tensor<1x64x112x112xbf16>
    %51 = "ttir.broadcast"(%49, %50) <{broadcast_dimensions = array<i32: 1, 1, 112, 112>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %52 = tensor.empty() : tensor<1x64x112x112xbf16>
    %53 = "ttir.add"(%47, %51, %52) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %54 = tensor.empty() : tensor<1x64x112x112xbf16>
    %55 = "ttir.typecast"(%53, %54) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %56 = tensor.empty() : tensor<1x64x112x112xbf16>
    %57 = "ttir.maximum"(%55, %0, %56) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %58 = tensor.empty() : tensor<1x64x56x56xbf16>
    %59 = "ttir.pooling"(%57, %58) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 1, 1, 1, 1>, pooling_method = #ttir<pooling_method Max>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> : (tensor<1x64x112x112xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %60 = tensor.empty() : tensor<1x64x56x56xbf16>
    %61 = "ttir.convolution"(%59, %arg2, %60) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<64x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %62 = tensor.empty() : tensor<1x64x56x56xbf16>
    %63 = "ttir.typecast"(%61, %62) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %64 = tensor.empty() : tensor<1x64x56x56xbf16>
    %65 = "ttir.broadcast"(%63, %64) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %66 = tensor.empty() : tensor<1x64x1x1xbf16>
    %67 = "ttir.reshape"(%arg58, %66) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %68 = tensor.empty() : tensor<1x64x56x56xbf16>
    %69 = "ttir.broadcast"(%67, %68) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %70 = tensor.empty() : tensor<1x64x56x56xbf16>
    %71 = "ttir.subtract"(%65, %69, %70) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %72 = tensor.empty() : tensor<1x64x56x56xbf16>
    %73 = "ttir.broadcast"(%71, %72) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %74 = tensor.empty() : tensor<1x64x1x1xbf16>
    %75 = "ttir.reshape"(%arg59, %74) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %76 = tensor.empty() : tensor<1x64x56x56xbf16>
    %77 = "ttir.broadcast"(%75, %76) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %78 = tensor.empty() : tensor<1x64x56x56xbf16>
    %79 = "ttir.multiply"(%73, %77, %78) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %80 = tensor.empty() : tensor<64x1x1xbf16>
    %81 = "ttir.typecast"(%arg60, %80) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x1x1xbf16>, tensor<64x1x1xbf16>) -> tensor<64x1x1xbf16>
    %82 = tensor.empty() : tensor<1x64x56x56xbf16>
    %83 = "ttir.broadcast"(%79, %82) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %84 = tensor.empty() : tensor<1x64x1x1xbf16>
    %85 = "ttir.reshape"(%81, %84) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %86 = tensor.empty() : tensor<1x64x56x56xbf16>
    %87 = "ttir.broadcast"(%85, %86) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %88 = tensor.empty() : tensor<1x64x56x56xbf16>
    %89 = "ttir.multiply"(%83, %87, %88) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %90 = tensor.empty() : tensor<64x1x1xbf16>
    %91 = "ttir.typecast"(%arg61, %90) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x1x1xbf16>, tensor<64x1x1xbf16>) -> tensor<64x1x1xbf16>
    %92 = tensor.empty() : tensor<1x64x56x56xbf16>
    %93 = "ttir.broadcast"(%89, %92) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %94 = tensor.empty() : tensor<1x64x1x1xbf16>
    %95 = "ttir.reshape"(%91, %94) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %96 = tensor.empty() : tensor<1x64x56x56xbf16>
    %97 = "ttir.broadcast"(%95, %96) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %98 = tensor.empty() : tensor<1x64x56x56xbf16>
    %99 = "ttir.add"(%93, %97, %98) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %100 = tensor.empty() : tensor<1x64x56x56xbf16>
    %101 = "ttir.typecast"(%99, %100) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %102 = tensor.empty() : tensor<1x64x56x56xbf16>
    %103 = "ttir.maximum"(%101, %1, %102) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %104 = tensor.empty() : tensor<1x64x56x56xbf16>
    %105 = "ttir.convolution"(%103, %arg3, %104) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<64x64x3x3xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %106 = tensor.empty() : tensor<1x64x56x56xbf16>
    %107 = "ttir.typecast"(%105, %106) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %108 = tensor.empty() : tensor<1x64x56x56xbf16>
    %109 = "ttir.broadcast"(%107, %108) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %110 = tensor.empty() : tensor<1x64x1x1xbf16>
    %111 = "ttir.reshape"(%arg62, %110) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %112 = tensor.empty() : tensor<1x64x56x56xbf16>
    %113 = "ttir.broadcast"(%111, %112) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %114 = tensor.empty() : tensor<1x64x56x56xbf16>
    %115 = "ttir.subtract"(%109, %113, %114) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %116 = tensor.empty() : tensor<1x64x56x56xbf16>
    %117 = "ttir.broadcast"(%115, %116) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %118 = tensor.empty() : tensor<1x64x1x1xbf16>
    %119 = "ttir.reshape"(%arg63, %118) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %120 = tensor.empty() : tensor<1x64x56x56xbf16>
    %121 = "ttir.broadcast"(%119, %120) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %122 = tensor.empty() : tensor<1x64x56x56xbf16>
    %123 = "ttir.multiply"(%117, %121, %122) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %124 = tensor.empty() : tensor<64x1x1xbf16>
    %125 = "ttir.typecast"(%arg64, %124) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x1x1xbf16>, tensor<64x1x1xbf16>) -> tensor<64x1x1xbf16>
    %126 = tensor.empty() : tensor<1x64x56x56xbf16>
    %127 = "ttir.broadcast"(%123, %126) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %128 = tensor.empty() : tensor<1x64x1x1xbf16>
    %129 = "ttir.reshape"(%125, %128) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %130 = tensor.empty() : tensor<1x64x56x56xbf16>
    %131 = "ttir.broadcast"(%129, %130) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %132 = tensor.empty() : tensor<1x64x56x56xbf16>
    %133 = "ttir.multiply"(%127, %131, %132) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %134 = tensor.empty() : tensor<64x1x1xbf16>
    %135 = "ttir.typecast"(%arg65, %134) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x1x1xbf16>, tensor<64x1x1xbf16>) -> tensor<64x1x1xbf16>
    %136 = tensor.empty() : tensor<1x64x56x56xbf16>
    %137 = "ttir.broadcast"(%133, %136) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %138 = tensor.empty() : tensor<1x64x1x1xbf16>
    %139 = "ttir.reshape"(%135, %138) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %140 = tensor.empty() : tensor<1x64x56x56xbf16>
    %141 = "ttir.broadcast"(%139, %140) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %142 = tensor.empty() : tensor<1x64x56x56xbf16>
    %143 = "ttir.add"(%137, %141, %142) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %144 = tensor.empty() : tensor<1x64x56x56xbf16>
    %145 = "ttir.typecast"(%143, %144) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %146 = tensor.empty() : tensor<1x64x56x56xbf16>
    %147 = "ttir.maximum"(%145, %1, %146) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %148 = tensor.empty() : tensor<1x256x56x56xbf16>
    %149 = "ttir.convolution"(%147, %arg4, %148) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<256x64x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %150 = tensor.empty() : tensor<1x256x56x56xbf16>
    %151 = "ttir.typecast"(%149, %150) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %152 = tensor.empty() : tensor<1x256x56x56xbf16>
    %153 = "ttir.broadcast"(%151, %152) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %154 = tensor.empty() : tensor<1x256x1x1xbf16>
    %155 = "ttir.reshape"(%arg66, %154) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %156 = tensor.empty() : tensor<1x256x56x56xbf16>
    %157 = "ttir.broadcast"(%155, %156) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %158 = tensor.empty() : tensor<1x256x56x56xbf16>
    %159 = "ttir.subtract"(%153, %157, %158) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %160 = tensor.empty() : tensor<1x256x56x56xbf16>
    %161 = "ttir.broadcast"(%159, %160) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %162 = tensor.empty() : tensor<1x256x1x1xbf16>
    %163 = "ttir.reshape"(%arg67, %162) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %164 = tensor.empty() : tensor<1x256x56x56xbf16>
    %165 = "ttir.broadcast"(%163, %164) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %166 = tensor.empty() : tensor<1x256x56x56xbf16>
    %167 = "ttir.multiply"(%161, %165, %166) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %168 = tensor.empty() : tensor<256x1x1xbf16>
    %169 = "ttir.typecast"(%arg68, %168) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %170 = tensor.empty() : tensor<1x256x56x56xbf16>
    %171 = "ttir.broadcast"(%167, %170) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %172 = tensor.empty() : tensor<1x256x1x1xbf16>
    %173 = "ttir.reshape"(%169, %172) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %174 = tensor.empty() : tensor<1x256x56x56xbf16>
    %175 = "ttir.broadcast"(%173, %174) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %176 = tensor.empty() : tensor<1x256x56x56xbf16>
    %177 = "ttir.multiply"(%171, %175, %176) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %178 = tensor.empty() : tensor<256x1x1xbf16>
    %179 = "ttir.typecast"(%arg69, %178) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %180 = tensor.empty() : tensor<1x256x56x56xbf16>
    %181 = "ttir.broadcast"(%177, %180) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %182 = tensor.empty() : tensor<1x256x1x1xbf16>
    %183 = "ttir.reshape"(%179, %182) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %184 = tensor.empty() : tensor<1x256x56x56xbf16>
    %185 = "ttir.broadcast"(%183, %184) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %186 = tensor.empty() : tensor<1x256x56x56xbf16>
    %187 = "ttir.add"(%181, %185, %186) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %188 = tensor.empty() : tensor<1x256x56x56xbf16>
    %189 = "ttir.typecast"(%187, %188) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %190 = tensor.empty() : tensor<1x256x56x56xbf16>
    %191 = "ttir.convolution"(%59, %arg5, %190) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<256x64x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %192 = tensor.empty() : tensor<1x256x56x56xbf16>
    %193 = "ttir.typecast"(%191, %192) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %194 = tensor.empty() : tensor<1x256x56x56xbf16>
    %195 = "ttir.broadcast"(%193, %194) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %196 = tensor.empty() : tensor<1x256x1x1xbf16>
    %197 = "ttir.reshape"(%arg70, %196) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %198 = tensor.empty() : tensor<1x256x56x56xbf16>
    %199 = "ttir.broadcast"(%197, %198) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %200 = tensor.empty() : tensor<1x256x56x56xbf16>
    %201 = "ttir.subtract"(%195, %199, %200) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %202 = tensor.empty() : tensor<1x256x56x56xbf16>
    %203 = "ttir.broadcast"(%201, %202) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %204 = tensor.empty() : tensor<1x256x1x1xbf16>
    %205 = "ttir.reshape"(%arg71, %204) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %206 = tensor.empty() : tensor<1x256x56x56xbf16>
    %207 = "ttir.broadcast"(%205, %206) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %208 = tensor.empty() : tensor<1x256x56x56xbf16>
    %209 = "ttir.multiply"(%203, %207, %208) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %210 = tensor.empty() : tensor<256x1x1xbf16>
    %211 = "ttir.typecast"(%arg72, %210) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %212 = tensor.empty() : tensor<1x256x56x56xbf16>
    %213 = "ttir.broadcast"(%209, %212) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %214 = tensor.empty() : tensor<1x256x1x1xbf16>
    %215 = "ttir.reshape"(%211, %214) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %216 = tensor.empty() : tensor<1x256x56x56xbf16>
    %217 = "ttir.broadcast"(%215, %216) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %218 = tensor.empty() : tensor<1x256x56x56xbf16>
    %219 = "ttir.multiply"(%213, %217, %218) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %220 = tensor.empty() : tensor<256x1x1xbf16>
    %221 = "ttir.typecast"(%arg73, %220) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %222 = tensor.empty() : tensor<1x256x56x56xbf16>
    %223 = "ttir.broadcast"(%219, %222) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %224 = tensor.empty() : tensor<1x256x1x1xbf16>
    %225 = "ttir.reshape"(%221, %224) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %226 = tensor.empty() : tensor<1x256x56x56xbf16>
    %227 = "ttir.broadcast"(%225, %226) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %228 = tensor.empty() : tensor<1x256x56x56xbf16>
    %229 = "ttir.add"(%223, %227, %228) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %230 = tensor.empty() : tensor<1x256x56x56xbf16>
    %231 = "ttir.typecast"(%229, %230) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %232 = tensor.empty() : tensor<1x256x56x56xbf16>
    %233 = "ttir.add"(%189, %231, %232) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %234 = tensor.empty() : tensor<1x256x56x56xbf16>
    %235 = "ttir.maximum"(%233, %2, %234) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %236 = tensor.empty() : tensor<1x64x56x56xbf16>
    %237 = "ttir.convolution"(%235, %arg6, %236) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<64x256x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %238 = tensor.empty() : tensor<1x64x56x56xbf16>
    %239 = "ttir.typecast"(%237, %238) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %240 = tensor.empty() : tensor<1x64x56x56xbf16>
    %241 = "ttir.broadcast"(%239, %240) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %242 = tensor.empty() : tensor<1x64x1x1xbf16>
    %243 = "ttir.reshape"(%arg74, %242) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %244 = tensor.empty() : tensor<1x64x56x56xbf16>
    %245 = "ttir.broadcast"(%243, %244) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %246 = tensor.empty() : tensor<1x64x56x56xbf16>
    %247 = "ttir.subtract"(%241, %245, %246) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %248 = tensor.empty() : tensor<1x64x56x56xbf16>
    %249 = "ttir.broadcast"(%247, %248) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %250 = tensor.empty() : tensor<1x64x1x1xbf16>
    %251 = "ttir.reshape"(%arg75, %250) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %252 = tensor.empty() : tensor<1x64x56x56xbf16>
    %253 = "ttir.broadcast"(%251, %252) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %254 = tensor.empty() : tensor<1x64x56x56xbf16>
    %255 = "ttir.multiply"(%249, %253, %254) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %256 = tensor.empty() : tensor<64x1x1xbf16>
    %257 = "ttir.typecast"(%arg76, %256) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x1x1xbf16>, tensor<64x1x1xbf16>) -> tensor<64x1x1xbf16>
    %258 = tensor.empty() : tensor<1x64x56x56xbf16>
    %259 = "ttir.broadcast"(%255, %258) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %260 = tensor.empty() : tensor<1x64x1x1xbf16>
    %261 = "ttir.reshape"(%257, %260) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %262 = tensor.empty() : tensor<1x64x56x56xbf16>
    %263 = "ttir.broadcast"(%261, %262) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %264 = tensor.empty() : tensor<1x64x56x56xbf16>
    %265 = "ttir.multiply"(%259, %263, %264) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %266 = tensor.empty() : tensor<64x1x1xbf16>
    %267 = "ttir.typecast"(%arg77, %266) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x1x1xbf16>, tensor<64x1x1xbf16>) -> tensor<64x1x1xbf16>
    %268 = tensor.empty() : tensor<1x64x56x56xbf16>
    %269 = "ttir.broadcast"(%265, %268) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %270 = tensor.empty() : tensor<1x64x1x1xbf16>
    %271 = "ttir.reshape"(%267, %270) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %272 = tensor.empty() : tensor<1x64x56x56xbf16>
    %273 = "ttir.broadcast"(%271, %272) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %274 = tensor.empty() : tensor<1x64x56x56xbf16>
    %275 = "ttir.add"(%269, %273, %274) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %276 = tensor.empty() : tensor<1x64x56x56xbf16>
    %277 = "ttir.typecast"(%275, %276) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %278 = tensor.empty() : tensor<1x64x56x56xbf16>
    %279 = "ttir.maximum"(%277, %1, %278) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %280 = tensor.empty() : tensor<1x64x56x56xbf16>
    %281 = "ttir.convolution"(%279, %arg7, %280) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<64x64x3x3xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %282 = tensor.empty() : tensor<1x64x56x56xbf16>
    %283 = "ttir.typecast"(%281, %282) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %284 = tensor.empty() : tensor<1x64x56x56xbf16>
    %285 = "ttir.broadcast"(%283, %284) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %286 = tensor.empty() : tensor<1x64x1x1xbf16>
    %287 = "ttir.reshape"(%arg78, %286) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %288 = tensor.empty() : tensor<1x64x56x56xbf16>
    %289 = "ttir.broadcast"(%287, %288) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %290 = tensor.empty() : tensor<1x64x56x56xbf16>
    %291 = "ttir.subtract"(%285, %289, %290) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %292 = tensor.empty() : tensor<1x64x56x56xbf16>
    %293 = "ttir.broadcast"(%291, %292) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %294 = tensor.empty() : tensor<1x64x1x1xbf16>
    %295 = "ttir.reshape"(%arg79, %294) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %296 = tensor.empty() : tensor<1x64x56x56xbf16>
    %297 = "ttir.broadcast"(%295, %296) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %298 = tensor.empty() : tensor<1x64x56x56xbf16>
    %299 = "ttir.multiply"(%293, %297, %298) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %300 = tensor.empty() : tensor<64x1x1xbf16>
    %301 = "ttir.typecast"(%arg80, %300) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x1x1xbf16>, tensor<64x1x1xbf16>) -> tensor<64x1x1xbf16>
    %302 = tensor.empty() : tensor<1x64x56x56xbf16>
    %303 = "ttir.broadcast"(%299, %302) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %304 = tensor.empty() : tensor<1x64x1x1xbf16>
    %305 = "ttir.reshape"(%301, %304) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %306 = tensor.empty() : tensor<1x64x56x56xbf16>
    %307 = "ttir.broadcast"(%305, %306) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %308 = tensor.empty() : tensor<1x64x56x56xbf16>
    %309 = "ttir.multiply"(%303, %307, %308) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %310 = tensor.empty() : tensor<64x1x1xbf16>
    %311 = "ttir.typecast"(%arg81, %310) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x1x1xbf16>, tensor<64x1x1xbf16>) -> tensor<64x1x1xbf16>
    %312 = tensor.empty() : tensor<1x64x56x56xbf16>
    %313 = "ttir.broadcast"(%309, %312) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %314 = tensor.empty() : tensor<1x64x1x1xbf16>
    %315 = "ttir.reshape"(%311, %314) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %316 = tensor.empty() : tensor<1x64x56x56xbf16>
    %317 = "ttir.broadcast"(%315, %316) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %318 = tensor.empty() : tensor<1x64x56x56xbf16>
    %319 = "ttir.add"(%313, %317, %318) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %320 = tensor.empty() : tensor<1x64x56x56xbf16>
    %321 = "ttir.typecast"(%319, %320) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %322 = tensor.empty() : tensor<1x64x56x56xbf16>
    %323 = "ttir.maximum"(%321, %1, %322) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %324 = tensor.empty() : tensor<1x256x56x56xbf16>
    %325 = "ttir.convolution"(%323, %arg8, %324) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<256x64x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %326 = tensor.empty() : tensor<1x256x56x56xbf16>
    %327 = "ttir.typecast"(%325, %326) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %328 = tensor.empty() : tensor<1x256x56x56xbf16>
    %329 = "ttir.broadcast"(%327, %328) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %330 = tensor.empty() : tensor<1x256x1x1xbf16>
    %331 = "ttir.reshape"(%arg82, %330) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %332 = tensor.empty() : tensor<1x256x56x56xbf16>
    %333 = "ttir.broadcast"(%331, %332) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %334 = tensor.empty() : tensor<1x256x56x56xbf16>
    %335 = "ttir.subtract"(%329, %333, %334) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %336 = tensor.empty() : tensor<1x256x56x56xbf16>
    %337 = "ttir.broadcast"(%335, %336) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %338 = tensor.empty() : tensor<1x256x1x1xbf16>
    %339 = "ttir.reshape"(%arg83, %338) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %340 = tensor.empty() : tensor<1x256x56x56xbf16>
    %341 = "ttir.broadcast"(%339, %340) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %342 = tensor.empty() : tensor<1x256x56x56xbf16>
    %343 = "ttir.multiply"(%337, %341, %342) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %344 = tensor.empty() : tensor<256x1x1xbf16>
    %345 = "ttir.typecast"(%arg84, %344) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %346 = tensor.empty() : tensor<1x256x56x56xbf16>
    %347 = "ttir.broadcast"(%343, %346) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %348 = tensor.empty() : tensor<1x256x1x1xbf16>
    %349 = "ttir.reshape"(%345, %348) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %350 = tensor.empty() : tensor<1x256x56x56xbf16>
    %351 = "ttir.broadcast"(%349, %350) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %352 = tensor.empty() : tensor<1x256x56x56xbf16>
    %353 = "ttir.multiply"(%347, %351, %352) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %354 = tensor.empty() : tensor<256x1x1xbf16>
    %355 = "ttir.typecast"(%arg85, %354) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %356 = tensor.empty() : tensor<1x256x56x56xbf16>
    %357 = "ttir.broadcast"(%353, %356) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %358 = tensor.empty() : tensor<1x256x1x1xbf16>
    %359 = "ttir.reshape"(%355, %358) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %360 = tensor.empty() : tensor<1x256x56x56xbf16>
    %361 = "ttir.broadcast"(%359, %360) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %362 = tensor.empty() : tensor<1x256x56x56xbf16>
    %363 = "ttir.add"(%357, %361, %362) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %364 = tensor.empty() : tensor<1x256x56x56xbf16>
    %365 = "ttir.typecast"(%363, %364) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %366 = tensor.empty() : tensor<1x256x56x56xbf16>
    %367 = "ttir.add"(%365, %235, %366) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %368 = tensor.empty() : tensor<1x256x56x56xbf16>
    %369 = "ttir.maximum"(%367, %2, %368) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %370 = tensor.empty() : tensor<1x64x56x56xbf16>
    %371 = "ttir.convolution"(%369, %arg9, %370) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<64x256x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %372 = tensor.empty() : tensor<1x64x56x56xbf16>
    %373 = "ttir.typecast"(%371, %372) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %374 = tensor.empty() : tensor<1x64x56x56xbf16>
    %375 = "ttir.broadcast"(%373, %374) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %376 = tensor.empty() : tensor<1x64x1x1xbf16>
    %377 = "ttir.reshape"(%arg86, %376) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %378 = tensor.empty() : tensor<1x64x56x56xbf16>
    %379 = "ttir.broadcast"(%377, %378) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %380 = tensor.empty() : tensor<1x64x56x56xbf16>
    %381 = "ttir.subtract"(%375, %379, %380) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %382 = tensor.empty() : tensor<1x64x56x56xbf16>
    %383 = "ttir.broadcast"(%381, %382) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %384 = tensor.empty() : tensor<1x64x1x1xbf16>
    %385 = "ttir.reshape"(%arg87, %384) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %386 = tensor.empty() : tensor<1x64x56x56xbf16>
    %387 = "ttir.broadcast"(%385, %386) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %388 = tensor.empty() : tensor<1x64x56x56xbf16>
    %389 = "ttir.multiply"(%383, %387, %388) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %390 = tensor.empty() : tensor<64x1x1xbf16>
    %391 = "ttir.typecast"(%arg88, %390) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x1x1xbf16>, tensor<64x1x1xbf16>) -> tensor<64x1x1xbf16>
    %392 = tensor.empty() : tensor<1x64x56x56xbf16>
    %393 = "ttir.broadcast"(%389, %392) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %394 = tensor.empty() : tensor<1x64x1x1xbf16>
    %395 = "ttir.reshape"(%391, %394) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %396 = tensor.empty() : tensor<1x64x56x56xbf16>
    %397 = "ttir.broadcast"(%395, %396) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %398 = tensor.empty() : tensor<1x64x56x56xbf16>
    %399 = "ttir.multiply"(%393, %397, %398) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %400 = tensor.empty() : tensor<64x1x1xbf16>
    %401 = "ttir.typecast"(%arg89, %400) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x1x1xbf16>, tensor<64x1x1xbf16>) -> tensor<64x1x1xbf16>
    %402 = tensor.empty() : tensor<1x64x56x56xbf16>
    %403 = "ttir.broadcast"(%399, %402) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %404 = tensor.empty() : tensor<1x64x1x1xbf16>
    %405 = "ttir.reshape"(%401, %404) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %406 = tensor.empty() : tensor<1x64x56x56xbf16>
    %407 = "ttir.broadcast"(%405, %406) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %408 = tensor.empty() : tensor<1x64x56x56xbf16>
    %409 = "ttir.add"(%403, %407, %408) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %410 = tensor.empty() : tensor<1x64x56x56xbf16>
    %411 = "ttir.typecast"(%409, %410) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %412 = tensor.empty() : tensor<1x64x56x56xbf16>
    %413 = "ttir.maximum"(%411, %1, %412) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %414 = tensor.empty() : tensor<1x64x56x56xbf16>
    %415 = "ttir.convolution"(%413, %arg10, %414) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<64x64x3x3xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %416 = tensor.empty() : tensor<1x64x56x56xbf16>
    %417 = "ttir.typecast"(%415, %416) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %418 = tensor.empty() : tensor<1x64x56x56xbf16>
    %419 = "ttir.broadcast"(%417, %418) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %420 = tensor.empty() : tensor<1x64x1x1xbf16>
    %421 = "ttir.reshape"(%arg90, %420) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %422 = tensor.empty() : tensor<1x64x56x56xbf16>
    %423 = "ttir.broadcast"(%421, %422) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %424 = tensor.empty() : tensor<1x64x56x56xbf16>
    %425 = "ttir.subtract"(%419, %423, %424) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %426 = tensor.empty() : tensor<1x64x56x56xbf16>
    %427 = "ttir.broadcast"(%425, %426) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %428 = tensor.empty() : tensor<1x64x1x1xbf16>
    %429 = "ttir.reshape"(%arg91, %428) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %430 = tensor.empty() : tensor<1x64x56x56xbf16>
    %431 = "ttir.broadcast"(%429, %430) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %432 = tensor.empty() : tensor<1x64x56x56xbf16>
    %433 = "ttir.multiply"(%427, %431, %432) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %434 = tensor.empty() : tensor<64x1x1xbf16>
    %435 = "ttir.typecast"(%arg92, %434) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x1x1xbf16>, tensor<64x1x1xbf16>) -> tensor<64x1x1xbf16>
    %436 = tensor.empty() : tensor<1x64x56x56xbf16>
    %437 = "ttir.broadcast"(%433, %436) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %438 = tensor.empty() : tensor<1x64x1x1xbf16>
    %439 = "ttir.reshape"(%435, %438) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %440 = tensor.empty() : tensor<1x64x56x56xbf16>
    %441 = "ttir.broadcast"(%439, %440) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %442 = tensor.empty() : tensor<1x64x56x56xbf16>
    %443 = "ttir.multiply"(%437, %441, %442) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %444 = tensor.empty() : tensor<64x1x1xbf16>
    %445 = "ttir.typecast"(%arg93, %444) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x1x1xbf16>, tensor<64x1x1xbf16>) -> tensor<64x1x1xbf16>
    %446 = tensor.empty() : tensor<1x64x56x56xbf16>
    %447 = "ttir.broadcast"(%443, %446) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %448 = tensor.empty() : tensor<1x64x1x1xbf16>
    %449 = "ttir.reshape"(%445, %448) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %450 = tensor.empty() : tensor<1x64x56x56xbf16>
    %451 = "ttir.broadcast"(%449, %450) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %452 = tensor.empty() : tensor<1x64x56x56xbf16>
    %453 = "ttir.add"(%447, %451, %452) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %454 = tensor.empty() : tensor<1x64x56x56xbf16>
    %455 = "ttir.typecast"(%453, %454) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %456 = tensor.empty() : tensor<1x64x56x56xbf16>
    %457 = "ttir.maximum"(%455, %1, %456) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    %458 = tensor.empty() : tensor<1x256x56x56xbf16>
    %459 = "ttir.convolution"(%457, %arg11, %458) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x64x56x56xbf16>, tensor<256x64x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %460 = tensor.empty() : tensor<1x256x56x56xbf16>
    %461 = "ttir.typecast"(%459, %460) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %462 = tensor.empty() : tensor<1x256x56x56xbf16>
    %463 = "ttir.broadcast"(%461, %462) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %464 = tensor.empty() : tensor<1x256x1x1xbf16>
    %465 = "ttir.reshape"(%arg94, %464) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %466 = tensor.empty() : tensor<1x256x56x56xbf16>
    %467 = "ttir.broadcast"(%465, %466) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %468 = tensor.empty() : tensor<1x256x56x56xbf16>
    %469 = "ttir.subtract"(%463, %467, %468) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %470 = tensor.empty() : tensor<1x256x56x56xbf16>
    %471 = "ttir.broadcast"(%469, %470) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %472 = tensor.empty() : tensor<1x256x1x1xbf16>
    %473 = "ttir.reshape"(%arg95, %472) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %474 = tensor.empty() : tensor<1x256x56x56xbf16>
    %475 = "ttir.broadcast"(%473, %474) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %476 = tensor.empty() : tensor<1x256x56x56xbf16>
    %477 = "ttir.multiply"(%471, %475, %476) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %478 = tensor.empty() : tensor<256x1x1xbf16>
    %479 = "ttir.typecast"(%arg96, %478) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %480 = tensor.empty() : tensor<1x256x56x56xbf16>
    %481 = "ttir.broadcast"(%477, %480) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %482 = tensor.empty() : tensor<1x256x1x1xbf16>
    %483 = "ttir.reshape"(%479, %482) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %484 = tensor.empty() : tensor<1x256x56x56xbf16>
    %485 = "ttir.broadcast"(%483, %484) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %486 = tensor.empty() : tensor<1x256x56x56xbf16>
    %487 = "ttir.multiply"(%481, %485, %486) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %488 = tensor.empty() : tensor<256x1x1xbf16>
    %489 = "ttir.typecast"(%arg97, %488) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %490 = tensor.empty() : tensor<1x256x56x56xbf16>
    %491 = "ttir.broadcast"(%487, %490) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %492 = tensor.empty() : tensor<1x256x1x1xbf16>
    %493 = "ttir.reshape"(%489, %492) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %494 = tensor.empty() : tensor<1x256x56x56xbf16>
    %495 = "ttir.broadcast"(%493, %494) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %496 = tensor.empty() : tensor<1x256x56x56xbf16>
    %497 = "ttir.add"(%491, %495, %496) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %498 = tensor.empty() : tensor<1x256x56x56xbf16>
    %499 = "ttir.typecast"(%497, %498) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %500 = tensor.empty() : tensor<1x256x56x56xbf16>
    %501 = "ttir.add"(%499, %369, %500) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %502 = tensor.empty() : tensor<1x256x56x56xbf16>
    %503 = "ttir.maximum"(%501, %2, %502) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>, tensor<1x256x56x56xbf16>) -> tensor<1x256x56x56xbf16>
    %504 = tensor.empty() : tensor<1x128x56x56xbf16>
    %505 = "ttir.convolution"(%503, %arg12, %504) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x256x56x56xbf16>, tensor<128x256x1x1xbf16>, tensor<1x128x56x56xbf16>) -> tensor<1x128x56x56xbf16>
    %506 = tensor.empty() : tensor<1x128x56x56xbf16>
    %507 = "ttir.typecast"(%505, %506) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x128x56x56xbf16>, tensor<1x128x56x56xbf16>) -> tensor<1x128x56x56xbf16>
    %508 = tensor.empty() : tensor<1x128x56x56xbf16>
    %509 = "ttir.broadcast"(%507, %508) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x56x56xbf16>, tensor<1x128x56x56xbf16>) -> tensor<1x128x56x56xbf16>
    %510 = tensor.empty() : tensor<1x128x1x1xbf16>
    %511 = "ttir.reshape"(%arg98, %510) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %512 = tensor.empty() : tensor<1x128x56x56xbf16>
    %513 = "ttir.broadcast"(%511, %512) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x56x56xbf16>) -> tensor<1x128x56x56xbf16>
    %514 = tensor.empty() : tensor<1x128x56x56xbf16>
    %515 = "ttir.subtract"(%509, %513, %514) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x56x56xbf16>, tensor<1x128x56x56xbf16>, tensor<1x128x56x56xbf16>) -> tensor<1x128x56x56xbf16>
    %516 = tensor.empty() : tensor<1x128x56x56xbf16>
    %517 = "ttir.broadcast"(%515, %516) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x56x56xbf16>, tensor<1x128x56x56xbf16>) -> tensor<1x128x56x56xbf16>
    %518 = tensor.empty() : tensor<1x128x1x1xbf16>
    %519 = "ttir.reshape"(%arg99, %518) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %520 = tensor.empty() : tensor<1x128x56x56xbf16>
    %521 = "ttir.broadcast"(%519, %520) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x56x56xbf16>) -> tensor<1x128x56x56xbf16>
    %522 = tensor.empty() : tensor<1x128x56x56xbf16>
    %523 = "ttir.multiply"(%517, %521, %522) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x56x56xbf16>, tensor<1x128x56x56xbf16>, tensor<1x128x56x56xbf16>) -> tensor<1x128x56x56xbf16>
    %524 = tensor.empty() : tensor<128x1x1xbf16>
    %525 = "ttir.typecast"(%arg100, %524) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<128x1x1xbf16>, tensor<128x1x1xbf16>) -> tensor<128x1x1xbf16>
    %526 = tensor.empty() : tensor<1x128x56x56xbf16>
    %527 = "ttir.broadcast"(%523, %526) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x56x56xbf16>, tensor<1x128x56x56xbf16>) -> tensor<1x128x56x56xbf16>
    %528 = tensor.empty() : tensor<1x128x1x1xbf16>
    %529 = "ttir.reshape"(%525, %528) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %530 = tensor.empty() : tensor<1x128x56x56xbf16>
    %531 = "ttir.broadcast"(%529, %530) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x56x56xbf16>) -> tensor<1x128x56x56xbf16>
    %532 = tensor.empty() : tensor<1x128x56x56xbf16>
    %533 = "ttir.multiply"(%527, %531, %532) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x56x56xbf16>, tensor<1x128x56x56xbf16>, tensor<1x128x56x56xbf16>) -> tensor<1x128x56x56xbf16>
    %534 = tensor.empty() : tensor<128x1x1xbf16>
    %535 = "ttir.typecast"(%arg101, %534) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<128x1x1xbf16>, tensor<128x1x1xbf16>) -> tensor<128x1x1xbf16>
    %536 = tensor.empty() : tensor<1x128x56x56xbf16>
    %537 = "ttir.broadcast"(%533, %536) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x56x56xbf16>, tensor<1x128x56x56xbf16>) -> tensor<1x128x56x56xbf16>
    %538 = tensor.empty() : tensor<1x128x1x1xbf16>
    %539 = "ttir.reshape"(%535, %538) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %540 = tensor.empty() : tensor<1x128x56x56xbf16>
    %541 = "ttir.broadcast"(%539, %540) <{broadcast_dimensions = array<i32: 1, 1, 56, 56>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x56x56xbf16>) -> tensor<1x128x56x56xbf16>
    %542 = tensor.empty() : tensor<1x128x56x56xbf16>
    %543 = "ttir.add"(%537, %541, %542) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x56x56xbf16>, tensor<1x128x56x56xbf16>, tensor<1x128x56x56xbf16>) -> tensor<1x128x56x56xbf16>
    %544 = tensor.empty() : tensor<1x128x56x56xbf16>
    %545 = "ttir.typecast"(%543, %544) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x128x56x56xbf16>, tensor<1x128x56x56xbf16>) -> tensor<1x128x56x56xbf16>
    %546 = tensor.empty() : tensor<1x128x56x56xbf16>
    %547 = "ttir.maximum"(%545, %3, %546) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x56x56xbf16>, tensor<1x128x56x56xbf16>, tensor<1x128x56x56xbf16>) -> tensor<1x128x56x56xbf16>
    %548 = tensor.empty() : tensor<1x128x28x28xbf16>
    %549 = "ttir.convolution"(%547, %arg13, %548) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<1x128x56x56xbf16>, tensor<128x128x3x3xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %550 = tensor.empty() : tensor<1x128x28x28xbf16>
    %551 = "ttir.typecast"(%549, %550) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %552 = tensor.empty() : tensor<1x128x28x28xbf16>
    %553 = "ttir.broadcast"(%551, %552) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %554 = tensor.empty() : tensor<1x128x1x1xbf16>
    %555 = "ttir.reshape"(%arg102, %554) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %556 = tensor.empty() : tensor<1x128x28x28xbf16>
    %557 = "ttir.broadcast"(%555, %556) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %558 = tensor.empty() : tensor<1x128x28x28xbf16>
    %559 = "ttir.subtract"(%553, %557, %558) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %560 = tensor.empty() : tensor<1x128x28x28xbf16>
    %561 = "ttir.broadcast"(%559, %560) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %562 = tensor.empty() : tensor<1x128x1x1xbf16>
    %563 = "ttir.reshape"(%arg103, %562) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %564 = tensor.empty() : tensor<1x128x28x28xbf16>
    %565 = "ttir.broadcast"(%563, %564) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %566 = tensor.empty() : tensor<1x128x28x28xbf16>
    %567 = "ttir.multiply"(%561, %565, %566) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %568 = tensor.empty() : tensor<128x1x1xbf16>
    %569 = "ttir.typecast"(%arg104, %568) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<128x1x1xbf16>, tensor<128x1x1xbf16>) -> tensor<128x1x1xbf16>
    %570 = tensor.empty() : tensor<1x128x28x28xbf16>
    %571 = "ttir.broadcast"(%567, %570) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %572 = tensor.empty() : tensor<1x128x1x1xbf16>
    %573 = "ttir.reshape"(%569, %572) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %574 = tensor.empty() : tensor<1x128x28x28xbf16>
    %575 = "ttir.broadcast"(%573, %574) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %576 = tensor.empty() : tensor<1x128x28x28xbf16>
    %577 = "ttir.multiply"(%571, %575, %576) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %578 = tensor.empty() : tensor<128x1x1xbf16>
    %579 = "ttir.typecast"(%arg105, %578) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<128x1x1xbf16>, tensor<128x1x1xbf16>) -> tensor<128x1x1xbf16>
    %580 = tensor.empty() : tensor<1x128x28x28xbf16>
    %581 = "ttir.broadcast"(%577, %580) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %582 = tensor.empty() : tensor<1x128x1x1xbf16>
    %583 = "ttir.reshape"(%579, %582) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %584 = tensor.empty() : tensor<1x128x28x28xbf16>
    %585 = "ttir.broadcast"(%583, %584) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %586 = tensor.empty() : tensor<1x128x28x28xbf16>
    %587 = "ttir.add"(%581, %585, %586) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %588 = tensor.empty() : tensor<1x128x28x28xbf16>
    %589 = "ttir.typecast"(%587, %588) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %590 = tensor.empty() : tensor<1x128x28x28xbf16>
    %591 = "ttir.maximum"(%589, %4, %590) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %592 = tensor.empty() : tensor<1x512x28x28xbf16>
    %593 = "ttir.convolution"(%591, %arg14, %592) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<512x128x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %594 = tensor.empty() : tensor<1x512x28x28xbf16>
    %595 = "ttir.typecast"(%593, %594) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %596 = tensor.empty() : tensor<1x512x28x28xbf16>
    %597 = "ttir.broadcast"(%595, %596) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %598 = tensor.empty() : tensor<1x512x1x1xbf16>
    %599 = "ttir.reshape"(%arg106, %598) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %600 = tensor.empty() : tensor<1x512x28x28xbf16>
    %601 = "ttir.broadcast"(%599, %600) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %602 = tensor.empty() : tensor<1x512x28x28xbf16>
    %603 = "ttir.subtract"(%597, %601, %602) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %604 = tensor.empty() : tensor<1x512x28x28xbf16>
    %605 = "ttir.broadcast"(%603, %604) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %606 = tensor.empty() : tensor<1x512x1x1xbf16>
    %607 = "ttir.reshape"(%arg107, %606) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %608 = tensor.empty() : tensor<1x512x28x28xbf16>
    %609 = "ttir.broadcast"(%607, %608) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %610 = tensor.empty() : tensor<1x512x28x28xbf16>
    %611 = "ttir.multiply"(%605, %609, %610) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %612 = tensor.empty() : tensor<512x1x1xbf16>
    %613 = "ttir.typecast"(%arg108, %612) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %614 = tensor.empty() : tensor<1x512x28x28xbf16>
    %615 = "ttir.broadcast"(%611, %614) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %616 = tensor.empty() : tensor<1x512x1x1xbf16>
    %617 = "ttir.reshape"(%613, %616) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %618 = tensor.empty() : tensor<1x512x28x28xbf16>
    %619 = "ttir.broadcast"(%617, %618) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %620 = tensor.empty() : tensor<1x512x28x28xbf16>
    %621 = "ttir.multiply"(%615, %619, %620) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %622 = tensor.empty() : tensor<512x1x1xbf16>
    %623 = "ttir.typecast"(%arg109, %622) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %624 = tensor.empty() : tensor<1x512x28x28xbf16>
    %625 = "ttir.broadcast"(%621, %624) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %626 = tensor.empty() : tensor<1x512x1x1xbf16>
    %627 = "ttir.reshape"(%623, %626) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %628 = tensor.empty() : tensor<1x512x28x28xbf16>
    %629 = "ttir.broadcast"(%627, %628) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %630 = tensor.empty() : tensor<1x512x28x28xbf16>
    %631 = "ttir.add"(%625, %629, %630) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %632 = tensor.empty() : tensor<1x512x28x28xbf16>
    %633 = "ttir.typecast"(%631, %632) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %634 = tensor.empty() : tensor<1x512x28x28xbf16>
    %635 = "ttir.convolution"(%503, %arg15, %634) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<1x256x56x56xbf16>, tensor<512x256x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %636 = tensor.empty() : tensor<1x512x28x28xbf16>
    %637 = "ttir.typecast"(%635, %636) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %638 = tensor.empty() : tensor<1x512x28x28xbf16>
    %639 = "ttir.broadcast"(%637, %638) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %640 = tensor.empty() : tensor<1x512x1x1xbf16>
    %641 = "ttir.reshape"(%arg110, %640) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %642 = tensor.empty() : tensor<1x512x28x28xbf16>
    %643 = "ttir.broadcast"(%641, %642) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %644 = tensor.empty() : tensor<1x512x28x28xbf16>
    %645 = "ttir.subtract"(%639, %643, %644) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %646 = tensor.empty() : tensor<1x512x28x28xbf16>
    %647 = "ttir.broadcast"(%645, %646) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %648 = tensor.empty() : tensor<1x512x1x1xbf16>
    %649 = "ttir.reshape"(%arg111, %648) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %650 = tensor.empty() : tensor<1x512x28x28xbf16>
    %651 = "ttir.broadcast"(%649, %650) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %652 = tensor.empty() : tensor<1x512x28x28xbf16>
    %653 = "ttir.multiply"(%647, %651, %652) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %654 = tensor.empty() : tensor<512x1x1xbf16>
    %655 = "ttir.typecast"(%arg112, %654) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %656 = tensor.empty() : tensor<1x512x28x28xbf16>
    %657 = "ttir.broadcast"(%653, %656) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %658 = tensor.empty() : tensor<1x512x1x1xbf16>
    %659 = "ttir.reshape"(%655, %658) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %660 = tensor.empty() : tensor<1x512x28x28xbf16>
    %661 = "ttir.broadcast"(%659, %660) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %662 = tensor.empty() : tensor<1x512x28x28xbf16>
    %663 = "ttir.multiply"(%657, %661, %662) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %664 = tensor.empty() : tensor<512x1x1xbf16>
    %665 = "ttir.typecast"(%arg113, %664) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %666 = tensor.empty() : tensor<1x512x28x28xbf16>
    %667 = "ttir.broadcast"(%663, %666) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %668 = tensor.empty() : tensor<1x512x1x1xbf16>
    %669 = "ttir.reshape"(%665, %668) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %670 = tensor.empty() : tensor<1x512x28x28xbf16>
    %671 = "ttir.broadcast"(%669, %670) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %672 = tensor.empty() : tensor<1x512x28x28xbf16>
    %673 = "ttir.add"(%667, %671, %672) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %674 = tensor.empty() : tensor<1x512x28x28xbf16>
    %675 = "ttir.typecast"(%673, %674) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %676 = tensor.empty() : tensor<1x512x28x28xbf16>
    %677 = "ttir.add"(%633, %675, %676) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %678 = tensor.empty() : tensor<1x512x28x28xbf16>
    %679 = "ttir.maximum"(%677, %5, %678) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %680 = tensor.empty() : tensor<1x128x28x28xbf16>
    %681 = "ttir.convolution"(%679, %arg16, %680) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<128x512x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %682 = tensor.empty() : tensor<1x128x28x28xbf16>
    %683 = "ttir.typecast"(%681, %682) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %684 = tensor.empty() : tensor<1x128x28x28xbf16>
    %685 = "ttir.broadcast"(%683, %684) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %686 = tensor.empty() : tensor<1x128x1x1xbf16>
    %687 = "ttir.reshape"(%arg114, %686) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %688 = tensor.empty() : tensor<1x128x28x28xbf16>
    %689 = "ttir.broadcast"(%687, %688) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %690 = tensor.empty() : tensor<1x128x28x28xbf16>
    %691 = "ttir.subtract"(%685, %689, %690) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %692 = tensor.empty() : tensor<1x128x28x28xbf16>
    %693 = "ttir.broadcast"(%691, %692) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %694 = tensor.empty() : tensor<1x128x1x1xbf16>
    %695 = "ttir.reshape"(%arg115, %694) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %696 = tensor.empty() : tensor<1x128x28x28xbf16>
    %697 = "ttir.broadcast"(%695, %696) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %698 = tensor.empty() : tensor<1x128x28x28xbf16>
    %699 = "ttir.multiply"(%693, %697, %698) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %700 = tensor.empty() : tensor<128x1x1xbf16>
    %701 = "ttir.typecast"(%arg116, %700) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<128x1x1xbf16>, tensor<128x1x1xbf16>) -> tensor<128x1x1xbf16>
    %702 = tensor.empty() : tensor<1x128x28x28xbf16>
    %703 = "ttir.broadcast"(%699, %702) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %704 = tensor.empty() : tensor<1x128x1x1xbf16>
    %705 = "ttir.reshape"(%701, %704) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %706 = tensor.empty() : tensor<1x128x28x28xbf16>
    %707 = "ttir.broadcast"(%705, %706) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %708 = tensor.empty() : tensor<1x128x28x28xbf16>
    %709 = "ttir.multiply"(%703, %707, %708) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %710 = tensor.empty() : tensor<128x1x1xbf16>
    %711 = "ttir.typecast"(%arg117, %710) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<128x1x1xbf16>, tensor<128x1x1xbf16>) -> tensor<128x1x1xbf16>
    %712 = tensor.empty() : tensor<1x128x28x28xbf16>
    %713 = "ttir.broadcast"(%709, %712) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %714 = tensor.empty() : tensor<1x128x1x1xbf16>
    %715 = "ttir.reshape"(%711, %714) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %716 = tensor.empty() : tensor<1x128x28x28xbf16>
    %717 = "ttir.broadcast"(%715, %716) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %718 = tensor.empty() : tensor<1x128x28x28xbf16>
    %719 = "ttir.add"(%713, %717, %718) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %720 = tensor.empty() : tensor<1x128x28x28xbf16>
    %721 = "ttir.typecast"(%719, %720) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %722 = tensor.empty() : tensor<1x128x28x28xbf16>
    %723 = "ttir.maximum"(%721, %4, %722) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %724 = tensor.empty() : tensor<1x128x28x28xbf16>
    %725 = "ttir.convolution"(%723, %arg17, %724) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %726 = tensor.empty() : tensor<1x128x28x28xbf16>
    %727 = "ttir.typecast"(%725, %726) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %728 = tensor.empty() : tensor<1x128x28x28xbf16>
    %729 = "ttir.broadcast"(%727, %728) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %730 = tensor.empty() : tensor<1x128x1x1xbf16>
    %731 = "ttir.reshape"(%arg118, %730) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %732 = tensor.empty() : tensor<1x128x28x28xbf16>
    %733 = "ttir.broadcast"(%731, %732) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %734 = tensor.empty() : tensor<1x128x28x28xbf16>
    %735 = "ttir.subtract"(%729, %733, %734) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %736 = tensor.empty() : tensor<1x128x28x28xbf16>
    %737 = "ttir.broadcast"(%735, %736) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %738 = tensor.empty() : tensor<1x128x1x1xbf16>
    %739 = "ttir.reshape"(%arg119, %738) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %740 = tensor.empty() : tensor<1x128x28x28xbf16>
    %741 = "ttir.broadcast"(%739, %740) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %742 = tensor.empty() : tensor<1x128x28x28xbf16>
    %743 = "ttir.multiply"(%737, %741, %742) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %744 = tensor.empty() : tensor<128x1x1xbf16>
    %745 = "ttir.typecast"(%arg120, %744) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<128x1x1xbf16>, tensor<128x1x1xbf16>) -> tensor<128x1x1xbf16>
    %746 = tensor.empty() : tensor<1x128x28x28xbf16>
    %747 = "ttir.broadcast"(%743, %746) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %748 = tensor.empty() : tensor<1x128x1x1xbf16>
    %749 = "ttir.reshape"(%745, %748) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %750 = tensor.empty() : tensor<1x128x28x28xbf16>
    %751 = "ttir.broadcast"(%749, %750) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %752 = tensor.empty() : tensor<1x128x28x28xbf16>
    %753 = "ttir.multiply"(%747, %751, %752) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %754 = tensor.empty() : tensor<128x1x1xbf16>
    %755 = "ttir.typecast"(%arg121, %754) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<128x1x1xbf16>, tensor<128x1x1xbf16>) -> tensor<128x1x1xbf16>
    %756 = tensor.empty() : tensor<1x128x28x28xbf16>
    %757 = "ttir.broadcast"(%753, %756) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %758 = tensor.empty() : tensor<1x128x1x1xbf16>
    %759 = "ttir.reshape"(%755, %758) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %760 = tensor.empty() : tensor<1x128x28x28xbf16>
    %761 = "ttir.broadcast"(%759, %760) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %762 = tensor.empty() : tensor<1x128x28x28xbf16>
    %763 = "ttir.add"(%757, %761, %762) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %764 = tensor.empty() : tensor<1x128x28x28xbf16>
    %765 = "ttir.typecast"(%763, %764) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %766 = tensor.empty() : tensor<1x128x28x28xbf16>
    %767 = "ttir.maximum"(%765, %4, %766) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %768 = tensor.empty() : tensor<1x512x28x28xbf16>
    %769 = "ttir.convolution"(%767, %arg18, %768) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<512x128x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %770 = tensor.empty() : tensor<1x512x28x28xbf16>
    %771 = "ttir.typecast"(%769, %770) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %772 = tensor.empty() : tensor<1x512x28x28xbf16>
    %773 = "ttir.broadcast"(%771, %772) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %774 = tensor.empty() : tensor<1x512x1x1xbf16>
    %775 = "ttir.reshape"(%arg122, %774) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %776 = tensor.empty() : tensor<1x512x28x28xbf16>
    %777 = "ttir.broadcast"(%775, %776) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %778 = tensor.empty() : tensor<1x512x28x28xbf16>
    %779 = "ttir.subtract"(%773, %777, %778) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %780 = tensor.empty() : tensor<1x512x28x28xbf16>
    %781 = "ttir.broadcast"(%779, %780) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %782 = tensor.empty() : tensor<1x512x1x1xbf16>
    %783 = "ttir.reshape"(%arg123, %782) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %784 = tensor.empty() : tensor<1x512x28x28xbf16>
    %785 = "ttir.broadcast"(%783, %784) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %786 = tensor.empty() : tensor<1x512x28x28xbf16>
    %787 = "ttir.multiply"(%781, %785, %786) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %788 = tensor.empty() : tensor<512x1x1xbf16>
    %789 = "ttir.typecast"(%arg124, %788) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %790 = tensor.empty() : tensor<1x512x28x28xbf16>
    %791 = "ttir.broadcast"(%787, %790) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %792 = tensor.empty() : tensor<1x512x1x1xbf16>
    %793 = "ttir.reshape"(%789, %792) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %794 = tensor.empty() : tensor<1x512x28x28xbf16>
    %795 = "ttir.broadcast"(%793, %794) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %796 = tensor.empty() : tensor<1x512x28x28xbf16>
    %797 = "ttir.multiply"(%791, %795, %796) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %798 = tensor.empty() : tensor<512x1x1xbf16>
    %799 = "ttir.typecast"(%arg125, %798) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %800 = tensor.empty() : tensor<1x512x28x28xbf16>
    %801 = "ttir.broadcast"(%797, %800) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %802 = tensor.empty() : tensor<1x512x1x1xbf16>
    %803 = "ttir.reshape"(%799, %802) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %804 = tensor.empty() : tensor<1x512x28x28xbf16>
    %805 = "ttir.broadcast"(%803, %804) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %806 = tensor.empty() : tensor<1x512x28x28xbf16>
    %807 = "ttir.add"(%801, %805, %806) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %808 = tensor.empty() : tensor<1x512x28x28xbf16>
    %809 = "ttir.typecast"(%807, %808) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %810 = tensor.empty() : tensor<1x512x28x28xbf16>
    %811 = "ttir.add"(%809, %679, %810) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %812 = tensor.empty() : tensor<1x512x28x28xbf16>
    %813 = "ttir.maximum"(%811, %5, %812) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %814 = tensor.empty() : tensor<1x128x28x28xbf16>
    %815 = "ttir.convolution"(%813, %arg19, %814) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<128x512x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %816 = tensor.empty() : tensor<1x128x28x28xbf16>
    %817 = "ttir.typecast"(%815, %816) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %818 = tensor.empty() : tensor<1x128x28x28xbf16>
    %819 = "ttir.broadcast"(%817, %818) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %820 = tensor.empty() : tensor<1x128x1x1xbf16>
    %821 = "ttir.reshape"(%arg126, %820) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %822 = tensor.empty() : tensor<1x128x28x28xbf16>
    %823 = "ttir.broadcast"(%821, %822) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %824 = tensor.empty() : tensor<1x128x28x28xbf16>
    %825 = "ttir.subtract"(%819, %823, %824) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %826 = tensor.empty() : tensor<1x128x28x28xbf16>
    %827 = "ttir.broadcast"(%825, %826) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %828 = tensor.empty() : tensor<1x128x1x1xbf16>
    %829 = "ttir.reshape"(%arg127, %828) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %830 = tensor.empty() : tensor<1x128x28x28xbf16>
    %831 = "ttir.broadcast"(%829, %830) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %832 = tensor.empty() : tensor<1x128x28x28xbf16>
    %833 = "ttir.multiply"(%827, %831, %832) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %834 = tensor.empty() : tensor<128x1x1xbf16>
    %835 = "ttir.typecast"(%arg128, %834) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<128x1x1xbf16>, tensor<128x1x1xbf16>) -> tensor<128x1x1xbf16>
    %836 = tensor.empty() : tensor<1x128x28x28xbf16>
    %837 = "ttir.broadcast"(%833, %836) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %838 = tensor.empty() : tensor<1x128x1x1xbf16>
    %839 = "ttir.reshape"(%835, %838) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %840 = tensor.empty() : tensor<1x128x28x28xbf16>
    %841 = "ttir.broadcast"(%839, %840) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %842 = tensor.empty() : tensor<1x128x28x28xbf16>
    %843 = "ttir.multiply"(%837, %841, %842) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %844 = tensor.empty() : tensor<128x1x1xbf16>
    %845 = "ttir.typecast"(%arg129, %844) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<128x1x1xbf16>, tensor<128x1x1xbf16>) -> tensor<128x1x1xbf16>
    %846 = tensor.empty() : tensor<1x128x28x28xbf16>
    %847 = "ttir.broadcast"(%843, %846) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %848 = tensor.empty() : tensor<1x128x1x1xbf16>
    %849 = "ttir.reshape"(%845, %848) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %850 = tensor.empty() : tensor<1x128x28x28xbf16>
    %851 = "ttir.broadcast"(%849, %850) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %852 = tensor.empty() : tensor<1x128x28x28xbf16>
    %853 = "ttir.add"(%847, %851, %852) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %854 = tensor.empty() : tensor<1x128x28x28xbf16>
    %855 = "ttir.typecast"(%853, %854) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %856 = tensor.empty() : tensor<1x128x28x28xbf16>
    %857 = "ttir.maximum"(%855, %4, %856) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %858 = tensor.empty() : tensor<1x128x28x28xbf16>
    %859 = "ttir.convolution"(%857, %arg20, %858) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %860 = tensor.empty() : tensor<1x128x28x28xbf16>
    %861 = "ttir.typecast"(%859, %860) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %862 = tensor.empty() : tensor<1x128x28x28xbf16>
    %863 = "ttir.broadcast"(%861, %862) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %864 = tensor.empty() : tensor<1x128x1x1xbf16>
    %865 = "ttir.reshape"(%arg130, %864) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %866 = tensor.empty() : tensor<1x128x28x28xbf16>
    %867 = "ttir.broadcast"(%865, %866) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %868 = tensor.empty() : tensor<1x128x28x28xbf16>
    %869 = "ttir.subtract"(%863, %867, %868) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %870 = tensor.empty() : tensor<1x128x28x28xbf16>
    %871 = "ttir.broadcast"(%869, %870) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %872 = tensor.empty() : tensor<1x128x1x1xbf16>
    %873 = "ttir.reshape"(%arg131, %872) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %874 = tensor.empty() : tensor<1x128x28x28xbf16>
    %875 = "ttir.broadcast"(%873, %874) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %876 = tensor.empty() : tensor<1x128x28x28xbf16>
    %877 = "ttir.multiply"(%871, %875, %876) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %878 = tensor.empty() : tensor<128x1x1xbf16>
    %879 = "ttir.typecast"(%arg132, %878) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<128x1x1xbf16>, tensor<128x1x1xbf16>) -> tensor<128x1x1xbf16>
    %880 = tensor.empty() : tensor<1x128x28x28xbf16>
    %881 = "ttir.broadcast"(%877, %880) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %882 = tensor.empty() : tensor<1x128x1x1xbf16>
    %883 = "ttir.reshape"(%879, %882) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %884 = tensor.empty() : tensor<1x128x28x28xbf16>
    %885 = "ttir.broadcast"(%883, %884) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %886 = tensor.empty() : tensor<1x128x28x28xbf16>
    %887 = "ttir.multiply"(%881, %885, %886) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %888 = tensor.empty() : tensor<128x1x1xbf16>
    %889 = "ttir.typecast"(%arg133, %888) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<128x1x1xbf16>, tensor<128x1x1xbf16>) -> tensor<128x1x1xbf16>
    %890 = tensor.empty() : tensor<1x128x28x28xbf16>
    %891 = "ttir.broadcast"(%887, %890) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %892 = tensor.empty() : tensor<1x128x1x1xbf16>
    %893 = "ttir.reshape"(%889, %892) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %894 = tensor.empty() : tensor<1x128x28x28xbf16>
    %895 = "ttir.broadcast"(%893, %894) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %896 = tensor.empty() : tensor<1x128x28x28xbf16>
    %897 = "ttir.add"(%891, %895, %896) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %898 = tensor.empty() : tensor<1x128x28x28xbf16>
    %899 = "ttir.typecast"(%897, %898) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %900 = tensor.empty() : tensor<1x128x28x28xbf16>
    %901 = "ttir.maximum"(%899, %4, %900) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %902 = tensor.empty() : tensor<1x512x28x28xbf16>
    %903 = "ttir.convolution"(%901, %arg21, %902) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<512x128x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %904 = tensor.empty() : tensor<1x512x28x28xbf16>
    %905 = "ttir.typecast"(%903, %904) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %906 = tensor.empty() : tensor<1x512x28x28xbf16>
    %907 = "ttir.broadcast"(%905, %906) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %908 = tensor.empty() : tensor<1x512x1x1xbf16>
    %909 = "ttir.reshape"(%arg134, %908) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %910 = tensor.empty() : tensor<1x512x28x28xbf16>
    %911 = "ttir.broadcast"(%909, %910) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %912 = tensor.empty() : tensor<1x512x28x28xbf16>
    %913 = "ttir.subtract"(%907, %911, %912) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %914 = tensor.empty() : tensor<1x512x28x28xbf16>
    %915 = "ttir.broadcast"(%913, %914) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %916 = tensor.empty() : tensor<1x512x1x1xbf16>
    %917 = "ttir.reshape"(%arg135, %916) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %918 = tensor.empty() : tensor<1x512x28x28xbf16>
    %919 = "ttir.broadcast"(%917, %918) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %920 = tensor.empty() : tensor<1x512x28x28xbf16>
    %921 = "ttir.multiply"(%915, %919, %920) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %922 = tensor.empty() : tensor<512x1x1xbf16>
    %923 = "ttir.typecast"(%arg136, %922) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %924 = tensor.empty() : tensor<1x512x28x28xbf16>
    %925 = "ttir.broadcast"(%921, %924) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %926 = tensor.empty() : tensor<1x512x1x1xbf16>
    %927 = "ttir.reshape"(%923, %926) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %928 = tensor.empty() : tensor<1x512x28x28xbf16>
    %929 = "ttir.broadcast"(%927, %928) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %930 = tensor.empty() : tensor<1x512x28x28xbf16>
    %931 = "ttir.multiply"(%925, %929, %930) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %932 = tensor.empty() : tensor<512x1x1xbf16>
    %933 = "ttir.typecast"(%arg137, %932) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %934 = tensor.empty() : tensor<1x512x28x28xbf16>
    %935 = "ttir.broadcast"(%931, %934) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %936 = tensor.empty() : tensor<1x512x1x1xbf16>
    %937 = "ttir.reshape"(%933, %936) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %938 = tensor.empty() : tensor<1x512x28x28xbf16>
    %939 = "ttir.broadcast"(%937, %938) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %940 = tensor.empty() : tensor<1x512x28x28xbf16>
    %941 = "ttir.add"(%935, %939, %940) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %942 = tensor.empty() : tensor<1x512x28x28xbf16>
    %943 = "ttir.typecast"(%941, %942) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %944 = tensor.empty() : tensor<1x512x28x28xbf16>
    %945 = "ttir.add"(%943, %813, %944) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %946 = tensor.empty() : tensor<1x512x28x28xbf16>
    %947 = "ttir.maximum"(%945, %5, %946) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %948 = tensor.empty() : tensor<1x128x28x28xbf16>
    %949 = "ttir.convolution"(%947, %arg22, %948) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<128x512x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %950 = tensor.empty() : tensor<1x128x28x28xbf16>
    %951 = "ttir.typecast"(%949, %950) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %952 = tensor.empty() : tensor<1x128x28x28xbf16>
    %953 = "ttir.broadcast"(%951, %952) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %954 = tensor.empty() : tensor<1x128x1x1xbf16>
    %955 = "ttir.reshape"(%arg138, %954) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %956 = tensor.empty() : tensor<1x128x28x28xbf16>
    %957 = "ttir.broadcast"(%955, %956) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %958 = tensor.empty() : tensor<1x128x28x28xbf16>
    %959 = "ttir.subtract"(%953, %957, %958) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %960 = tensor.empty() : tensor<1x128x28x28xbf16>
    %961 = "ttir.broadcast"(%959, %960) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %962 = tensor.empty() : tensor<1x128x1x1xbf16>
    %963 = "ttir.reshape"(%arg139, %962) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %964 = tensor.empty() : tensor<1x128x28x28xbf16>
    %965 = "ttir.broadcast"(%963, %964) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %966 = tensor.empty() : tensor<1x128x28x28xbf16>
    %967 = "ttir.multiply"(%961, %965, %966) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %968 = tensor.empty() : tensor<128x1x1xbf16>
    %969 = "ttir.typecast"(%arg140, %968) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<128x1x1xbf16>, tensor<128x1x1xbf16>) -> tensor<128x1x1xbf16>
    %970 = tensor.empty() : tensor<1x128x28x28xbf16>
    %971 = "ttir.broadcast"(%967, %970) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %972 = tensor.empty() : tensor<1x128x1x1xbf16>
    %973 = "ttir.reshape"(%969, %972) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %974 = tensor.empty() : tensor<1x128x28x28xbf16>
    %975 = "ttir.broadcast"(%973, %974) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %976 = tensor.empty() : tensor<1x128x28x28xbf16>
    %977 = "ttir.multiply"(%971, %975, %976) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %978 = tensor.empty() : tensor<128x1x1xbf16>
    %979 = "ttir.typecast"(%arg141, %978) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<128x1x1xbf16>, tensor<128x1x1xbf16>) -> tensor<128x1x1xbf16>
    %980 = tensor.empty() : tensor<1x128x28x28xbf16>
    %981 = "ttir.broadcast"(%977, %980) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %982 = tensor.empty() : tensor<1x128x1x1xbf16>
    %983 = "ttir.reshape"(%979, %982) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %984 = tensor.empty() : tensor<1x128x28x28xbf16>
    %985 = "ttir.broadcast"(%983, %984) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %986 = tensor.empty() : tensor<1x128x28x28xbf16>
    %987 = "ttir.add"(%981, %985, %986) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %988 = tensor.empty() : tensor<1x128x28x28xbf16>
    %989 = "ttir.typecast"(%987, %988) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %990 = tensor.empty() : tensor<1x128x28x28xbf16>
    %991 = "ttir.maximum"(%989, %4, %990) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %992 = tensor.empty() : tensor<1x128x28x28xbf16>
    %993 = "ttir.convolution"(%991, %arg23, %992) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<128x128x3x3xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %994 = tensor.empty() : tensor<1x128x28x28xbf16>
    %995 = "ttir.typecast"(%993, %994) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %996 = tensor.empty() : tensor<1x128x28x28xbf16>
    %997 = "ttir.broadcast"(%995, %996) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %998 = tensor.empty() : tensor<1x128x1x1xbf16>
    %999 = "ttir.reshape"(%arg142, %998) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %1000 = tensor.empty() : tensor<1x128x28x28xbf16>
    %1001 = "ttir.broadcast"(%999, %1000) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %1002 = tensor.empty() : tensor<1x128x28x28xbf16>
    %1003 = "ttir.subtract"(%997, %1001, %1002) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %1004 = tensor.empty() : tensor<1x128x28x28xbf16>
    %1005 = "ttir.broadcast"(%1003, %1004) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %1006 = tensor.empty() : tensor<1x128x1x1xbf16>
    %1007 = "ttir.reshape"(%arg143, %1006) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %1008 = tensor.empty() : tensor<1x128x28x28xbf16>
    %1009 = "ttir.broadcast"(%1007, %1008) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %1010 = tensor.empty() : tensor<1x128x28x28xbf16>
    %1011 = "ttir.multiply"(%1005, %1009, %1010) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %1012 = tensor.empty() : tensor<128x1x1xbf16>
    %1013 = "ttir.typecast"(%arg144, %1012) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<128x1x1xbf16>, tensor<128x1x1xbf16>) -> tensor<128x1x1xbf16>
    %1014 = tensor.empty() : tensor<1x128x28x28xbf16>
    %1015 = "ttir.broadcast"(%1011, %1014) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %1016 = tensor.empty() : tensor<1x128x1x1xbf16>
    %1017 = "ttir.reshape"(%1013, %1016) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %1018 = tensor.empty() : tensor<1x128x28x28xbf16>
    %1019 = "ttir.broadcast"(%1017, %1018) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %1020 = tensor.empty() : tensor<1x128x28x28xbf16>
    %1021 = "ttir.multiply"(%1015, %1019, %1020) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %1022 = tensor.empty() : tensor<128x1x1xbf16>
    %1023 = "ttir.typecast"(%arg145, %1022) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<128x1x1xbf16>, tensor<128x1x1xbf16>) -> tensor<128x1x1xbf16>
    %1024 = tensor.empty() : tensor<1x128x28x28xbf16>
    %1025 = "ttir.broadcast"(%1021, %1024) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %1026 = tensor.empty() : tensor<1x128x1x1xbf16>
    %1027 = "ttir.reshape"(%1023, %1026) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %1028 = tensor.empty() : tensor<1x128x28x28xbf16>
    %1029 = "ttir.broadcast"(%1027, %1028) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %1030 = tensor.empty() : tensor<1x128x28x28xbf16>
    %1031 = "ttir.add"(%1025, %1029, %1030) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %1032 = tensor.empty() : tensor<1x128x28x28xbf16>
    %1033 = "ttir.typecast"(%1031, %1032) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %1034 = tensor.empty() : tensor<1x128x28x28xbf16>
    %1035 = "ttir.maximum"(%1033, %4, %1034) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %1036 = tensor.empty() : tensor<1x512x28x28xbf16>
    %1037 = "ttir.convolution"(%1035, %arg24, %1036) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x128x28x28xbf16>, tensor<512x128x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %1038 = tensor.empty() : tensor<1x512x28x28xbf16>
    %1039 = "ttir.typecast"(%1037, %1038) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %1040 = tensor.empty() : tensor<1x512x28x28xbf16>
    %1041 = "ttir.broadcast"(%1039, %1040) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %1042 = tensor.empty() : tensor<1x512x1x1xbf16>
    %1043 = "ttir.reshape"(%arg146, %1042) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %1044 = tensor.empty() : tensor<1x512x28x28xbf16>
    %1045 = "ttir.broadcast"(%1043, %1044) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %1046 = tensor.empty() : tensor<1x512x28x28xbf16>
    %1047 = "ttir.subtract"(%1041, %1045, %1046) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %1048 = tensor.empty() : tensor<1x512x28x28xbf16>
    %1049 = "ttir.broadcast"(%1047, %1048) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %1050 = tensor.empty() : tensor<1x512x1x1xbf16>
    %1051 = "ttir.reshape"(%arg147, %1050) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %1052 = tensor.empty() : tensor<1x512x28x28xbf16>
    %1053 = "ttir.broadcast"(%1051, %1052) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %1054 = tensor.empty() : tensor<1x512x28x28xbf16>
    %1055 = "ttir.multiply"(%1049, %1053, %1054) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %1056 = tensor.empty() : tensor<512x1x1xbf16>
    %1057 = "ttir.typecast"(%arg148, %1056) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %1058 = tensor.empty() : tensor<1x512x28x28xbf16>
    %1059 = "ttir.broadcast"(%1055, %1058) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %1060 = tensor.empty() : tensor<1x512x1x1xbf16>
    %1061 = "ttir.reshape"(%1057, %1060) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %1062 = tensor.empty() : tensor<1x512x28x28xbf16>
    %1063 = "ttir.broadcast"(%1061, %1062) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %1064 = tensor.empty() : tensor<1x512x28x28xbf16>
    %1065 = "ttir.multiply"(%1059, %1063, %1064) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %1066 = tensor.empty() : tensor<512x1x1xbf16>
    %1067 = "ttir.typecast"(%arg149, %1066) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %1068 = tensor.empty() : tensor<1x512x28x28xbf16>
    %1069 = "ttir.broadcast"(%1065, %1068) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %1070 = tensor.empty() : tensor<1x512x1x1xbf16>
    %1071 = "ttir.reshape"(%1067, %1070) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %1072 = tensor.empty() : tensor<1x512x28x28xbf16>
    %1073 = "ttir.broadcast"(%1071, %1072) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %1074 = tensor.empty() : tensor<1x512x28x28xbf16>
    %1075 = "ttir.add"(%1069, %1073, %1074) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %1076 = tensor.empty() : tensor<1x512x28x28xbf16>
    %1077 = "ttir.typecast"(%1075, %1076) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %1078 = tensor.empty() : tensor<1x512x28x28xbf16>
    %1079 = "ttir.add"(%1077, %947, %1078) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %1080 = tensor.empty() : tensor<1x512x28x28xbf16>
    %1081 = "ttir.maximum"(%1079, %5, %1080) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %1082 = tensor.empty() : tensor<1x256x28x28xbf16>
    %1083 = "ttir.convolution"(%1081, %arg25, %1082) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x512x28x28xbf16>, tensor<256x512x1x1xbf16>, tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xbf16>
    %1084 = tensor.empty() : tensor<1x256x28x28xbf16>
    %1085 = "ttir.typecast"(%1083, %1084) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x28x28xbf16>, tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xbf16>
    %1086 = tensor.empty() : tensor<1x256x28x28xbf16>
    %1087 = "ttir.broadcast"(%1085, %1086) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x28x28xbf16>, tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xbf16>
    %1088 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1089 = "ttir.reshape"(%arg150, %1088) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1090 = tensor.empty() : tensor<1x256x28x28xbf16>
    %1091 = "ttir.broadcast"(%1089, %1090) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xbf16>
    %1092 = tensor.empty() : tensor<1x256x28x28xbf16>
    %1093 = "ttir.subtract"(%1087, %1091, %1092) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x28x28xbf16>, tensor<1x256x28x28xbf16>, tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xbf16>
    %1094 = tensor.empty() : tensor<1x256x28x28xbf16>
    %1095 = "ttir.broadcast"(%1093, %1094) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x28x28xbf16>, tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xbf16>
    %1096 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1097 = "ttir.reshape"(%arg151, %1096) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1098 = tensor.empty() : tensor<1x256x28x28xbf16>
    %1099 = "ttir.broadcast"(%1097, %1098) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xbf16>
    %1100 = tensor.empty() : tensor<1x256x28x28xbf16>
    %1101 = "ttir.multiply"(%1095, %1099, %1100) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x28x28xbf16>, tensor<1x256x28x28xbf16>, tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xbf16>
    %1102 = tensor.empty() : tensor<256x1x1xbf16>
    %1103 = "ttir.typecast"(%arg152, %1102) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1104 = tensor.empty() : tensor<1x256x28x28xbf16>
    %1105 = "ttir.broadcast"(%1101, %1104) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x28x28xbf16>, tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xbf16>
    %1106 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1107 = "ttir.reshape"(%1103, %1106) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1108 = tensor.empty() : tensor<1x256x28x28xbf16>
    %1109 = "ttir.broadcast"(%1107, %1108) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xbf16>
    %1110 = tensor.empty() : tensor<1x256x28x28xbf16>
    %1111 = "ttir.multiply"(%1105, %1109, %1110) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x28x28xbf16>, tensor<1x256x28x28xbf16>, tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xbf16>
    %1112 = tensor.empty() : tensor<256x1x1xbf16>
    %1113 = "ttir.typecast"(%arg153, %1112) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1114 = tensor.empty() : tensor<1x256x28x28xbf16>
    %1115 = "ttir.broadcast"(%1111, %1114) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x28x28xbf16>, tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xbf16>
    %1116 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1117 = "ttir.reshape"(%1113, %1116) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1118 = tensor.empty() : tensor<1x256x28x28xbf16>
    %1119 = "ttir.broadcast"(%1117, %1118) <{broadcast_dimensions = array<i32: 1, 1, 28, 28>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xbf16>
    %1120 = tensor.empty() : tensor<1x256x28x28xbf16>
    %1121 = "ttir.add"(%1115, %1119, %1120) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x28x28xbf16>, tensor<1x256x28x28xbf16>, tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xbf16>
    %1122 = tensor.empty() : tensor<1x256x28x28xbf16>
    %1123 = "ttir.typecast"(%1121, %1122) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x28x28xbf16>, tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xbf16>
    %1124 = tensor.empty() : tensor<1x256x28x28xbf16>
    %1125 = "ttir.maximum"(%1123, %6, %1124) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x28x28xbf16>, tensor<1x256x28x28xbf16>, tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xbf16>
    %1126 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1127 = "ttir.convolution"(%1125, %arg26, %1126) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<1x256x28x28xbf16>, tensor<256x256x3x3xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1128 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1129 = "ttir.typecast"(%1127, %1128) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1130 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1131 = "ttir.broadcast"(%1129, %1130) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1132 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1133 = "ttir.reshape"(%arg154, %1132) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1134 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1135 = "ttir.broadcast"(%1133, %1134) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1136 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1137 = "ttir.subtract"(%1131, %1135, %1136) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1138 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1139 = "ttir.broadcast"(%1137, %1138) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1140 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1141 = "ttir.reshape"(%arg155, %1140) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1142 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1143 = "ttir.broadcast"(%1141, %1142) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1144 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1145 = "ttir.multiply"(%1139, %1143, %1144) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1146 = tensor.empty() : tensor<256x1x1xbf16>
    %1147 = "ttir.typecast"(%arg156, %1146) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1148 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1149 = "ttir.broadcast"(%1145, %1148) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1150 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1151 = "ttir.reshape"(%1147, %1150) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1152 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1153 = "ttir.broadcast"(%1151, %1152) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1154 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1155 = "ttir.multiply"(%1149, %1153, %1154) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1156 = tensor.empty() : tensor<256x1x1xbf16>
    %1157 = "ttir.typecast"(%arg157, %1156) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1158 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1159 = "ttir.broadcast"(%1155, %1158) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1160 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1161 = "ttir.reshape"(%1157, %1160) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1162 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1163 = "ttir.broadcast"(%1161, %1162) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1164 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1165 = "ttir.add"(%1159, %1163, %1164) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1166 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1167 = "ttir.typecast"(%1165, %1166) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1168 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1169 = "ttir.maximum"(%1167, %7, %1168) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1170 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1171 = "ttir.convolution"(%1169, %arg27, %1170) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1024x256x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1172 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1173 = "ttir.typecast"(%1171, %1172) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1174 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1175 = "ttir.broadcast"(%1173, %1174) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1176 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1177 = "ttir.reshape"(%arg158, %1176) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1178 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1179 = "ttir.broadcast"(%1177, %1178) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1180 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1181 = "ttir.subtract"(%1175, %1179, %1180) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1182 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1183 = "ttir.broadcast"(%1181, %1182) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1184 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1185 = "ttir.reshape"(%arg159, %1184) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1186 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1187 = "ttir.broadcast"(%1185, %1186) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1188 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1189 = "ttir.multiply"(%1183, %1187, %1188) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1190 = tensor.empty() : tensor<1024x1x1xbf16>
    %1191 = "ttir.typecast"(%arg160, %1190) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1024x1x1xbf16>, tensor<1024x1x1xbf16>) -> tensor<1024x1x1xbf16>
    %1192 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1193 = "ttir.broadcast"(%1189, %1192) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1194 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1195 = "ttir.reshape"(%1191, %1194) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1196 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1197 = "ttir.broadcast"(%1195, %1196) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1198 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1199 = "ttir.multiply"(%1193, %1197, %1198) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1200 = tensor.empty() : tensor<1024x1x1xbf16>
    %1201 = "ttir.typecast"(%arg161, %1200) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1024x1x1xbf16>, tensor<1024x1x1xbf16>) -> tensor<1024x1x1xbf16>
    %1202 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1203 = "ttir.broadcast"(%1199, %1202) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1204 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1205 = "ttir.reshape"(%1201, %1204) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1206 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1207 = "ttir.broadcast"(%1205, %1206) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1208 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1209 = "ttir.add"(%1203, %1207, %1208) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1210 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1211 = "ttir.typecast"(%1209, %1210) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1212 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1213 = "ttir.convolution"(%1081, %arg28, %1212) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<1x512x28x28xbf16>, tensor<1024x512x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1214 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1215 = "ttir.typecast"(%1213, %1214) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1216 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1217 = "ttir.broadcast"(%1215, %1216) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1218 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1219 = "ttir.reshape"(%arg162, %1218) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1220 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1221 = "ttir.broadcast"(%1219, %1220) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1222 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1223 = "ttir.subtract"(%1217, %1221, %1222) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1224 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1225 = "ttir.broadcast"(%1223, %1224) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1226 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1227 = "ttir.reshape"(%arg163, %1226) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1228 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1229 = "ttir.broadcast"(%1227, %1228) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1230 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1231 = "ttir.multiply"(%1225, %1229, %1230) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1232 = tensor.empty() : tensor<1024x1x1xbf16>
    %1233 = "ttir.typecast"(%arg164, %1232) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1024x1x1xbf16>, tensor<1024x1x1xbf16>) -> tensor<1024x1x1xbf16>
    %1234 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1235 = "ttir.broadcast"(%1231, %1234) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1236 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1237 = "ttir.reshape"(%1233, %1236) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1238 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1239 = "ttir.broadcast"(%1237, %1238) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1240 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1241 = "ttir.multiply"(%1235, %1239, %1240) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1242 = tensor.empty() : tensor<1024x1x1xbf16>
    %1243 = "ttir.typecast"(%arg165, %1242) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1024x1x1xbf16>, tensor<1024x1x1xbf16>) -> tensor<1024x1x1xbf16>
    %1244 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1245 = "ttir.broadcast"(%1241, %1244) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1246 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1247 = "ttir.reshape"(%1243, %1246) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1248 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1249 = "ttir.broadcast"(%1247, %1248) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1250 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1251 = "ttir.add"(%1245, %1249, %1250) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1252 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1253 = "ttir.typecast"(%1251, %1252) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1254 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1255 = "ttir.add"(%1211, %1253, %1254) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1256 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1257 = "ttir.maximum"(%1255, %8, %1256) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1258 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1259 = "ttir.convolution"(%1257, %arg29, %1258) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1260 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1261 = "ttir.typecast"(%1259, %1260) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1262 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1263 = "ttir.broadcast"(%1261, %1262) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1264 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1265 = "ttir.reshape"(%arg166, %1264) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1266 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1267 = "ttir.broadcast"(%1265, %1266) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1268 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1269 = "ttir.subtract"(%1263, %1267, %1268) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1270 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1271 = "ttir.broadcast"(%1269, %1270) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1272 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1273 = "ttir.reshape"(%arg167, %1272) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1274 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1275 = "ttir.broadcast"(%1273, %1274) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1276 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1277 = "ttir.multiply"(%1271, %1275, %1276) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1278 = tensor.empty() : tensor<256x1x1xbf16>
    %1279 = "ttir.typecast"(%arg168, %1278) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1280 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1281 = "ttir.broadcast"(%1277, %1280) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1282 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1283 = "ttir.reshape"(%1279, %1282) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1284 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1285 = "ttir.broadcast"(%1283, %1284) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1286 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1287 = "ttir.multiply"(%1281, %1285, %1286) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1288 = tensor.empty() : tensor<256x1x1xbf16>
    %1289 = "ttir.typecast"(%arg169, %1288) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1290 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1291 = "ttir.broadcast"(%1287, %1290) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1292 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1293 = "ttir.reshape"(%1289, %1292) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1294 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1295 = "ttir.broadcast"(%1293, %1294) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1296 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1297 = "ttir.add"(%1291, %1295, %1296) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1298 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1299 = "ttir.typecast"(%1297, %1298) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1300 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1301 = "ttir.maximum"(%1299, %7, %1300) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1302 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1303 = "ttir.convolution"(%1301, %arg30, %1302) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<256x256x3x3xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1304 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1305 = "ttir.typecast"(%1303, %1304) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1306 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1307 = "ttir.broadcast"(%1305, %1306) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1308 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1309 = "ttir.reshape"(%arg170, %1308) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1310 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1311 = "ttir.broadcast"(%1309, %1310) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1312 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1313 = "ttir.subtract"(%1307, %1311, %1312) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1314 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1315 = "ttir.broadcast"(%1313, %1314) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1316 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1317 = "ttir.reshape"(%arg171, %1316) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1318 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1319 = "ttir.broadcast"(%1317, %1318) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1320 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1321 = "ttir.multiply"(%1315, %1319, %1320) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1322 = tensor.empty() : tensor<256x1x1xbf16>
    %1323 = "ttir.typecast"(%arg172, %1322) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1324 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1325 = "ttir.broadcast"(%1321, %1324) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1326 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1327 = "ttir.reshape"(%1323, %1326) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1328 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1329 = "ttir.broadcast"(%1327, %1328) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1330 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1331 = "ttir.multiply"(%1325, %1329, %1330) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1332 = tensor.empty() : tensor<256x1x1xbf16>
    %1333 = "ttir.typecast"(%arg173, %1332) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1334 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1335 = "ttir.broadcast"(%1331, %1334) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1336 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1337 = "ttir.reshape"(%1333, %1336) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1338 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1339 = "ttir.broadcast"(%1337, %1338) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1340 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1341 = "ttir.add"(%1335, %1339, %1340) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1342 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1343 = "ttir.typecast"(%1341, %1342) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1344 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1345 = "ttir.maximum"(%1343, %7, %1344) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1346 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1347 = "ttir.convolution"(%1345, %arg31, %1346) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1024x256x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1348 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1349 = "ttir.typecast"(%1347, %1348) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1350 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1351 = "ttir.broadcast"(%1349, %1350) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1352 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1353 = "ttir.reshape"(%arg174, %1352) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1354 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1355 = "ttir.broadcast"(%1353, %1354) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1356 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1357 = "ttir.subtract"(%1351, %1355, %1356) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1358 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1359 = "ttir.broadcast"(%1357, %1358) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1360 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1361 = "ttir.reshape"(%arg175, %1360) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1362 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1363 = "ttir.broadcast"(%1361, %1362) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1364 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1365 = "ttir.multiply"(%1359, %1363, %1364) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1366 = tensor.empty() : tensor<1024x1x1xbf16>
    %1367 = "ttir.typecast"(%arg176, %1366) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1024x1x1xbf16>, tensor<1024x1x1xbf16>) -> tensor<1024x1x1xbf16>
    %1368 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1369 = "ttir.broadcast"(%1365, %1368) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1370 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1371 = "ttir.reshape"(%1367, %1370) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1372 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1373 = "ttir.broadcast"(%1371, %1372) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1374 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1375 = "ttir.multiply"(%1369, %1373, %1374) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1376 = tensor.empty() : tensor<1024x1x1xbf16>
    %1377 = "ttir.typecast"(%arg177, %1376) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1024x1x1xbf16>, tensor<1024x1x1xbf16>) -> tensor<1024x1x1xbf16>
    %1378 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1379 = "ttir.broadcast"(%1375, %1378) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1380 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1381 = "ttir.reshape"(%1377, %1380) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1382 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1383 = "ttir.broadcast"(%1381, %1382) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1384 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1385 = "ttir.add"(%1379, %1383, %1384) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1386 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1387 = "ttir.typecast"(%1385, %1386) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1388 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1389 = "ttir.add"(%1387, %1257, %1388) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1390 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1391 = "ttir.maximum"(%1389, %8, %1390) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1392 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1393 = "ttir.convolution"(%1391, %arg32, %1392) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1394 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1395 = "ttir.typecast"(%1393, %1394) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1396 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1397 = "ttir.broadcast"(%1395, %1396) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1398 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1399 = "ttir.reshape"(%arg178, %1398) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1400 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1401 = "ttir.broadcast"(%1399, %1400) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1402 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1403 = "ttir.subtract"(%1397, %1401, %1402) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1404 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1405 = "ttir.broadcast"(%1403, %1404) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1406 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1407 = "ttir.reshape"(%arg179, %1406) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1408 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1409 = "ttir.broadcast"(%1407, %1408) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1410 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1411 = "ttir.multiply"(%1405, %1409, %1410) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1412 = tensor.empty() : tensor<256x1x1xbf16>
    %1413 = "ttir.typecast"(%arg180, %1412) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1414 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1415 = "ttir.broadcast"(%1411, %1414) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1416 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1417 = "ttir.reshape"(%1413, %1416) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1418 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1419 = "ttir.broadcast"(%1417, %1418) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1420 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1421 = "ttir.multiply"(%1415, %1419, %1420) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1422 = tensor.empty() : tensor<256x1x1xbf16>
    %1423 = "ttir.typecast"(%arg181, %1422) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1424 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1425 = "ttir.broadcast"(%1421, %1424) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1426 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1427 = "ttir.reshape"(%1423, %1426) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1428 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1429 = "ttir.broadcast"(%1427, %1428) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1430 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1431 = "ttir.add"(%1425, %1429, %1430) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1432 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1433 = "ttir.typecast"(%1431, %1432) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1434 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1435 = "ttir.maximum"(%1433, %7, %1434) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1436 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1437 = "ttir.convolution"(%1435, %arg33, %1436) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<256x256x3x3xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1438 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1439 = "ttir.typecast"(%1437, %1438) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1440 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1441 = "ttir.broadcast"(%1439, %1440) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1442 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1443 = "ttir.reshape"(%arg182, %1442) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1444 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1445 = "ttir.broadcast"(%1443, %1444) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1446 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1447 = "ttir.subtract"(%1441, %1445, %1446) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1448 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1449 = "ttir.broadcast"(%1447, %1448) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1450 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1451 = "ttir.reshape"(%arg183, %1450) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1452 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1453 = "ttir.broadcast"(%1451, %1452) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1454 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1455 = "ttir.multiply"(%1449, %1453, %1454) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1456 = tensor.empty() : tensor<256x1x1xbf16>
    %1457 = "ttir.typecast"(%arg184, %1456) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1458 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1459 = "ttir.broadcast"(%1455, %1458) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1460 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1461 = "ttir.reshape"(%1457, %1460) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1462 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1463 = "ttir.broadcast"(%1461, %1462) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1464 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1465 = "ttir.multiply"(%1459, %1463, %1464) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1466 = tensor.empty() : tensor<256x1x1xbf16>
    %1467 = "ttir.typecast"(%arg185, %1466) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1468 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1469 = "ttir.broadcast"(%1465, %1468) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1470 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1471 = "ttir.reshape"(%1467, %1470) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1472 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1473 = "ttir.broadcast"(%1471, %1472) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1474 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1475 = "ttir.add"(%1469, %1473, %1474) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1476 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1477 = "ttir.typecast"(%1475, %1476) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1478 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1479 = "ttir.maximum"(%1477, %7, %1478) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1480 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1481 = "ttir.convolution"(%1479, %arg34, %1480) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1024x256x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1482 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1483 = "ttir.typecast"(%1481, %1482) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1484 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1485 = "ttir.broadcast"(%1483, %1484) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1486 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1487 = "ttir.reshape"(%arg186, %1486) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1488 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1489 = "ttir.broadcast"(%1487, %1488) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1490 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1491 = "ttir.subtract"(%1485, %1489, %1490) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1492 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1493 = "ttir.broadcast"(%1491, %1492) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1494 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1495 = "ttir.reshape"(%arg187, %1494) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1496 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1497 = "ttir.broadcast"(%1495, %1496) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1498 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1499 = "ttir.multiply"(%1493, %1497, %1498) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1500 = tensor.empty() : tensor<1024x1x1xbf16>
    %1501 = "ttir.typecast"(%arg188, %1500) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1024x1x1xbf16>, tensor<1024x1x1xbf16>) -> tensor<1024x1x1xbf16>
    %1502 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1503 = "ttir.broadcast"(%1499, %1502) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1504 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1505 = "ttir.reshape"(%1501, %1504) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1506 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1507 = "ttir.broadcast"(%1505, %1506) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1508 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1509 = "ttir.multiply"(%1503, %1507, %1508) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1510 = tensor.empty() : tensor<1024x1x1xbf16>
    %1511 = "ttir.typecast"(%arg189, %1510) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1024x1x1xbf16>, tensor<1024x1x1xbf16>) -> tensor<1024x1x1xbf16>
    %1512 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1513 = "ttir.broadcast"(%1509, %1512) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1514 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1515 = "ttir.reshape"(%1511, %1514) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1516 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1517 = "ttir.broadcast"(%1515, %1516) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1518 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1519 = "ttir.add"(%1513, %1517, %1518) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1520 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1521 = "ttir.typecast"(%1519, %1520) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1522 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1523 = "ttir.add"(%1521, %1391, %1522) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1524 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1525 = "ttir.maximum"(%1523, %8, %1524) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1526 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1527 = "ttir.convolution"(%1525, %arg35, %1526) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1528 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1529 = "ttir.typecast"(%1527, %1528) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1530 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1531 = "ttir.broadcast"(%1529, %1530) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1532 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1533 = "ttir.reshape"(%arg190, %1532) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1534 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1535 = "ttir.broadcast"(%1533, %1534) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1536 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1537 = "ttir.subtract"(%1531, %1535, %1536) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1538 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1539 = "ttir.broadcast"(%1537, %1538) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1540 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1541 = "ttir.reshape"(%arg191, %1540) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1542 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1543 = "ttir.broadcast"(%1541, %1542) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1544 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1545 = "ttir.multiply"(%1539, %1543, %1544) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1546 = tensor.empty() : tensor<256x1x1xbf16>
    %1547 = "ttir.typecast"(%arg192, %1546) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1548 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1549 = "ttir.broadcast"(%1545, %1548) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1550 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1551 = "ttir.reshape"(%1547, %1550) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1552 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1553 = "ttir.broadcast"(%1551, %1552) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1554 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1555 = "ttir.multiply"(%1549, %1553, %1554) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1556 = tensor.empty() : tensor<256x1x1xbf16>
    %1557 = "ttir.typecast"(%arg193, %1556) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1558 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1559 = "ttir.broadcast"(%1555, %1558) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1560 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1561 = "ttir.reshape"(%1557, %1560) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1562 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1563 = "ttir.broadcast"(%1561, %1562) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1564 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1565 = "ttir.add"(%1559, %1563, %1564) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1566 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1567 = "ttir.typecast"(%1565, %1566) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1568 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1569 = "ttir.maximum"(%1567, %7, %1568) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1570 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1571 = "ttir.convolution"(%1569, %arg36, %1570) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<256x256x3x3xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1572 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1573 = "ttir.typecast"(%1571, %1572) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1574 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1575 = "ttir.broadcast"(%1573, %1574) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1576 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1577 = "ttir.reshape"(%arg194, %1576) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1578 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1579 = "ttir.broadcast"(%1577, %1578) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1580 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1581 = "ttir.subtract"(%1575, %1579, %1580) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1582 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1583 = "ttir.broadcast"(%1581, %1582) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1584 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1585 = "ttir.reshape"(%arg195, %1584) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1586 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1587 = "ttir.broadcast"(%1585, %1586) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1588 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1589 = "ttir.multiply"(%1583, %1587, %1588) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1590 = tensor.empty() : tensor<256x1x1xbf16>
    %1591 = "ttir.typecast"(%arg196, %1590) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1592 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1593 = "ttir.broadcast"(%1589, %1592) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1594 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1595 = "ttir.reshape"(%1591, %1594) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1596 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1597 = "ttir.broadcast"(%1595, %1596) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1598 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1599 = "ttir.multiply"(%1593, %1597, %1598) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1600 = tensor.empty() : tensor<256x1x1xbf16>
    %1601 = "ttir.typecast"(%arg197, %1600) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1602 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1603 = "ttir.broadcast"(%1599, %1602) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1604 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1605 = "ttir.reshape"(%1601, %1604) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1606 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1607 = "ttir.broadcast"(%1605, %1606) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1608 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1609 = "ttir.add"(%1603, %1607, %1608) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1610 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1611 = "ttir.typecast"(%1609, %1610) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1612 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1613 = "ttir.maximum"(%1611, %7, %1612) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1614 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1615 = "ttir.convolution"(%1613, %arg37, %1614) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1024x256x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1616 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1617 = "ttir.typecast"(%1615, %1616) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1618 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1619 = "ttir.broadcast"(%1617, %1618) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1620 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1621 = "ttir.reshape"(%arg198, %1620) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1622 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1623 = "ttir.broadcast"(%1621, %1622) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1624 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1625 = "ttir.subtract"(%1619, %1623, %1624) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1626 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1627 = "ttir.broadcast"(%1625, %1626) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1628 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1629 = "ttir.reshape"(%arg199, %1628) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1630 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1631 = "ttir.broadcast"(%1629, %1630) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1632 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1633 = "ttir.multiply"(%1627, %1631, %1632) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1634 = tensor.empty() : tensor<1024x1x1xbf16>
    %1635 = "ttir.typecast"(%arg200, %1634) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1024x1x1xbf16>, tensor<1024x1x1xbf16>) -> tensor<1024x1x1xbf16>
    %1636 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1637 = "ttir.broadcast"(%1633, %1636) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1638 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1639 = "ttir.reshape"(%1635, %1638) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1640 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1641 = "ttir.broadcast"(%1639, %1640) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1642 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1643 = "ttir.multiply"(%1637, %1641, %1642) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1644 = tensor.empty() : tensor<1024x1x1xbf16>
    %1645 = "ttir.typecast"(%arg201, %1644) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1024x1x1xbf16>, tensor<1024x1x1xbf16>) -> tensor<1024x1x1xbf16>
    %1646 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1647 = "ttir.broadcast"(%1643, %1646) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1648 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1649 = "ttir.reshape"(%1645, %1648) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1650 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1651 = "ttir.broadcast"(%1649, %1650) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1652 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1653 = "ttir.add"(%1647, %1651, %1652) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1654 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1655 = "ttir.typecast"(%1653, %1654) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1656 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1657 = "ttir.add"(%1655, %1525, %1656) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1658 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1659 = "ttir.maximum"(%1657, %8, %1658) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1660 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1661 = "ttir.convolution"(%1659, %arg38, %1660) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1662 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1663 = "ttir.typecast"(%1661, %1662) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1664 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1665 = "ttir.broadcast"(%1663, %1664) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1666 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1667 = "ttir.reshape"(%arg202, %1666) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1668 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1669 = "ttir.broadcast"(%1667, %1668) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1670 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1671 = "ttir.subtract"(%1665, %1669, %1670) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1672 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1673 = "ttir.broadcast"(%1671, %1672) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1674 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1675 = "ttir.reshape"(%arg203, %1674) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1676 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1677 = "ttir.broadcast"(%1675, %1676) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1678 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1679 = "ttir.multiply"(%1673, %1677, %1678) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1680 = tensor.empty() : tensor<256x1x1xbf16>
    %1681 = "ttir.typecast"(%arg204, %1680) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1682 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1683 = "ttir.broadcast"(%1679, %1682) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1684 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1685 = "ttir.reshape"(%1681, %1684) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1686 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1687 = "ttir.broadcast"(%1685, %1686) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1688 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1689 = "ttir.multiply"(%1683, %1687, %1688) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1690 = tensor.empty() : tensor<256x1x1xbf16>
    %1691 = "ttir.typecast"(%arg205, %1690) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1692 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1693 = "ttir.broadcast"(%1689, %1692) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1694 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1695 = "ttir.reshape"(%1691, %1694) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1696 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1697 = "ttir.broadcast"(%1695, %1696) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1698 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1699 = "ttir.add"(%1693, %1697, %1698) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1700 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1701 = "ttir.typecast"(%1699, %1700) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1702 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1703 = "ttir.maximum"(%1701, %7, %1702) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1704 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1705 = "ttir.convolution"(%1703, %arg39, %1704) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<256x256x3x3xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1706 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1707 = "ttir.typecast"(%1705, %1706) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1708 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1709 = "ttir.broadcast"(%1707, %1708) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1710 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1711 = "ttir.reshape"(%arg206, %1710) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1712 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1713 = "ttir.broadcast"(%1711, %1712) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1714 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1715 = "ttir.subtract"(%1709, %1713, %1714) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1716 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1717 = "ttir.broadcast"(%1715, %1716) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1718 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1719 = "ttir.reshape"(%arg207, %1718) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1720 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1721 = "ttir.broadcast"(%1719, %1720) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1722 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1723 = "ttir.multiply"(%1717, %1721, %1722) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1724 = tensor.empty() : tensor<256x1x1xbf16>
    %1725 = "ttir.typecast"(%arg208, %1724) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1726 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1727 = "ttir.broadcast"(%1723, %1726) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1728 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1729 = "ttir.reshape"(%1725, %1728) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1730 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1731 = "ttir.broadcast"(%1729, %1730) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1732 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1733 = "ttir.multiply"(%1727, %1731, %1732) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1734 = tensor.empty() : tensor<256x1x1xbf16>
    %1735 = "ttir.typecast"(%arg209, %1734) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1736 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1737 = "ttir.broadcast"(%1733, %1736) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1738 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1739 = "ttir.reshape"(%1735, %1738) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1740 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1741 = "ttir.broadcast"(%1739, %1740) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1742 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1743 = "ttir.add"(%1737, %1741, %1742) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1744 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1745 = "ttir.typecast"(%1743, %1744) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1746 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1747 = "ttir.maximum"(%1745, %7, %1746) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1748 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1749 = "ttir.convolution"(%1747, %arg40, %1748) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1024x256x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1750 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1751 = "ttir.typecast"(%1749, %1750) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1752 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1753 = "ttir.broadcast"(%1751, %1752) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1754 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1755 = "ttir.reshape"(%arg210, %1754) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1756 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1757 = "ttir.broadcast"(%1755, %1756) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1758 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1759 = "ttir.subtract"(%1753, %1757, %1758) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1760 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1761 = "ttir.broadcast"(%1759, %1760) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1762 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1763 = "ttir.reshape"(%arg211, %1762) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1764 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1765 = "ttir.broadcast"(%1763, %1764) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1766 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1767 = "ttir.multiply"(%1761, %1765, %1766) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1768 = tensor.empty() : tensor<1024x1x1xbf16>
    %1769 = "ttir.typecast"(%arg212, %1768) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1024x1x1xbf16>, tensor<1024x1x1xbf16>) -> tensor<1024x1x1xbf16>
    %1770 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1771 = "ttir.broadcast"(%1767, %1770) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1772 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1773 = "ttir.reshape"(%1769, %1772) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1774 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1775 = "ttir.broadcast"(%1773, %1774) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1776 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1777 = "ttir.multiply"(%1771, %1775, %1776) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1778 = tensor.empty() : tensor<1024x1x1xbf16>
    %1779 = "ttir.typecast"(%arg213, %1778) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1024x1x1xbf16>, tensor<1024x1x1xbf16>) -> tensor<1024x1x1xbf16>
    %1780 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1781 = "ttir.broadcast"(%1777, %1780) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1782 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1783 = "ttir.reshape"(%1779, %1782) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1784 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1785 = "ttir.broadcast"(%1783, %1784) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1786 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1787 = "ttir.add"(%1781, %1785, %1786) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1788 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1789 = "ttir.typecast"(%1787, %1788) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1790 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1791 = "ttir.add"(%1789, %1659, %1790) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1792 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1793 = "ttir.maximum"(%1791, %8, %1792) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1794 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1795 = "ttir.convolution"(%1793, %arg41, %1794) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1796 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1797 = "ttir.typecast"(%1795, %1796) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1798 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1799 = "ttir.broadcast"(%1797, %1798) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1800 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1801 = "ttir.reshape"(%arg214, %1800) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1802 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1803 = "ttir.broadcast"(%1801, %1802) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1804 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1805 = "ttir.subtract"(%1799, %1803, %1804) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1806 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1807 = "ttir.broadcast"(%1805, %1806) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1808 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1809 = "ttir.reshape"(%arg215, %1808) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1810 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1811 = "ttir.broadcast"(%1809, %1810) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1812 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1813 = "ttir.multiply"(%1807, %1811, %1812) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1814 = tensor.empty() : tensor<256x1x1xbf16>
    %1815 = "ttir.typecast"(%arg216, %1814) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1816 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1817 = "ttir.broadcast"(%1813, %1816) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1818 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1819 = "ttir.reshape"(%1815, %1818) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1820 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1821 = "ttir.broadcast"(%1819, %1820) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1822 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1823 = "ttir.multiply"(%1817, %1821, %1822) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1824 = tensor.empty() : tensor<256x1x1xbf16>
    %1825 = "ttir.typecast"(%arg217, %1824) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1826 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1827 = "ttir.broadcast"(%1823, %1826) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1828 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1829 = "ttir.reshape"(%1825, %1828) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1830 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1831 = "ttir.broadcast"(%1829, %1830) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1832 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1833 = "ttir.add"(%1827, %1831, %1832) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1834 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1835 = "ttir.typecast"(%1833, %1834) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1836 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1837 = "ttir.maximum"(%1835, %7, %1836) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1838 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1839 = "ttir.convolution"(%1837, %arg42, %1838) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<256x256x3x3xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1840 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1841 = "ttir.typecast"(%1839, %1840) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1842 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1843 = "ttir.broadcast"(%1841, %1842) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1844 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1845 = "ttir.reshape"(%arg218, %1844) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1846 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1847 = "ttir.broadcast"(%1845, %1846) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1848 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1849 = "ttir.subtract"(%1843, %1847, %1848) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1850 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1851 = "ttir.broadcast"(%1849, %1850) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1852 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1853 = "ttir.reshape"(%arg219, %1852) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1854 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1855 = "ttir.broadcast"(%1853, %1854) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1856 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1857 = "ttir.multiply"(%1851, %1855, %1856) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1858 = tensor.empty() : tensor<256x1x1xbf16>
    %1859 = "ttir.typecast"(%arg220, %1858) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1860 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1861 = "ttir.broadcast"(%1857, %1860) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1862 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1863 = "ttir.reshape"(%1859, %1862) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1864 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1865 = "ttir.broadcast"(%1863, %1864) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1866 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1867 = "ttir.multiply"(%1861, %1865, %1866) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1868 = tensor.empty() : tensor<256x1x1xbf16>
    %1869 = "ttir.typecast"(%arg221, %1868) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<256x1x1xbf16>, tensor<256x1x1xbf16>) -> tensor<256x1x1xbf16>
    %1870 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1871 = "ttir.broadcast"(%1867, %1870) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1872 = tensor.empty() : tensor<1x256x1x1xbf16>
    %1873 = "ttir.reshape"(%1869, %1872) <{shape = [1 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<256x1x1xbf16>, tensor<1x256x1x1xbf16>) -> tensor<1x256x1x1xbf16>
    %1874 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1875 = "ttir.broadcast"(%1873, %1874) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x256x1x1xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1876 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1877 = "ttir.add"(%1871, %1875, %1876) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1878 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1879 = "ttir.typecast"(%1877, %1878) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1880 = tensor.empty() : tensor<1x256x14x14xbf16>
    %1881 = "ttir.maximum"(%1879, %7, %1880) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>, tensor<1x256x14x14xbf16>) -> tensor<1x256x14x14xbf16>
    %1882 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1883 = "ttir.convolution"(%1881, %arg43, %1882) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x256x14x14xbf16>, tensor<1024x256x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1884 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1885 = "ttir.typecast"(%1883, %1884) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1886 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1887 = "ttir.broadcast"(%1885, %1886) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1888 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1889 = "ttir.reshape"(%arg222, %1888) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1890 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1891 = "ttir.broadcast"(%1889, %1890) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1892 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1893 = "ttir.subtract"(%1887, %1891, %1892) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1894 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1895 = "ttir.broadcast"(%1893, %1894) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1896 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1897 = "ttir.reshape"(%arg223, %1896) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1898 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1899 = "ttir.broadcast"(%1897, %1898) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1900 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1901 = "ttir.multiply"(%1895, %1899, %1900) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1902 = tensor.empty() : tensor<1024x1x1xbf16>
    %1903 = "ttir.typecast"(%arg224, %1902) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1024x1x1xbf16>, tensor<1024x1x1xbf16>) -> tensor<1024x1x1xbf16>
    %1904 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1905 = "ttir.broadcast"(%1901, %1904) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1906 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1907 = "ttir.reshape"(%1903, %1906) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1908 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1909 = "ttir.broadcast"(%1907, %1908) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1910 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1911 = "ttir.multiply"(%1905, %1909, %1910) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1912 = tensor.empty() : tensor<1024x1x1xbf16>
    %1913 = "ttir.typecast"(%arg225, %1912) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1024x1x1xbf16>, tensor<1024x1x1xbf16>) -> tensor<1024x1x1xbf16>
    %1914 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1915 = "ttir.broadcast"(%1911, %1914) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1916 = tensor.empty() : tensor<1x1024x1x1xbf16>
    %1917 = "ttir.reshape"(%1913, %1916) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>, tensor<1x1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %1918 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1919 = "ttir.broadcast"(%1917, %1918) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1920 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1921 = "ttir.add"(%1915, %1919, %1920) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1922 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1923 = "ttir.typecast"(%1921, %1922) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1924 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1925 = "ttir.add"(%1923, %1793, %1924) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1926 = tensor.empty() : tensor<1x1024x14x14xbf16>
    %1927 = "ttir.maximum"(%1925, %8, %1926) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %1928 = tensor.empty() : tensor<1x512x14x14xbf16>
    %1929 = "ttir.convolution"(%1927, %arg44, %1928) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<512x1024x1x1xbf16>, tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xbf16>
    %1930 = tensor.empty() : tensor<1x512x14x14xbf16>
    %1931 = "ttir.typecast"(%1929, %1930) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x14x14xbf16>, tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xbf16>
    %1932 = tensor.empty() : tensor<1x512x14x14xbf16>
    %1933 = "ttir.broadcast"(%1931, %1932) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x14x14xbf16>, tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xbf16>
    %1934 = tensor.empty() : tensor<1x512x1x1xbf16>
    %1935 = "ttir.reshape"(%arg226, %1934) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %1936 = tensor.empty() : tensor<1x512x14x14xbf16>
    %1937 = "ttir.broadcast"(%1935, %1936) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xbf16>
    %1938 = tensor.empty() : tensor<1x512x14x14xbf16>
    %1939 = "ttir.subtract"(%1933, %1937, %1938) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x14x14xbf16>, tensor<1x512x14x14xbf16>, tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xbf16>
    %1940 = tensor.empty() : tensor<1x512x14x14xbf16>
    %1941 = "ttir.broadcast"(%1939, %1940) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x14x14xbf16>, tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xbf16>
    %1942 = tensor.empty() : tensor<1x512x1x1xbf16>
    %1943 = "ttir.reshape"(%arg227, %1942) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %1944 = tensor.empty() : tensor<1x512x14x14xbf16>
    %1945 = "ttir.broadcast"(%1943, %1944) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xbf16>
    %1946 = tensor.empty() : tensor<1x512x14x14xbf16>
    %1947 = "ttir.multiply"(%1941, %1945, %1946) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x14x14xbf16>, tensor<1x512x14x14xbf16>, tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xbf16>
    %1948 = tensor.empty() : tensor<512x1x1xbf16>
    %1949 = "ttir.typecast"(%arg228, %1948) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %1950 = tensor.empty() : tensor<1x512x14x14xbf16>
    %1951 = "ttir.broadcast"(%1947, %1950) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x14x14xbf16>, tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xbf16>
    %1952 = tensor.empty() : tensor<1x512x1x1xbf16>
    %1953 = "ttir.reshape"(%1949, %1952) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %1954 = tensor.empty() : tensor<1x512x14x14xbf16>
    %1955 = "ttir.broadcast"(%1953, %1954) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xbf16>
    %1956 = tensor.empty() : tensor<1x512x14x14xbf16>
    %1957 = "ttir.multiply"(%1951, %1955, %1956) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x14x14xbf16>, tensor<1x512x14x14xbf16>, tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xbf16>
    %1958 = tensor.empty() : tensor<512x1x1xbf16>
    %1959 = "ttir.typecast"(%arg229, %1958) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %1960 = tensor.empty() : tensor<1x512x14x14xbf16>
    %1961 = "ttir.broadcast"(%1957, %1960) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x14x14xbf16>, tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xbf16>
    %1962 = tensor.empty() : tensor<1x512x1x1xbf16>
    %1963 = "ttir.reshape"(%1959, %1962) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %1964 = tensor.empty() : tensor<1x512x14x14xbf16>
    %1965 = "ttir.broadcast"(%1963, %1964) <{broadcast_dimensions = array<i32: 1, 1, 14, 14>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xbf16>
    %1966 = tensor.empty() : tensor<1x512x14x14xbf16>
    %1967 = "ttir.add"(%1961, %1965, %1966) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x14x14xbf16>, tensor<1x512x14x14xbf16>, tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xbf16>
    %1968 = tensor.empty() : tensor<1x512x14x14xbf16>
    %1969 = "ttir.typecast"(%1967, %1968) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x14x14xbf16>, tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xbf16>
    %1970 = tensor.empty() : tensor<1x512x14x14xbf16>
    %1971 = "ttir.maximum"(%1969, %9, %1970) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x14x14xbf16>, tensor<1x512x14x14xbf16>, tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xbf16>
    %1972 = tensor.empty() : tensor<1x512x7x7xbf16>
    %1973 = "ttir.convolution"(%1971, %arg45, %1972) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<1x512x14x14xbf16>, tensor<512x512x3x3xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %1974 = tensor.empty() : tensor<1x512x7x7xbf16>
    %1975 = "ttir.typecast"(%1973, %1974) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %1976 = tensor.empty() : tensor<1x512x7x7xbf16>
    %1977 = "ttir.broadcast"(%1975, %1976) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %1978 = tensor.empty() : tensor<1x512x1x1xbf16>
    %1979 = "ttir.reshape"(%arg230, %1978) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %1980 = tensor.empty() : tensor<1x512x7x7xbf16>
    %1981 = "ttir.broadcast"(%1979, %1980) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %1982 = tensor.empty() : tensor<1x512x7x7xbf16>
    %1983 = "ttir.subtract"(%1977, %1981, %1982) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %1984 = tensor.empty() : tensor<1x512x7x7xbf16>
    %1985 = "ttir.broadcast"(%1983, %1984) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %1986 = tensor.empty() : tensor<1x512x1x1xbf16>
    %1987 = "ttir.reshape"(%arg231, %1986) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %1988 = tensor.empty() : tensor<1x512x7x7xbf16>
    %1989 = "ttir.broadcast"(%1987, %1988) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %1990 = tensor.empty() : tensor<1x512x7x7xbf16>
    %1991 = "ttir.multiply"(%1985, %1989, %1990) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %1992 = tensor.empty() : tensor<512x1x1xbf16>
    %1993 = "ttir.typecast"(%arg232, %1992) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %1994 = tensor.empty() : tensor<1x512x7x7xbf16>
    %1995 = "ttir.broadcast"(%1991, %1994) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %1996 = tensor.empty() : tensor<1x512x1x1xbf16>
    %1997 = "ttir.reshape"(%1993, %1996) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %1998 = tensor.empty() : tensor<1x512x7x7xbf16>
    %1999 = "ttir.broadcast"(%1997, %1998) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2000 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2001 = "ttir.multiply"(%1995, %1999, %2000) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2002 = tensor.empty() : tensor<512x1x1xbf16>
    %2003 = "ttir.typecast"(%arg233, %2002) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %2004 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2005 = "ttir.broadcast"(%2001, %2004) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2006 = tensor.empty() : tensor<1x512x1x1xbf16>
    %2007 = "ttir.reshape"(%2003, %2006) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %2008 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2009 = "ttir.broadcast"(%2007, %2008) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2010 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2011 = "ttir.add"(%2005, %2009, %2010) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2012 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2013 = "ttir.typecast"(%2011, %2012) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2014 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2015 = "ttir.maximum"(%2013, %10, %2014) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2016 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2017 = "ttir.convolution"(%2015, %arg46, %2016) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<2048x512x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2018 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2019 = "ttir.typecast"(%2017, %2018) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2020 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2021 = "ttir.broadcast"(%2019, %2020) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2022 = tensor.empty() : tensor<1x2048x1x1xbf16>
    %2023 = "ttir.reshape"(%arg234, %2022) <{shape = [1 : i32, 2048 : i32, 1 : i32, 1 : i32]}> : (tensor<2048x1x1xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %2024 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2025 = "ttir.broadcast"(%2023, %2024) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2026 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2027 = "ttir.subtract"(%2021, %2025, %2026) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2028 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2029 = "ttir.broadcast"(%2027, %2028) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2030 = tensor.empty() : tensor<1x2048x1x1xbf16>
    %2031 = "ttir.reshape"(%arg235, %2030) <{shape = [1 : i32, 2048 : i32, 1 : i32, 1 : i32]}> : (tensor<2048x1x1xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %2032 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2033 = "ttir.broadcast"(%2031, %2032) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2034 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2035 = "ttir.multiply"(%2029, %2033, %2034) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2036 = tensor.empty() : tensor<2048x1x1xbf16>
    %2037 = "ttir.typecast"(%arg236, %2036) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<2048x1x1xbf16>, tensor<2048x1x1xbf16>) -> tensor<2048x1x1xbf16>
    %2038 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2039 = "ttir.broadcast"(%2035, %2038) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2040 = tensor.empty() : tensor<1x2048x1x1xbf16>
    %2041 = "ttir.reshape"(%2037, %2040) <{shape = [1 : i32, 2048 : i32, 1 : i32, 1 : i32]}> : (tensor<2048x1x1xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %2042 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2043 = "ttir.broadcast"(%2041, %2042) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2044 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2045 = "ttir.multiply"(%2039, %2043, %2044) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2046 = tensor.empty() : tensor<2048x1x1xbf16>
    %2047 = "ttir.typecast"(%arg237, %2046) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<2048x1x1xbf16>, tensor<2048x1x1xbf16>) -> tensor<2048x1x1xbf16>
    %2048 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2049 = "ttir.broadcast"(%2045, %2048) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2050 = tensor.empty() : tensor<1x2048x1x1xbf16>
    %2051 = "ttir.reshape"(%2047, %2050) <{shape = [1 : i32, 2048 : i32, 1 : i32, 1 : i32]}> : (tensor<2048x1x1xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %2052 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2053 = "ttir.broadcast"(%2051, %2052) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2054 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2055 = "ttir.add"(%2049, %2053, %2054) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2056 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2057 = "ttir.typecast"(%2055, %2056) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2058 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2059 = "ttir.convolution"(%1927, %arg47, %2058) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<1x1024x14x14xbf16>, tensor<2048x1024x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2060 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2061 = "ttir.typecast"(%2059, %2060) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2062 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2063 = "ttir.broadcast"(%2061, %2062) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2064 = tensor.empty() : tensor<1x2048x1x1xbf16>
    %2065 = "ttir.reshape"(%arg238, %2064) <{shape = [1 : i32, 2048 : i32, 1 : i32, 1 : i32]}> : (tensor<2048x1x1xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %2066 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2067 = "ttir.broadcast"(%2065, %2066) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2068 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2069 = "ttir.subtract"(%2063, %2067, %2068) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2070 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2071 = "ttir.broadcast"(%2069, %2070) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2072 = tensor.empty() : tensor<1x2048x1x1xbf16>
    %2073 = "ttir.reshape"(%arg239, %2072) <{shape = [1 : i32, 2048 : i32, 1 : i32, 1 : i32]}> : (tensor<2048x1x1xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %2074 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2075 = "ttir.broadcast"(%2073, %2074) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2076 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2077 = "ttir.multiply"(%2071, %2075, %2076) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2078 = tensor.empty() : tensor<2048x1x1xbf16>
    %2079 = "ttir.typecast"(%arg240, %2078) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<2048x1x1xbf16>, tensor<2048x1x1xbf16>) -> tensor<2048x1x1xbf16>
    %2080 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2081 = "ttir.broadcast"(%2077, %2080) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2082 = tensor.empty() : tensor<1x2048x1x1xbf16>
    %2083 = "ttir.reshape"(%2079, %2082) <{shape = [1 : i32, 2048 : i32, 1 : i32, 1 : i32]}> : (tensor<2048x1x1xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %2084 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2085 = "ttir.broadcast"(%2083, %2084) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2086 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2087 = "ttir.multiply"(%2081, %2085, %2086) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2088 = tensor.empty() : tensor<2048x1x1xbf16>
    %2089 = "ttir.typecast"(%arg241, %2088) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<2048x1x1xbf16>, tensor<2048x1x1xbf16>) -> tensor<2048x1x1xbf16>
    %2090 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2091 = "ttir.broadcast"(%2087, %2090) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2092 = tensor.empty() : tensor<1x2048x1x1xbf16>
    %2093 = "ttir.reshape"(%2089, %2092) <{shape = [1 : i32, 2048 : i32, 1 : i32, 1 : i32]}> : (tensor<2048x1x1xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %2094 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2095 = "ttir.broadcast"(%2093, %2094) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2096 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2097 = "ttir.add"(%2091, %2095, %2096) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2098 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2099 = "ttir.typecast"(%2097, %2098) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2100 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2101 = "ttir.add"(%2057, %2099, %2100) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2102 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2103 = "ttir.maximum"(%2101, %11, %2102) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2104 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2105 = "ttir.convolution"(%2103, %arg48, %2104) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2106 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2107 = "ttir.typecast"(%2105, %2106) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2108 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2109 = "ttir.broadcast"(%2107, %2108) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2110 = tensor.empty() : tensor<1x512x1x1xbf16>
    %2111 = "ttir.reshape"(%arg242, %2110) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %2112 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2113 = "ttir.broadcast"(%2111, %2112) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2114 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2115 = "ttir.subtract"(%2109, %2113, %2114) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2116 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2117 = "ttir.broadcast"(%2115, %2116) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2118 = tensor.empty() : tensor<1x512x1x1xbf16>
    %2119 = "ttir.reshape"(%arg243, %2118) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %2120 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2121 = "ttir.broadcast"(%2119, %2120) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2122 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2123 = "ttir.multiply"(%2117, %2121, %2122) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2124 = tensor.empty() : tensor<512x1x1xbf16>
    %2125 = "ttir.typecast"(%arg244, %2124) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %2126 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2127 = "ttir.broadcast"(%2123, %2126) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2128 = tensor.empty() : tensor<1x512x1x1xbf16>
    %2129 = "ttir.reshape"(%2125, %2128) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %2130 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2131 = "ttir.broadcast"(%2129, %2130) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2132 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2133 = "ttir.multiply"(%2127, %2131, %2132) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2134 = tensor.empty() : tensor<512x1x1xbf16>
    %2135 = "ttir.typecast"(%arg245, %2134) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %2136 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2137 = "ttir.broadcast"(%2133, %2136) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2138 = tensor.empty() : tensor<1x512x1x1xbf16>
    %2139 = "ttir.reshape"(%2135, %2138) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %2140 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2141 = "ttir.broadcast"(%2139, %2140) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2142 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2143 = "ttir.add"(%2137, %2141, %2142) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2144 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2145 = "ttir.typecast"(%2143, %2144) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2146 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2147 = "ttir.maximum"(%2145, %10, %2146) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2148 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2149 = "ttir.convolution"(%2147, %arg49, %2148) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<512x512x3x3xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2150 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2151 = "ttir.typecast"(%2149, %2150) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2152 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2153 = "ttir.broadcast"(%2151, %2152) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2154 = tensor.empty() : tensor<1x512x1x1xbf16>
    %2155 = "ttir.reshape"(%arg246, %2154) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %2156 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2157 = "ttir.broadcast"(%2155, %2156) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2158 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2159 = "ttir.subtract"(%2153, %2157, %2158) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2160 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2161 = "ttir.broadcast"(%2159, %2160) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2162 = tensor.empty() : tensor<1x512x1x1xbf16>
    %2163 = "ttir.reshape"(%arg247, %2162) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %2164 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2165 = "ttir.broadcast"(%2163, %2164) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2166 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2167 = "ttir.multiply"(%2161, %2165, %2166) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2168 = tensor.empty() : tensor<512x1x1xbf16>
    %2169 = "ttir.typecast"(%arg248, %2168) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %2170 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2171 = "ttir.broadcast"(%2167, %2170) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2172 = tensor.empty() : tensor<1x512x1x1xbf16>
    %2173 = "ttir.reshape"(%2169, %2172) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %2174 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2175 = "ttir.broadcast"(%2173, %2174) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2176 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2177 = "ttir.multiply"(%2171, %2175, %2176) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2178 = tensor.empty() : tensor<512x1x1xbf16>
    %2179 = "ttir.typecast"(%arg249, %2178) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %2180 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2181 = "ttir.broadcast"(%2177, %2180) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2182 = tensor.empty() : tensor<1x512x1x1xbf16>
    %2183 = "ttir.reshape"(%2179, %2182) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %2184 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2185 = "ttir.broadcast"(%2183, %2184) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2186 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2187 = "ttir.add"(%2181, %2185, %2186) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2188 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2189 = "ttir.typecast"(%2187, %2188) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2190 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2191 = "ttir.maximum"(%2189, %10, %2190) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2192 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2193 = "ttir.convolution"(%2191, %arg50, %2192) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<2048x512x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2194 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2195 = "ttir.typecast"(%2193, %2194) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2196 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2197 = "ttir.broadcast"(%2195, %2196) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2198 = tensor.empty() : tensor<1x2048x1x1xbf16>
    %2199 = "ttir.reshape"(%arg250, %2198) <{shape = [1 : i32, 2048 : i32, 1 : i32, 1 : i32]}> : (tensor<2048x1x1xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %2200 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2201 = "ttir.broadcast"(%2199, %2200) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2202 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2203 = "ttir.subtract"(%2197, %2201, %2202) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2204 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2205 = "ttir.broadcast"(%2203, %2204) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2206 = tensor.empty() : tensor<1x2048x1x1xbf16>
    %2207 = "ttir.reshape"(%arg251, %2206) <{shape = [1 : i32, 2048 : i32, 1 : i32, 1 : i32]}> : (tensor<2048x1x1xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %2208 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2209 = "ttir.broadcast"(%2207, %2208) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2210 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2211 = "ttir.multiply"(%2205, %2209, %2210) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2212 = tensor.empty() : tensor<2048x1x1xbf16>
    %2213 = "ttir.typecast"(%arg252, %2212) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<2048x1x1xbf16>, tensor<2048x1x1xbf16>) -> tensor<2048x1x1xbf16>
    %2214 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2215 = "ttir.broadcast"(%2211, %2214) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2216 = tensor.empty() : tensor<1x2048x1x1xbf16>
    %2217 = "ttir.reshape"(%2213, %2216) <{shape = [1 : i32, 2048 : i32, 1 : i32, 1 : i32]}> : (tensor<2048x1x1xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %2218 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2219 = "ttir.broadcast"(%2217, %2218) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2220 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2221 = "ttir.multiply"(%2215, %2219, %2220) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2222 = tensor.empty() : tensor<2048x1x1xbf16>
    %2223 = "ttir.typecast"(%arg253, %2222) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<2048x1x1xbf16>, tensor<2048x1x1xbf16>) -> tensor<2048x1x1xbf16>
    %2224 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2225 = "ttir.broadcast"(%2221, %2224) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2226 = tensor.empty() : tensor<1x2048x1x1xbf16>
    %2227 = "ttir.reshape"(%2223, %2226) <{shape = [1 : i32, 2048 : i32, 1 : i32, 1 : i32]}> : (tensor<2048x1x1xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %2228 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2229 = "ttir.broadcast"(%2227, %2228) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2230 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2231 = "ttir.add"(%2225, %2229, %2230) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2232 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2233 = "ttir.typecast"(%2231, %2232) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2234 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2235 = "ttir.add"(%2233, %2103, %2234) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2236 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2237 = "ttir.maximum"(%2235, %11, %2236) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2238 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2239 = "ttir.convolution"(%2237, %arg51, %2238) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2240 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2241 = "ttir.typecast"(%2239, %2240) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2242 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2243 = "ttir.broadcast"(%2241, %2242) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2244 = tensor.empty() : tensor<1x512x1x1xbf16>
    %2245 = "ttir.reshape"(%arg254, %2244) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %2246 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2247 = "ttir.broadcast"(%2245, %2246) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2248 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2249 = "ttir.subtract"(%2243, %2247, %2248) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2250 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2251 = "ttir.broadcast"(%2249, %2250) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2252 = tensor.empty() : tensor<1x512x1x1xbf16>
    %2253 = "ttir.reshape"(%arg255, %2252) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %2254 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2255 = "ttir.broadcast"(%2253, %2254) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2256 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2257 = "ttir.multiply"(%2251, %2255, %2256) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2258 = tensor.empty() : tensor<512x1x1xbf16>
    %2259 = "ttir.typecast"(%arg256, %2258) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %2260 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2261 = "ttir.broadcast"(%2257, %2260) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2262 = tensor.empty() : tensor<1x512x1x1xbf16>
    %2263 = "ttir.reshape"(%2259, %2262) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %2264 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2265 = "ttir.broadcast"(%2263, %2264) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2266 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2267 = "ttir.multiply"(%2261, %2265, %2266) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2268 = tensor.empty() : tensor<512x1x1xbf16>
    %2269 = "ttir.typecast"(%arg257, %2268) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %2270 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2271 = "ttir.broadcast"(%2267, %2270) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2272 = tensor.empty() : tensor<1x512x1x1xbf16>
    %2273 = "ttir.reshape"(%2269, %2272) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %2274 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2275 = "ttir.broadcast"(%2273, %2274) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2276 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2277 = "ttir.add"(%2271, %2275, %2276) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2278 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2279 = "ttir.typecast"(%2277, %2278) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2280 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2281 = "ttir.maximum"(%2279, %10, %2280) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2282 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2283 = "ttir.convolution"(%2281, %arg52, %2282) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<512x512x3x3xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2284 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2285 = "ttir.typecast"(%2283, %2284) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2286 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2287 = "ttir.broadcast"(%2285, %2286) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2288 = tensor.empty() : tensor<1x512x1x1xbf16>
    %2289 = "ttir.reshape"(%arg258, %2288) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %2290 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2291 = "ttir.broadcast"(%2289, %2290) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2292 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2293 = "ttir.subtract"(%2287, %2291, %2292) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2294 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2295 = "ttir.broadcast"(%2293, %2294) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2296 = tensor.empty() : tensor<1x512x1x1xbf16>
    %2297 = "ttir.reshape"(%arg259, %2296) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %2298 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2299 = "ttir.broadcast"(%2297, %2298) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2300 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2301 = "ttir.multiply"(%2295, %2299, %2300) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2302 = tensor.empty() : tensor<512x1x1xbf16>
    %2303 = "ttir.typecast"(%arg260, %2302) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %2304 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2305 = "ttir.broadcast"(%2301, %2304) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2306 = tensor.empty() : tensor<1x512x1x1xbf16>
    %2307 = "ttir.reshape"(%2303, %2306) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %2308 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2309 = "ttir.broadcast"(%2307, %2308) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2310 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2311 = "ttir.multiply"(%2305, %2309, %2310) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2312 = tensor.empty() : tensor<512x1x1xbf16>
    %2313 = "ttir.typecast"(%arg261, %2312) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<512x1x1xbf16>, tensor<512x1x1xbf16>) -> tensor<512x1x1xbf16>
    %2314 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2315 = "ttir.broadcast"(%2311, %2314) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2316 = tensor.empty() : tensor<1x512x1x1xbf16>
    %2317 = "ttir.reshape"(%2313, %2316) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512x1x1xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %2318 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2319 = "ttir.broadcast"(%2317, %2318) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2320 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2321 = "ttir.add"(%2315, %2319, %2320) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2322 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2323 = "ttir.typecast"(%2321, %2322) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2324 = tensor.empty() : tensor<1x512x7x7xbf16>
    %2325 = "ttir.maximum"(%2323, %10, %2324) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %2326 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2327 = "ttir.convolution"(%2325, %arg53, %2326) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<2048x512x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2328 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2329 = "ttir.typecast"(%2327, %2328) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2330 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2331 = "ttir.broadcast"(%2329, %2330) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2332 = tensor.empty() : tensor<1x2048x1x1xbf16>
    %2333 = "ttir.reshape"(%arg262, %2332) <{shape = [1 : i32, 2048 : i32, 1 : i32, 1 : i32]}> : (tensor<2048x1x1xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %2334 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2335 = "ttir.broadcast"(%2333, %2334) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2336 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2337 = "ttir.subtract"(%2331, %2335, %2336) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2338 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2339 = "ttir.broadcast"(%2337, %2338) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2340 = tensor.empty() : tensor<1x2048x1x1xbf16>
    %2341 = "ttir.reshape"(%arg263, %2340) <{shape = [1 : i32, 2048 : i32, 1 : i32, 1 : i32]}> : (tensor<2048x1x1xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %2342 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2343 = "ttir.broadcast"(%2341, %2342) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2344 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2345 = "ttir.multiply"(%2339, %2343, %2344) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2346 = tensor.empty() : tensor<2048x1x1xbf16>
    %2347 = "ttir.typecast"(%arg264, %2346) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<2048x1x1xbf16>, tensor<2048x1x1xbf16>) -> tensor<2048x1x1xbf16>
    %2348 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2349 = "ttir.broadcast"(%2345, %2348) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2350 = tensor.empty() : tensor<1x2048x1x1xbf16>
    %2351 = "ttir.reshape"(%2347, %2350) <{shape = [1 : i32, 2048 : i32, 1 : i32, 1 : i32]}> : (tensor<2048x1x1xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %2352 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2353 = "ttir.broadcast"(%2351, %2352) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2354 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2355 = "ttir.multiply"(%2349, %2353, %2354) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2356 = tensor.empty() : tensor<2048x1x1xbf16>
    %2357 = "ttir.typecast"(%arg265, %2356) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<2048x1x1xbf16>, tensor<2048x1x1xbf16>) -> tensor<2048x1x1xbf16>
    %2358 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2359 = "ttir.broadcast"(%2355, %2358) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2360 = tensor.empty() : tensor<1x2048x1x1xbf16>
    %2361 = "ttir.reshape"(%2357, %2360) <{shape = [1 : i32, 2048 : i32, 1 : i32, 1 : i32]}> : (tensor<2048x1x1xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %2362 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2363 = "ttir.broadcast"(%2361, %2362) <{broadcast_dimensions = array<i32: 1, 1, 7, 7>}> : (tensor<1x2048x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2364 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2365 = "ttir.add"(%2359, %2363, %2364) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2366 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2367 = "ttir.typecast"(%2365, %2366) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2368 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2369 = "ttir.add"(%2367, %2237, %2368) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2370 = tensor.empty() : tensor<1x2048x7x7xbf16>
    %2371 = "ttir.maximum"(%2369, %11, %2370) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %2372 = tensor.empty() : tensor<1x2048xbf16>
    %2373 = "ttir.sum"(%2371, %2372) <{dim_arg = [2 : i32, 3 : i32], keep_dim = false}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048xbf16>) -> tensor<1x2048xbf16>
    %2374 = tensor.empty() : tensor<1x2048x1x1xbf16>
    %2375 = "ttir.reshape"(%2373, %2374) <{shape = [1 : i32, 2048 : i32, 1 : i32, 1 : i32]}> : (tensor<1x2048xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %2376 = tensor.empty() : tensor<1xbf16>
    %2377 = "ttir.typecast"(%12, %2376) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1xi32>, tensor<1xbf16>) -> tensor<1xbf16>
    %2378 = tensor.empty() : tensor<1xbf16>
    %2379 = "ttir.reshape"(%2377, %2378) <{shape = [1 : i32]}> : (tensor<1xbf16>, tensor<1xbf16>) -> tensor<1xbf16>
    %2380 = tensor.empty() : tensor<1x2048x1x1xbf16>
    %2381 = "ttir.broadcast"(%2375, %2380) <{broadcast_dimensions = array<i32: 1, 1, 1, 1>}> : (tensor<1x2048x1x1xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %2382 = tensor.empty() : tensor<1x1x1x1xbf16>
    %2383 = "ttir.reshape"(%2379, %2382) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x1x1x1xbf16>
    %2384 = tensor.empty() : tensor<1x2048x1x1xbf16>
    %2385 = "ttir.broadcast"(%2383, %2384) <{broadcast_dimensions = array<i32: 1, 2048, 1, 1>}> : (tensor<1x1x1x1xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %2386 = tensor.empty() : tensor<1x2048x1x1xbf16>
    %2387 = "ttir.div"(%2381, %2385, %2386) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x2048x1x1xbf16>, tensor<1x2048x1x1xbf16>, tensor<1x2048x1x1xbf16>) -> tensor<1x2048x1x1xbf16>
    %2388 = tensor.empty() : tensor<1x2048xbf16>
    %2389 = "ttir.reshape"(%2387, %2388) <{shape = [1 : i32, 2048 : i32]}> : (tensor<1x2048x1x1xbf16>, tensor<1x2048xbf16>) -> tensor<1x2048xbf16>
    %2390 = tensor.empty() : tensor<1x2048xbf16>
    %2391 = "ttir.typecast"(%2389, %2390) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x2048xbf16>, tensor<1x2048xbf16>) -> tensor<1x2048xbf16>
    %2392 = "ttir.dot_general"(%2391, %arg266) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x2048xbf16>, tensor<2048x1000xbf16>) -> tensor<1x1000xbf16>
    %2393 = tensor.empty() : tensor<1xbf16>
    %2394 = "ttir.typecast"(%13, %2393) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1xi32>, tensor<1xbf16>) -> tensor<1xbf16>
    %2395 = tensor.empty() : tensor<1xbf16>
    %2396 = "ttir.reshape"(%2394, %2395) <{shape = [1 : i32]}> : (tensor<1xbf16>, tensor<1xbf16>) -> tensor<1xbf16>
    %2397 = tensor.empty() : tensor<1x1000xbf16>
    %2398 = "ttir.broadcast"(%2392, %2397) <{broadcast_dimensions = array<i32: 1, 1>}> : (tensor<1x1000xbf16>, tensor<1x1000xbf16>) -> tensor<1x1000xbf16>
    %2399 = tensor.empty() : tensor<1x1xbf16>
    %2400 = "ttir.reshape"(%2396, %2399) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xbf16>, tensor<1x1xbf16>) -> tensor<1x1xbf16>
    %2401 = tensor.empty() : tensor<1x1000xbf16>
    %2402 = "ttir.broadcast"(%2400, %2401) <{broadcast_dimensions = array<i32: 1, 1000>}> : (tensor<1x1xbf16>, tensor<1x1000xbf16>) -> tensor<1x1000xbf16>
    %2403 = tensor.empty() : tensor<1x1000xbf16>
    %2404 = "ttir.multiply"(%2398, %2402, %2403) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1000xbf16>, tensor<1x1000xbf16>, tensor<1x1000xbf16>) -> tensor<1x1000xbf16>
    %2405 = tensor.empty() : tensor<1x1000xbf16>
    %2406 = "ttir.broadcast"(%2404, %2405) <{broadcast_dimensions = array<i32: 1, 1>}> : (tensor<1x1000xbf16>, tensor<1x1000xbf16>) -> tensor<1x1000xbf16>
    %2407 = tensor.empty() : tensor<1x1000xbf16>
    %2408 = "ttir.reshape"(%arg267, %2407) <{shape = [1 : i32, 1000 : i32]}> : (tensor<1000xbf16>, tensor<1x1000xbf16>) -> tensor<1x1000xbf16>
    %2409 = tensor.empty() : tensor<1x1000xbf16>
    %2410 = "ttir.broadcast"(%2408, %2409) <{broadcast_dimensions = array<i32: 1, 1>}> : (tensor<1x1000xbf16>, tensor<1x1000xbf16>) -> tensor<1x1000xbf16>
    %2411 = tensor.empty() : tensor<1x1000xbf16>
    %2412 = "ttir.add"(%2406, %2410, %2411) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1000xbf16>, tensor<1x1000xbf16>, tensor<1x1000xbf16>) -> tensor<1x1000xbf16>
    %2413 = tensor.empty() : tensor<1x1000xbf16>
    %2414 = "ttir.typecast"(%2412, %2413) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x1000xbf16>, tensor<1x1000xbf16>) -> tensor<1x1000xbf16>
    return %2414 : tensor<1x1000xbf16>
  }
}
