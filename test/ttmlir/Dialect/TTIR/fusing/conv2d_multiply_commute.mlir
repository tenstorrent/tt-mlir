// RUN: ttmlir-opt --ttir-implicit-broadcast-fold --ttir-fusing="ttnn-enable-conv2d-with-multiply-pattern=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Commute multiply before conv2d.
module {
  // CHECK-LABEL: func.func @commute_multiply
  func.func @commute_multiply(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x30x30x64xbf16> {
    // CHECK: %[[SCALE:.*]] = "ttir.reshape"
    // CHECK-SAME: (%arg2
    // CHECK: %[[WEIGHT_SCALED:.*]] = "ttir.multiply"
    // CHECK-SAME: (%arg1, %[[SCALE]]
    // CHECK: %[[CONV:.*]] = "ttir.conv2d"
    // CHECK: (%arg0, %[[WEIGHT_SCALED]]
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>
    %4 = "ttir.multiply"(%1, %arg2, %2) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>

    // CHECK: return %[[CONV]]
    return %4: tensor<1x30x30x64xbf16>
  }
  // Check that we can commute multiply before conv2d with bias.
  // CHECK-LABEL: func.func @commute_multiply_with_bias
  func.func @commute_multiply_with_bias(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg3: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x30x30x64xbf16> {
    // CHECK: %[[SCALE:.*]] = "ttir.reshape"
    // CHECK-SAME: (%arg2
    // CHECK: %[[WEIGHT_SCALED:.*]] = "ttir.multiply"
    // CHECK-SAME: (%arg1, %[[SCALE]]
    // CHECK: %[[BIAS_SCALED:.*]] = "ttir.multiply"
    // CHECK-SAME: (%arg3, %arg2
    // CHECK: %[[CONV:.*]] = "ttir.conv2d"
    // CHECK-SAME: (%arg0, %[[WEIGHT_SCALED]], %[[BIAS_SCALED]]
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg3, %0)
        <{
          stride = 1: i32,
          padding = 0: i32,
          dilation = 1: i32,
          groups = 1: i32
        }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>
    %4 = "ttir.multiply"(%1, %arg2, %2) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>

    // CHECK: return %[[CONV]]
    return %4: tensor<1x30x30x64xbf16>
  }

  // Check that we can't commute because conv has more than one use.
  // CHECK-LABEL: func.func @conv2d_with_multiple_uses
  func.func @conv2d_with_multiple_uses(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x30x30x64xbf16> {
    // CHECK: "ttir.conv2d"
    // CHECK: "ttir.multiply"
    // CHECK: "ttir.multiply"
    // CHECK: "ttir.add"
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>
    %4 = "ttir.multiply"(%1, %arg2, %2) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %5 = ttir.empty() : tensor<1x30x30x64xbf16>
    %6 = "ttir.multiply"(%1, %arg2, %5) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %7 = ttir.empty() : tensor<1x30x30x64xbf16>
    %8 = "ttir.add"(%6, %4, %7) : (tensor<1x30x30x64xbf16>, tensor<1x30x30x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %8: tensor<1x30x30x64xbf16>
  }

  // Check that we can't commute because scale operand is not in format (1, 1, 1, out_channels).
  // CHECK-LABEL: func.func @conv2d_with_invalid_scale
  func.func @conv2d_with_invalid_scale(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x1x30x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x30x30x64xbf16> {
    // CHECK: "ttir.conv2d"
    // CHECK: "ttir.multiply"
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>
    %4 = "ttir.multiply"(%1, %arg2, %2) : (tensor<1x30x30x64xbf16>, tensor<1x1x30x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %4: tensor<1x30x30x64xbf16>
  }

  // Verify that we can't commute because function arguments are not constants.
  // %arg2 is not constant in this case.
  func.func @conv2d_with_non_constant_scale(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    // CHECK: "ttir.conv2d"
    // CHECK: "ttir.multiply"
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>
    %4 = "ttir.multiply"(%1, %arg2, %2) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %4: tensor<1x30x30x64xbf16>
  }

  // Check that we can commute consecutive multiply.
  // CHECK-LABEL: func.func @conv2d_with_chain_of_multiply
  func.func @conv2d_with_chain_of_multiply(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x30x30x64xbf16> {
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.multiply"
    // CHECK: "ttir.multiply"
    // CHECK: "ttir.conv2d"
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>
    %4 = "ttir.multiply"(%1, %arg2, %2) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>

    %5 = ttir.empty() : tensor<1x30x30x64xbf16>
    %6 = "ttir.multiply"(%4, %arg2, %5) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %6: tensor<1x30x30x64xbf16>
  }

  // Verify that we can commute when weight and scale are creation ops.
  // CHECK-LABEL: func.func @conv2d_creation_op_commute
  func.func @conv2d_creation_op_commute(%input: tensor<1x32x32x64xbf16>) -> tensor<1x30x30x64xbf16> {
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.multiply"
    // CHECK: "ttir.conv2d"
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    %weight = "ttir.zeros"() <{shape = array<i32: 64, 64, 3, 3>, dtype = bf16}> : () -> tensor<64x64x3x3xbf16>
    %scale = "ttir.ones"() <{shape = array<i32: 1, 1, 1, 64>, dtype = bf16}> : () -> tensor<1x1x1x64xbf16>
    %conv = "ttir.conv2d"(%input, %weight, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    // CHECK-NOT: "ttir.multiply"
    %1 = ttir.empty() : tensor<1x30x30x64xbf16>
    %2 = "ttir.multiply"(%conv, %scale, %1) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>

    return %2: tensor<1x30x30x64xbf16>
  }

  // Check that we can commute by moving scale before conv2d.
  // CHECK-LABEL: func.func @conv2d_creation_op_commutable
  func.func @conv2d_creation_op_commutable(%input: tensor<1x32x32x64xbf16>) -> tensor<1x30x30x64xbf16> {
    // CHECK: "ttir.ones"
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.multiply"
    // CHECK: "ttir.conv2d"
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    %weight = "ttir.zeros"() <{shape = array<i32: 64, 64, 3, 3>, dtype = bf16}> : () -> tensor<64x64x3x3xbf16>
    %conv = "ttir.conv2d"(%input, %weight, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %scale = "ttir.ones"() <{shape = array<i32: 1, 1, 1, 64>, dtype = bf16}> : () -> tensor<1x1x1x64xbf16>
    %1 = ttir.empty() : tensor<1x30x30x64xbf16>
    %2 = "ttir.multiply"(%conv, %scale, %1) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>

    return %2: tensor<1x30x30x64xbf16>
  }

  // Check that we can commute const-eval subgraph which generates scale for conv2d output.
  // CHECK-LABEL: func.func @conv2d_subgraph_commute
  func.func @conv2d_subgraph_commute(%arg0: tensor<1x3x224x224xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<1x32x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x32x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg3: tensor<32x3x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.conv2d_weight}, %arg4: tensor<32x1x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.conv2d_weight}) -> tensor<1x112x112x32xbf16> {
    // Ignore first reshape which is for conv input.
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.transpose"
    // CHECK: "ttir.transpose"
    // CHECK: %[[RESHAPE:.*]] = "ttir.reshape"
    // CHECK-SAME: {shape = [32 : i32, 1 : i32, 1 : i32, 1 : i32]}
    // CHECK: %[[MUL:.*]] = "ttir.multiply"
    // CHECK-SAME: (%arg3, %[[RESHAPE]]
    // CHECK: "ttir.conv2d"
    // CHECK-SAME: %[[MUL]]
    %0 = ttir.empty() : tensor<1x224x3x224xbf16>
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x3x224x224xbf16>, tensor<1x224x3x224xbf16>) -> tensor<1x224x3x224xbf16>
    %2 = ttir.empty() : tensor<1x224x224x3xbf16>
    %3 = "ttir.transpose"(%1, %2) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x224x3x224xbf16>, tensor<1x224x224x3xbf16>) -> tensor<1x224x224x3xbf16>
    %4 = ttir.empty() : tensor<1x1x50176x3xbf16>
    %5 = "ttir.reshape"(%3, %4) <{shape = [1 : i32, 1 : i32, 50176 : i32, 3 : i32]}> : (tensor<1x224x224x3xbf16>, tensor<1x1x50176x3xbf16>) -> tensor<1x1x50176x3xbf16>
    %6 = ttir.empty() : tensor<1x1x12544x32xbf16>
    %7 = "ttir.conv2d"(%5, %arg3, %6) <{dilation = array<i32: 1, 1>, flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 224, input_width = 224>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> : (tensor<1x1x50176x3xbf16>, tensor<32x3x3x3xbf16>, tensor<1x1x12544x32xbf16>) -> tensor<1x1x12544x32xbf16>
    %8 = ttir.empty() : tensor<1x1x32x1xbf16>
    %9 = "ttir.transpose"(%arg1, %8) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x1x1xbf16>, tensor<1x1x32x1xbf16>) -> tensor<1x1x32x1xbf16>
    %10 = ttir.empty() : tensor<1x1x1x32xbf16>
    %11 = "ttir.transpose"(%9, %10) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x1x32x1xbf16>, tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xbf16>
    %12 = ttir.empty() : tensor<1x1x12544x32xbf16>
    %13 = "ttir.broadcast"(%11, %12) <{broadcast_dimensions = array<i64: 1, 1, 12544, 1>}> : (tensor<1x1x1x32xbf16>, tensor<1x1x12544x32xbf16>) -> tensor<1x1x12544x32xbf16>
    %14 = ttir.empty() : tensor<1x1x12544x32xbf16>
    %15 = "ttir.multiply"(%7, %13, %14) : (tensor<1x1x12544x32xbf16>, tensor<1x1x12544x32xbf16>, tensor<1x1x12544x32xbf16>) -> tensor<1x1x12544x32xbf16>
    %16 = ttir.empty() : tensor<1x112x112x32xbf16>
    %17 = "ttir.reshape"(%15, %16) <{shape = [1 : i32, 112 : i32, 112 : i32, 32 : i32]}> : (tensor<1x1x12544x32xbf16>, tensor<1x112x112x32xbf16>) -> tensor<1x112x112x32xbf16>
    return %17 : tensor<1x112x112x32xbf16>
  }

  // Check that we can't commute const-eval since arg1 is not constant.
  // CHECK-LABEL: func.func @conv2d_subgraph_not_commuteable
  func.func @conv2d_subgraph_not_commuteable(%arg0: tensor<1x3x224x224xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<1x32x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg2: tensor<1x32x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg3: tensor<32x3x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.conv2d_weight}, %arg4: tensor<32x1x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.conv2d_weight}) -> tensor<1x112x112x32xbf16> {
    // Ignore first reshape which is for conv input.
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.conv2d"
    // CHECK: "ttir.transpose"
    // CHECK: "ttir.transpose"
    // CHECK: "ttir.multiply"
    %0 = ttir.empty() : tensor<1x224x3x224xbf16>
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x3x224x224xbf16>, tensor<1x224x3x224xbf16>) -> tensor<1x224x3x224xbf16>
    %2 = ttir.empty() : tensor<1x224x224x3xbf16>
    %3 = "ttir.transpose"(%1, %2) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x224x3x224xbf16>, tensor<1x224x224x3xbf16>) -> tensor<1x224x224x3xbf16>
    %4 = ttir.empty() : tensor<1x1x50176x3xbf16>
    %5 = "ttir.reshape"(%3, %4) <{shape = [1 : i32, 1 : i32, 50176 : i32, 3 : i32]}> : (tensor<1x224x224x3xbf16>, tensor<1x1x50176x3xbf16>) -> tensor<1x1x50176x3xbf16>
    %6 = ttir.empty() : tensor<1x1x12544x32xbf16>
    %7 = "ttir.conv2d"(%5, %arg3, %6) <{dilation = array<i32: 1, 1>, flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 224, input_width = 224>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> : (tensor<1x1x50176x3xbf16>, tensor<32x3x3x3xbf16>, tensor<1x1x12544x32xbf16>) -> tensor<1x1x12544x32xbf16>
    %8 = ttir.empty() : tensor<1x1x32x1xbf16>
    %9 = "ttir.transpose"(%arg1, %8) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x1x1xbf16>, tensor<1x1x32x1xbf16>) -> tensor<1x1x32x1xbf16>
    %10 = ttir.empty() : tensor<1x1x1x32xbf16>
    %11 = "ttir.transpose"(%9, %10) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x1x32x1xbf16>, tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xbf16>
    %12 = ttir.empty() : tensor<1x1x12544x32xbf16>
    %13 = "ttir.broadcast"(%11, %12) <{broadcast_dimensions = array<i64: 1, 1, 12544, 1>}> : (tensor<1x1x1x32xbf16>, tensor<1x1x12544x32xbf16>) -> tensor<1x1x12544x32xbf16>
    %14 = ttir.empty() : tensor<1x1x12544x32xbf16>
    %15 = "ttir.multiply"(%7, %13, %14) : (tensor<1x1x12544x32xbf16>, tensor<1x1x12544x32xbf16>, tensor<1x1x12544x32xbf16>) -> tensor<1x1x12544x32xbf16>
    %16 = ttir.empty() : tensor<1x112x112x32xbf16>
    %17 = "ttir.reshape"(%15, %16) <{shape = [1 : i32, 112 : i32, 112 : i32, 32 : i32]}> : (tensor<1x1x12544x32xbf16>, tensor<1x112x112x32xbf16>) -> tensor<1x112x112x32xbf16>
    return %17 : tensor<1x112x112x32xbf16>
  }
}
