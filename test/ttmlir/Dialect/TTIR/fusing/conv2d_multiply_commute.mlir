// RUN: ttmlir-opt --ttir-fusing %s | FileCheck %s

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

  // Verify that we can commute only first multiply.
  // This is because when rewriting we check if weight argument to conv is constant block argument, which is not the case after we commute first multiply.
  // Ideally we would commute second one also, but it would require more complex analysis.
  // CHECK-LABEL: func.func @conv2d_with_chain_of_multiply
  func.func @conv2d_with_chain_of_multiply(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x30x30x64xbf16> {
    // CHECK: "ttir.reshape"
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

    // CHECK: "ttir.multiply"
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
    %weight = "ttir.zeros"() <{shape = array<i32: 64, 64, 3, 3>}> : () -> tensor<64x64x3x3xbf16>
    %scale = "ttir.ones"() <{shape = array<i32: 1, 1, 1, 64>}> : () -> tensor<1x1x1x64xbf16>
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

  // Check that we can't commute since %scale is not before %conv in block.
  // CHECK-LABEL: func.func @conv2d_creation_op_non_commutable
  func.func @conv2d_creation_op_non_commutable(%input: tensor<1x32x32x64xbf16>) -> tensor<1x30x30x64xbf16> {
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.conv2d"
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    %weight = "ttir.zeros"() <{shape = array<i32: 64, 64, 3, 3>}> : () -> tensor<64x64x3x3xbf16>
    %conv = "ttir.conv2d"(%input, %weight, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    // CHECK: "ttir.multiply"
    %scale = "ttir.ones"() <{shape = array<i32: 1, 1, 1, 64>}> : () -> tensor<1x1x1x64xbf16>
    %1 = ttir.empty() : tensor<1x30x30x64xbf16>
    %2 = "ttir.multiply"(%conv, %scale, %1) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>

    return %2: tensor<1x30x30x64xbf16>
  }
}
