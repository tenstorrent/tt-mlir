// RUN: ttmlir-opt --ttir-fold-sliced-bias-into-conv -o %t %s
// RUN: FileCheck %s --input-file=%t

// Bias-less NHWC conv -> NCHW transposes -> channel ranges, each with its own
// parameter bias add: the biases become the conv's bias, the adds go away.

module {
  // CHECK-LABEL: func.func @head
  func.func @head(%x: tensor<1x3x13x80xbf16>,
                  %w: tensor<15x80x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                  %b0: tensor<1x3x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                  %b1: tensor<1x4x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                  %b2: tensor<1x8x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>})
      -> (tensor<1x3x3x13xbf16>, tensor<1x4x3x13xbf16>, tensor<1x8x3x13xbf16>) {
    // CHECK-DAG: %[[R0:.*]] = "ttir.reshape"(%arg2) <{shape = [1 : i32, 1 : i32, 1 : i32, 3 : i32]}>
    // CHECK-DAG: %[[R1:.*]] = "ttir.reshape"(%arg3) <{shape = [1 : i32, 1 : i32, 1 : i32, 4 : i32]}>
    // CHECK-DAG: %[[R2:.*]] = "ttir.reshape"(%arg4) <{shape = [1 : i32, 1 : i32, 1 : i32, 8 : i32]}>
    // CHECK: %[[B:.*]] = "ttir.concat"(%[[R0]], %[[R1]], %[[R2]]) <{dim = 3 : si32}> {{.*}} -> tensor<1x1x1x15xbf16>
    // CHECK: "ttir.conv2d"(%arg0, %arg1, %[[B]])
    // CHECK-NOT: "ttir.add"
    // CHECK: return
    %0 = "ttir.conv2d"(%x, %w) <{batch_dim = 0 : i64, channel_dim = 3 : i64, dilation = array<i32: 1, 1>, groups = 1 : i32, height_dim = 1 : i64, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>, width_dim = 2 : i64}> : (tensor<1x3x13x80xbf16>, tensor<15x80x3x3xbf16>) -> tensor<1x3x13x15xbf16>
    %1 = "ttir.transpose"(%0) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x3x13x15xbf16>) -> tensor<1x3x15x13xbf16>
    %2 = "ttir.transpose"(%1) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x3x15x13xbf16>) -> tensor<1x15x3x13xbf16>
    %3 = "ttir.index"(%2) <{begin = 0 : i32, dim = 1 : i32, end = 3 : i32, step = 1 : i32}> : (tensor<1x15x3x13xbf16>) -> tensor<1x3x3x13xbf16>
    %4 = "ttir.add"(%3, %b0) : (tensor<1x3x3x13xbf16>, tensor<1x3x1x1xbf16>) -> tensor<1x3x3x13xbf16>
    %5 = "ttir.index"(%2) <{begin = 3 : i32, dim = 1 : i32, end = 7 : i32, step = 1 : i32}> : (tensor<1x15x3x13xbf16>) -> tensor<1x4x3x13xbf16>
    %6 = "ttir.add"(%5, %b1) : (tensor<1x4x3x13xbf16>, tensor<1x4x1x1xbf16>) -> tensor<1x4x3x13xbf16>
    %7 = "ttir.index"(%2) <{begin = 7 : i32, dim = 1 : i32, end = 15 : i32, step = 1 : i32}> : (tensor<1x15x3x13xbf16>) -> tensor<1x8x3x13xbf16>
    %8 = "ttir.add"(%7, %b2) : (tensor<1x8x3x13xbf16>, tensor<1x8x1x1xbf16>) -> tensor<1x8x3x13xbf16>
    return %4, %6, %8 : tensor<1x3x3x13xbf16>, tensor<1x4x3x13xbf16>, tensor<1x8x3x13xbf16>
  }

  // The ranges leave channels 3..6 uncovered: nothing to fold.
  // CHECK-LABEL: func.func @gap
  // CHECK: "ttir.conv2d"(%arg0, %arg1) <{
  // CHECK: "ttir.add"
  func.func @gap(%x: tensor<1x3x13x80xbf16>,
                 %w: tensor<15x80x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                 %b0: tensor<1x3x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                 %b2: tensor<1x8x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>})
      -> (tensor<1x3x3x13xbf16>, tensor<1x8x3x13xbf16>) {
    %0 = "ttir.conv2d"(%x, %w) <{batch_dim = 0 : i64, channel_dim = 3 : i64, dilation = array<i32: 1, 1>, groups = 1 : i32, height_dim = 1 : i64, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>, width_dim = 2 : i64}> : (tensor<1x3x13x80xbf16>, tensor<15x80x3x3xbf16>) -> tensor<1x3x13x15xbf16>
    %1 = "ttir.transpose"(%0) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x3x13x15xbf16>) -> tensor<1x3x15x13xbf16>
    %2 = "ttir.transpose"(%1) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x3x15x13xbf16>) -> tensor<1x15x3x13xbf16>
    %3 = "ttir.index"(%2) <{begin = 0 : i32, dim = 1 : i32, end = 3 : i32, step = 1 : i32}> : (tensor<1x15x3x13xbf16>) -> tensor<1x3x3x13xbf16>
    %4 = "ttir.add"(%3, %b0) : (tensor<1x3x3x13xbf16>, tensor<1x3x1x1xbf16>) -> tensor<1x3x3x13xbf16>
    %5 = "ttir.index"(%2) <{begin = 7 : i32, dim = 1 : i32, end = 15 : i32, step = 1 : i32}> : (tensor<1x15x3x13xbf16>) -> tensor<1x8x3x13xbf16>
    %6 = "ttir.add"(%5, %b2) : (tensor<1x8x3x13xbf16>, tensor<1x8x1x1xbf16>) -> tensor<1x8x3x13xbf16>
    return %4, %6 : tensor<1x3x3x13xbf16>, tensor<1x8x3x13xbf16>
  }

  // A bias that is a runtime input (not a parameter) would make the folded
  // bias a per-inference concat: not folded.
  // CHECK-LABEL: func.func @runtime_bias
  // CHECK: "ttir.conv2d"(%arg0, %arg1) <{
  // CHECK: "ttir.add"
  func.func @runtime_bias(%x: tensor<1x3x13x80xbf16>,
                          %w: tensor<7x80x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                          %b0: tensor<1x3x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                          %b1: tensor<1x4x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>})
      -> (tensor<1x3x3x13xbf16>, tensor<1x4x3x13xbf16>) {
    %0 = "ttir.conv2d"(%x, %w) <{batch_dim = 0 : i64, channel_dim = 3 : i64, dilation = array<i32: 1, 1>, groups = 1 : i32, height_dim = 1 : i64, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>, width_dim = 2 : i64}> : (tensor<1x3x13x80xbf16>, tensor<7x80x3x3xbf16>) -> tensor<1x3x13x7xbf16>
    %1 = "ttir.permute"(%0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x3x13x7xbf16>) -> tensor<1x7x3x13xbf16>
    %2 = "ttir.index"(%1) <{begin = 0 : i32, dim = 1 : i32, end = 3 : i32, step = 1 : i32}> : (tensor<1x7x3x13xbf16>) -> tensor<1x3x3x13xbf16>
    %3 = "ttir.add"(%2, %b0) : (tensor<1x3x3x13xbf16>, tensor<1x3x1x1xbf16>) -> tensor<1x3x3x13xbf16>
    %4 = "ttir.index"(%1) <{begin = 3 : i32, dim = 1 : i32, end = 7 : i32, step = 1 : i32}> : (tensor<1x7x3x13xbf16>) -> tensor<1x4x3x13xbf16>
    %5 = "ttir.add"(%4, %b1) : (tensor<1x4x3x13xbf16>, tensor<1x4x1x1xbf16>) -> tensor<1x4x3x13xbf16>
    return %3, %5 : tensor<1x3x3x13xbf16>, tensor<1x4x3x13xbf16>
  }

  // An inner (non-output) range 8:40 is 32 channels wide but starts off a
  // tile boundary: the conv's output channels are reordered so it comes first.
  // CHECK-LABEL: func.func @reorder
  // CHECK-DAG: %[[W1:.*]] = "ttir.slice_static"(%arg1) <{begins = [8 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [40 : i32, 80 : i32, 3 : i32, 3 : i32]
  // CHECK-DAG: %[[W0:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [8 : i32, 80 : i32, 3 : i32, 3 : i32]
  // CHECK: %[[W:.*]] = "ttir.concat"(%[[W1]], %[[W0]]) <{dim = 0 : si32}>
  // CHECK-DAG: %[[B1:.*]] = "ttir.reshape"(%arg3)
  // CHECK-DAG: %[[B0:.*]] = "ttir.reshape"(%arg2)
  // CHECK: %[[B:.*]] = "ttir.concat"(%[[B1]], %[[B0]]) <{dim = 3 : si32}>
  // CHECK: "ttir.conv2d"(%arg0, %[[W]], %[[B]])
  // CHECK-DAG: "ttir.index"({{.*}}) <{begin = 32 : i32, dim = 1 : i32, end = 40 : i32
  // CHECK-DAG: "ttir.index"({{.*}}) <{begin = 0 : i32, dim = 1 : i32, end = 32 : i32
  // CHECK-NOT: "ttir.add"
  func.func @reorder(%x: tensor<1x3x13x80xbf16>,
                     %w: tensor<40x80x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                     %b0: tensor<1x8x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                     %b1: tensor<1x32x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>})
      -> (tensor<1x8x3x13xbf16>, tensor<1x32x3x13xbf16>) {
    %0 = "ttir.conv2d"(%x, %w) <{batch_dim = 0 : i64, channel_dim = 3 : i64, dilation = array<i32: 1, 1>, groups = 1 : i32, height_dim = 1 : i64, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>, width_dim = 2 : i64}> : (tensor<1x3x13x80xbf16>, tensor<40x80x3x3xbf16>) -> tensor<1x3x13x40xbf16>
    %1 = "ttir.permute"(%0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x3x13x40xbf16>) -> tensor<1x40x3x13xbf16>
    %2 = "ttir.index"(%1) <{begin = 0 : i32, dim = 1 : i32, end = 8 : i32, step = 1 : i32}> : (tensor<1x40x3x13xbf16>) -> tensor<1x8x3x13xbf16>
    %3 = "ttir.add"(%2, %b0) : (tensor<1x8x3x13xbf16>, tensor<1x8x1x1xbf16>) -> tensor<1x8x3x13xbf16>
    %4 = "ttir.index"(%1) <{begin = 8 : i32, dim = 1 : i32, end = 40 : i32, step = 1 : i32}> : (tensor<1x40x3x13xbf16>) -> tensor<1x32x3x13xbf16>
    %5 = "ttir.add"(%4, %b1) : (tensor<1x32x3x13xbf16>, tensor<1x32x1x1xbf16>) -> tensor<1x32x3x13xbf16>
    %6 = "ttir.relu"(%5) : (tensor<1x32x3x13xbf16>) -> tensor<1x32x3x13xbf16>
    return %3, %6 : tensor<1x8x3x13xbf16>, tensor<1x32x3x13xbf16>
  }
}
