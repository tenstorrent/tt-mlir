// RUN: ttmlir-opt --canonicalize --ttir-fusing %s | FileCheck %s

// ReshapeBroadcastReshapeToRepeatPattern normalizes an
// insert-size-1-dim -> broadcast -> reshape chain into a repeat or a
// repeat_interleave.

// Unsqueeze right of the expanded dim, final reshape merges left -> repeat_interleave.
module {
  func.func @unsqueeze_broadcast_reshape_to_repeat_interleave(%arg0: tensor<1x8x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: @unsqueeze_broadcast_reshape_to_repeat_interleave
    // CHECK: "ttir.repeat_interleave"(%arg0)
    // CHECK-SAME: dim = 1 : si32
    // CHECK-SAME: repeats = 4 : ui32
    // CHECK-NOT: ttir.broadcast
    %0 = "ttir.unsqueeze"(%arg0) <{dim = 2 : si32}> : (tensor<1x8x128x64xbf16>) -> tensor<1x8x1x128x64xbf16>
    %1 = "ttir.broadcast"(%0) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x128x64xbf16>) -> tensor<1x8x4x128x64xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 32 : i32, 128 : i32, 64 : i32]}> : (tensor<1x8x4x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %2 : tensor<1x32x128x64xbf16>
  }
}

// Unsqueeze left of the expanded dim, final reshape merges right -> repeat.
module {
  func.func @unsqueeze_broadcast_reshape_to_repeat(%arg0: tensor<1x8x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: @unsqueeze_broadcast_reshape_to_repeat
    // CHECK: "ttir.repeat"(%arg0)
    // CHECK-SAME: repeat_dimensions = array<i64: 1, 4, 1, 1>
    // CHECK-NOT: ttir.broadcast
    %0 = "ttir.unsqueeze"(%arg0) <{dim = 1 : si32}> : (tensor<1x8x128x64xbf16>) -> tensor<1x1x8x128x64xbf16>
    %1 = "ttir.broadcast"(%0) <{broadcast_dimensions = array<i64: 1, 4, 1, 1, 1>}> : (tensor<1x1x8x128x64xbf16>) -> tensor<1x4x8x128x64xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 32 : i32, 128 : i32, 64 : i32]}> : (tensor<1x4x8x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %2 : tensor<1x32x128x64xbf16>
  }
}

// The reshape spelling of the same two shape changes.
module {
  func.func @reshape_broadcast_reshape_to_repeat_interleave(%arg0: tensor<1x8x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: @reshape_broadcast_reshape_to_repeat_interleave
    // CHECK: "ttir.repeat_interleave"(%arg0)
    // CHECK-SAME: dim = 1 : si32
    // CHECK-SAME: repeats = 4 : ui32
    // CHECK-NOT: ttir.broadcast
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 8 : i32, 1 : i32, 128 : i32, 64 : i32]}> : (tensor<1x8x128x64xbf16>) -> tensor<1x8x1x128x64xbf16>
    %1 = "ttir.broadcast"(%0) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x128x64xbf16>) -> tensor<1x8x4x128x64xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 32 : i32, 128 : i32, 64 : i32]}> : (tensor<1x8x4x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %2 : tensor<1x32x128x64xbf16>
  }
}

module {
  func.func @reshape_broadcast_reshape_to_repeat(%arg0: tensor<1x8x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: @reshape_broadcast_reshape_to_repeat
    // CHECK: "ttir.repeat"(%arg0)
    // CHECK-SAME: repeat_dimensions = array<i64: 1, 4, 1, 1>
    // CHECK-NOT: ttir.broadcast
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 8 : i32, 128 : i32, 64 : i32]}> : (tensor<1x8x128x64xbf16>) -> tensor<1x1x8x128x64xbf16>
    %1 = "ttir.broadcast"(%0) <{broadcast_dimensions = array<i64: 1, 4, 1, 1, 1>}> : (tensor<1x1x8x128x64xbf16>) -> tensor<1x4x8x128x64xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 32 : i32, 128 : i32, 64 : i32]}> : (tensor<1x4x8x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %2 : tensor<1x32x128x64xbf16>
  }
}

// The expanded dim has size 1.
module {
  func.func @unsqueeze_broadcast_reshape_size_one_dim(%arg0: tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: @unsqueeze_broadcast_reshape_size_one_dim
    // CHECK: "ttir.repeat_interleave"(%arg0)
    // CHECK-SAME: dim = 1 : si32
    // CHECK-SAME: repeats = 32 : ui32
    // CHECK-NOT: ttir.broadcast
    %0 = "ttir.unsqueeze"(%arg0) <{dim = 2 : si32}> : (tensor<1x1x128x64xbf16>) -> tensor<1x1x1x128x64xbf16>
    %1 = "ttir.broadcast"(%0) <{broadcast_dimensions = array<i64: 1, 1, 32, 1, 1>}> : (tensor<1x1x1x128x64xbf16>) -> tensor<1x1x32x128x64xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 32 : i32, 128 : i32, 64 : i32]}> : (tensor<1x1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %2 : tensor<1x32x128x64xbf16>
  }
}

// Negative: the inserted dim feeds a second consumer, so the broadcast would
// survive anyway and the rewrite is not a saving.
module {
  func.func @unsqueeze_broadcast_reshape_multiple_uses(%arg0: tensor<1x8x128x64xbf16>) -> (tensor<1x32x128x64xbf16>, tensor<1x8x1x128x64xbf16>) {
    // CHECK-LABEL: @unsqueeze_broadcast_reshape_multiple_uses
    // CHECK: ttir.broadcast
    // CHECK-NOT: ttir.repeat_interleave
    %0 = "ttir.unsqueeze"(%arg0) <{dim = 2 : si32}> : (tensor<1x8x128x64xbf16>) -> tensor<1x8x1x128x64xbf16>
    %1 = "ttir.broadcast"(%0) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x128x64xbf16>) -> tensor<1x8x4x128x64xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 32 : i32, 128 : i32, 64 : i32]}> : (tensor<1x8x4x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %2, %0 : tensor<1x32x128x64xbf16>, tensor<1x8x1x128x64xbf16>
  }
}

// Negative: the broadcast expands two dims at once. Only a single expanded
// dim is modelled, so the chain is left alone rather than split into a
// repeat_interleave plus a repeat.
module {
  func.func @broadcast_of_two_dims_not_fused(%arg0: tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: @broadcast_of_two_dims_not_fused
    // CHECK: ttir.broadcast
    // CHECK-NOT: ttir.repeat_interleave
    %0 = "ttir.unsqueeze"(%arg0) <{dim = 2 : si32}> : (tensor<1x1x128x64xbf16>) -> tensor<1x1x1x128x64xbf16>
    %1 = "ttir.broadcast"(%0) <{broadcast_dimensions = array<i64: 1, 4, 8, 1, 1>}> : (tensor<1x1x1x128x64xbf16>) -> tensor<1x4x8x128x64xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 32 : i32, 128 : i32, 64 : i32]}> : (tensor<1x4x8x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %2 : tensor<1x32x128x64xbf16>
  }
}

// An expansion merged into the last dim normalizes the same way.
module {
  func.func @unsqueeze_broadcast_reshape_on_last_dim(%arg0: tensor<1x8x128x64xbf16>) -> tensor<1x8x128x256xbf16> {
    // CHECK-LABEL: @unsqueeze_broadcast_reshape_on_last_dim
    // CHECK: "ttir.repeat_interleave"(%arg0)
    // CHECK-SAME: dim = 3 : si32
    // CHECK-SAME: repeats = 4 : ui32
    // CHECK-NOT: ttir.broadcast
    %0 = "ttir.unsqueeze"(%arg0) <{dim = 4 : si32}> : (tensor<1x8x128x64xbf16>) -> tensor<1x8x128x64x1xbf16>
    %1 = "ttir.broadcast"(%0) <{broadcast_dimensions = array<i64: 1, 1, 1, 1, 4>}> : (tensor<1x8x128x64x1xbf16>) -> tensor<1x8x128x64x4xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 8 : i32, 128 : i32, 256 : i32]}> : (tensor<1x8x128x64x4xbf16>) -> tensor<1x8x128x256xbf16>
    return %2 : tensor<1x8x128x256xbf16>
  }
}
