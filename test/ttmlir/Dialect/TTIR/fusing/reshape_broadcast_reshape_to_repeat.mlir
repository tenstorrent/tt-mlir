// RUN: ttmlir-opt --ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

// Single reshape -> broadcast -> reshape pattern that fuses to
// repeat_interleave. The size-1 dim is inserted to the right of the dim being
// repeated, so the merge collapses leftward.
module {
  func.func @repeat_interleave_simple(%arg0: tensor<1x512x120x208xbf16>) -> tensor<1x512x240x208xbf16> {
    // CHECK-LABEL: func.func @repeat_interleave_simple
    // CHECK-NOT: ttir.reshape
    // CHECK-NOT: ttir.broadcast
    // CHECK: %[[R:.*]] = "ttir.repeat_interleave"(%arg0) <{dim = 2 : si32, repeats = 2 : ui32}>
    // CHECK: return %[[R]]
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 512 : i32, 120 : i32, 1 : i32, 208 : i32]}> : (tensor<1x512x120x208xbf16>) -> tensor<1x512x120x1x208xbf16>
    %1 = "ttir.broadcast"(%0) <{broadcast_dimensions = array<i64: 1, 1, 1, 2, 1>}> : (tensor<1x512x120x1x208xbf16>) -> tensor<1x512x120x2x208xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 512 : i32, 240 : i32, 208 : i32]}> : (tensor<1x512x120x2x208xbf16>) -> tensor<1x512x240x208xbf16>
    return %2 : tensor<1x512x240x208xbf16>
  }
}

// Two chained reshape -> broadcast -> reshape patterns. The merge reshape of
// the first chain and the unsqueeze reshape of the second chain are folded
// together by ReshapeOp::fold (foldConsecutiveReshape) before the fusing
// pattern runs, so the input rank into the second chain's broadcast does not
// match the simple "rank+1" structure of an unsqueeze. Both chains must still
// fuse to repeat_interleave.
module {
  func.func @repeat_interleave_chained(%arg0: tensor<1x512x120x208xbf16>) -> tensor<1x512x240x416xbf16> {
    // CHECK-LABEL: func.func @repeat_interleave_chained
    // CHECK-NOT: ttir.broadcast
    // CHECK: %[[R0:.*]] = "ttir.repeat_interleave"(%arg0) <{dim = 2 : si32, repeats = 2 : ui32}> : (tensor<1x512x120x208xbf16>) -> tensor<1x512x240x208xbf16>
    // CHECK: %[[R1:.*]] = "ttir.repeat_interleave"(%[[R0]]) <{dim = 3 : si32, repeats = 2 : ui32}> : (tensor<1x512x240x208xbf16>) -> tensor<1x512x240x416xbf16>
    // CHECK: return %[[R1]]
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 512 : i32, 120 : i32, 1 : i32, 208 : i32]}> : (tensor<1x512x120x208xbf16>) -> tensor<1x512x120x1x208xbf16>
    %1 = "ttir.broadcast"(%0) <{broadcast_dimensions = array<i64: 1, 1, 1, 2, 1>}> : (tensor<1x512x120x1x208xbf16>) -> tensor<1x512x120x2x208xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 512 : i32, 240 : i32, 208 : i32]}> : (tensor<1x512x120x2x208xbf16>) -> tensor<1x512x240x208xbf16>
    %3 = "ttir.reshape"(%2) <{shape = [1 : i32, 512 : i32, 240 : i32, 208 : i32, 1 : i32]}> : (tensor<1x512x240x208xbf16>) -> tensor<1x512x240x208x1xbf16>
    %4 = "ttir.broadcast"(%3) <{broadcast_dimensions = array<i64: 1, 1, 1, 1, 2>}> : (tensor<1x512x240x208x1xbf16>) -> tensor<1x512x240x208x2xbf16>
    %5 = "ttir.reshape"(%4) <{shape = [1 : i32, 512 : i32, 240 : i32, 416 : i32]}> : (tensor<1x512x240x208x2xbf16>) -> tensor<1x512x240x416xbf16>
    return %5 : tensor<1x512x240x416xbf16>
  }
}

// GQA-style reshape -> broadcast -> reshape that should fuse to repeat (size-1
// dim inserted at the merge position, merge collapses rightward).
module {
  func.func @repeat_gqa(%arg0: tensor<1x8x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: func.func @repeat_gqa
    // CHECK-NOT: ttir.reshape
    // CHECK-NOT: ttir.broadcast
    // CHECK: %[[R:.*]] = "ttir.repeat"(%arg0) <{repeat_dimensions = array<i64: 1, 4, 1, 1>}>
    // CHECK: return %[[R]]
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 8 : i32, 128 : i32, 64 : i32]}> : (tensor<1x8x128x64xbf16>) -> tensor<1x1x8x128x64xbf16>
    %1 = "ttir.broadcast"(%0) <{broadcast_dimensions = array<i64: 1, 4, 1, 1, 1>}> : (tensor<1x1x8x128x64xbf16>) -> tensor<1x4x8x128x64xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 32 : i32, 128 : i32, 64 : i32]}> : (tensor<1x4x8x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %2 : tensor<1x32x128x64xbf16>
  }
}
