// RUN: ttmlir-opt --ttnn-force-final-deallocs -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Test for the --ttnn-force-final-deallocs pass.
// A view-eligible ttnn.reshape aliases its input's buffer, so the input and the
// reshape result get separate ttnn.deallocate ops that both target one buffer.
// The pass forces the last deallocation (bottom-most in program order) of each
// such buffer so the memory is actually freed, while leaving buffers that
// escape the function untouched. All other deallocations of that buffer are no-ops
// and are removed.

#dram = #ttnn.buffer_type<dram>
#l2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#l3 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#l4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 64 + d2, d3), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  // %0 and its view %1 share one buffer; the last deallocate (%1's) is forced,
  // the earlier one (%0's) is a redundant no-op and is removed.
  // CHECK-LABEL: func.func @aliased
  func.func @aliased(%arg0: tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3> {
    %0 = "ttnn.add"(%arg0, %arg0) : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    %1 = "ttnn.reshape"(%0) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
    %2 = "ttnn.add"(%1, %1) : (tensor<1x64x128xbf16, #l3>, tensor<1x64x128xbf16, #l3>) -> tensor<1x64x128xbf16, #l3>
    // CHECK-NOT: "ttnn.deallocate"
    // CHECK: "ttnn.deallocate"(%1) <{force = true}>
    // CHECK-NOT: "ttnn.deallocate"
    "ttnn.deallocate"(%0) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x64x128xbf16, #l3>) -> ()
    return %2 : tensor<1x64x128xbf16, #l3>
  }

  // A view of the buffer is returned, so the buffer escapes the function and is
  // freed by the caller. All of its (no-op) deallocates are removed.
  // CHECK-LABEL: func.func @returned_aliased
  func.func @returned_aliased(%arg0: tensor<64x128xbf16, #l2>) -> tensor<1x1x64x128xbf16, #l4> {
    %0 = "ttnn.add"(%arg0, %arg0) : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    %1 = "ttnn.reshape"(%0) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
    %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 1 : i32, 64 : i32, 128 : i32]}> : (tensor<1x64x128xbf16, #l3>) -> tensor<1x1x64x128xbf16, #l4>
    // CHECK-NOT: "ttnn.deallocate"
    "ttnn.deallocate"(%0) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x64x128xbf16, #l3>) -> ()
    return %2 : tensor<1x1x64x128xbf16, #l4>
  }

  // No aliasing: a single deallocate per buffer already frees with force = false
  // (refcount 1), so the pass leaves it untouched.
  // CHECK-LABEL: func.func @single
  func.func @single(%arg0: tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2> {
    %0 = "ttnn.add"(%arg0, %arg0) : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    %1 = "ttnn.add"(%0, %0) : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    // CHECK: "ttnn.deallocate"(%0) <{force = false}>
    "ttnn.deallocate"(%0) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    return %1 : tensor<64x128xbf16, #l2>
  }
}
