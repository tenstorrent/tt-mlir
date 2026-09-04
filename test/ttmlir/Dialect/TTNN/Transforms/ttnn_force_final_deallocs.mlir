// RUN: ttmlir-opt --ttnn-force-final-deallocs -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Test for the --ttnn-force-final-deallocs pass.
// A view-eligible ttnn.reshape aliases its input's buffer, so the input and the
// reshape result get separate ttnn.deallocate ops that both target one buffer.
// The pass forces the last deallocation (bottom-most in program order) of each
// such buffer so the memory is actually freed. Other deallocations of that buffer
// are no-ops and are removed. Buffers freed elsewhere are never forced and all of
// their no-op deallocations are removed: returned values freed by the caller,
// values yielded out of a region, buffers a region borrows through its block
// arguments, and conv activations the conv op force-deallocates itself.

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#system_memory = #ttnn.buffer_type<system_memory>
#l2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#l3 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#l4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 64 + d2, d3), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Conv activation (L1) and a view of it, plus weight/bias/output layouts.
#conv_in = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 852800 + d1 * 852800 + d2, d3), <1x1>, memref<26650x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#conv_in_view = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 852800 + d1, d2), <1x1>, memref<26650x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#weight = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 21 + d1 * 7 + d2, d3), <1x1>, memref<1344x7xbf16, #system_memory>>
// A ttnn.while condition result: a single-element host-resident ui32 tensor.
#pred = #ttnn.ttnn_layout<() -> (0, 0), <1x1>, memref<1x1xui32, #system_memory>>
#bias = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x64xbf16, #system_memory>>
#conv_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 213216 + d1 * 213216 + d2, d3), <1x1>, memref<6663x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  // %0 and its view %1 share one buffer; the last deallocate (%1's) is forced,
  // the earlier one (%0's) is a redundant no-op and is removed.
  // CHECK-LABEL: func.func @aliased
  func.func @aliased(%arg0: tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3> {
    %0 = "ttnn.add"(%arg0, %arg0) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    %1 = "ttnn.reshape"(%0) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
    %2 = "ttnn.add"(%1, %1) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<1x64x128xbf16, #l3>, tensor<1x64x128xbf16, #l3>) -> tensor<1x64x128xbf16, #l3>
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
    %0 = "ttnn.add"(%arg0, %arg0) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    %1 = "ttnn.reshape"(%0) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
    %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 1 : i32, 64 : i32, 128 : i32]}> : (tensor<1x64x128xbf16, #l3>) -> tensor<1x1x64x128xbf16, #l4>
    // CHECK-NOT: "ttnn.deallocate"
    "ttnn.deallocate"(%0) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x64x128xbf16, #l3>) -> ()
    return %2 : tensor<1x1x64x128xbf16, #l4>
  }

  // The conv2d has deallocate_activation=true and an L1 input, so the conv frees
  // that buffer itself. All of its (no-op) deallocates are removed.
  // CHECK-LABEL: func.func @conv_activation
  func.func @conv_activation(%arg0: tensor<1x1x852800x3xbf16, #conv_in>, %arg1: tensor<64x3x7x7xbf16, #weight>, %arg2: tensor<1x1x1x64xbf16, #bias>) -> tensor<1x1x213200x64xbf16, #conv_out> {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %view = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 852800 : i32, 3 : i32]}> : (tensor<1x1x852800x3xbf16, #conv_in>) -> tensor<1x852800x3xbf16, #conv_in_view>
    // CHECK: "ttnn.conv2d"
    %result = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0) <{batch_size = 1 : i32, conv2d_config = #ttnn.conv2d_config<weights_dtype = bf16, deallocate_activation = true, enable_kernel_stride_folding = false>, dilation = array<i32: 1, 1>, dtype = #ttcore.supportedDataTypes<bf16>, groups = 1 : i32, in_channels = 3 : i32, input_height = 800 : i32, input_width = 1066 : i32, kernel_size = array<i32: 7, 7>, out_channels = 64 : i32, padding = array<i32: 3, 3, 3, 3>, stride = array<i32: 2, 2>}> : (tensor<1x1x852800x3xbf16, #conv_in>, tensor<64x3x7x7xbf16, #weight>, tensor<1x1x1x64xbf16, #bias>, !ttnn.device) -> tensor<1x1x213200x64xbf16, #conv_out>
    // CHECK-NOT: "ttnn.deallocate"
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x1x852800x3xbf16, #conv_in>) -> ()
    "ttnn.deallocate"(%view) <{force = false}> : (tensor<1x852800x3xbf16, #conv_in_view>) -> ()
    return %result : tensor<1x1x213200x64xbf16, #conv_out>
  }

  // A ttnn.while body only borrows the buffers behind its block arguments: they
  // belong to the enclosing program on the first iteration and to the previous
  // iteration afterwards. Forcing a deallocation of one from inside the region
  // would free it while the loop still needs it, so both aliasing deallocations
  // of the carried value are removed and neither is forced.
  // CHECK-LABEL: func.func @while_borrowed_carry
  func.func @while_borrowed_carry(%arg0: tensor<64x128xbf16, #l2>, %arg1: tensor<ui32, #pred>) -> tensor<64x128xbf16, #l2> {
    // CHECK: ttnn.while
    %0 = ttnn.while inits(%arg0 : tensor<64x128xbf16, #l2>) captures(%arg1 : tensor<ui32, #pred>) {trip_count = 2 : i64} cond {
    ^bb0(%acc: tensor<64x128xbf16, #l2>, %p: tensor<ui32, #pred>):
      ttnn.yield %p : tensor<ui32, #pred>
    } do {
    ^bb0(%acc: tensor<64x128xbf16, #l2>, %p: tensor<ui32, #pred>):
      %v1 = "ttnn.reshape"(%acc) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
      %v2 = "ttnn.reshape"(%acc) <{shape = [1 : i32, 1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x1x64x128xbf16, #l4>
      %s1 = "ttnn.add"(%v1, %v1) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<1x64x128xbf16, #l3>, tensor<1x64x128xbf16, #l3>) -> tensor<1x64x128xbf16, #l3>
      %s2 = "ttnn.add"(%v2, %v2) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<1x1x64x128xbf16, #l4>, tensor<1x1x64x128xbf16, #l4>) -> tensor<1x1x64x128xbf16, #l4>
      // Both of these resolve to root %acc, a borrowed block argument.
      // CHECK-NOT: "ttnn.deallocate"
      "ttnn.deallocate"(%v1) <{force = false}> : (tensor<1x64x128xbf16, #l3>) -> ()
      "ttnn.deallocate"(%v2) <{force = false}> : (tensor<1x1x64x128xbf16, #l4>) -> ()
      %o1 = "ttnn.reshape"(%s1) <{shape = [64 : i32, 128 : i32]}> : (tensor<1x64x128xbf16, #l3>) -> tensor<64x128xbf16, #l2>
      %o2 = "ttnn.reshape"(%s2) <{shape = [64 : i32, 128 : i32]}> : (tensor<1x1x64x128xbf16, #l4>) -> tensor<64x128xbf16, #l2>
      %o = "ttnn.add"(%o1, %o2) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
      // CHECK: ttnn.yield
      ttnn.yield %o : tensor<64x128xbf16, #l2>
    } -> (tensor<64x128xbf16, #l2>)
    return %0 : tensor<64x128xbf16, #l2>
  }

  // A value yielded out of a region escapes it exactly as a returned value
  // escapes the function: it becomes the next iteration's carried value, so the
  // region must not free it. Views of %t are deallocated, but %t is yielded, so
  // nothing is forced.
  // CHECK-LABEL: func.func @while_yielded_escapes
  func.func @while_yielded_escapes(%arg0: tensor<64x128xbf16, #l2>, %arg1: tensor<ui32, #pred>) -> tensor<64x128xbf16, #l2> {
    // CHECK: ttnn.while
    %0 = ttnn.while inits(%arg0 : tensor<64x128xbf16, #l2>) captures(%arg1 : tensor<ui32, #pred>) {trip_count = 2 : i64} cond {
    ^bb0(%acc: tensor<64x128xbf16, #l2>, %p: tensor<ui32, #pred>):
      ttnn.yield %p : tensor<ui32, #pred>
    } do {
    ^bb0(%acc: tensor<64x128xbf16, #l2>, %p: tensor<ui32, #pred>):
      %t = "ttnn.add"(%acc, %acc) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
      %w1 = "ttnn.reshape"(%t) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
      %w2 = "ttnn.reshape"(%t) <{shape = [1 : i32, 1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x1x64x128xbf16, #l4>
      // CHECK-NOT: "ttnn.deallocate"
      "ttnn.deallocate"(%w1) <{force = false}> : (tensor<1x64x128xbf16, #l3>) -> ()
      "ttnn.deallocate"(%w2) <{force = false}> : (tensor<1x1x64x128xbf16, #l4>) -> ()
      // CHECK: ttnn.yield
      ttnn.yield %t : tensor<64x128xbf16, #l2>
    } -> (tensor<64x128xbf16, #l2>)
    return %0 : tensor<64x128xbf16, #l2>
  }

  // A buffer the body allocates itself and does not yield is owned by the body:
  // it is reallocated every iteration, so its last aliasing deallocation still
  // gets forced. Region ops do not disable forcing wholesale.
  // CHECK-LABEL: func.func @while_body_owned
  func.func @while_body_owned(%arg0: tensor<64x128xbf16, #l2>, %arg1: tensor<ui32, #pred>) -> tensor<64x128xbf16, #l2> {
    // CHECK: ttnn.while
    %0 = ttnn.while inits(%arg0 : tensor<64x128xbf16, #l2>) captures(%arg1 : tensor<ui32, #pred>) {trip_count = 2 : i64} cond {
    ^bb0(%acc: tensor<64x128xbf16, #l2>, %p: tensor<ui32, #pred>):
      ttnn.yield %p : tensor<ui32, #pred>
    } do {
    ^bb0(%acc: tensor<64x128xbf16, #l2>, %p: tensor<ui32, #pred>):
      %t = "ttnn.add"(%acc, %acc) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
      %w1 = "ttnn.reshape"(%t) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
      %u = "ttnn.add"(%w1, %w1) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<1x64x128xbf16, #l3>, tensor<1x64x128xbf16, #l3>) -> tensor<1x64x128xbf16, #l3>
      %o = "ttnn.reshape"(%u) <{shape = [64 : i32, 128 : i32]}> : (tensor<1x64x128xbf16, #l3>) -> tensor<64x128xbf16, #l2>
      %out = "ttnn.add"(%o, %o) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
      // %t is body-local and not yielded, so its final deallocation is forced.
      // CHECK-NOT: "ttnn.deallocate"
      // CHECK: "ttnn.deallocate"(%{{[0-9]+}}) <{force = true}>
      "ttnn.deallocate"(%t) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
      "ttnn.deallocate"(%w1) <{force = false}> : (tensor<1x64x128xbf16, #l3>) -> ()
      // CHECK: ttnn.yield
      ttnn.yield %out : tensor<64x128xbf16, #l2>
    } -> (tensor<64x128xbf16, #l2>)
    return %0 : tensor<64x128xbf16, #l2>
  }

  // No aliasing: a single deallocate per buffer already frees with force = false
  // (refcount 1), so the pass leaves it untouched.
  // CHECK-LABEL: func.func @single
  func.func @single(%arg0: tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2> {
    %0 = "ttnn.add"(%arg0, %arg0) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    %1 = "ttnn.add"(%0, %0) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    // CHECK: "ttnn.deallocate"(%0) <{force = false}>
    "ttnn.deallocate"(%0) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    return %1 : tensor<64x128xbf16, #l2>
  }
}
