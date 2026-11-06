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
// A ttnn.case index: a single-element host-resident si32 tensor.
#index = #ttnn.ttnn_layout<() -> (0, 0), <1x1>, memref<1x1xsi32, #system_memory>>
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

  // A case branch borrows its captures the same way a while region borrows its
  // block arguments, so the exemption has to cover branch regions too - the
  // pass keys off block arguments of any region op, not off `ttnn.while`.
  // CHECK-LABEL: func.func @case_borrowed_capture
  func.func @case_borrowed_capture(%arg0: tensor<64x128xbf16, #l2>, %arg1: tensor<si32, #index>) -> tensor<64x128xbf16, #l2> {
    // CHECK: ttnn.case
    %0 = ttnn.case index(%arg1 : tensor<si32, #index>) captures(%arg0 : tensor<64x128xbf16, #l2>) branches {
    ^bb0(%cap: tensor<64x128xbf16, #l2>):
      %v1 = "ttnn.reshape"(%cap) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
      %v2 = "ttnn.reshape"(%cap) <{shape = [1 : i32, 1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x1x64x128xbf16, #l4>
      %out = "ttnn.add"(%cap, %cap) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
      // The views alias the capture, which the caller owns, so none of these
      // deallocations may be forced.
      // CHECK-NOT: force = true
      "ttnn.deallocate"(%v1) <{force = false}> : (tensor<1x64x128xbf16, #l3>) -> ()
      "ttnn.deallocate"(%v2) <{force = false}> : (tensor<1x1x64x128xbf16, #l4>) -> ()
      // CHECK: ttnn.yield
      ttnn.yield %out : tensor<64x128xbf16, #l2>
    }, {
    ^bb0(%cap: tensor<64x128xbf16, #l2>):
      // CHECK: ttnn.yield
      ttnn.yield %cap : tensor<64x128xbf16, #l2>
    } -> (tensor<64x128xbf16, #l2>)
    return %0 : tensor<64x128xbf16, #l2>
  }

  // A region that yields one of its block arguments unchanged makes the op's
  // result a second handle on that operand's buffer, which the runtime publishes
  // as such. The two therefore share a root, and here that root escapes through
  // the return, so neither deallocation may be forced. Here every branch
  // forwards %arg0, the second through views, so the result is %arg0's buffer
  // whichever branch runs.
  //
  // Without the aliasing, %arg0's deallocation would look like the last use of a
  // buffer of its own and get forced, freeing the buffer %0 still names.
  // CHECK-LABEL: func.func @case_forwards_capture
  func.func @case_forwards_capture(%arg0: tensor<64x128xbf16, #l2>, %arg1: tensor<si32, #index>) -> tensor<64x128xbf16, #l2> {
    // CHECK: ttnn.case
    %v = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
    %0 = ttnn.case index(%arg1 : tensor<si32, #index>) captures(%arg0 : tensor<64x128xbf16, #l2>) branches {
    ^bb0(%cap: tensor<64x128xbf16, #l2>):
      ttnn.yield %cap : tensor<64x128xbf16, #l2>
    }, {
    ^bb0(%cap: tensor<64x128xbf16, #l2>):
      %v1 = "ttnn.reshape"(%cap) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
      %v2 = "ttnn.reshape"(%v1) <{shape = [64 : i32, 128 : i32]}> : (tensor<1x64x128xbf16, #l3>) -> tensor<64x128xbf16, #l2>
      ttnn.yield %v2 : tensor<64x128xbf16, #l2>
    } -> (tensor<64x128xbf16, #l2>)
    // CHECK-NOT: "ttnn.deallocate"
    "ttnn.deallocate"(%v) <{force = false}> : (tensor<1x64x128xbf16, #l3>) -> ()
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    // CHECK: return
    return %0 : tensor<64x128xbf16, #l2>
  }

  // The same for a loop body that carries a value through untouched: result 0
  // aliases init %arg0, so %arg0's deallocation must not be forced while the
  // returned result still names that buffer.
  // CHECK-LABEL: func.func @while_forwards_carry
  func.func @while_forwards_carry(%arg0: tensor<64x128xbf16, #l2>, %arg1: tensor<ui32, #pred>) -> tensor<64x128xbf16, #l2> {
    // CHECK: ttnn.while
    %v = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
    %0 = ttnn.while inits(%arg0 : tensor<64x128xbf16, #l2>) captures(%arg1 : tensor<ui32, #pred>) {trip_count = 2 : i64} cond {
    ^bb0(%acc: tensor<64x128xbf16, #l2>, %p: tensor<ui32, #pred>):
      ttnn.yield %p : tensor<ui32, #pred>
    } do {
    ^bb0(%acc: tensor<64x128xbf16, #l2>, %p: tensor<ui32, #pred>):
      ttnn.yield %acc : tensor<64x128xbf16, #l2>
    } -> (tensor<64x128xbf16, #l2>)
    // CHECK-NOT: "ttnn.deallocate"
    "ttnn.deallocate"(%v) <{force = false}> : (tensor<1x64x128xbf16, #l3>) -> ()
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    // CHECK: return
    return %0 : tensor<64x128xbf16, #l2>
  }

  // Branches that forward *different* captures leave the result aliasing one of
  // them, but which is only known at runtime. The handles are grouped rather
  // than merged: they name distinct buffers, only one of which is shared.
  //
  // Here the result escapes through the return, so nothing may be forced - but
  // the deallocations must still be kept. Each one frees whichever buffer it
  // alone owns, and dropping them would leak the captures.
  //
  // Each capture is given a view so that its root has two deallocations: without
  // the grouping the bottom-most of each pair would be forced, freeing a buffer
  // the returned result may still name.
  // CHECK-LABEL: func.func @case_forwards_ambiguous
  func.func @case_forwards_ambiguous(%arg0: tensor<64x128xbf16, #l2>, %arg1: tensor<si32, #index>) -> tensor<64x128xbf16, #l2> {
    %a = "ttnn.add"(%arg0, %arg0) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    %b = "ttnn.add"(%a, %a) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    %va = "ttnn.reshape"(%a) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
    %vb = "ttnn.reshape"(%b) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
    // CHECK: ttnn.case
    %0 = ttnn.case index(%arg1 : tensor<si32, #index>) captures(%a, %b : tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) branches {
    ^bb0(%c0: tensor<64x128xbf16, #l2>, %c1: tensor<64x128xbf16, #l2>):
      ttnn.yield %c0 : tensor<64x128xbf16, #l2>
    }, {
    ^bb0(%c0: tensor<64x128xbf16, #l2>, %c1: tensor<64x128xbf16, #l2>):
      ttnn.yield %c1 : tensor<64x128xbf16, #l2>
    } -> (tensor<64x128xbf16, #l2>)
    // All four survive, none forced.
    // CHECK-NOT: force = true
    // CHECK-COUNT-4: "ttnn.deallocate"
    // CHECK-NOT: "ttnn.deallocate"
    "ttnn.deallocate"(%va) <{force = false}> : (tensor<1x64x128xbf16, #l3>) -> ()
    "ttnn.deallocate"(%vb) <{force = false}> : (tensor<1x64x128xbf16, #l3>) -> ()
    "ttnn.deallocate"(%a) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    "ttnn.deallocate"(%b) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    // CHECK: return
    return %0 : tensor<64x128xbf16, #l2>
  }

  // The same ambiguity, but nothing escapes: the result is consumed here and
  // only %r is returned. Every deallocation is kept, since each frees whichever
  // buffer it alone owns, and the bottom-most is forced to free the one that is
  // shared - whose refcount never drops to zero on its own. Dropping any of
  // them would hold a buffer to the end of the function.
  // CHECK-LABEL: func.func @case_forwards_ambiguous_local
  func.func @case_forwards_ambiguous_local(%arg0: tensor<64x128xbf16, #l2>, %arg1: tensor<si32, #index>) -> tensor<64x128xbf16, #l2> {
    %a = "ttnn.add"(%arg0, %arg0) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    %b = "ttnn.add"(%a, %a) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    // CHECK: ttnn.case
    %0 = ttnn.case index(%arg1 : tensor<si32, #index>) captures(%a, %b : tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) branches {
    ^bb0(%c0: tensor<64x128xbf16, #l2>, %c1: tensor<64x128xbf16, #l2>):
      ttnn.yield %c0 : tensor<64x128xbf16, #l2>
    }, {
    ^bb0(%c0: tensor<64x128xbf16, #l2>, %c1: tensor<64x128xbf16, #l2>):
      ttnn.yield %c1 : tensor<64x128xbf16, #l2>
    } -> (tensor<64x128xbf16, #l2>)
    %r = "ttnn.multiply"(%0, %0) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    // CHECK: "ttnn.deallocate"(%{{[0-9]+}}) <{force = false}>
    // CHECK: "ttnn.deallocate"(%{{[0-9]+}}) <{force = false}>
    // CHECK: "ttnn.deallocate"(%{{[0-9]+}}) <{force = true}>
    "ttnn.deallocate"(%a) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    "ttnn.deallocate"(%b) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    "ttnn.deallocate"(%0) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    // CHECK: return
    return %r : tensor<64x128xbf16, #l2>
  }

  // A branch forwards a capture however it hands it back: through views, as
  // the result of a nested while that carries it through, or as the result of
  // a nested case whose own branches forward different captures. One branch
  // does each here, so the result may alias any of %a to %d, and all five are
  // grouped: every deallocation is kept and only the bottom-most, the
  // result's, is forced.
  //
  // Each capture is reached one way only, and its root has a second
  // deallocation, its view's. A capture the pass fails to trace would get the
  // last of those, right after the case, forced while the result may still
  // name it.
  // CHECK-LABEL: func.func @case_yields_forwarded_capture
  func.func @case_yields_forwarded_capture(%arg0: tensor<64x128xbf16, #l2>, %arg1: tensor<si32, #index>, %arg2: tensor<ui32, #pred>) -> tensor<64x128xbf16, #l2> {
    // CHECK: %[[A:[0-9]+]] = "ttnn.add"(%arg0, %arg0)
    // CHECK: %[[B:[0-9]+]] = "ttnn.multiply"(%arg0, %arg0)
    // CHECK: %[[C:[0-9]+]] = "ttnn.subtract"(%arg0, %arg0)
    // CHECK: %[[D:[0-9]+]] = "ttnn.add"(%[[A]], %[[B]])
    // CHECK: %[[R:[0-9]+]] = ttnn.case
    %a = "ttnn.add"(%arg0, %arg0) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    %b = "ttnn.multiply"(%arg0, %arg0) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    %c = "ttnn.subtract"(%arg0, %arg0) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    %d = "ttnn.add"(%a, %b) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    %va = "ttnn.reshape"(%a) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
    %vb = "ttnn.reshape"(%b) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
    %vc = "ttnn.reshape"(%c) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
    %vd = "ttnn.reshape"(%d) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
    "ttnn.deallocate"(%va) <{force = false}> : (tensor<1x64x128xbf16, #l3>) -> ()
    "ttnn.deallocate"(%vb) <{force = false}> : (tensor<1x64x128xbf16, #l3>) -> ()
    "ttnn.deallocate"(%vc) <{force = false}> : (tensor<1x64x128xbf16, #l3>) -> ()
    "ttnn.deallocate"(%vd) <{force = false}> : (tensor<1x64x128xbf16, #l3>) -> ()
    %r = ttnn.case index(%arg1 : tensor<si32, #index>) captures(%a, %b, %c, %d, %arg1, %arg2 : tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>, tensor<si32, #index>, tensor<ui32, #pred>) branches {
    ^bb0(%ca: tensor<64x128xbf16, #l2>, %cb: tensor<64x128xbf16, #l2>, %cc: tensor<64x128xbf16, #l2>, %cd: tensor<64x128xbf16, #l2>, %i: tensor<si32, #index>, %p: tensor<ui32, #pred>):
      %v1 = "ttnn.reshape"(%ca) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
      %v2 = "ttnn.reshape"(%v1) <{shape = [64 : i32, 128 : i32]}> : (tensor<1x64x128xbf16, #l3>) -> tensor<64x128xbf16, #l2>
      ttnn.yield %v2 : tensor<64x128xbf16, #l2>
    }, {
    ^bb0(%ca: tensor<64x128xbf16, #l2>, %cb: tensor<64x128xbf16, #l2>, %cc: tensor<64x128xbf16, #l2>, %cd: tensor<64x128xbf16, #l2>, %i: tensor<si32, #index>, %p: tensor<ui32, #pred>):
      %w = ttnn.while inits(%cb : tensor<64x128xbf16, #l2>) captures(%p : tensor<ui32, #pred>) {trip_count = 2 : i64} cond {
      ^bb0(%acc: tensor<64x128xbf16, #l2>, %q: tensor<ui32, #pred>):
        ttnn.yield %q : tensor<ui32, #pred>
      } do {
      ^bb0(%acc: tensor<64x128xbf16, #l2>, %q: tensor<ui32, #pred>):
        ttnn.yield %acc : tensor<64x128xbf16, #l2>
      } -> (tensor<64x128xbf16, #l2>)
      ttnn.yield %w : tensor<64x128xbf16, #l2>
    }, {
    ^bb0(%ca: tensor<64x128xbf16, #l2>, %cb: tensor<64x128xbf16, #l2>, %cc: tensor<64x128xbf16, #l2>, %cd: tensor<64x128xbf16, #l2>, %i: tensor<si32, #index>, %p: tensor<ui32, #pred>):
      %n = ttnn.case index(%i : tensor<si32, #index>) captures(%cc, %cd : tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) branches {
      ^bb0(%e0: tensor<64x128xbf16, #l2>, %e1: tensor<64x128xbf16, #l2>):
        ttnn.yield %e0 : tensor<64x128xbf16, #l2>
      }, {
      ^bb0(%e0: tensor<64x128xbf16, #l2>, %e1: tensor<64x128xbf16, #l2>):
        ttnn.yield %e1 : tensor<64x128xbf16, #l2>
      } -> (tensor<64x128xbf16, #l2>)
      ttnn.yield %n : tensor<64x128xbf16, #l2>
    } -> (tensor<64x128xbf16, #l2>)
    // CHECK: "ttnn.deallocate"(%[[D]]) <{force = false}>
    // CHECK-NEXT: "ttnn.deallocate"(%[[C]]) <{force = false}>
    // CHECK-NEXT: "ttnn.deallocate"(%[[B]]) <{force = false}>
    // CHECK-NEXT: "ttnn.deallocate"(%[[A]]) <{force = false}>
    // CHECK-NEXT: "ttnn.multiply"(%[[R]], %[[R]])
    // CHECK-NEXT: "ttnn.deallocate"(%[[R]]) <{force = true}>
    // CHECK-NEXT: return
    "ttnn.deallocate"(%d) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    "ttnn.deallocate"(%c) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    "ttnn.deallocate"(%b) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    "ttnn.deallocate"(%a) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    %u = "ttnn.multiply"(%r, %r) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    "ttnn.deallocate"(%r) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    return %u : tensor<64x128xbf16, #l2>
  }

  // A while result can be an init even though the body computes its own
  // values: after an odd number of iterations of a body that swaps two slots,
  // result 0 is init 1, and a loop without a trip count may not run at all.
  // Each result is therefore grouped with the inits it may be: every
  // deallocation is kept and only the bottom-most of each group is forced.
  //
  // %y and %z each have a second deallocation, their view's. Without the
  // grouping, the last of those, right after the loops, would be forced while
  // a result may still name that buffer.
  // CHECK-LABEL: func.func @while_may_return_an_init
  func.func @while_may_return_an_init(%arg0: tensor<64x128xbf16, #l2>, %arg1: tensor<ui32, #pred>) -> tensor<64x128xbf16, #l2> {
    // CHECK: %[[X:[0-9]+]] = "ttnn.add"(%arg0, %arg0)
    // CHECK: %[[Y:[0-9]+]] = "ttnn.multiply"(%arg0, %arg0)
    // CHECK: %[[Z:[0-9]+]] = "ttnn.subtract"(%arg0, %arg0)
    // CHECK: %[[S:[0-9]+]]:2 = ttnn.while
    // CHECK: %[[T:[0-9]+]] = ttnn.while
    %x = "ttnn.add"(%arg0, %arg0) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    %y = "ttnn.multiply"(%arg0, %arg0) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    %z = "ttnn.subtract"(%arg0, %arg0) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    %vy = "ttnn.reshape"(%y) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
    %vz = "ttnn.reshape"(%z) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
    "ttnn.deallocate"(%vy) <{force = false}> : (tensor<1x64x128xbf16, #l3>) -> ()
    "ttnn.deallocate"(%vz) <{force = false}> : (tensor<1x64x128xbf16, #l3>) -> ()
    %s:2 = ttnn.while inits(%x, %y : tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) captures(%arg1 : tensor<ui32, #pred>) {trip_count = 1 : i64} cond {
    ^bb0(%a: tensor<64x128xbf16, #l2>, %b: tensor<64x128xbf16, #l2>, %p: tensor<ui32, #pred>):
      ttnn.yield %p : tensor<ui32, #pred>
    } do {
    ^bb0(%a: tensor<64x128xbf16, #l2>, %b: tensor<64x128xbf16, #l2>, %p: tensor<ui32, #pred>):
      ttnn.yield %b, %a : tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>
    } -> (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>)
    %t = ttnn.while inits(%z : tensor<64x128xbf16, #l2>) captures(%arg1 : tensor<ui32, #pred>) cond {
    ^bb0(%a: tensor<64x128xbf16, #l2>, %p: tensor<ui32, #pred>):
      ttnn.yield %p : tensor<ui32, #pred>
    } do {
    ^bb0(%a: tensor<64x128xbf16, #l2>, %p: tensor<ui32, #pred>):
      %n = "ttnn.add"(%a, %a) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
      ttnn.yield %n : tensor<64x128xbf16, #l2>
    } -> (tensor<64x128xbf16, #l2>)
    // CHECK: "ttnn.deallocate"(%[[Z]]) <{force = false}>
    // CHECK-NEXT: "ttnn.deallocate"(%[[Y]]) <{force = false}>
    // CHECK-NEXT: "ttnn.deallocate"(%[[X]]) <{force = false}>
    // CHECK-NEXT: %[[U:[0-9]+]] = "ttnn.add"(%[[S]]#0, %[[S]]#1)
    // CHECK-NEXT: "ttnn.deallocate"(%[[S]]#1) <{force = false}>
    // CHECK-NEXT: "ttnn.deallocate"(%[[S]]#0) <{force = true}>
    // CHECK-NEXT: "ttnn.multiply"(%[[U]], %[[T]])
    // CHECK-NEXT: "ttnn.deallocate"(%[[U]]) <{force = false}>
    // CHECK-NEXT: "ttnn.deallocate"(%[[T]]) <{force = true}>
    // CHECK-NEXT: return
    "ttnn.deallocate"(%z) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    "ttnn.deallocate"(%y) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    "ttnn.deallocate"(%x) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    %u = "ttnn.add"(%s#0, %s#1) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    "ttnn.deallocate"(%s#1) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    "ttnn.deallocate"(%s#0) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    %w = "ttnn.multiply"(%u, %t) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    "ttnn.deallocate"(%u) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    "ttnn.deallocate"(%t) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    return %w : tensor<64x128xbf16, #l2>
  }

  // A branch that yields one value twice makes the two results one buffer when
  // it runs, and two when the other branch does, so the results are grouped:
  // both deallocations are kept, and the bottom-most is forced to free the
  // shared one, whose refcount never drops to zero on its own.
  // CHECK-LABEL: func.func @case_yields_twice
  func.func @case_yields_twice(%arg0: tensor<64x128xbf16, #l2>, %arg1: tensor<si32, #index>) -> tensor<64x128xbf16, #l2> {
    // CHECK: %[[R:[0-9]+]]:2 = ttnn.case
    %r:2 = ttnn.case index(%arg1 : tensor<si32, #index>) captures(%arg0 : tensor<64x128xbf16, #l2>) branches {
    ^bb0(%c: tensor<64x128xbf16, #l2>):
      %a = "ttnn.add"(%c, %c) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
      ttnn.yield %a, %a : tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>
    }, {
    ^bb0(%c: tensor<64x128xbf16, #l2>):
      %m = "ttnn.multiply"(%c, %c) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
      %s = "ttnn.subtract"(%c, %c) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
      ttnn.yield %m, %s : tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>
    } -> (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>)
    // CHECK: "ttnn.deallocate"(%[[R]]#0) <{force = false}>
    // CHECK: "ttnn.deallocate"(%[[R]]#1) <{force = true}>
    // CHECK-NEXT: return
    %e = "ttnn.add"(%r#0, %r#0) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    "ttnn.deallocate"(%r#0) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    %t = "ttnn.add"(%e, %r#1) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    "ttnn.deallocate"(%e) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    "ttnn.deallocate"(%r#1) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    return %t : tensor<64x128xbf16, #l2>
  }

  // A parameter stays with the caller for the next call, and a const-eval
  // result with the cache, so a handle that may be one is never force-freed.
  // Each result here is grouped with what it may forward, a parameter and a
  // cached value, and without the exemption the group's bottom-most
  // deallocation would be forced.
  // CHECK-LABEL: func.func @case_forwards_outliving
  func.func @case_forwards_outliving(%arg0: tensor<64x128xbf16, #l2> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg1: tensor<si32, #index>) -> tensor<64x128xbf16, #l2> {
    %cached = ttcore.load_cached(@outliving_const_eval_0, []) : () -> tensor<64x128xbf16, #l2>
    %r:2 = ttnn.case index(%arg1 : tensor<si32, #index>) captures(%arg0, %cached : tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) branches {
    ^bb0(%p: tensor<64x128xbf16, #l2>, %c: tensor<64x128xbf16, #l2>):
      ttnn.yield %p, %c : tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>
    }, {
    ^bb0(%p: tensor<64x128xbf16, #l2>, %c: tensor<64x128xbf16, #l2>):
      %a = "ttnn.add"(%p, %c) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
      %m = "ttnn.multiply"(%p, %c) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
      ttnn.yield %a, %m : tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>
    } -> (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>)
    // CHECK-NOT: force = true
    // CHECK: return
    %u = "ttnn.multiply"(%r#0, %r#1) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    "ttnn.deallocate"(%r#1) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    "ttnn.deallocate"(%r#0) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    return %u : tensor<64x128xbf16, #l2>
  }

  func.func private @outliving_const_eval_0() -> tensor<64x128xbf16, #l2> attributes {tt.function_type = "const_eval"} {
    %0 = "ttnn.ones"() <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, shape = #ttnn.shape<64x128>}> : () -> tensor<64x128xbf16, #l2>
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
