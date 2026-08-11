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
      %s1 = "ttnn.add"(%v1, %v1) : (tensor<1x64x128xbf16, #l3>, tensor<1x64x128xbf16, #l3>) -> tensor<1x64x128xbf16, #l3>
      %s2 = "ttnn.add"(%v2, %v2) : (tensor<1x1x64x128xbf16, #l4>, tensor<1x1x64x128xbf16, #l4>) -> tensor<1x1x64x128xbf16, #l4>
      // Both of these resolve to root %acc, a borrowed block argument.
      // CHECK-NOT: "ttnn.deallocate"
      "ttnn.deallocate"(%v1) <{force = false}> : (tensor<1x64x128xbf16, #l3>) -> ()
      "ttnn.deallocate"(%v2) <{force = false}> : (tensor<1x1x64x128xbf16, #l4>) -> ()
      %o1 = "ttnn.reshape"(%s1) <{shape = [64 : i32, 128 : i32]}> : (tensor<1x64x128xbf16, #l3>) -> tensor<64x128xbf16, #l2>
      %o2 = "ttnn.reshape"(%s2) <{shape = [64 : i32, 128 : i32]}> : (tensor<1x1x64x128xbf16, #l4>) -> tensor<64x128xbf16, #l2>
      %o = "ttnn.add"(%o1, %o2) : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
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
      %t = "ttnn.add"(%acc, %acc) : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
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
      %t = "ttnn.add"(%acc, %acc) : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
      %w1 = "ttnn.reshape"(%t) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16, #l2>) -> tensor<1x64x128xbf16, #l3>
      %u = "ttnn.add"(%w1, %w1) : (tensor<1x64x128xbf16, #l3>, tensor<1x64x128xbf16, #l3>) -> tensor<1x64x128xbf16, #l3>
      %o = "ttnn.reshape"(%u) <{shape = [64 : i32, 128 : i32]}> : (tensor<1x64x128xbf16, #l3>) -> tensor<64x128xbf16, #l2>
      %out = "ttnn.add"(%o, %o) : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
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
      %out = "ttnn.add"(%cap, %cap) : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
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
  // the return, so neither deallocation may be forced.
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
      %a = "ttnn.add"(%cap, %cap) : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
      ttnn.yield %a : tensor<64x128xbf16, #l2>
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
    %a = "ttnn.add"(%arg0, %arg0) : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    %b = "ttnn.add"(%a, %a) : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
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
    %a = "ttnn.add"(%arg0, %arg0) : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    %b = "ttnn.add"(%a, %a) : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    // CHECK: ttnn.case
    %0 = ttnn.case index(%arg1 : tensor<si32, #index>) captures(%a, %b : tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) branches {
    ^bb0(%c0: tensor<64x128xbf16, #l2>, %c1: tensor<64x128xbf16, #l2>):
      ttnn.yield %c0 : tensor<64x128xbf16, #l2>
    }, {
    ^bb0(%c0: tensor<64x128xbf16, #l2>, %c1: tensor<64x128xbf16, #l2>):
      ttnn.yield %c1 : tensor<64x128xbf16, #l2>
    } -> (tensor<64x128xbf16, #l2>)
    %r = "ttnn.multiply"(%0, %0) : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
    // CHECK: "ttnn.deallocate"(%{{[0-9]+}}) <{force = false}>
    // CHECK: "ttnn.deallocate"(%{{[0-9]+}}) <{force = false}>
    // CHECK: "ttnn.deallocate"(%{{[0-9]+}}) <{force = true}>
    "ttnn.deallocate"(%a) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    "ttnn.deallocate"(%b) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    "ttnn.deallocate"(%0) <{force = false}> : (tensor<64x128xbf16, #l2>) -> ()
    // CHECK: return
    return %r : tensor<64x128xbf16, #l2>
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
