// RUN: ttmlir-opt --ttnn-force-final-deallocs -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Test for the --ttnn-force-final-deallocs pass.
// A view-eligible ttnn.reshape aliases its input's buffer, so the input and the
// reshape result get separate ttnn.deallocate ops that both target one buffer.
// The pass forces the last deallocation (bottom-most in program order) of each
// such buffer so the memory is actually freed. Other deallocations of that buffer
// are no-ops and are removed. Buffers freed elsewhere (returned values that are
// freed by the caller or conv activations the conv op force-deallocates itself)
// are never forced and all of their no-op deallocations are removed.

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
