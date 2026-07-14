// RUN: ttmlir-opt --ttcore-register-device --ttnn-allocate-distributed-op-semaphores -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Host-side compiler test for the TTNN-only `all_gather_minimal_matmul_async`
// op. There is no TTIR op / frontend path yet (see issue #8984, "we probably
// need just the TTNN op"), so the op is hand-authored at the TTNN level. This
// exercises three compiler stages with no silicon:
//   1. the op verifier (runs on parse),
//   2. the `--ttnn-allocate-distributed-op-semaphores` pass, which materializes
//      the two all-gather semaphores + barrier semaphore via the
//      DistributedOpInterface when they are left unbound, and
//   3. flatbuffer serialization (`ttmlir-translate --ttnn-to-flatbuffer`).

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

// Activation A[M=32, K=128] sharded along K across a 1x4 worker grid; the
// semaphore pass derives the semaphore core range from this layout.
#ttnn_layout_in = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x4>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (3,0)>]>>
// Weight B[K=128, N=64].
#ttnn_layout_w = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Output [M=32, N=64].
#ttnn_layout_out = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Row-broadcast operands [*, N=64].
#ttnn_layout_row = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module @test_all_gather_minimal_matmul_async attributes {} {
  // Minimal form: just activation + weight, semaphores left unbound.
  // CHECK-LABEL: func.func @minimal
  func.func @minimal(
      %input: tensor<32x128xbf16, #ttnn_layout_in>,
      %weight: tensor<128x64xbf16, #ttnn_layout_w>)
      -> tensor<32x64xbf16, #ttnn_layout_out> attributes {tt.function_type = "forward_device"} {
    // CHECK: "ttnn.get_device"
    // The pass allocates two all-gather semaphores plus one barrier semaphore.
    // CHECK-COUNT-3: "ttnn.create_global_semaphore"
    // CHECK: "ttnn.all_gather_minimal_matmul_async"
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 2, 1, 1>
    %device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %0 = "ttnn.all_gather_minimal_matmul_async"(%input, %weight, %device) <{
      cluster_axis = 1 : ui32,
      operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 1>
    }> : (tensor<32x128xbf16, #ttnn_layout_in>, tensor<128x64xbf16, #ttnn_layout_w>, !ttnn.device) -> tensor<32x64xbf16, #ttnn_layout_out>
    return %0 : tensor<32x64xbf16, #ttnn_layout_out>
  }

  // Fused form: bias + gated residual (addcmul) epilogue, exercising every
  // optional tensor operand through the verifier and flatbuffer serializer.
  // CHECK-LABEL: func.func @fused
  func.func @fused(
      %input: tensor<32x128xbf16, #ttnn_layout_in>,
      %weight: tensor<128x64xbf16, #ttnn_layout_w>,
      %bias: tensor<1x64xbf16, #ttnn_layout_row>,
      %res: tensor<32x64xbf16, #ttnn_layout_out>,
      %gate: tensor<1x64xbf16, #ttnn_layout_row>)
      -> tensor<32x64xbf16, #ttnn_layout_out> attributes {tt.function_type = "forward_device"} {
    // CHECK-COUNT-3: "ttnn.create_global_semaphore"
    // CHECK: "ttnn.all_gather_minimal_matmul_async"
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 2, 1, 1>
    %device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %0 = "ttnn.all_gather_minimal_matmul_async"(%input, %weight, %bias, %res, %gate, %device) <{
      cluster_axis = 1 : ui32,
      scalar = 5.000000e-01 : f32,
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 1>
    }> : (tensor<32x128xbf16, #ttnn_layout_in>, tensor<128x64xbf16, #ttnn_layout_w>, tensor<1x64xbf16, #ttnn_layout_row>, tensor<32x64xbf16, #ttnn_layout_out>, tensor<1x64xbf16, #ttnn_layout_row>, !ttnn.device) -> tensor<32x64xbf16, #ttnn_layout_out>
    return %0 : tensor<32x64xbf16, #ttnn_layout_out>
  }
}
