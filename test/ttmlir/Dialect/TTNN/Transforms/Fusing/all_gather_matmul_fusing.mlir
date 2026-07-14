// RUN: ttmlir-opt --ttcore-register-device --ttnn-fusing="enable-all-gather-matmul-fusion=true" %s | FileCheck %s

// Fuses an all_gather feeding a matmul/linear into a single
// ttnn.all_gather_minimal_matmul_async op. The gathered activation is never
// materialized as a standalone tensor; the collective and the matmul run
// together. The fused op's semaphores are left unbound here (materialized later
// by TTNNAllocateDistributedOpSemaphores).
//
// A single top-level module is used (not nested modules) so the device op
// registered by --ttcore-register-device is visible to the fusion pattern,
// which synthesizes the get_device the fused op needs.

#dram = #ttnn.buffer_type<dram>
// Sharded activation A[M=32, K_local=128] (gathered to K=512 across 4 devices).
#a  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Gathered activation A[M=32, K=512].
#ag = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Weight B[K=512, N=64].
#w  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<16x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Weight transposed B[N=64, K=512] for the transpose negative case.
#wt = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Output [M=32, N=64].
#o  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Bias [1, N=64].
#b  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// matmul(all_gather(x), W) -> fused. The all_gather is consumed and folded in.
// CHECK-LABEL: func.func @all_gather_matmul
func.func @all_gather_matmul(%x: tensor<32x128xbf16, #a>, %w: tensor<512x64xbf16, #w>)
    -> tensor<32x64xbf16, #o> {
  // CHECK: "ttnn.all_gather_minimal_matmul_async"
  // CHECK-NOT: "ttnn.all_gather"
  // CHECK-NOT: "ttnn.matmul"
  %0 = "ttnn.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16, #a>) -> tensor<32x512xbf16, #ag>
  %1 = "ttnn.matmul"(%0, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16, #ag>, tensor<512x64xbf16, #w>) -> tensor<32x64xbf16, #o>
  return %1 : tensor<32x64xbf16, #o>
}

// linear(all_gather(x), W, bias) -> fused, bias carried through.
// CHECK-LABEL: func.func @all_gather_linear
func.func @all_gather_linear(%x: tensor<32x128xbf16, #a>, %w: tensor<512x64xbf16, #w>, %bias: tensor<1x64xbf16, #b>)
    -> tensor<32x64xbf16, #o> {
  // CHECK: "ttnn.all_gather_minimal_matmul_async"
  // CHECK-NOT: "ttnn.all_gather"
  // CHECK-NOT: "ttnn.linear"
  %0 = "ttnn.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16, #a>) -> tensor<32x512xbf16, #ag>
  %1 = "ttnn.linear"(%0, %w, %bias) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16, #ag>, tensor<512x64xbf16, #w>, tensor<1x64xbf16, #b>) -> tensor<32x64xbf16, #o>
  return %1 : tensor<32x64xbf16, #o>
}

// transpose_b set -> not fused (fused op does not model transpose).
// CHECK-LABEL: func.func @no_fuse_transpose
func.func @no_fuse_transpose(%x: tensor<32x128xbf16, #a>, %w: tensor<64x512xbf16, #wt>)
    -> tensor<32x64xbf16, #o> {
  // CHECK: "ttnn.all_gather"
  // CHECK: "ttnn.matmul"
  // CHECK-NOT: all_gather_minimal_matmul_async
  %0 = "ttnn.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16, #a>) -> tensor<32x512xbf16, #ag>
  %1 = "ttnn.matmul"(%0, %w) <{transpose_a = false, transpose_b = true}> : (tensor<32x512xbf16, #ag>, tensor<64x512xbf16, #wt>) -> tensor<32x64xbf16, #o>
  return %1 : tensor<32x64xbf16, #o>
}

// all_gather result has a second use -> not fused (would duplicate the gather).
// CHECK-LABEL: func.func @no_fuse_multiuse
func.func @no_fuse_multiuse(%x: tensor<32x128xbf16, #a>, %w: tensor<512x64xbf16, #w>)
    -> (tensor<32x64xbf16, #o>, tensor<32x512xbf16, #ag>) {
  // CHECK-NOT: all_gather_minimal_matmul_async
  %0 = "ttnn.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16, #a>) -> tensor<32x512xbf16, #ag>
  %1 = "ttnn.matmul"(%0, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16, #ag>, tensor<512x64xbf16, #w>) -> tensor<32x64xbf16, #o>
  return %1, %0 : tensor<32x64xbf16, #o>, tensor<32x512xbf16, #ag>
}

// Gated-residual epilogue: residual + gate * matmul(all_gather(x), W) folds the
// whole thing (gather + matmul + multiply + add) into one op, with
// residual/gate mapped to the addcmul operands. Mirrors tt-metal's addcmul
// epilogue on the same kernel.
// CHECK-LABEL: func.func @all_gather_matmul_addcmul
func.func @all_gather_matmul_addcmul(%x: tensor<32x128xbf16, #a>, %w: tensor<512x64xbf16, #w>,
                                     %gate: tensor<32x64xbf16, #o>, %res: tensor<32x64xbf16, #o>)
    -> tensor<32x64xbf16, #o> {
  // CHECK: "ttnn.all_gather_minimal_matmul_async"
  // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0, 1>
  // CHECK-SAME: scalar = 1.000000e+00 : f32
  // CHECK-NOT: "ttnn.all_gather"
  // CHECK-NOT: "ttnn.matmul"
  // CHECK-NOT: "ttnn.multiply"
  // CHECK-NOT: "ttnn.add"
  %0 = "ttnn.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16, #a>) -> tensor<32x512xbf16, #ag>
  %1 = "ttnn.matmul"(%0, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16, #ag>, tensor<512x64xbf16, #w>) -> tensor<32x64xbf16, #o>
  %2 = "ttnn.multiply"(%1, %gate) : (tensor<32x64xbf16, #o>, tensor<32x64xbf16, #o>) -> tensor<32x64xbf16, #o>
  %3 = "ttnn.add"(%res, %2) : (tensor<32x64xbf16, #o>, tensor<32x64xbf16, #o>) -> tensor<32x64xbf16, #o>
  return %3 : tensor<32x64xbf16, #o>
}

// linear (with bias) + gated-residual epilogue, multiply operands reversed:
// exercises the linear path and both commutative branches.
// CHECK-LABEL: func.func @all_gather_linear_addcmul
func.func @all_gather_linear_addcmul(%x: tensor<32x128xbf16, #a>, %w: tensor<512x64xbf16, #w>, %bias: tensor<1x64xbf16, #b>,
                                     %gate: tensor<32x64xbf16, #o>, %res: tensor<32x64xbf16, #o>)
    -> tensor<32x64xbf16, #o> {
  // CHECK: "ttnn.all_gather_minimal_matmul_async"
  // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 1>
  // CHECK-NOT: "ttnn.linear"
  // CHECK-NOT: "ttnn.multiply"
  %0 = "ttnn.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16, #a>) -> tensor<32x512xbf16, #ag>
  %1 = "ttnn.linear"(%0, %w, %bias) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16, #ag>, tensor<512x64xbf16, #w>, tensor<1x64xbf16, #b>) -> tensor<32x64xbf16, #o>
  %2 = "ttnn.multiply"(%gate, %1) : (tensor<32x64xbf16, #o>, tensor<32x64xbf16, #o>) -> tensor<32x64xbf16, #o>
  %3 = "ttnn.add"(%2, %res) : (tensor<32x64xbf16, #o>, tensor<32x64xbf16, #o>) -> tensor<32x64xbf16, #o>
  return %3 : tensor<32x64xbf16, #o>
}
