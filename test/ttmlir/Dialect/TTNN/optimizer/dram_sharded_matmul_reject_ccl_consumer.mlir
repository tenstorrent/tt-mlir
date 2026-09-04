// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8 mock-system-desc-arch=blackhole" -o %t %s
// RUN: FileCheck %s --input-file=%t

// The DS path declines when the matmul result feeds a collective.
//
// No CCL implements the op-model interface -- they all carry OpModelExempt
// because tt-metal exposes no constraint query for them (tt-mlir#4392). The
// optimizer can therefore cost the matmul but not the collective behind it, so
// a DRAM-sharded output is chosen on the matmul's own merits and the layout
// mismatch is left to an inserted ToMemoryConfigOp sitting on the collective's
// critical path.
//
// The shape is llama_3_1_70b's row-parallel down projection under TP=4:
// K = 7168 is 224 tiles, so K-per-core is 224/8 = 28 and the fitted
// in0_block_w is 7 -- comfortably above kMinBlockWidth, i.e. this op is
// declined for its consumer, not for its geometry. Measured on qb2 the reshard
// costs ~71 us per op across the 80 layers.
//
// As with the other declines, the matmul must still get *some* program config;
// falling back to the 1D/2D mcast configs is the point.

module attributes {} {
  // CHECK-LABEL: func.func @ds_matmul_feeds_all_reduce
  // CHECK: "ttnn.matmul"
  // CHECK-NOT: dram_sharded_program_config
  func.func @ds_matmul_feeds_all_reduce(
      %act: tensor<32x7168xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %weight: tensor<7168x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x8192xbf16> {
    %0 = "ttir.matmul"(%act, %weight) : (tensor<32x7168xbf16>, tensor<7168x8192xbf16>) -> tensor<32x8192xbf16>
    %1 = "ttir.all_reduce"(%0) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<32x8192xbf16>) -> tensor<32x8192xbf16>
    return %1 : tensor<32x8192xbf16>
  }
}
