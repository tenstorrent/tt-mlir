// REQUIRES: opmodel
// RUN: ttmlir-opt --ttnn-fusing="enable-op-constraints=true" %s | FileCheck %s

// Regression test: the QKV weight reorder must not leak when fused-op
// validation fails.
//
// SplitQueryKeyValueAndSplitHeadsFusing brings the QKV projection weight into
// [Q|K|V] column order only after the fused op is validated. Validation can
// return failure (here it fails deterministically because the module has no
// ttcore.system_desc, so IsolatedIRValidationWrapper reports a precondition
// failure), and the greedy rewriter does not roll back ops a pattern already
// *created*. Reordering before validation would therefore leave a reordered
// weight (extra slice_static + concat on the const-eval'd LoadCachedOp, rebound
// into the matmul) while the downstream unfused slicing still assumed the
// original [K|Q|V] layout — silently reading the wrong K/V columns and
// producing garbage generation.
//
// The weight here is GQA in [K|Q|V] order (K: 2 heads, Q: 8 heads, V: 2 heads),
// which forces the reorder path. After the fix the reorder is deferred until
// after validation succeeds, so when validation fails nothing is mutated: the
// matmul must still consume the LoadCachedOp weight directly, with no reorder
// slice_static/concat inserted into the forward function, and no fused op.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<16x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<24x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x512xbf16, #dram>, <interleaved>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 32 + d2, d3), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <1x1>, memref<8x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout9 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout10 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout11 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x24x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout12 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout13 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  // Const-eval'd QKV weight concatenated in [K|Q|V] column order.
  func.func private @qkv_weight_const_eval(%arg0: tensor<128x512xbf16, #ttnn_layout>, %arg1: tensor<512x512xbf16, #ttnn_layout1>, %arg2: tensor<128x512xbf16, #ttnn_layout>) -> tensor<768x512xbf16, #ttnn_layout2> attributes {tt.function_type = "const_eval"} {
    %0 = "ttnn.concat"(%arg0, %arg1, %arg2) <{dim = 0 : si32}> : (tensor<128x512xbf16, #ttnn_layout>, tensor<512x512xbf16, #ttnn_layout1>, tensor<128x512xbf16, #ttnn_layout>) -> tensor<768x512xbf16, #ttnn_layout2>
    return %0 : tensor<768x512xbf16, #ttnn_layout2>
  }

  // CHECK-LABEL: func.func @split_qkv_reorder_no_leak_on_failure
  // The const-eval'd weight is consumed directly by the matmul: no reorder
  // slice_static/concat is inserted into the forward function, and no fused op
  // is created.
  // CHECK: %[[W:.*]] = ttcore.load_cached(@qkv_weight_const_eval,
  // CHECK-NOT: "ttnn.concat"
  // CHECK: "ttnn.matmul"(%{{.*}}, %[[W]])
  // CHECK-NOT: "ttnn.split_query_key_value_and_split_heads"
  func.func @split_qkv_reorder_no_leak_on_failure(%arg0: tensor<1x32x512xbf16, #ttnn_layout6> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<128x512xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<512x512xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<128x512xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> (tensor<1x2x32x64xbf16, #ttnn_layout7>, tensor<1x8x32x64xbf16, #ttnn_layout8>, tensor<1x2x32x64xbf16, #ttnn_layout7>) attributes {tt.function_type = "forward_device"} {
    %0 = ttcore.load_cached(@qkv_weight_const_eval, [%arg1, %arg2, %arg3]) : (tensor<128x512xbf16, #ttnn_layout>, tensor<512x512xbf16, #ttnn_layout1>, tensor<128x512xbf16, #ttnn_layout>) -> tensor<768x512xbf16, #ttnn_layout2>
    %1 = "ttnn.to_layout"(%arg0) : (tensor<1x32x512xbf16, #ttnn_layout6>) -> tensor<1x32x512xbf16, #ttnn_layout9>
    %2 = "ttnn.reshape"(%1) <{shape = [32 : i32, 512 : i32]}> : (tensor<1x32x512xbf16, #ttnn_layout9>) -> tensor<32x512xbf16, #ttnn_layout10>
    %3 = "ttnn.matmul"(%2, %0) <{transpose_a = false, transpose_b = true}> : (tensor<32x512xbf16, #ttnn_layout10>, tensor<768x512xbf16, #ttnn_layout2>) -> tensor<32x768xbf16, #ttnn_layout11>
    // K slice (2 heads): columns [0, 128).
    %4 = "ttnn.slice_static"(%3) <{begins = [0 : i32, 0 : i32], ends = [32 : i32, 128 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x768xbf16, #ttnn_layout11>) -> tensor<32x128xbf16, #ttnn_layout12>
    // Q slice (8 heads): columns [128, 640).
    %5 = "ttnn.slice_static"(%3) <{begins = [0 : i32, 128 : i32], ends = [32 : i32, 640 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x768xbf16, #ttnn_layout11>) -> tensor<32x512xbf16, #ttnn_layout10>
    // V slice (2 heads): columns [640, 768).
    %6 = "ttnn.slice_static"(%3) <{begins = [0 : i32, 640 : i32], ends = [32 : i32, 768 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x768xbf16, #ttnn_layout11>) -> tensor<32x128xbf16, #ttnn_layout12>
    %7 = "ttnn.reshape"(%6) <{shape = [1 : i32, 32 : i32, 2 : i32, 64 : i32]}> : (tensor<32x128xbf16, #ttnn_layout12>) -> tensor<1x32x2x64xbf16, #ttnn_layout13>
    %8 = "ttnn.permute"(%7) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x2x64xbf16, #ttnn_layout13>) -> tensor<1x2x32x64xbf16, #ttnn_layout7>
    %9 = "ttnn.reshape"(%5) <{shape = [1 : i32, 32 : i32, 8 : i32, 64 : i32]}> : (tensor<32x512xbf16, #ttnn_layout10>) -> tensor<1x32x8x64xbf16, #ttnn_layout13>
    %10 = "ttnn.permute"(%9) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x8x64xbf16, #ttnn_layout13>) -> tensor<1x8x32x64xbf16, #ttnn_layout8>
    %11 = "ttnn.reshape"(%4) <{shape = [1 : i32, 32 : i32, 2 : i32, 64 : i32]}> : (tensor<32x128xbf16, #ttnn_layout12>) -> tensor<1x32x2x64xbf16, #ttnn_layout13>
    %12 = "ttnn.permute"(%11) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x2x64xbf16, #ttnn_layout13>) -> tensor<1x2x32x64xbf16, #ttnn_layout7>
    return %8, %10, %12 : tensor<1x2x32x64xbf16, #ttnn_layout7>, tensor<1x8x32x64xbf16, #ttnn_layout8>, tensor<1x2x32x64xbf16, #ttnn_layout7>
  }
}
