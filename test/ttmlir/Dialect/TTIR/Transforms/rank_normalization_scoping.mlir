// RUN: ttmlir-opt --split-input-file --ttir-rank-normalization %s | FileCheck %s
// RUN: ttmlir-opt --split-input-file --ttir-rank-normalization %s | FileCheck %s --check-prefix=NOCAST
//
// Consolidated scoping tests for TTIRRankNormalization.
//
// Each section below is a self-contained module separated by a split marker.
// '--split-input-file' runs the pass independently on each chunk, so symbol
// and layout names can repeat across sections without conflict and tests
// remain fully isolated.
//
// Sections in this file:
//   1. Selective promotion: TTIR-using funcs vs. TTNN-only funcs (incl.
//      const_eval helpers, external CPU-hoisted decls, d2m_subgraph mixes,
//      already-rank-2 ttir).
//   2. External-call regression: a module with no participating funcs must be
//      a complete no-op (regression for the bug where external CPU-hoisted
//      decls were promoted even without a TTIR caller).
//   3. d2m_subgraph callee with rank-1 arg + pre-existing reshape (Kimi K2
//      d2m_subgraph_4 reproducer): callee signature stays byte-identical via
//      Plan B; a `ttir.reshape` is materialized at the entry-block boundary.
//   4. d2m_subgraph callee with a rank-1 arg flowing DIRECTLY into a TTIR op
//      (no pre-existing reshape): exercises both source and target
//      materializer hooks (entry-block bridge + pre-return bridge).
//
// No `unrealized_conversion_cast` ops should be inserted anywhere across any
// section.
// NOCAST: module
// NOCAST-NOT: unrealized_conversion_cast

// =============================================================================
// SECTION 1: Selective promotion of TTIR-using funcs vs. TTNN-only funcs.
//
// Test that TTIRRankNormalization correctly scopes its rewrites to functions
// that contain TTIR ops. Functions containing only TTNN ops (e.g. const_eval
// helpers) must be left entirely untouched.
//
// This is needed in the D2M + Optimizer integration path where the pass would
// promote function signatures of TTNN-only functions, causing func.return type
// mismatches or unrealized_conversion_cast insertions.
// =============================================================================

#dram = #ttnn.buffer_type<dram>
#layout_rank1_bf16 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_rank1_f32 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// TTNN-only const_eval functions which should not be promoted.
// CHECK-LABEL: func.func private @main_const_eval_0
// CHECK-SAME: (%arg0: tensor<32xbf16, #{{.*}}>) -> tensor<32xf32, #{{.*}}>
// CHECK: "ttnn.typecast"(%arg0)
// CHECK-SAME: (tensor<32xbf16, #{{.*}}>) -> tensor<32xf32, #{{.*}}>
// CHECK: return %{{.*}} : tensor<32xf32, #{{.*}}>
func.func private @main_const_eval_0(%arg0: tensor<32xbf16, #layout_rank1_bf16>)
    -> tensor<32xf32, #layout_rank1_f32>
    attributes {tt.function_type = "const_eval"} {
  %0 = "ttnn.typecast"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>}>
      : (tensor<32xbf16, #layout_rank1_bf16>) -> tensor<32xf32, #layout_rank1_f32>
  return %0 : tensor<32xf32, #layout_rank1_f32>
}

// CHECK-LABEL: func.func private @main_const_eval_1
// CHECK-SAME: (%arg0: tensor<64xbf16, #{{.*}}>) -> tensor<64xf32, #{{.*}}>
// CHECK: "ttnn.typecast"(%arg0)
// CHECK-SAME: (tensor<64xbf16, #{{.*}}>) -> tensor<64xf32, #{{.*}}>
// CHECK: return %{{.*}} : tensor<64xf32, #{{.*}}>
func.func private @main_const_eval_1(%arg0: tensor<64xbf16, #layout_rank1_bf16>)
    -> tensor<64xf32, #layout_rank1_f32>
    attributes {tt.function_type = "const_eval"} {
  %0 = "ttnn.typecast"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>}>
      : (tensor<64xbf16, #layout_rank1_bf16>) -> tensor<64xf32, #layout_rank1_f32>
  return %0 : tensor<64xf32, #layout_rank1_f32>
}

// Function with TTIR ops which should be promoted to rank-2. This represents a
// D2M subgraph function after ConvertTTNNToTTIR.
// CHECK-LABEL: func.func @ttir_eltwise_chain
// CHECK-SAME: (%arg0: tensor<1x32xf32>, %arg1: tensor<1x32xf32>) -> tensor<1x32xf32>
// CHECK: "ttir.add"(%arg0, %arg1) : (tensor<1x32xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>
// CHECK: "ttir.multiply"
// CHECK-SAME: (tensor<1x32xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>
// CHECK: return %{{.*}} : tensor<1x32xf32>
func.func @ttir_eltwise_chain(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> tensor<32xf32> {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  %1 = "ttir.multiply"(%0, %arg0) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  return %1 : tensor<32xf32>
}


// External declaration NOT called from any TTIR function. Should remain
// unpromoted - its callers are non-TTIR functions that also won't be promoted.
// CHECK-LABEL: func.func private @cpu_hoisted_decl
// CHECK-SAME: (tensor<32xf32>) -> tensor<32xf32>
func.func private @cpu_hoisted_decl(tensor<32xf32>) -> tensor<32xf32>
    attributes {tt.function_type = "ForwardCPUDeclaration"}

// External declaration called from a TTIR (d2m_subgraph) function. Should be
// promoted so the call site in the participating function stays in sync.
// CHECK-LABEL: func.func private @cpu_hoisted_called_from_ttir
// CHECK-SAME: (tensor<1x32xf32>) -> tensor<1x32xf32>
func.func private @cpu_hoisted_called_from_ttir(tensor<32xf32>) -> tensor<32xf32>
    attributes {tt.function_type = "ForwardCPUDeclaration"}

// CHECK-LABEL: func.func @d2m_subgraph_with_call
// CHECK-SAME: (%arg0: tensor<1x32xf32>) -> tensor<1x32xf32>
// CHECK: %[[ADD:.*]] = "ttir.add"(%arg0, %arg0) : (tensor<1x32xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>
// CHECK: %[[CALL:.*]] = call @cpu_hoisted_called_from_ttir(%[[ADD]]) : (tensor<1x32xf32>) -> tensor<1x32xf32>
// CHECK: return %[[CALL]] : tensor<1x32xf32>
func.func @d2m_subgraph_with_call(%arg0: tensor<32xf32>) -> tensor<32xf32> {
  %0 = "ttir.add"(%arg0, %arg0) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  %1 = func.call @cpu_hoisted_called_from_ttir(%0) : (tensor<32xf32>) -> tensor<32xf32>
  return %1 : tensor<32xf32>
}

// @main-like function with only TTNN ops that calls both a d2m_subgraph
// function and a const_eval function. The pass should NOT touch @main_caller
// or the const_eval callee - only @d2m_subgraph_callee should be promoted.
// D2M subgraph with rank-1 tensors that should be promoted.
// CHECK-LABEL: func.func private @d2m_subgraph_rank1
// CHECK-SAME: (%arg0: tensor<1x32xf32>) -> tensor<1x32xf32>
// CHECK: "ttir.multiply"(%arg0, %arg0) : (tensor<1x32xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>
// CHECK: return
func.func private @d2m_subgraph_rank1(%arg0: tensor<32xf32>) -> tensor<32xf32> {
  %0 = "ttir.multiply"(%arg0, %arg0) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// D2M subgraph already at rank-2 (realistic case). Should be untouched.
// CHECK-LABEL: func.func private @d2m_subgraph_rank2
// CHECK-SAME: (%arg0: tensor<4x32xf32>) -> tensor<4x32xf32>
// CHECK: "ttir.multiply"
// CHECK: return
func.func private @d2m_subgraph_rank2(%arg0: tensor<4x32xf32>) -> tensor<4x32xf32> {
  %0 = "ttir.multiply"(%arg0, %arg0) : (tensor<4x32xf32>, tensor<4x32xf32>) -> tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// Const_eval function (TTNN-only). Should NOT be promoted.
// CHECK-LABEL: func.func private @const_eval_callee
// CHECK-SAME: (%arg0: tensor<32xbf16, #{{.*}}>) -> tensor<32xf32, #{{.*}}>
// CHECK: "ttnn.typecast"
// CHECK: return
func.func private @const_eval_callee(%arg0: tensor<32xbf16, #layout_rank1_bf16>)
    -> tensor<32xf32, #layout_rank1_f32>
    attributes {tt.function_type = "const_eval"} {
  %0 = "ttnn.typecast"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>}>
      : (tensor<32xbf16, #layout_rank1_bf16>) -> tensor<32xf32, #layout_rank1_f32>
  return %0 : tensor<32xf32, #layout_rank1_f32>
}

// @main-like function (no TTIR ops). Calls const_eval via ttcore.load_cached
// and d2m_subgraph via ttnn.d2m_subgraph. Nothing here should be promoted.
// CHECK-LABEL: func.func @main_caller
// CHECK-SAME: (%arg0: tensor<32xbf16, #{{.*}}>, %arg1: tensor<4x32xf32>, %arg2: tensor<4x32xf32>)
// CHECK: ttcore.load_cached(@const_eval_callee, [%arg0])
// CHECK-SAME: (tensor<32xbf16, #{{.*}}>) -> tensor<32xf32, #{{.*}}>
// CHECK: ttnn.d2m_subgraph @d2m_subgraph_rank2
// CHECK-NEXT: ins(%arg1 : tensor<4x32xf32>)
// CHECK-NEXT: outs(%arg2 : tensor<4x32xf32>) : tensor<4x32xf32>
// CHECK: return
func.func @main_caller(%arg0: tensor<32xbf16, #layout_rank1_bf16>, %arg1: tensor<4x32xf32>, %arg2: tensor<4x32xf32>)
    -> tensor<4x32xf32> {
  %0 = ttcore.load_cached(@const_eval_callee, [%arg0]) : (tensor<32xbf16, #layout_rank1_bf16>) -> tensor<32xf32, #layout_rank1_f32>
  %1 = ttnn.d2m_subgraph @d2m_subgraph_rank2
      ins(%arg1 : tensor<4x32xf32>)
      outs(%arg2 : tensor<4x32xf32>) : tensor<4x32xf32>
  return %1 : tensor<4x32xf32>
}

// Function with rank-2 TTIR ops which should be left untouched.
// CHECK-LABEL: func.func @ttir_already_rank2
// CHECK-SAME: (%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK: "ttir.add"
// CHECK: return
func.func @ttir_already_rank2(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// -----

// =============================================================================
// SECTION 2: External-call regression - module with no participating funcs is
// a complete no-op.
//
// Regression test for the bug where TTIRRankNormalization promoted external
// CPU-hoisted declarations even when their callers were non-participating
// functions, causing func.call operand type mismatches like:
//
//   error: 'func.call' op operand type mismatch:
//     expected operand type 'tensor<1x64xf32, #ttnn.ttnn_layout<...>>',
//     but provided 'tensor<64xf32, #ttnn.ttnn_layout<...>>' for operand number 0
//
// With the scoping fix in place, when the module contains no TTIR-using
// functions, the pass must be a complete no-op - neither the external decl
// nor the call site should be rewritten.
// =============================================================================

#system_memory = #ttnn.buffer_type<system_memory>
#layout_rank1 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x64xf32, #system_memory>>

// External CPU-hoisted declaration with a rank-1 signature.
// The OLD logic would have promoted this to (tensor<1x64xf32>) -> tensor<1x64xf32>.
// The NEW logic must leave it untouched because no participating (TTIR) function calls it.
// CHECK-LABEL: func.func private @cpu_hoisted_helper
// CHECK-SAME: (tensor<64xf32, #{{.*}}>) -> tensor<64xf32, #{{.*}}>
func.func private @cpu_hoisted_helper(tensor<64xf32, #layout_rank1>) -> tensor<64xf32, #layout_rank1>
    attributes {tt.function_type = "ForwardCPUDeclaration"}

// Non-TTIR function (mimicking @main / a const_eval helper) that calls
// the external decl. With the OLD logic, the call site would have stayed at
// rank-1 while the callee was promoted, producing the operand type mismatch.
// CHECK-LABEL: func.func @main
// CHECK-SAME: (%arg0: tensor<64xf32, #{{.*}}>) -> tensor<64xf32, #{{.*}}>
// CHECK: %[[CALL:.*]] = call @cpu_hoisted_helper(%arg0) : (tensor<64xf32, #{{.*}}>) -> tensor<64xf32, #{{.*}}>
// CHECK: return %[[CALL]] : tensor<64xf32, #{{.*}}>
func.func @main(%arg0: tensor<64xf32, #layout_rank1>) -> tensor<64xf32, #layout_rank1> {
  %0 = func.call @cpu_hoisted_helper(%arg0)
      : (tensor<64xf32, #layout_rank1>) -> tensor<64xf32, #layout_rank1>
  return %0 : tensor<64xf32, #layout_rank1>
}

// -----

// =============================================================================
// SECTION 3: d2m_subgraph callee with rank-1 arg + pre-existing reshape.
// (Kimi K2 d2m_subgraph_4 reproducer.)
//
// Background:
//   In the production pipeline (Kimi K2, OSS-20B, etc.) the order is roughly:
//     ConvertTTNNToTTIR  -> body of @d2m_subgraph_* now contains TTIR ops.
//     TTIRExplicateTMs   -> inserts an explicit `ttir.reshape` on the rank-1
//                           operand (%arg3 in @d2m_subgraph_4).
//     TTIRRankNormalization
//
// What today's pass does (the bug):
//   The callee @d2m_subgraph_4 contains TTIR ops, so the pass treats it as a
//   participating function and rewrites its signature, promoting %arg3 from
//   `tensor<896xbf16>` to `tensor<1x896xbf16>`. The `ttnn.d2m_subgraph` op in
//   @main lives in the legal `ttnn` dialect, so the pass leaves the call site
//   alone. The result is an operand-type mismatch caught by
//   D2MSubgraphOp::verify():
//
//     error: 'ttnn.d2m_subgraph' op D2M function argument type 3 mismatch:
//       expected 'tensor<896xbf16, ...>',
//       got      'tensor<1x896xbf16, ...>'
//
// What Plan B does:
//   Pin the callee `func.func` signature and `func.return` so the matching
//   `ttnn.d2m_subgraph` call site stays well-typed. Promote the body interior
//   to rank-2. The dialect-conversion framework auto-inserts a `ttir.reshape`
//   at the entry-block boundary (via the type converter's source/target
//   materializer customized to emit `ttir.reshape` for rank-only conversions).
//
// For Kimi K2 d2m_subgraph_4, the body's pre-existing `ttir.reshape` on
// %arg3 (`tensor<896xbf16>` -> `tensor<1x1x896xbf16>`) gets matched by the
// rank-norm pattern (its rank-1 operand makes it illegal). The framework
// promotes its operand from rank-1 to rank-2 via the materializer:
//
//   %0 = "ttir.reshape"(%arg3)                   <- NEW: rank-1 -> rank-2 bridge
//        : (tensor<896xbf16>) -> tensor<1x896xbf16>
//   ...
//   %5 = "ttir.reshape"(%0)                      <- existing reshape, input bumped
//        : (tensor<1x896xbf16>) -> tensor<1x1x896xbf16>
//
// The two reshapes compose to the original `(rank-1) -> (rank-3)` reshape,
// and a later canonicalization pass folds them. Crucially:
//   - `@d2m_subgraph_4`'s function signature stays byte-identical (%arg3 is
//      still `tensor<896xbf16>`).
//   - `@main` is byte-identical (no TTIR ops -> non-participating).
//   - The `ttnn.d2m_subgraph` op's operand types are byte-identical -> the
//      `D2MSubgraphOp` verifier mismatch is gone.
// =============================================================================

#dram = #ttnn.buffer_type<dram>

// Layout for `tensor<896xbf16>` (rank-1, broadcast bias-style operand).
#layout_rank1_896 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>,
                                       memref<1x28x!ttcore.tile<32x32, bf16>, #dram>,
                                       <interleaved>>

// Layout for `tensor<16x16x1xbf16>` (rank-3, reduction-axis operand).
#layout_rank3_16x16x1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2),
                                          <1x1>,
                                          memref<16x1x!ttcore.tile<32x32, bf16>, #dram>,
                                          <interleaved>>

// Layout for `tensor<16x16x896xbf16>` (rank-3, full activation tensor).
#layout_rank3_16x16x896 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2),
                                            <1x1>,
                                            memref<16x28x!ttcore.tile<32x32, bf16>, #dram>,
                                            <interleaved>>

// d2m_subgraph callee. Body contains TTIR ops (post-ConvertTTNNToTTIR) and an
// explicit `ttir.reshape` on the rank-1 arg %arg3 (post-TTIRExplicateTMs).
// Modeled on Kimi K2 @d2m_subgraph_4.
//
// Post-fix expectation: signature unchanged. In particular %arg3 stays at
// `tensor<896xbf16>`. A new `ttir.reshape : (rank-1) -> (rank-2)` bridges %arg3
// into the body, and the existing reshape's input is bumped to rank-2.
// CHECK-LABEL: func.func private @d2m_subgraph_4
// CHECK-SAME: %arg3: tensor<896xbf16
// CHECK: %[[BRIDGE:.*]] = "ttir.reshape"(%arg3)
// CHECK-SAME: -> tensor<1x896xbf16
// CHECK: "ttir.reshape"(%[[BRIDGE]])
// CHECK-SAME: -> tensor<1x1x896xbf16
// CHECK: return %{{.*}} : tensor<16x16x896xbf16
func.func private @d2m_subgraph_4(
    %arg0: tensor<16x16x1xbf16, #layout_rank3_16x16x1>,
    %arg1: tensor<16x16x1xbf16, #layout_rank3_16x16x1>,
    %arg2: tensor<16x16x896xbf16, #layout_rank3_16x16x896>,
    %arg3: tensor<896xbf16, #layout_rank1_896>) -> tensor<16x16x896xbf16, #layout_rank3_16x16x896> {
  %0 = "ttir.add"(%arg0, %arg1)
      : (tensor<16x16x1xbf16, #layout_rank3_16x16x1>,
         tensor<16x16x1xbf16, #layout_rank3_16x16x1>) -> tensor<16x16x1xbf16, #layout_rank3_16x16x1>
  %1 = "ttir.rsqrt"(%0)
      : (tensor<16x16x1xbf16, #layout_rank3_16x16x1>) -> tensor<16x16x1xbf16, #layout_rank3_16x16x1>
  %2 = "ttir.broadcast"(%1) <{broadcast_dimensions = array<i64: 1, 1, 896>}>
      : (tensor<16x16x1xbf16, #layout_rank3_16x16x1>) -> tensor<16x16x896xbf16, #layout_rank3_16x16x1>
  %3 = "ttir.multiply"(%arg2, %2)
      : (tensor<16x16x896xbf16, #layout_rank3_16x16x896>,
         tensor<16x16x896xbf16, #layout_rank3_16x16x1>) -> tensor<16x16x896xbf16, #layout_rank3_16x16x896>
  %4 = "ttir.reshape"(%arg3) <{shape = [1 : i32, 1 : i32, 896 : i32]}>
      : (tensor<896xbf16, #layout_rank1_896>) -> tensor<1x1x896xbf16, #layout_rank1_896>
  %5 = "ttir.broadcast"(%4) <{broadcast_dimensions = array<i64: 16, 16, 1>}>
      : (tensor<1x1x896xbf16, #layout_rank1_896>) -> tensor<16x16x896xbf16, #layout_rank1_896>
  %6 = "ttir.multiply"(%3, %5)
      : (tensor<16x16x896xbf16, #layout_rank3_16x16x896>,
         tensor<16x16x896xbf16, #layout_rank1_896>) -> tensor<16x16x896xbf16, #layout_rank3_16x16x896>
  return %6 : tensor<16x16x896xbf16, #layout_rank3_16x16x896>
}

// @main: dispatches @d2m_subgraph_4 via ttnn.d2m_subgraph. No TTIR ops here -
// @main is a non-participating function and must stay byte-identical, in
// particular the rank-1 4th operand (%arg3) must still be `tensor<896xbf16>`.
// CHECK-LABEL: func.func @main
// CHECK: ttnn.d2m_subgraph @d2m_subgraph_4
// CHECK-NEXT: ins(%arg0, %arg1, %arg2, %arg3
// CHECK-NOT: ttir.reshape
// CHECK: return
func.func @main(
    %arg0: tensor<16x16x1xbf16, #layout_rank3_16x16x1>,
    %arg1: tensor<16x16x1xbf16, #layout_rank3_16x16x1>,
    %arg2: tensor<16x16x896xbf16, #layout_rank3_16x16x896>,
    %arg3: tensor<896xbf16, #layout_rank1_896>,
    %arg4: tensor<16x16x896xbf16, #layout_rank3_16x16x896>) -> tensor<16x16x896xbf16, #layout_rank3_16x16x896> {
  %0 = ttnn.d2m_subgraph @d2m_subgraph_4
      ins(%arg0, %arg1, %arg2, %arg3 :
          tensor<16x16x1xbf16, #layout_rank3_16x16x1>,
          tensor<16x16x1xbf16, #layout_rank3_16x16x1>,
          tensor<16x16x896xbf16, #layout_rank3_16x16x896>,
          tensor<896xbf16, #layout_rank1_896>)
      outs(%arg4 : tensor<16x16x896xbf16, #layout_rank3_16x16x896>)
      : tensor<16x16x896xbf16, #layout_rank3_16x16x896>
  return %0 : tensor<16x16x896xbf16, #layout_rank3_16x16x896>
}

// -----

// =============================================================================
// SECTION 4: d2m_subgraph callee with rank-1 arg flowing DIRECTLY into a TTIR
// op (no pre-existing reshape from TTIRExplicateTMs).
//
// Synthetic reproducer for the case Plan B's materializer hooks must handle:
// the callee signature and `func.return` stay byte-identical, the body gets
// promoted to rank-2 internally, and `ttir.reshape` ops are inserted at the
// entry block (rank-1 -> rank-2) and before the return (rank-2 -> rank-1).
//
// Today (pre-fix) the pass promotes the callee signature, the
// `ttnn.d2m_subgraph` operand stays rank-1, and `D2MSubgraphOp::verify()`
// rejects the mismatch:
//
//   error: 'ttnn.d2m_subgraph' op D2M function argument type 0 mismatch:
//     expected 'tensor<32xf32>', got 'tensor<1x32xf32>'
//
// Post-fix expectations:
//   - `@sub` signature unchanged: (tensor<32xf32>) -> tensor<32xf32>.
//   - Entry block first op is `ttir.reshape : tensor<32xf32> -> tensor<1x32xf32>`.
//   - `ttir.add` operates at rank-2.
//   - `ttir.reshape : tensor<1x32xf32> -> tensor<32xf32>` immediately before
//     `func.return`.
//   - `@main` and the `ttnn.d2m_subgraph` op byte-identical to the input.
// =============================================================================

// d2m_subgraph callee: rank-1 arg flows directly into ttir.add (no reshape).
// CHECK-LABEL: func.func private @sub
// CHECK-SAME: (%arg0: tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[R0:.*]] = "ttir.reshape"(%arg0)
// CHECK-SAME: (tensor<32xf32>) -> tensor<1x32xf32>
// CHECK: %[[ADD:.*]] = "ttir.add"(%[[R0]], %[[R0]])
// CHECK-SAME: (tensor<1x32xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>
// CHECK: %[[R1:.*]] = "ttir.reshape"(%[[ADD]])
// CHECK-SAME: (tensor<1x32xf32>) -> tensor<32xf32>
// CHECK: return %[[R1]] : tensor<32xf32>
func.func private @sub(%arg0: tensor<32xf32>) -> tensor<32xf32> {
  %0 = "ttir.add"(%arg0, %arg0) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// @main caller: byte-identical to input. The rank-1 operand stays rank-1.
// CHECK-LABEL: func.func @main
// CHECK-SAME: (%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[OUT:.*]] = ttnn.d2m_subgraph @sub
// CHECK-NEXT: ins(%arg0 : tensor<32xf32>)
// CHECK-NEXT: outs(%arg1 : tensor<32xf32>) : tensor<32xf32>
// CHECK-NOT: tensor<1x32xf32
// CHECK: return %[[OUT]] : tensor<32xf32>
func.func @main(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> tensor<32xf32> {
  %0 = ttnn.d2m_subgraph @sub
       ins(%arg0 : tensor<32xf32>)
       outs(%arg1 : tensor<32xf32>) : tensor<32xf32>
  return %0 : tensor<32xf32>
}
