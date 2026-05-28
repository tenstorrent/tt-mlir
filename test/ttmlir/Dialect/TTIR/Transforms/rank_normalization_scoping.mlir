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
//      d2m_subgraph_4 reproducer): callee signature is promoted to rank-2;
//      a ttnn.reshape is inserted before the call site in @main.
//   4. d2m_subgraph callee with a rank-1 arg flowing DIRECTLY into a TTIR op
//      (no pre-existing reshape): exercises the call-site walk for inputs,
//      outputs (DPS buffer), and result.
//
// No unrealized_conversion_cast ops should be inserted anywhere across any
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
//     TTIRExplicateTMs   -> inserts an explicit ttir.reshape on the rank-1
//                           operand (%arg3 in @d2m_subgraph_4).
//     TTIRRankNormalization
//
// Pre-fix bug:
//   The callee @d2m_subgraph_4 contains TTIR ops, so the pass treats it as a
//   participating function and rewrites its signature, promoting %arg3 from
//   tensor<896xbf16> to tensor<1x896xbf16>. The ttnn.d2m_subgraph op in
//   @main lives in the legal ttnn dialect, so the pass leaves the call site
//   alone. The result is an operand-type mismatch caught by
//   D2MSubgraphOp::verify().
//
// Plan C fix:
//   - The callee signature is promoted: %arg3 becomes tensor<1x896xbf16> with
//     a layout rebuilt by ttnn::utils::RankedTensorTypeFactory.
//   - The body's pre-existing ttir.reshape gets its rank-1 input replaced by
//     the rank-2 promoted %arg3 (the conversion pattern fires because the
//     reshape's operand was rank-1).
//   - At the call site in @main, a post-conversion walk inserts a
//     ttnn.reshape on %arg3 (rank-1 -> rank-2) so the ttnn.d2m_subgraph op
//     re-satisfies D2MSubgraphOp::verify().
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
// explicit ttir.reshape on the rank-1 arg %arg3 (post-TTIRExplicateTMs).
// Modeled on Kimi K2 @d2m_subgraph_4.
//
// Post-fix expectation (Plan C): signature is promoted, %arg3 becomes rank-2
// (tensor<1x896xbf16>) with a rebuilt layout. The pre-existing ttir.reshape's
// input now feeds directly from the rank-2 %arg3.
// CHECK-LABEL: func.func private @d2m_subgraph_4
// CHECK-SAME: %arg3: tensor<1x896xbf16
// CHECK: %[[RESHAPE:.*]] = "ttir.reshape"(%arg3)
// CHECK-SAME: (tensor<1x896xbf16{{.*}}) -> tensor<1x1x896xbf16
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

// @main: dispatches @d2m_subgraph_4 via ttnn.d2m_subgraph. @main has no TTIR
// ops (non-participating) and its signature stays untouched, but the
// post-conversion call-site walk inserts a ttnn.reshape on %arg3 (rank-1 ->
// rank-2) so the ttnn.d2m_subgraph operand types align with the (now
// rank-2-promoted) callee signature.
// CHECK-LABEL: func.func @main
// CHECK-SAME: %arg3: tensor<896xbf16
// CHECK: %[[BIAS:.*]] = "ttnn.reshape"(%arg3)
// CHECK-SAME: (tensor<896xbf16{{.*}}) -> tensor<1x896xbf16
// CHECK: ttnn.d2m_subgraph @d2m_subgraph_4
// CHECK-NEXT: ins(%arg0, %arg1, %arg2, %[[BIAS]]
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
// Synthetic minimal reproducer that exercises every leg of the Plan C call-
// site walk: input bridge, output (DPS buffer) bridge, and result bridge.
//
// Pre-fix bug:
//   The pass promotes the callee signature, the ttnn.d2m_subgraph operand
//   stays rank-1, and D2MSubgraphOp::verify() rejects the mismatch:
//     error: 'ttnn.d2m_subgraph' op D2M function argument type 0 mismatch:
//       expected 'tensor<1x32xf32>', got 'tensor<32xf32>'
//
// Plan C fix:
//   - @sub signature is promoted: (tensor<1x32xf32>) -> tensor<1x32xf32>.
//   - Body operates entirely at rank-2 (no internal reshape).
//   - In @main, the post-conversion walk inserts ttnn.reshape ops on each
//     leg: input %arg0 (rank-1 -> rank-2), DPS output %arg1 (rank-1 ->
//     rank-2), and the d2m_subgraph result (rank-2 -> rank-1) so downstream
//     uses see the original-rank type.
// =============================================================================

// d2m_subgraph callee: rank-1 arg flows directly into ttir.add (no reshape).
// Plan C promotes the entire signature and body to rank-2.
// CHECK-LABEL: func.func private @sub
// CHECK-SAME: (%arg0: tensor<1x32xf32>) -> tensor<1x32xf32>
// CHECK: %[[ADD:.*]] = "ttir.add"(%arg0, %arg0) : (tensor<1x32xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>
// CHECK: return %[[ADD]] : tensor<1x32xf32>
func.func private @sub(%arg0: tensor<32xf32>) -> tensor<32xf32> {
  %0 = "ttir.add"(%arg0, %arg0) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// @main caller: signature stays rank-1, but the call-site walk inserts
// ttnn.reshape ops to bridge the rank-1 operands/result against the rank-2
// callee signature.
// CHECK-LABEL: func.func @main
// CHECK-SAME: (%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[IN:.*]] = "ttnn.reshape"(%arg0)
// CHECK-SAME: (tensor<32xf32>) -> tensor<1x32xf32>
// CHECK: %[[BUF:.*]] = "ttnn.reshape"(%arg1)
// CHECK-SAME: (tensor<32xf32>) -> tensor<1x32xf32>
// CHECK: %[[CALL:.*]] = ttnn.d2m_subgraph @sub
// CHECK-NEXT: ins(%[[IN]] : tensor<1x32xf32>)
// CHECK-NEXT: outs(%[[BUF]] : tensor<1x32xf32>) : tensor<1x32xf32>
// CHECK: %[[FINAL:.*]] = "ttnn.reshape"(%[[CALL]])
// CHECK-SAME: (tensor<1x32xf32>) -> tensor<32xf32>
// CHECK: return %[[FINAL]] : tensor<32xf32>
func.func @main(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> tensor<32xf32> {
  %0 = ttnn.d2m_subgraph @sub
       ins(%arg0 : tensor<32xf32>)
       outs(%arg1 : tensor<32xf32>) : tensor<32xf32>
  return %0 : tensor<32xf32>
}
