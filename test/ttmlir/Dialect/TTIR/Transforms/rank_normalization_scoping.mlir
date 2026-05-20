// RUN: ttmlir-opt --ttir-rank-normalization %s | FileCheck %s
// RUN: ttmlir-opt --ttir-rank-normalization %s | FileCheck %s --check-prefix=NOCAST
//
// No unrealized_conversion_cast ops should be inserted anywhere in the module.
// NOCAST: module
// NOCAST-NOT: unrealized_conversion_cast

// Test that TTIRRankNormalization correctly scopes its rewrites to functions
// that contain TTIR ops. Functions containing only TTNN ops (e.g. const_eval
// helpers) must be left entirely untouched.
//
// This is needed in the D2M + Optimizer integration path where the pass would promote
// function signatures of TTNN-only functions, causing func.return type
// mismatches or unrealized_conversion_cast insertions.

#dram = #ttnn.buffer_type<dram>
#layout_rank1_bf16 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_rank1_f32 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// =============================================================================
// TTNN-only const_eval functions which should not be promoted.
// =============================================================================

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

// =============================================================================
// Function with TTIR ops which should be promoted to rank-2. This represents a D2M subgraph function after ConvertTTNNToTTIR.
// =============================================================================

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


// =============================================================================
// External declaration NOT called from any TTIR function. Should remain
// unpromoted — its callers are non-TTIR functions that also won't be promoted.
// =============================================================================

// CHECK-LABEL: func.func private @cpu_hoisted_decl
// CHECK-SAME: (tensor<32xf32>) -> tensor<32xf32>
func.func private @cpu_hoisted_decl(tensor<32xf32>) -> tensor<32xf32>
    attributes {tt.function_type = "ForwardCPUDeclaration"}

// =============================================================================
// External declaration called from a TTIR (d2m_subgraph) function. Should be
// promoted so the call site in the participating function stays in sync.
// =============================================================================

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

// =============================================================================
// @main-like function with only TTNN ops that calls both a d2m_subgraph
// function and a const_eval function. The pass should NOT touch @main_caller
// or the const_eval callee — only @d2m_subgraph_callee should be promoted.
// =============================================================================

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

// =============================================================================
// Function with rank-2 TTIR ops which should be left untouched.
// =============================================================================

// CHECK-LABEL: func.func @ttir_already_rank2
// CHECK-SAME: (%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK: "ttir.add"
// CHECK: return
func.func @ttir_already_rank2(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}
