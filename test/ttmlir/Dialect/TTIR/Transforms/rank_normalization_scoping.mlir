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
// External declaration with rank-1 signature — must be promoted so call sites
// in participating TTIR functions stay in sync.
// =============================================================================

// CHECK-LABEL: func.func private @cpu_hoisted_decl
// CHECK-SAME: (tensor<1x32xf32>) -> tensor<1x32xf32>
func.func private @cpu_hoisted_decl(tensor<32xf32>) -> tensor<32xf32>
    attributes {tt.function_type = "ForwardCPUDeclaration"}

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
