// RUN: ttmlir-opt --ttnn-fusing="disable-dram-sharded-matmul=false" -o %t.ds %s
// RUN: FileCheck %s --input-file=%t.ds --check-prefix=DSON
// RUN: ttmlir-opt --ttnn-fusing="enable-eltwise-activation-fusion=true" -o %t.gen %s
// RUN: FileCheck %s --input-file=%t.gen --check-prefix=GEN
// RUN: ttmlir-opt --ttnn-fusing -o %t.off %s
// RUN: FileCheck %s --input-file=%t.off --check-prefix=DSOFF

// Binary input-activation fusing has two callers with different scopes.
//
// enable-eltwise-activation-fusion asks for the general behaviour: any
// single-use unary folds into any binary op's operand, and output activations
// fuse too.
//
// DRAM sharding needs only to keep an activation off a matmul the optimizer may
// hand a DS config, so there the pattern is restricted to unaries fed by a
// matmul or linear.
//
// The pass defaults to true: DRAM sharding is chosen by the optimizer, so a bare
// ttnn-fusing run has no DS matmuls and leaves both halves off.

#dram = #ttnn.buffer_type<dram>
#l = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  // Unaries fed by block arguments, not by a matmul.
  // DSON-LABEL: func.func @unary_from_block_arg
  // DSON: "ttnn.relu"
  // DSON: "ttnn.sigmoid"
  // DSON: "ttnn.add"
  // DSON-SAME: input_tensor_a_activations = []
  // DSON-SAME: input_tensor_b_activations = []
  // DSON: "ttnn.tanh"

  // GEN-LABEL: func.func @unary_from_block_arg
  // GEN: "ttnn.add"
  // GEN-SAME: activations = [#ttnn.unary_with_param<op_type = tanh>]
  // GEN-SAME: input_tensor_a_activations = [#ttnn.unary_with_param<op_type = relu>]
  // GEN-SAME: input_tensor_b_activations = [#ttnn.unary_with_param<op_type = sigmoid>]
  // GEN-NOT: "ttnn.relu"
  func.func @unary_from_block_arg(%arg0: tensor<32x32xbf16, #l>, %arg1: tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l> {
    %0 = "ttnn.relu"(%arg0) : (tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l>
    %1 = "ttnn.sigmoid"(%arg1) : (tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l>
    %2 = "ttnn.add"(%0, %1) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<32x32xbf16, #l>, tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l>
    %3 = "ttnn.tanh"(%2) : (tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l>
    return %3 : tensor<32x32xbf16, #l>
  }

  // SwiGLU: the silu is fed by a matmul, so it folds in both modes. With the
  // option at its default the matmul pattern claims it instead, which is the
  // pre-existing behaviour.
  // DSON-LABEL: func.func @swiglu_silu_from_matmul
  // DSON: "ttnn.matmul"
  // DSON-NOT: activation = "silu"
  // DSON: "ttnn.multiply"
  // DSON-SAME: input_tensor_a_activations = [#ttnn.unary_with_param<op_type = silu>]

  // GEN-LABEL: func.func @swiglu_silu_from_matmul
  // GEN: "ttnn.multiply"
  // GEN-SAME: input_tensor_a_activations = [#ttnn.unary_with_param<op_type = silu>]

  // DSOFF-LABEL: func.func @swiglu_silu_from_matmul
  // DSOFF: "ttnn.matmul"
  // DSOFF-SAME: activation = "silu"
  // DSOFF-NOT: input_tensor_a_activations = [#ttnn.unary_with_param<op_type = silu>]
  func.func @swiglu_silu_from_matmul(%act: tensor<32x32xbf16, #l>, %gate_w: tensor<32x32xbf16, #l>, %up: tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l> {
    %0 = "ttnn.matmul"(%act, %gate_w) <{transpose_a = false, transpose_b = false}> : (tensor<32x32xbf16, #l>, tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l>
    %1 = "ttnn.silu"(%0) : (tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l>
    %2 = "ttnn.multiply"(%1, %up) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<32x32xbf16, #l>, tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l>
    return %2 : tensor<32x32xbf16, #l>
  }

  // Gemma-2's logit soft cap in miniature: the unary is fed by a divide, so the
  // DRAM-sharded scope leaves it alone and the general scope takes it.
  // DSON-LABEL: func.func @soft_cap_tanh_from_divide
  // DSON: "ttnn.tanh"
  // DSON: "ttnn.multiply"
  // DSON-SAME: input_tensor_a_activations = []

  // GEN-LABEL: func.func @soft_cap_tanh_from_divide
  // GEN: "ttnn.multiply"
  // GEN-SAME: input_tensor_a_activations = [#ttnn.unary_with_param<op_type = tanh>]
  func.func @soft_cap_tanh_from_divide(%logits: tensor<32x32xbf16, #l>, %cap: tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l> {
    %0 = "ttnn.divide"(%logits, %cap) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<32x32xbf16, #l>, tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l>
    %1 = "ttnn.tanh"(%0) : (tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l>
    %2 = "ttnn.multiply"(%1, %cap) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<32x32xbf16, #l>, tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l>
    return %2 : tensor<32x32xbf16, #l>
  }
}
