// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

// Verifier negative tests for `ttir.tt_lang_op`. Mirrors the TTNN-side
// negative suite so the StableHLOToTTIR pattern can never produce IR
// that the TTNN verifier or flatbuffer emitter would reject; the
// diagnostic surfaces at the layer that built the op.

// Zero "out" tokens, zero results: rejected. See TTNN-side
// equivalent for the runtime rationale.
module {
  func.func @no_out_operand(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) {
    // expected-error @+1 {{tt-lang kernel must have at least one operand tagged "out"}}
    "ttir.tt_lang_op"(%arg0, %arg1) <{
      kernel_id = "test.no_out::v1",
      version_tag = "1.0",
      arg_roles = "in,in",
      shard_spec = ""
    }> : (tensor<32x32xf32>, tensor<32x32xf32>) -> ()
    return
  }
}

// -----

// Token count must equal operand count.
module {
  func.func @arg_roles_count_mismatch(
      %arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // expected-error @+1 {{`arg_roles` token count (3) must match number of inputs (2)}}
    %0 = "ttir.tt_lang_op"(%arg0, %arg1) <{
      kernel_id = "test.count_mismatch::v1",
      version_tag = "1.0",
      arg_roles = "in,in,out",
      shard_spec = ""
    }> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}

// -----

// "out" count must equal number of results.
module {
  func.func @out_count_mismatch(
      %arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>)
      -> (tensor<32x32xf32>, tensor<32x32xf32>) {
    // expected-error @+1 {{number of "out" roles (1) must match number of results (2)}}
    %0:2 = "ttir.tt_lang_op"(%arg0, %arg1) <{
      kernel_id = "test.out_mismatch::v1",
      version_tag = "1.0",
      arg_roles = "in,out",
      shard_spec = ""
    }> : (tensor<32x32xf32>, tensor<32x32xf32>) -> (tensor<32x32xf32>, tensor<32x32xf32>)
    return %0#0, %0#1 : tensor<32x32xf32>, tensor<32x32xf32>
  }
}

// -----

// Interleaved ordering is rejected: the op is destination-passing style,
// so `arg_roles` must list all "in" operands before any "out" operand.
module {
  func.func @interleaved_arg_roles(
      %arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>,
      %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // expected-error @+1 {{`arg_roles` must list all "in" operands before any "out" operand}}
    %0 = "ttir.tt_lang_op"(%arg0, %arg1, %arg2) <{
      kernel_id = "test.interleaved::v1",
      version_tag = "1.0",
      arg_roles = "in,out,in",
      shard_spec = ""
    }> : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
