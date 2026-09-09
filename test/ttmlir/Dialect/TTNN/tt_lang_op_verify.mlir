// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

// Verifier negative tests for `ttnn.tt_lang_op`. Positive lowering /
// dealloc cases live in `tt_lang_op.mlir` and
// `Transforms/ttnn_deallocate_tt_lang_op.mlir`; these focus on
// diagnostics for malformed `arg_roles`.

// Zero "out" tokens, zero results: rejected. The runtime aliases the
// kernel's SSA result to the "out"-roled operand's TensorRef so
// `gatherOutputTensors()` finds the kernel's output at program end;
// an op with no "out" operand has nothing to alias and would fail
// later in the flatbuffer emitter. Reject here for a clearer
// diagnostic at the op site.
module {
  func.func @no_out_operand(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) {
    // expected-error @+1 {{tt-lang kernel must have at least one operand tagged "out"}}
    "ttnn.tt_lang_op"(%arg0, %arg1) <{
      kernel_id = "test.no_out::v1",
      version_tag = "1.0",
      arg_roles = "in,in",
      shard_spec = ""
    }> : (tensor<32x32xf32>, tensor<32x32xf32>) -> ()
    return
  }
}

// -----

// Token / operand count mismatch is still caught by the existing
// equality check; covered here so a future refactor that drops it
// does not silently regress.
module {
  func.func @arg_roles_count_mismatch(
      %arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // expected-error @+1 {{`arg_roles` token count (3) must match number of inputs (2)}}
    %0 = "ttnn.tt_lang_op"(%arg0, %arg1) <{
      kernel_id = "test.count_mismatch::v1",
      version_tag = "1.0",
      arg_roles = "in,in,out",
      shard_spec = ""
    }> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}

// -----

// "out" count must equal number of results. With one "out" but two
// results we hit the existing equality check (not the new zero-out
// guard).
module {
  func.func @out_count_mismatch(
      %arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>)
      -> (tensor<32x32xf32>, tensor<32x32xf32>) {
    // expected-error @+1 {{number of "out" roles (1) must match number of results (2)}}
    %0:2 = "ttnn.tt_lang_op"(%arg0, %arg1) <{
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
    %0 = "ttnn.tt_lang_op"(%arg0, %arg1, %arg2) <{
      kernel_id = "test.interleaved::v1",
      version_tag = "1.0",
      arg_roles = "in,out,in",
      shard_spec = ""
    }> : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
