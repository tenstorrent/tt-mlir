// RUN: ttmlir-opt --ttnn-resolve-tt-lang-kernels -o %t %s
// RUN: FileCheck %s --input-file=%t

// Smoke tests for `--ttnn-resolve-tt-lang-kernels` that don't require
// a live Python interpreter:
//
//   * A module with no `ttnn.tt_lang_op` is untouched.
//   * A `ttnn.tt_lang_op` whose `kernel_artifact` is already populated
//     is left alone (the resolver is skipped). This pins the
//     "composable with ahead-of-time artifact baking" contract.
//
// The Python-invocation path is exercised by tt-xla's hardware e2e
// tests (`tests/torch/ops/test_tt_lang_kernel_e2e.py`); covering it
// here would require ttmlir-opt to spin up a CPython interpreter,
// which lit doesn't do.

#dram = #ttnn.buffer_type<dram>
#dev = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: func.func @no_tt_lang_op
// A function without any tt_lang_op survives the pass byte-for-byte:
// the walk finds nothing to do and we never try to import the Python
// resolver. The CHECK-NOT pins the "no spurious ops materialised"
// half of the contract.
// CHECK-NOT: ttnn.tt_lang_op
func.func @no_tt_lang_op(%a: tensor<32x32xf32, #dev>)
    -> tensor<32x32xf32, #dev> {
  return %a : tensor<32x32xf32, #dev>
}

// CHECK-LABEL: func.func @prebaked_kernel_artifact_is_preserved
// When `kernel_artifact` is already populated (e.g. by a pre-bake
// step or a prior run of this pass), the resolver must NOT overwrite
// it. The artifact string survives the pass unchanged. This also
// implicitly exercises the "no work to do" branch -- the pass
// short-circuits without trying to import tt_torch.tt_lang.
// CHECK: ttnn.tt_lang_op
// CHECK-SAME: kernel_artifact = "prebaked-artifact"
// CHECK-SAME: kernel_id = "test.prebaked::v1"
func.func @prebaked_kernel_artifact_is_preserved(
    %a: tensor<32x32xf32, #dev>,
    %b: tensor<32x32xf32, #dev>,
    %out: tensor<32x32xf32, #dev>) -> tensor<32x32xf32, #dev> {
  %0 = "ttnn.tt_lang_op"(%a, %b, %out) <{
    kernel_id = "test.prebaked::v1",
    version_tag = "1.0",
    arg_roles = "in,in,out",
    shard_spec = "",
    kernel_artifact = "prebaked-artifact"
  }> : (tensor<32x32xf32, #dev>, tensor<32x32xf32, #dev>, tensor<32x32xf32, #dev>) -> tensor<32x32xf32, #dev>
  return %0 : tensor<32x32xf32, #dev>
}
