// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline --mlir-print-local-scope -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Verifies that `ttir.tt_lang_op` is lowered to `ttnn.tt_lang_op` with the
// four metadata attributes preserved verbatim. The `kernel_artifact`
// attribute is intentionally left absent here; the
// `--ttnn-resolve-tt-lang-kernels` pass populates it after the pipeline by
// calling the tt-lang Python resolver.

module {
  // CHECK-LABEL: func.func @forward
  func.func @forward(%arg0: tensor<1x32xf32>, %arg1: tensor<1x32xf32>)
      -> tensor<1x32xf32> {
    // CHECK: ttnn.tt_lang_op
    // CHECK-SAME: arg_roles = "in,out"
    // CHECK-SAME: kernel_id = "pkg.softmax::v1"
    // CHECK-SAME: version_tag = "1.0"
    // CHECK-NOT: kernel_artifact
    %0 = "ttir.tt_lang_op"(%arg0, %arg1) <{
      kernel_id = "pkg.softmax::v1",
      version_tag = "1.0",
      arg_roles = "in,out",
      shard_spec = ""
    }> : (tensor<1x32xf32>, tensor<1x32xf32>) -> (tensor<1x32xf32>)
    return %0 : tensor<1x32xf32>
  }

  // CHECK-LABEL: func.func @forward_multi
  // The op carries one operand per `arg_roles` token; entries tagged `out`
  // are pre-allocated buffers that surface as op results.
  func.func @forward_multi(
      %arg0: tensor<4x4xbf16>, %arg1: tensor<4x4xbf16>,
      %arg2: tensor<4x4xbf16>, %arg3: tensor<4x4xbf16>)
      -> (tensor<4x4xbf16>, tensor<4x4xbf16>) {
    // CHECK: ttnn.tt_lang_op
    // CHECK-SAME: arg_roles = "in,in,out,out"
    // CHECK-SAME: kernel_id = "pkg.dual_out::v1"
    // CHECK-SAME: shard_spec = "{\22axis\22:0}"
    %0:2 = "ttir.tt_lang_op"(%arg0, %arg1, %arg2, %arg3) <{
      kernel_id = "pkg.dual_out::v1",
      version_tag = "2.0",
      arg_roles = "in,in,out,out",
      shard_spec = "{\22axis\22:0}"
    }> : (tensor<4x4xbf16>, tensor<4x4xbf16>, tensor<4x4xbf16>, tensor<4x4xbf16>)
        -> (tensor<4x4xbf16>, tensor<4x4xbf16>)
    return %0#0, %0#1 : tensor<4x4xbf16>, tensor<4x4xbf16>
  }
}
