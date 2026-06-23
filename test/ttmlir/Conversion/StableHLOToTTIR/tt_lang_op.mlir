// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verifies that `stablehlo.custom_call @tt.tt_lang_op` is lowered to
// `ttir.tt_lang_op`, with all four metadata attributes (`kernel_id`,
// `version_tag`, `arg_roles`, `shard_spec`) lifted from
// `mhlo.frontend_attributes` onto the new op, and that operand / result
// types are preserved verbatim.

module {
  // One `in` operand + one `out` operand (pre-allocated buffer) -> one result.
  // No shard spec.
  func.func @tt_lang_op_simple(
      %arg0: tensor<1x32xf32>,
      %arg1: tensor<1x32xf32>) -> tensor<1x32xf32> {
    // CHECK-LABEL: func.func @tt_lang_op_simple
    // CHECK: ttir.tt_lang_op
    // CHECK-SAME: arg_roles = "in,out"
    // CHECK-SAME: kernel_id = "pkg.softmax::v1"
    // CHECK-SAME: version_tag = "1.0"
    // shard_spec defaults to "" and is elided when empty.
    // CHECK-NOT: shard_spec
    %0 = stablehlo.custom_call @tt.tt_lang_op(%arg0, %arg1) {
      api_version = 0 : i32,
      mhlo.frontend_attributes = {
        kernel_id = "pkg.softmax::v1",
        version_tag = "1.0",
        arg_roles = "in,out"
      }
    } : (tensor<1x32xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>
    return %0 : tensor<1x32xf32>
  }

  // Two inputs (one tagged `out`), one output, with shard_spec set.
  func.func @tt_lang_op_with_shard_spec(
      %arg0: tensor<8x16xbf16>,
      %arg1: tensor<8x16xbf16>) -> tensor<8x16xbf16> {
    // CHECK-LABEL: func.func @tt_lang_op_with_shard_spec
    // CHECK: ttir.tt_lang_op
    // CHECK-SAME: arg_roles = "in,out"
    // CHECK-SAME: kernel_id = "pkg.fused_add::v2"
    // CHECK-SAME: shard_spec = "{\22axis\22:0}"
    // CHECK-SAME: version_tag = "2.1"
    %0 = stablehlo.custom_call @tt.tt_lang_op(%arg0, %arg1) {
      api_version = 0 : i32,
      mhlo.frontend_attributes = {
        kernel_id = "pkg.fused_add::v2",
        version_tag = "2.1",
        arg_roles = "in,out",
        shard_spec = "{\22axis\22:0}"
      }
    } : (tensor<8x16xbf16>, tensor<8x16xbf16>) -> tensor<8x16xbf16>
    return %0 : tensor<8x16xbf16>
  }

  // Multi-output kernel: three operands (two `in`, one `out`) plus a second
  // result that the kernel writes to. The number of `out` tokens in
  // `arg_roles` matches the number of stablehlo results.
  func.func @tt_lang_op_multi_out(
      %arg0: tensor<4x4xf32>,
      %arg1: tensor<4x4xf32>,
      %arg2: tensor<4x4xf32>,
      %arg3: tensor<4xf32>)
      -> (tensor<4x4xf32>, tensor<4xf32>) {
    // CHECK-LABEL: func.func @tt_lang_op_multi_out
    // CHECK: ttir.tt_lang_op
    // CHECK-SAME: arg_roles = "in,in,out,out"
    // CHECK-SAME: kernel_id = "pkg.dual_out::v1"
    // CHECK-SAME: version_tag = "1.0"
    %0:2 = stablehlo.custom_call @tt.tt_lang_op(%arg0, %arg1, %arg2, %arg3) {
      api_version = 0 : i32,
      mhlo.frontend_attributes = {
        kernel_id = "pkg.dual_out::v1",
        version_tag = "1.0",
        arg_roles = "in,in,out,out"
      }
    } : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>)
        -> (tensor<4x4xf32>, tensor<4xf32>)
    return %0#0, %0#1 : tensor<4x4xf32>, tensor<4xf32>
  }
}
