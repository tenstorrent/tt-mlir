// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verifies that a *functional* `stablehlo.custom_call @tt.tt_lang_op`
// (operands = "in" tensors only) is lowered to DPS `ttir.tt_lang_op`, with
// one `ttir.empty` synthesized per result as the trailing DPS init, and
// that metadata (`kernel_id`, `version_tag`, `arg_roles`, `shard_spec`) is
// lifted from `mhlo.frontend_attributes` onto the new op.
//
// Legacy DPS-on-SHLO (outs as custom_call operands) is intentionally rejected
// by the conversion pattern and is not covered here.

module {
  // Functional SHLO: one `in` operand -> one result. Conversion synthesizes
  // a matching `ttir.empty` DPS init from the result type.
  func.func @tt_lang_op_simple(
      %arg0: tensor<1x32xf32>) -> tensor<1x32xf32> {
    // CHECK-LABEL: func.func @tt_lang_op_simple
    // CHECK: %[[OUT:.*]] = ttir.empty() : tensor<1x32xf32>
    // CHECK: ttir.tt_lang_op(%{{.*}}, %[[OUT]])
    // CHECK-SAME: arg_roles = "in,out"
    // CHECK-SAME: kernel_id = "pkg.softmax::v1"
    // CHECK-SAME: version_tag = "1.0"
    // shard_spec defaults to "" and is elided when empty.
    // CHECK-NOT: shard_spec
    %0 = stablehlo.custom_call @tt.tt_lang_op(%arg0) {
      api_version = 0 : i32,
      mhlo.frontend_attributes = {
        kernel_id = "pkg.softmax::v1",
        version_tag = "1.0",
        arg_roles = "in,out"
      }
    } : (tensor<1x32xf32>) -> tensor<1x32xf32>
    return %0 : tensor<1x32xf32>
  }

  // Same functional shape with shard_spec set.
  func.func @tt_lang_op_with_shard_spec(
      %arg0: tensor<8x16xbf16>) -> tensor<8x16xbf16> {
    // CHECK-LABEL: func.func @tt_lang_op_with_shard_spec
    // CHECK: %[[OUT:.*]] = ttir.empty() : tensor<8x16xbf16>
    // CHECK: ttir.tt_lang_op(%{{.*}}, %[[OUT]])
    // CHECK-SAME: arg_roles = "in,out"
    // CHECK-SAME: kernel_id = "pkg.fused_add::v2"
    // CHECK-SAME: shard_spec = "{\22axis\22:0}"
    // CHECK-SAME: version_tag = "2.1"
    %0 = stablehlo.custom_call @tt.tt_lang_op(%arg0) {
      api_version = 0 : i32,
      mhlo.frontend_attributes = {
        kernel_id = "pkg.fused_add::v2",
        version_tag = "2.1",
        arg_roles = "in,out",
        shard_spec = "{\"axis\":0}"
      }
    } : (tensor<8x16xbf16>) -> tensor<8x16xbf16>
    return %0 : tensor<8x16xbf16>
  }

  // Multi-output: two functional `in` operands, two results. Conversion
  // synthesizes one `ttir.empty` per result; `arg_roles` still lists the
  // logical DPS outs.
  func.func @tt_lang_op_multi_out(
      %arg0: tensor<4x4xf32>,
      %arg1: tensor<4x4xf32>)
      -> (tensor<4x4xf32>, tensor<4xf32>) {
    // CHECK-LABEL: func.func @tt_lang_op_multi_out
    // CHECK: %[[OUT0:.*]] = ttir.empty() : tensor<4x4xf32>
    // CHECK: %[[OUT1:.*]] = ttir.empty() : tensor<4xf32>
    // CHECK: ttir.tt_lang_op(%{{.*}}, %{{.*}}, %[[OUT0]], %[[OUT1]])
    // CHECK-SAME: arg_roles = "in,in,out,out"
    // CHECK-SAME: kernel_id = "pkg.dual_out::v1"
    // CHECK-SAME: version_tag = "1.0"
    %0:2 = stablehlo.custom_call @tt.tt_lang_op(%arg0, %arg1) {
      api_version = 0 : i32,
      mhlo.frontend_attributes = {
        kernel_id = "pkg.dual_out::v1",
        version_tag = "1.0",
        arg_roles = "in,in,out,out"
      }
    } : (tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4xf32>)
    return %0#0, %0#1 : tensor<4x4xf32>, tensor<4xf32>
  }
}
