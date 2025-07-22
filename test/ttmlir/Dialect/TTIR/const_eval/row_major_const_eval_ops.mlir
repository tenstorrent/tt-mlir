// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-row-major-const-eval-ops=true" %s | FileCheck %s

module {
  // CHECK-LABEL: sanity_test_const_eval_0
  // 1. Check that const eval function converts inputs to RM.
  // CHECK: ttnn.to_layout
  // CHECK-SAME: layout = #ttnn.layout<row_major>

  // 2. Check that we are updating layout attribute.
  // CHECK: ttnn.zeros
  // CHECK-SAME: layout = #ttnn.layout<row_major>
  // CHECK: ttnn.add

  // 3. Check that return type is tiled.
  // CHECK: %[[RET:.*]] = "ttnn.to_layout
  // CHECK-SAME: layout = #ttnn.layout<tile>
  // CHECK: return %[[RET]]

  // CHECK-LABEL: sanity_test(
  func.func @sanity_test(%arg0: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32xbf16> {
    %0 = ttir.empty() : tensor<32xbf16>
    %1 = "ttir.zeros"() <{ shape = array<i32 : 32>}> : () -> tensor<32xbf16>
    %2 = "ttir.add"(%1, %arg0, %0) : (tensor<32xbf16>, tensor<32xbf16>, tensor<32xbf16>) -> tensor<32xbf16>
    return %2 : tensor<32xbf16>
  }
}
