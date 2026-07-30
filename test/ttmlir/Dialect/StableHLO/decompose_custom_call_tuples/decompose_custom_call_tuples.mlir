// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --decompose-custom-call-tuples -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test stablehlo.tuple op is eliminated by forwarding operands directly.
module @TupleForward {
  func.func @main(%arg0: tensor<3x3xf32>, %arg1: tensor<3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>) {
    // CHECK-NOT: stablehlo.tuple
    %0 = stablehlo.tuple %arg0, %arg1 : tuple<tensor<3x3xf32>, tensor<3xf32>>
    // CHECK-NOT: stablehlo.get_tuple_element
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<3x3xf32>, tensor<3xf32>>) -> tensor<3x3xf32>
    %2 = stablehlo.get_tuple_element %0[1] : (tuple<tensor<3x3xf32>, tensor<3xf32>>) -> tensor<3xf32>
    // CHECK: return %arg0, %arg1 : tensor<3x3xf32>, tensor<3xf32>
    return %1, %2 : tensor<3x3xf32>, tensor<3xf32>
  }
}

// -----

// Test stablehlo.tuple with partial use forwards only the used operand.
module @TuplePartialUse {
  func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4xi32>, %arg2: tensor<i32>) -> tensor<4xi32> {
    // CHECK-NOT: stablehlo.tuple
    %0 = stablehlo.tuple %arg0, %arg1, %arg2 : tuple<tensor<4x4xf32>, tensor<4xi32>, tensor<i32>>
    // CHECK-NOT: stablehlo.get_tuple_element
    %1 = stablehlo.get_tuple_element %0[1] : (tuple<tensor<4x4xf32>, tensor<4xi32>, tensor<i32>>) -> tensor<4xi32>
    // CHECK: return %arg1 : tensor<4xi32>
    return %1 : tensor<4xi32>
  }
}

// -----

// A tt-lang kernel with two "out" roles arrives as a single tuple-typed result
// and must become a multi-result custom call, with the frontend attributes that
// carry the kernel metadata preserved.
module @TTLangMultiOutput {
  func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4xf32>) {
    // CHECK: %[[CALL:.*]]:2 = stablehlo.custom_call @tt.tt_lang_op(%arg0, %arg1)
    // CHECK-SAME: arg_roles = "in,in,out,out"
    // CHECK-SAME: kernel_id = "pkg.dual_out::v1"
    // CHECK-SAME: version_tag = "1.0"
    // CHECK-SAME: -> (tensor<4x4xf32>, tensor<4xf32>)
    // CHECK-NOT: stablehlo.get_tuple_element
    %0 = stablehlo.custom_call @tt.tt_lang_op(%arg0, %arg1) {
      api_version = 0 : i32,
      mhlo.frontend_attributes = {
        arg_roles = "in,in,out,out",
        kernel_id = "pkg.dual_out::v1",
        version_tag = "1.0"
      }
    } : (tensor<4x4xf32>, tensor<4x4xf32>) -> tuple<tensor<4x4xf32>, tensor<4xf32>>
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<4x4xf32>, tensor<4xf32>>) -> tensor<4x4xf32>
    %2 = stablehlo.get_tuple_element %0[1] : (tuple<tensor<4x4xf32>, tensor<4xf32>>) -> tensor<4xf32>
    // CHECK: return %[[CALL]]#0, %[[CALL]]#1
    return %1, %2 : tensor<4x4xf32>, tensor<4xf32>
  }
}

// -----

// A single-element tuple is still a tuple as far as Shardy is concerned, so it
// is split into a one-result custom call rather than left alone.
module @TTLangSingleElementTuple {
  func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK: %[[CALL:.*]] = stablehlo.custom_call @tt.tt_lang_op(%arg0)
    // CHECK-SAME: -> tensor<4x4xf32>
    // CHECK-NOT: tuple
    %0 = stablehlo.custom_call @tt.tt_lang_op(%arg0) {
      api_version = 0 : i32,
      mhlo.frontend_attributes = {
        arg_roles = "in,out",
        kernel_id = "pkg.single_out::v1",
        version_tag = "1.0"
      }
    } : (tensor<4x4xf32>) -> tuple<tensor<4x4xf32>>
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<4x4xf32>>) -> tensor<4x4xf32>
    // CHECK: return %[[CALL]]
    return %1 : tensor<4x4xf32>
  }
}

// -----

// Layout attributes are dropped as a pair: stablehlo requires operand_layouts
// and result_layouts to be set together or not at all, and nothing downstream
// of this pass consults them on the tt-lang path.
module @TTLangLayoutsDropped {
  func.func @main(%arg0: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4xf32>) {
    // CHECK: stablehlo.custom_call @tt.tt_lang_op
    // CHECK-NOT: operand_layouts
    // CHECK-NOT: result_layouts
    %0 = stablehlo.custom_call @tt.tt_lang_op(%arg0) {
      api_version = 0 : i32,
      operand_layouts = [dense<[1, 0]> : tensor<2xindex>],
      result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[0]> : tensor<1xindex>],
      mhlo.frontend_attributes = {
        arg_roles = "in,out,out",
        kernel_id = "pkg.layouts::v1",
        version_tag = "1.0"
      }
    } : (tensor<4x4xf32>) -> tuple<tensor<4x4xf32>, tensor<4xf32>>
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<4x4xf32>, tensor<4xf32>>) -> tensor<4x4xf32>
    %2 = stablehlo.get_tuple_element %0[1] : (tuple<tensor<4x4xf32>, tensor<4xf32>>) -> tensor<4xf32>
    return %1, %2 : tensor<4x4xf32>, tensor<4xf32>
  }
}

// -----

// Rewrite 2 is scoped to @tt.tt_lang_op. Other tuple-returning custom calls are
// unpacked by their own StableHLOToTTIR patterns and must be left untouched.
module @NonTTLangTupleUntouched {
  func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK: stablehlo.custom_call @some.other.target
    // CHECK-SAME: -> tuple<tensor<4x4xf32>, tensor<4xf32>>
    %0 = stablehlo.custom_call @some.other.target(%arg0) {
      api_version = 0 : i32
    } : (tensor<4x4xf32>) -> tuple<tensor<4x4xf32>, tensor<4xf32>>
    // CHECK: stablehlo.get_tuple_element
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<4x4xf32>, tensor<4xf32>>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}

// -----

// A tt-lang call that already uses multi-result form is left as-is.
module @TTLangAlreadyMultiResult {
  func.func @main(%arg0: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4xf32>) {
    // CHECK: %[[CALL:.*]]:2 = stablehlo.custom_call @tt.tt_lang_op(%arg0)
    // CHECK-SAME: -> (tensor<4x4xf32>, tensor<4xf32>)
    %0:2 = stablehlo.custom_call @tt.tt_lang_op(%arg0) {
      api_version = 0 : i32,
      mhlo.frontend_attributes = {
        arg_roles = "in,out,out",
        kernel_id = "pkg.dual_out::v1",
        version_tag = "1.0"
      }
    } : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4xf32>)
    // CHECK: return %[[CALL]]#0, %[[CALL]]#1
    return %0#0, %0#1 : tensor<4x4xf32>, tensor<4xf32>
  }
}
