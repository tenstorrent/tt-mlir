// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --decompose-custom-call-tuples --verify-diagnostics %s

// Cases where a tuple-returning `@tt.tt_lang_op` cannot be split faithfully.
// The pass reports them here rather than leaving a tuple behind for Shardy to
// reject later with a diagnostic that points at propagation instead of the
// custom call.

// A user other than get_tuple_element consumes the tuple value itself, which is
// exactly what Shardy cannot accept, so there is nothing useful to rewire it to.
module {
  func.func @non_get_tuple_element_user(%arg0: tensor<4x4xf32>) -> tuple<tensor<4x4xf32>, tensor<4xf32>> {
    // expected-error @+1 {{expected every user to be a stablehlo.get_tuple_element, but found 'func.return'}}
    %0 = stablehlo.custom_call @tt.tt_lang_op(%arg0) {
      api_version = 0 : i32,
      mhlo.frontend_attributes = {
        arg_roles = "in,out,out",
        kernel_id = "pkg.dual_out::v1",
        version_tag = "1.0"
      }
    } : (tensor<4x4xf32>) -> tuple<tensor<4x4xf32>, tensor<4xf32>>
    return %0 : tuple<tensor<4x4xf32>, tensor<4xf32>>
  }
}

// -----

// The split flattens exactly one level, so a nested tuple would leave a
// tuple-typed result behind.
module {
  func.func @nested_tuple(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
    // expected-error @+1 {{cannot decompose nested tuple result}}
    %0 = stablehlo.custom_call @tt.tt_lang_op(%arg0) {
      api_version = 0 : i32,
      mhlo.frontend_attributes = {
        arg_roles = "in,out,out",
        kernel_id = "pkg.nested::v1",
        version_tag = "1.0"
      }
    } : (tensor<4x4xf32>) -> tuple<tuple<tensor<4x4xf32>>, tensor<4xf32>>
    %1 = stablehlo.get_tuple_element %0[1] : (tuple<tuple<tensor<4x4xf32>>, tensor<4xf32>>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}

// -----

// `output_operand_aliases` addresses results through tuple indices. A
// single-element tuple splits into one plain tensor result, so index [0] no
// longer resolves and the aliasing contract cannot be carried over.
module {
  func.func @single_element_tuple_with_aliases(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // expected-error @+1 {{cannot decompose single-element tuple result}}
    %0 = stablehlo.custom_call @tt.tt_lang_op(%arg0) {
      api_version = 0 : i32,
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>
      ],
      mhlo.frontend_attributes = {
        arg_roles = "in,out",
        kernel_id = "pkg.aliased::v1",
        version_tag = "1.0"
      }
    } : (tensor<4x4xf32>) -> tuple<tensor<4x4xf32>>
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<4x4xf32>>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}
