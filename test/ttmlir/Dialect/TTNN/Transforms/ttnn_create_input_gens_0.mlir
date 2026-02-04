// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" --ttcore-unwrap-device-module --ttnn-tuplify-tensors --ttnn-create-input-gens -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK-LABEL: @add_const_eval_0
  // CHECK-LABEL: @add(

  // CHECK-NOT: @create_inputs_for_func_no_inputs
  // CHECK-LABEL: @create_inputs_for_add
  // CHECK: %[[ARG0:.*]] = "ttnn.ones"
  // CHECK: %[[ARG1:.*]] = "ttnn.ones"
  // CHECK: %[[RES:.*]] = ttcore.tuple %[[ARG0]], %[[ARG1]]

  func.func @add(%arg0 : tensor<32x32xbf16> { ttcore.argument_type = #ttcore.argument_type<constant>}, %arg1 : tensor<32x32xbf16> { ttcore.argument_type = #ttcore.argument_type<input> }) -> tensor<32x32xbf16> {
    %0 = "ttir.add"(%arg0, %arg0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1 = "ttir.subtract"(%arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }

  func.func @func_no_inputs() -> (tensor<64x64xbf16>) {
    %0 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<64x64xbf16>}> :
        () -> tensor<64x64xbf16>
    return %0 : tensor<64x64xbf16>
  }
}
