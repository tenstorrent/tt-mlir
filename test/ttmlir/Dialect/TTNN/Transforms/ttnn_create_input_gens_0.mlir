// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" --ttcore-unwrap-device-module --ttnn-tuplify-tensors --ttnn-create-input-gens -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK-LABEL: @add_const_eval_0
  // CHECK-LABEL: @add(

  // CHECK-LABEL: @create_inputs_for_add
  // CHECK: %[[ARG0:.*]] = "ttnn.ones"
  // CHECK: %[[ARG1:.*]] = "ttnn.ones"
  // CHECK: %[[RES:.*]] = ttcore.tuple %[[ARG0]], %[[ARG1]]

  func.func @add(%arg0 : tensor<32x32xbf16> { ttcore.argument_type = #ttcore.argument_type<constant>}, %arg1 : tensor<32x32xbf16> { ttcore.argument_type = #ttcore.argument_type<input> }) -> tensor<32x32xbf16> {
    %0 = ttir.empty() : tensor<32x32xbf16>
    %1 = "ttir.add"(%arg0, %arg0, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = ttir.empty() : tensor<32x32xbf16>
    %3 = "ttir.subtract"(%arg1, %1, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %3 : tensor<32x32xbf16>
  }
}
