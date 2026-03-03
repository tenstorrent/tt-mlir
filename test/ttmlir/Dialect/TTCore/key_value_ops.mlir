// RUN: ttmlir-opt -o %t %s
// RUN: FileCheck %s --input-file=%t

// ===== SetKeyValueOp =====

// CHECK-LABEL: func.func @set_key_value_string_key
func.func @set_key_value_string_key(%dict: !ttcore.dict, %val: tensor<32x32xbf16>) {
  // CHECK: ttcore.set_key_value %{{.*}}["key"] = %{{.*}} : !ttcore.dict, tensor<32x32xbf16>
  ttcore.set_key_value %dict["key"] = %val : !ttcore.dict, tensor<32x32xbf16>
  return
}

// CHECK-LABEL: func.func @set_key_value_index_key
func.func @set_key_value_index_key(%dict: !ttcore.dict, %val: tensor<32x32xbf16>) {
  // CHECK: ttcore.set_key_value %{{.*}}[3 : index] = %{{.*}} : !ttcore.dict, tensor<32x32xbf16>
  ttcore.set_key_value %dict[3 : index] = %val : !ttcore.dict, tensor<32x32xbf16>
  return
}

// CHECK-LABEL: func.func @set_key_value_multiple_values
func.func @set_key_value_multiple_values(%dict: !ttcore.dict, %v0: tensor<32x32xbf16>, %v1: tensor<64x64xbf16>) {
  // CHECK: ttcore.set_key_value %{{.*}}["multi"] = %{{.*}}, %{{.*}} : !ttcore.dict, tensor<32x32xbf16>, tensor<64x64xbf16>
  ttcore.set_key_value %dict["multi"] = %v0, %v1 : !ttcore.dict, tensor<32x32xbf16>, tensor<64x64xbf16>
  return
}

// ===== GetKeyValueOp =====

// CHECK-LABEL: func.func @get_key_value_string_key
func.func @get_key_value_string_key(%dict: !ttcore.dict) -> tensor<32x32xbf16> {
  // CHECK: ttcore.get_key_value %{{.*}}["key"] : (!ttcore.dict) -> tensor<32x32xbf16>
  %0 = ttcore.get_key_value %dict["key"] : (!ttcore.dict) -> tensor<32x32xbf16>
  return %0 : tensor<32x32xbf16>
}

// CHECK-LABEL: func.func @get_key_value_index_key
func.func @get_key_value_index_key(%dict: !ttcore.dict) -> tensor<32x32xbf16> {
  // CHECK: ttcore.get_key_value %{{.*}}[3 : index] : (!ttcore.dict) -> tensor<32x32xbf16>
  %0 = ttcore.get_key_value %dict[3 : index] : (!ttcore.dict) -> tensor<32x32xbf16>
  return %0 : tensor<32x32xbf16>
}

// CHECK-LABEL: func.func @get_key_value_multiple_results
func.func @get_key_value_multiple_results(%dict: !ttcore.dict) -> (tensor<32x32xbf16>, tensor<64x64xbf16>) {
  // CHECK: ttcore.get_key_value %{{.*}}["multi"] : (!ttcore.dict) -> (tensor<32x32xbf16>, tensor<64x64xbf16>)
  %0, %1 = ttcore.get_key_value %dict["multi"] : (!ttcore.dict) -> (tensor<32x32xbf16>, tensor<64x64xbf16>)
  return %0, %1 : tensor<32x32xbf16>, tensor<64x64xbf16>
}

// ===== TTCore_DictType =====

// CHECK-LABEL: func.func @dict_as_func_arg_and_result
func.func @dict_as_func_arg_and_result(%dict: !ttcore.dict) -> !ttcore.dict {
  // CHECK: return %{{.*}} : !ttcore.dict
  return %dict : !ttcore.dict
}
