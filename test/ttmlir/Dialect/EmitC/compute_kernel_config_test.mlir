// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path% compute-cfg-math-fidelity=lofi" --ttnn-common-to-emitc-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK: ::MathFidelity::LoFi
func.func @sum_with_compute_config(%arg0: tensor<512x1024xbf16>) -> tensor<512x1xbf16> {
  %0 = "ttir.sum"(%arg0) <{dim_arg = [1: i32], keep_dim = true}> : (tensor<512x1024xbf16>) -> tensor<512x1xbf16>
  return %0 : tensor<512x1xbf16>
}

// CHECK: ::MathFidelity::LoFi
func.func @mean_with_compute_config(%arg0: tensor<512x1024xbf16>) -> tensor<512x1xbf16> {
  %0 = "ttir.mean"(%arg0) <{dim_arg = [1: i32], keep_dim = true}> : (tensor<512x1024xbf16>) -> tensor<512x1xbf16>
  return %0 : tensor<512x1xbf16>
}

// CHECK: ::MathFidelity::LoFi
func.func @max_with_compute_config(%arg0: tensor<512x1024xbf16>) -> tensor<512x1xbf16> {
  %0 = "ttir.max"(%arg0) <{dim_arg = [1: i32], keep_dim = true}> : (tensor<512x1024xbf16>) -> tensor<512x1xbf16>
  return %0 : tensor<512x1xbf16>
}

// CHECK: ::MathFidelity::LoFi
func.func @min_with_compute_config(%arg0: tensor<512x1024xbf16>) -> tensor<512x1xbf16> {
  %0 = "ttir.min"(%arg0) <{dim_arg = [1: i32], keep_dim = true}> : (tensor<512x1024xbf16>) -> tensor<512x1xbf16>
  return %0 : tensor<512x1xbf16>
}

// CHECK: ::MathFidelity::LoFi
func.func @softmax_with_compute_config(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
  %0 = "ttir.softmax"(%arg0) <{dimension = 1 : si32}> : (tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
  return %0 : tensor<512x1024xbf16>
}

// CHECK: ::MathFidelity::LoFi
func.func @matmul_with_compute_config(%arg0: tensor<512x1024xbf16>, %arg1: tensor<1024x512xbf16>) -> tensor<512x512xbf16> {
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<512x1024xbf16>, tensor<1024x512xbf16>) -> tensor<512x512xbf16>
  return %0 : tensor<512x512xbf16>
}

// CHECK: ::MathFidelity::LoFi
func.func @linear_with_compute_config(%arg0: tensor<512x1024xbf16>, %arg1: tensor<1024x512xbf16>, %arg2: tensor<512xbf16>) -> tensor<512x512xbf16> {
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<512x1024xbf16>, tensor<1024x512xbf16>, tensor<512xbf16>) -> tensor<512x512xbf16>
  return %0 : tensor<512x512xbf16>
}

// rmsNorm has HiFi4 as default math fidelity
// CHECK: ::MathFidelity::HiFi4
func.func @rms_norm_with_compute_config(%arg0: tensor<32x2048xbf16>, %arg1: tensor<2048xbf16>) -> tensor<32x2048xbf16> {
  %0 = "ttir.rms_norm"(%arg0, %arg1) <{normalized_shape = array<i64: 2048>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x2048xbf16>, tensor<2048xbf16>) -> tensor<32x2048xbf16>
  return %0 : tensor<32x2048xbf16>
}

// CHECK: ::MathFidelity::LoFi
func.func @batch_norm_inference_with_compute_config(%arg0: tensor<2x2x2x2xbf16>, %arg1: tensor<2xbf16>, %arg2: tensor<2xbf16>, %arg3: tensor<2xbf16>, %arg4: tensor<2xbf16>) -> tensor<2x2x2x2xbf16> {
  %0 = "ttir.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4) <{dimension = 1 : i32, epsilon = 1.000000e-05 : f32}> : (tensor<2x2x2x2xbf16>, tensor<2xbf16>, tensor<2xbf16>, tensor<2xbf16>, tensor<2xbf16>) -> tensor<2x2x2x2xbf16>
  return %0 : tensor<2x2x2x2xbf16>
}

// CHECK: ::MathFidelity::LoFi
func.func @batch_norm_training_with_compute_config(%arg0: tensor<2x2x2x2xbf16>, %arg1: tensor<2xbf16>, %arg2: tensor<2xbf16>, %arg3: tensor<2xbf16>, %arg4: tensor<2xbf16>) -> (tensor<2x2x2x2xbf16>, tensor<2xbf16>, tensor<2xbf16>) {
  %0:3 = "ttir.batch_norm_training"(%arg0, %arg1, %arg2, %arg3, %arg4) <{dimension = 1 : i32, epsilon = 1.000000e-05 : f32, momentum = 1.000000e-01 : f32}> : (tensor<2x2x2x2xbf16>, tensor<2xbf16>, tensor<2xbf16>, tensor<2xbf16>, tensor<2xbf16>) -> (tensor<2x2x2x2xbf16>, tensor<2xbf16>, tensor<2xbf16>)
  return %0#0, %0#1, %0#2 : tensor<2x2x2x2xbf16>, tensor<2xbf16>, tensor<2xbf16>
}
