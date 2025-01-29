// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
func.func @sum(%arg0: tensor<1x1x512x64xbf16>) -> tensor<1x1x512x1xbf16> {
  %0 = tensor.empty() : tensor<1x1x512x1xbf16>
  // CHECK: %[[C:.*]] = "ttnn.sum"[[C:.*]]
  %1 = "ttir.sum"(%arg0, %0) <{dim_arg = [-1: i32], keep_dim = true}> : (tensor<1x1x512x64xbf16>, tensor<1x1x512x1xbf16>) -> tensor<1x1x512x1xbf16>
  return %1 : tensor<1x1x512x1xbf16>
}

func.func @sum_last_2_dims(%arg0: tensor<1x32x512x64xbf16>) -> tensor<1x32x1x1xbf16> {
  %0 = tensor.empty() : tensor<1x32x1x1xbf16>
  // CHECK: %[[C:.*]] = "ttnn.sum"[[C:.*]]
  %1 = "ttir.sum"(%arg0, %0) <{dim_arg = [-1: i32, -2: i32], keep_dim = true}> : (tensor<1x32x512x64xbf16>, tensor<1x32x1x1xbf16>) -> tensor<1x32x1x1xbf16>
  return %1 : tensor<1x32x1x1xbf16>
}

func.func @sum_first_dim(%arg0: tensor<64x10xf32>) -> tensor<1x10xf32> {
  %0 = tensor.empty() : tensor<1x10xf32>
  %1 = "ttir.sum"(%arg0, %0) <{dim_arg = [-2 : i32], keep_dim = true}> : (tensor<64x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
  return %1: tensor<1x10xf32>
}

func.func @mean(%arg0: tensor<1x1x512x64xbf16>) -> tensor<1x1x512x1xbf16> {
  %0 = tensor.empty() : tensor<1x1x512x1xbf16>
  // CHECK: %[[C:.*]] = "ttnn.mean"[[C:.*]]
  %1 = "ttir.mean"(%arg0, %0) <{dim_arg = [-1: i32], keep_dim = true}> : (tensor<1x1x512x64xbf16>, tensor<1x1x512x1xbf16>) -> tensor<1x1x512x1xbf16>
  return %1 : tensor<1x1x512x1xbf16>
}

func.func @mean_last_2_dims(%arg0: tensor<1x32x512x64xbf16>) -> tensor<1x32x1x1xbf16> {
  %0 = tensor.empty() : tensor<1x32x1x1xbf16>
  // CHECK: %[[C:.*]] = "ttnn.mean"[[C:.*]]
  %1 = "ttir.mean"(%arg0, %0) <{dim_arg = [-1: i32, -2: i32], keep_dim = true}> : (tensor<1x32x512x64xbf16>, tensor<1x32x1x1xbf16>) -> tensor<1x32x1x1xbf16>
  return %1 : tensor<1x32x1x1xbf16>
}

func.func @max(%arg0: tensor<1x1x512x64xbf16>) -> tensor<1x1x512x1xbf16> {
  %0 = tensor.empty() : tensor<1x1x512x1xbf16>
  // CHECK: %[[C:.*]] = "ttnn.max"[[C:.*]]
  %1 = "ttir.max"(%arg0, %0) <{dim_arg = [-1: i32], keep_dim = true}> : (tensor<1x1x512x64xbf16>, tensor<1x1x512x1xbf16>) -> tensor<1x1x512x1xbf16>
  return %1 : tensor<1x1x512x1xbf16>
}

func.func @max_last_2_dims(%arg0: tensor<1x32x512x64xbf16>) -> tensor<1x32x1x1xbf16> {
  %0 = tensor.empty() : tensor<1x32x1x1xbf16>
  // CHECK: %[[C:.*]] = "ttnn.max"[[C:.*]]
  %1 = "ttir.max"(%arg0, %0) <{dim_arg = [-1: i32, -2: i32], keep_dim = true}> : (tensor<1x32x512x64xbf16>, tensor<1x32x1x1xbf16>) -> tensor<1x32x1x1xbf16>
  return %1 : tensor<1x32x1x1xbf16>
}
