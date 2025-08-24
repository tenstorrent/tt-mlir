// RUN: ttmlir-opt --ttir-quant-data-type-conversion="target-bit-width=32" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir --check-prefix=CHECK-INT32
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t2.mlir %t.mlir

// This checks that the whole backend pipeline works with int16 quantization as well.
// RUN: ttmlir-opt --ttir-quant-data-type-conversion="target-bit-width=16" -o %t-int16.mlir %s
// RUN: FileCheck %s --input-file=%t-int16.mlir --check-prefix=CHECK-INT16
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o  %t2-int16.mlir %t-int16.mlir

func.func @quantize_datatype_transform_test(%arg0: tensor<1x3x320x320xf32>) -> tensor<1x3x320x320x!quant.uniform<i8:f32, 1.000000e-01>> {
    // CHECK-INT32-LABEL: func.func @quantize_datatype_transform_test
    // CHECK-INT32: %[[EMPTY:[0-9]+]] = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>
    // CHECK-INT32: %[[RET:[0-9]+]] = "ttir.quantize"(%arg0, %[[EMPTY]]) : (tensor<1x3x320x320xf32>, tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>
    // CHECK-INT32: return %[[RET]] : tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>

    // CHECK-INT16-LABEL: func.func @quantize_datatype_transform_test
    // CHECK-INT16: %[[EMPTY:[0-9]+]] = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i16:f32, 1.000000e-01>>
    // CHECK-INT16: %[[RET:[0-9]+]] = "ttir.quantize"(%arg0, %[[EMPTY]]) : (tensor<1x3x320x320xf32>, tensor<1x3x320x320x!quant.uniform<i16:f32, 1.000000e-01>>) -> tensor<1x3x320x320x!quant.uniform<i16:f32, 1.000000e-01>>
    // CHECK-INT16: return %[[RET]] : tensor<1x3x320x320x!quant.uniform<i16:f32, 1.000000e-01>>

    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i8:f32, 1.000000e-01>>
    %1 = "ttir.quantize"(%arg0, %0) : (tensor<1x3x320x320xf32>, tensor<1x3x320x320x!quant.uniform<i8:f32, 1.000000e-01>>) -> tensor<1x3x320x320x!quant.uniform<i8:f32, 1.000000e-01>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i8:f32, 1.000000e-01>>
  }

func.func @dequantize_datatype_transform_test(%arg0: tensor<1x3x320x320x!quant.uniform<i8:f32, 1.000000e-01>>) -> tensor<1x3x320x320xf32> {
  // CHECK-INT32-LABEL: func.func @dequantize_datatype_transform_test
  // CHECK-INT32: %[[EMPTY:[0-9]+]] = ttir.empty() : tensor<1x3x320x320xf32>
  // CHECK-INT32: %[[RET:[0-9]+]] = "ttir.dequantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
  // CHECK-INT32: return %[[RET]] : tensor<1x3x320x320xf32>

  // CHECK-INT16-LABEL: func.func @dequantize_datatype_transform_test
  // CHECK-INT16: %[[EMPTY:[0-9]+]] = ttir.empty() : tensor<1x3x320x320xf32>
  // CHECK-INT16: %[[RET:[0-9]+]] = "ttir.dequantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i16:f32, 1.000000e-01>>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
  // CHECK: return %[[RET]] : tensor<1x3x320x320xf32>

  %0 = ttir.empty() : tensor<1x3x320x320xf32>
  %1 = "ttir.dequantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i8:f32, 1.000000e-01>>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
  return %1 : tensor<1x3x320x320xf32>
}

func.func @requantize_datatype_transform_test(%arg0: tensor<1x3x320x320x!quant.uniform<i8:f32, 1.000000e-01>>) -> tensor<1x3x320x320x!quant.uniform<i8:f32, 2.000000e-01>> {
  // CHECK-INT32-LABEL: func.func @requantize_datatype_transform_test
  // CHECK-INT32: %[[EMPTY:[0-9]+]] = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>>
  // CHECK-INT32: %[[RET:[0-9]+]] = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>, tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>>
  // CHECK-INT32: return %[[RET]] : tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>>


  // CHECK-INT16-LABEL: func.func @requantize_datatype_transform_test
  // CHECK-INT16: %[[EMPTY:[0-9]+]] = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i16:f32, 2.000000e-01>>
  // CHECK-INT16: %[[RET:[0-9]+]] = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i16:f32, 1.000000e-01>>, tensor<1x3x320x320x!quant.uniform<i16:f32, 2.000000e-01>>) -> tensor<1x3x320x320x!quant.uniform<i16:f32, 2.000000e-01>>
  // CHECK-INT16: return %[[RET]] : tensor<1x3x320x320x!quant.uniform<i16:f32, 2.000000e-01>>

  %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i8:f32, 2.000000e-01>>
  %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i8:f32, 1.000000e-01>>, tensor<1x3x320x320x!quant.uniform<i8:f32, 2.000000e-01>>) -> tensor<1x3x320x320x!quant.uniform<i8:f32, 2.000000e-01>>
  return %1 : tensor<1x3x320x320x!quant.uniform<i8:f32, 2.000000e-01>>
}
