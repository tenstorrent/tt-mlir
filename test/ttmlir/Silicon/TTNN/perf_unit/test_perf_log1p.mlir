// REQUIRES: num-chips-1 || num-chips-2
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
func.func @log1p(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: [[VAL0:%[0-9]+]] = "ttnn.empty"(%{{[0-9]+}}) <{dtype = {{.*}}, layout = {{.*}}, memory_config = {{.*}}, <{{.*}}>>, shape = #ttnn.shape<[[TENSOR_SHAPE:[0-9]+x[0-9]+]]>}>
  %1 = "ttir.log1p"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: %{{[0-9]+}} = "ttnn.log1p"(%{{[0-9]+}}, [[VAL0]]) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<[[TENSOR_SHAPE]]x{{.*}}, {{.*}}>, tensor<[[TENSOR_SHAPE]]x{{.*}}, {{.*}}) -> tensor<[[TENSOR_SHAPE]]x{{.*}}, {{.*}}>
  return %1 : tensor<64x128xf32>
  // CHECK: return %{{[0-9]+}} : tensor<[[TENSOR_SHAPE]]xf32, {{.*}}>
}
