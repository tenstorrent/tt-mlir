// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>

func.func @add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.add"[[C:.*]]
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @ceil(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[C:.*]] = "ttnn.ceil"[[C:.*]]
  %1 = "ttir.ceil"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

func.func @clamp(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
  %0 = tensor.empty() : tensor<64x128xbf16>
  // CHECK: %[[DEVICE:.*]] = "ttnn.to_device"(%arg0,
  // CHECK: %[[LAYOUT:.*]] = "ttnn.to_layout"(%[[DEVICE]])
  // CHECK: = "ttnn.clamp"(%[[LAYOUT]])
  // CHECK-SAME: {max = 3.000000e+00 : f32, min = 2.000000e+00 : f32}
  // CHECK-SAME: [[TENSOR:tensor<64x128xbf16]], #ttnn_layout{{[0-9]+}}>) -> [[TENSOR]]
  %1 = "ttir.clamp"(%arg0, %0) <{max = 3.000000e+00 : f32, min = 2.000000e+00 : f32, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
  return %1 : tensor<64x128xbf16>
}

func.func @concat(%arg0: tensor<32x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<32x96xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<32x96xf32>
  // CHECK: %[[C:.*]] = "ttnn.concat"[[C:.*]]
  %1 = "ttir.concat"(%arg0, %arg1, %0) <{dim = 1 : si32, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<32x32xf32>, tensor<32x64xf32>, tensor<32x96xf32>) -> tensor<32x96xf32>
  return %1 : tensor<32x96xf32>
}

func.func @cosine(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[C:.*]] = "ttnn.cos"[[C:.*]]
  %1 = "ttir.cos"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

func.func @div(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.div"[[C:.*]]
  %1 = "ttir.div"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @floor(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %{{[0-9]+}} = "ttnn.empty"
  // CHECK-SAME: [[TENSOR:tensor<64x128xf32,]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %{{[0-9]+}} = "ttnn.floor"
  // CHECK-SAME: [[TENSOR]]
  // CHECK-SAME: [[TENSOR]]
  // CHECK-SAME: -> [[TENSOR]]
  %1 = "ttir.floor"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @is_finite(%arg0: tensor<64x128xf32>) -> tensor<64x128xbf16> {
  // CHECK: %[[C:.*]] = "ttnn.empty"
  // CHECK-SAME: [[TENSOR:tensor<64x128xbf16,]]
  %0 = tensor.empty() : tensor<64x128xbf16>
  // CHECK: %[[C:.*]] = "ttnn.isfinite"
  // CHECK-SAME: tensor<64x128xf32,
  // CHECK-SAME: [[TENSOR]]
  // CHECK-SAME: -> [[TENSOR]]
  %1 = "ttir.isfinite"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
  return %1 : tensor<64x128xbf16>
}

func.func @minimum(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"
  // CHECK-SAME: [[TENSOR:tensor<64x128xf32,]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.minimum"
  // CHECK-SAME: [[TENSOR]]
  // CHECK-SAME: [[TENSOR]]
  // CHECK-SAME: [[TENSOR]]
  // CHECK-SAME: -> [[TENSOR]]
  %1 = "ttir.minimum"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @ge(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.ge"[[C:.*]]
  %1 = "ttir.ge"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @maximum(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.maximum"[[C:.*]]
  %1 = "ttir.maximum"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @multiply(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.multiply"[[C:.*]]
  %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @negate(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[C:.*]] = "ttnn.neg"[[C:.*]]
  %1 = "ttir.neg"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

func.func @reciprocal(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.reciprocal"[[C:.*]]
  %1 = "ttir.reciprocal"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @relu(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.relu"[[C:.*]]
  %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @leaky_relu(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK: %[[C:.*]] = "ttnn.empty"
    %0 = tensor.empty() : tensor<64x128xf32>
    // CHECK: %[[C:.*]] = "ttnn.leaky_relu"
    %1 = "ttir.leaky_relu"(%arg0, %0) <{parameter = 0.01 : f32, operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
}

func.func @reshape(%arg0: tensor<4x2x32x32xbf16>) -> tensor<2x4x32x32xbf16> {
  %0 = tensor.empty() : tensor<2x4x32x32xbf16>
  // CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [2: i32, 4: i32, 32: i32, 32: i32] , operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<4x2x32x32xbf16>, tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16>
  return %1 : tensor<2x4x32x32xbf16>
}

func.func @squeeze(%arg0: tensor<1x2x1x32x32xbf16>) -> tensor<1x2x32x32xbf16> {
  %0 = tensor.empty() : tensor<1x2x32x32xbf16>
  // CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
  %1 = "ttir.squeeze"(%arg0, %0) <{dim = 2 : si32, operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<1x2x1x32x32xbf16>, tensor<1x2x32x32xbf16>) -> tensor<1x2x32x32xbf16>
  return %1 : tensor<1x2x32x32xbf16>
}

func.func @subtract(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.subtract"[[C:.*]]
  %1 = "ttir.subtract"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @rsqrt(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.rsqrt"[[C:.*]]
  %1 = "ttir.rsqrt"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @sigmoid(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.sigmoid"[[C:.*]]
  %1 = "ttir.sigmoid"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @sqrt(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.sqrt"[[C:.*]]
  %1 = "ttir.sqrt"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @sine(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[C:.*]] = "ttnn.sin"[[C:.*]]
  %1 = "ttir.sin"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

func.func @softmax(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
  %0 = tensor.empty() : tensor<512x1024xbf16>
  // CHECK: %[[C:.*]] = "ttnn.softmax"[[C:.*]]
  // Check for positive dimension attribute
  %1 = "ttir.softmax"(%arg0, %0) <{dimension = 1 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<512x1024xbf16>, tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
  %2 = tensor.empty() : tensor<512x1024xbf16>
  // CHECK: %[[C:.*]] = "ttnn.softmax"[[C:.*]]
  // Check for negative dimension attribute
  %3 = "ttir.softmax"(%1, %2) <{dimension = -1 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<512x1024xbf16>, tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
  return %3 : tensor<512x1024xbf16>
}

func.func @cbrt(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.cbrt"[[C:.*]]
  %1 = "ttir.cbrt"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @typecast(%arg0: tensor<64x128xf32>) -> tensor<64x128xbf16> {
  %0 = tensor.empty() : tensor<64x128xbf16>
  // CHECK: %[[C:.*]] = "ttnn.typecast"
  // CHECK-SAME: tensor<64x128xf32,
  // CHECK-SAME: tensor<64x128xbf16,
  %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
  return %1 : tensor<64x128xbf16>
}

func.func @log(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.log"[[C:.*]]
  %1 = "ttir.log"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @log1p(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: [[VAL0:%[0-9]+]] = "ttnn.empty"(%{{[0-9]+}}) <{dtype = {{.*}}, layout = {{.*}}, memory_config = {{.*}}, <{{.*}}>>, shape = #ttnn.shape<[[TENSOR_SHAPE:[0-9]+x[0-9]+]]>}>
  %1 = "ttir.log1p"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: %{{[0-9]+}} = "ttnn.log1p"(%{{[0-9]+}}, [[VAL0]]) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<[[TENSOR_SHAPE]]x{{.*}}, {{.*}}>, tensor<[[TENSOR_SHAPE]]x{{.*}}, {{.*}}) -> tensor<[[TENSOR_SHAPE]]x{{.*}}, {{.*}}>
  return %1 : tensor<64x128xf32>
  // CHECK: return %{{[0-9]+}} : tensor<[[TENSOR_SHAPE]]xf32, {{.*}}>
}

func.func @expm1(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: [[VAL0:%[0-9]+]] = "ttnn.empty"(%{{[0-9]+}}) <{dtype = {{.*}}, layout = {{.*}}, memory_config = {{.*}}, <{{.*}}>>, shape = #ttnn.shape<[[TENSOR_SHAPE:[0-9]+x[0-9]+]]>}>
  %1 = "ttir.expm1"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: %{{[0-9]+}} = "ttnn.expm1"(%{{[0-9]+}}, [[VAL0]]) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<[[TENSOR_SHAPE]]x{{.*}}, {{.*}}>, tensor<[[TENSOR_SHAPE]]x{{.*}}, {{.*}}) -> tensor<[[TENSOR_SHAPE]]x{{.*}}, {{.*}}>
  return %1 : tensor<64x128xf32>
  // CHECK: return %{{[0-9]+}} : tensor<[[TENSOR_SHAPE]]xf32, {{.*}}>
}

func.func @sign(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: [[VAL0:%[0-9]+]] = "ttnn.empty"(%{{[0-9]+}}) <{dtype = {{.*}}, layout = {{.*}}, memory_config = {{.*}}, <{{.*}}>>, shape = #ttnn.shape<[[TENSOR_SHAPE:[0-9]+x[0-9]+]]>}>
  %1 = "ttir.sign"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: %{{[0-9]+}} = "ttnn.sign"(%{{[0-9]+}}, [[VAL0]]) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<[[TENSOR_SHAPE]]x{{.*}}, {{.*}}>, tensor<[[TENSOR_SHAPE]]x{{.*}}, {{.*}}) -> tensor<[[TENSOR_SHAPE]]x{{.*}}, {{.*}}>
  return %1 : tensor<64x128xf32>
  // CHECK: return %{{[0-9]+}} : tensor<[[TENSOR_SHAPE]]xf32, {{.*}}>
}

func.func @remainder(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[EMPTY:.*]] = "ttnn.empty"{{.*}} -> tensor<32x32xf32, {{.*}}
  %1 = "ttir.remainder"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[REM:[0-9]+]] = "ttnn.remainder"({{.*}}, {{.*}}, %[[EMPTY]]){{.*}} -> tensor<32x32xf32, {{.*}}
  return %1 : tensor<32x32xf32>
  // CHECK: return {{.*}} : tensor<32x32xf32, {{.*}}
}

func.func @get_dimension_size(%arg0: tensor<13x21x3xf32>) -> tensor<1xi32> {
  %0 = "ttir.get_dimension_size"(%arg0) <{dimension = 1 : i32}> : (tensor<13x21x3xf32>) -> tensor<1xi32>
  // CHECK: [[VAL:%[0-9]+]] = "ttnn.full"(%{{[0-9]+}}) <{fillValue = 2.100000e+01 : f32}> : (!tt.device<#device>) -> tensor<1xi32, {{.*}}>
  return %0 : tensor<1xi32>
  // CHECK: return [[VAL]] : tensor<1xi32, {{.*}}>
}

func.func @test_where(%arg0: tensor<13x37xf32>, %arg1: tensor<13x37xf32>) -> tensor<13x37xf32> {
  %0 = tensor.empty() : tensor<13x37xbf16>
  %1 = "ttir.eq"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<13x37xf32>, tensor<13x37xf32>, tensor<13x37xbf16>) -> tensor<13x37xbf16>
  %2 = tensor.empty() : tensor<13x37xf32>
  %3 = "ttir.where"(%1, %arg0, %arg1, %2) <{operandSegmentSizes = array<i32: 3, 1>, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<13x37xbf16>, tensor<13x37xf32>, tensor<13x37xf32>, tensor<13x37xf32>) -> tensor<13x37xf32>
  // CHECK: %[[EMPTY:.*]] = "ttnn.empty"{{.*}}
  // CHECK: %[[VAL1:[0-9]+]] = "ttnn.eq"(%{{[0-9]+}}, %{{[0-9]+}}, %[[EMPTY]])
  // CHECK: %{{[0-9]+}} = "ttnn.where"(%[[VAL1]], %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}})
  return %3 : tensor<13x37xf32>
}

func.func @gelu(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: "ttnn.empty"
  // CHECK-SAME: tensor<64x128xf32,
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: "ttnn.gelu"
  // CHECK-SAME: tensor<64x128xf32,
  // CHECK-SAME: tensor<64x128xf32,
  // CHECK-SAME: tensor<64x128xf32,
  %1 = "ttir.gelu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @addint32(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
  %0 = tensor.empty() : tensor<64x128xi32>
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
  return %1 : tensor<64x128xi32>
}

func.func @scatter(%arg0: tensor<1x3x320x320xf32>, %arg1: tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32> {
  %0 = tensor.empty() : tensor<1x3x320x320xf32>
  %1 = tensor.empty() : tensor<1x1xi32>
  %2 = "ttir.scatter"(%arg0, %1, %arg1, %0) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile, #any_device_tile], scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 1, 2, 3>}> ({
  ^bb0(%arg3: tensor<1xf32>, %arg4: tensor<1xf32>):
    "ttir.yield"(%arg4) : (tensor<1xf32>) -> ()
  }) : (tensor<1x3x320x320xf32>, tensor<1x1xi32>, tensor<1x3x32x32xf32>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
  return %2 : tensor<1x3x320x320xf32>
}
