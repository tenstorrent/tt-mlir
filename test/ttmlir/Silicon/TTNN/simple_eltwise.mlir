// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>

func.func @subtract(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.subtract"[[C:.*]]
  %1 = "ttir.subtract"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @div(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.div"[[C:.*]]
  %1 = "ttir.div"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @multiply(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.multiply"[[C:.*]]
  %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @relu(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.relu"[[C:.*]]
  %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @ge(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.ge"[[C:.*]]
  %1 = "ttir.ge"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @concat(%arg0: tensor<32x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<32x96xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<32x96xf32>
  // CHECK: %[[C:.*]] = "ttnn.concat"[[C:.*]]
  %1 = "ttir.concat"(%arg0, %arg1, %0) <{dim = 1 : si32, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<32x32xf32>, tensor<32x64xf32>, tensor<32x96xf32>) -> tensor<32x96xf32>
  return %1 : tensor<32x96xf32>
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

func.func @reciprocal(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.reciprocal"[[C:.*]]
  %1 = "ttir.reciprocal"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
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

func.func @rsqrt(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.rsqrt"[[C:.*]]
  %1 = "ttir.rsqrt"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @softmax(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<512x1024xbf16>
  // CHECK: %[[C:.*]] = "ttnn.softmax"[[C:.*]]
  // Check for positive dimension attribute
  %1 = "ttir.softmax"(%arg0, %0) <{dimension = 1 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<512x1024xbf16>, tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %2 = tensor.empty() : tensor<512x1024xbf16>
  // CHECK: %[[C:.*]] = "ttnn.softmax"[[C:.*]]
  // Check for negative dimension attribute
  %3 = "ttir.softmax"(%1, %2) <{dimension = -1 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<512x1024xbf16>, tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
  return %3 : tensor<512x1024xbf16>
}

func.func @typecast(%arg0: tensor<64x128xf32>) -> tensor<64x128xbf16> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xbf16>
  // CHECK: %[[C:.*]] = "ttnn.typecast"
  // CHECK-SAME: tensor<64x128xf32,
  // CHECK-SAME: tensor<64x128xbf16,
  %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
  return %1 : tensor<64x128xbf16>
}

func.func @maximum(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.maximum"[[C:.*]]
  %1 = "ttir.maximum"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
