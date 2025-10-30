// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=false" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

func.func @subtract(%arg0: tensor<224x64xf32>, %arg1: tensor<224x64xf32>) -> tensor<224x64xf32> {
  %0 = ttir.empty() : tensor<224x64xf32>
  // CHECK: "ttnn.subtract"
  %1 = "ttir.subtract"(%arg0, %arg1, %0) : (tensor<224x64xf32>, tensor<224x64xf32>, tensor<224x64xf32>) -> tensor<224x64xf32>
  return %1 : tensor<224x64xf32>
}

func.func @div(%arg0: tensor<224x64xf32>, %arg1: tensor<224x64xf32>) -> tensor<224x64xf32> {
  %0 = ttir.empty() : tensor<224x64xf32>
  // CHECK: "ttnn.divide"
  %1 = "ttir.div"(%arg0, %arg1, %0) : (tensor<224x64xf32>, tensor<224x64xf32>, tensor<224x64xf32>) -> tensor<224x64xf32>
  return %1 : tensor<224x64xf32>
}

func.func @multiply(%arg0: tensor<224x64xf32>, %arg1: tensor<224x64xf32>) -> tensor<224x64xf32> {
  %0 = ttir.empty() : tensor<224x64xf32>
  // CHECK: "ttnn.multiply"
  %1 = "ttir.multiply"(%arg0, %arg1, %0) : (tensor<224x64xf32>, tensor<224x64xf32>, tensor<224x64xf32>) -> tensor<224x64xf32>
  return %1 : tensor<224x64xf32>
}

func.func @relu(%arg0: tensor<224x64xf32>) -> tensor<224x64xf32> {
  %0 = ttir.empty() : tensor<224x64xf32>
  // CHECK: "ttnn.relu"
  %1 = "ttir.relu"(%arg0, %0) : (tensor<224x64xf32>, tensor<224x64xf32>) -> tensor<224x64xf32>
  return %1 : tensor<224x64xf32>
}

func.func @ge(%arg0: tensor<224x64xf32>, %arg1: tensor<224x64xf32>) -> tensor<224x64xf32> {
  %0 = ttir.empty() : tensor<224x64xf32>
  // CHECK: "ttnn.ge"
  %1 = "ttir.ge"(%arg0, %arg1, %0) : (tensor<224x64xf32>, tensor<224x64xf32>, tensor<224x64xf32>) -> tensor<224x64xf32>
  return %1 : tensor<224x64xf32>
}

func.func @reshape(%arg0: tensor<4x2x224x64xbf16>) -> tensor<2x4x224x64xbf16> {
  %0 = ttir.empty() : tensor<2x4x224x64xbf16>
  // CHECK: "ttnn.reshape"
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [2: i32, 4: i32, 224: i32, 64: i32]}> : (tensor<4x2x224x64xbf16>, tensor<2x4x224x64xbf16>) -> tensor<2x4x224x64xbf16>
  return %1 : tensor<2x4x224x64xbf16>
}

func.func @squeeze(%arg0: tensor<1x2x1x224x64xbf16>) -> tensor<1x2x224x64xbf16> {
  %0 = ttir.empty() : tensor<1x2x224x64xbf16>
  // CHECK: "ttnn.reshape"
  %1 = "ttir.squeeze"(%arg0, %0) <{dim = 2 : si32}> : (tensor<1x2x1x224x64xbf16>, tensor<1x2x224x64xbf16>) -> tensor<1x2x224x64xbf16>
  return %1 : tensor<1x2x224x64xbf16>
}

func.func @reciprocal(%arg0: tensor<224x64xf32>) -> tensor<224x64xf32> {
  %0 = ttir.empty() : tensor<224x64xf32>
  // CHECK: "ttnn.reciprocal"
  %1 = "ttir.reciprocal"(%arg0, %0) : (tensor<224x64xf32>, tensor<224x64xf32>) -> tensor<224x64xf32>
  return %1 : tensor<224x64xf32>
}

func.func @sigmoid(%arg0: tensor<224x64xf32>) -> tensor<224x64xf32> {
  %0 = ttir.empty() : tensor<224x64xf32>
  // CHECK: "ttnn.sigmoid"
  %1 = "ttir.sigmoid"(%arg0, %0) : (tensor<224x64xf32>, tensor<224x64xf32>) -> tensor<224x64xf32>
  return %1 : tensor<224x64xf32>
}

func.func @sqrt(%arg0: tensor<224x64xf32>) -> tensor<224x64xf32> {
  %0 = ttir.empty() : tensor<224x64xf32>
  // CHECK: "ttnn.sqrt"
  %1 = "ttir.sqrt"(%arg0, %0) : (tensor<224x64xf32>, tensor<224x64xf32>) -> tensor<224x64xf32>
  return %1 : tensor<224x64xf32>
}

func.func @silu(%arg0: tensor<224x64xf32>) -> tensor<224x64xf32> {
  %0 = ttir.empty() : tensor<224x64xf32>
  // CHECK: "ttnn.silu"
  %1 = "ttir.silu"(%arg0, %0) : (tensor<224x64xf32>, tensor<224x64xf32>) -> tensor<224x64xf32>
  return %1 : tensor<224x64xf32>
}

/////////////////////////////////////////
// Unsupported eltwise ops with sharding
//  * Concat: Sharded concat requires ROW MAJOR layout
//  * Softmax: Sharded softmax produces incorrect values, TODO (#843)
/////////////////////////////////////////
