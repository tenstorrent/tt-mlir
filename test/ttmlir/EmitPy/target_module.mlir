// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="target-module=true" -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir
// RUN: FileCheck %s --input-file=%t.py

// Verify that with target-module=true:
// 1. The entry function is renamed to 'forward'
// 2. The inputs are tuplified into a single argument
// 3. A device argument is added as the second parameter
// 4. No DeviceGetter.get_device call is present (device comes from argument)

// CHECK: def forward(input, device):
// CHECK-NOT: utils.DeviceGetter.get_device

func.func @add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %1 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
