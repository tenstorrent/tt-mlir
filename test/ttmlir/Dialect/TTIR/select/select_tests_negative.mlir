// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @select_negative_invalid_stride(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tensor.empty() : tensor<4x4xf32>
    // CHECK: {{.*error.*Invalid stride.*}}
    %1 = "ttir.select"(%arg0, %0) <{dim = 1: si32, begin = 0: si32, length = 4: si32, stride = 7: si32, operand_constraints = [#any_device_tile, #any_device_tile]}>  :
        (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}

// -----

#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @select_negative_invalid_stride_2(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tensor.empty() : tensor<4x4xf32>
    // CHECK: {{.*error.*Invalid stride.*}}
    %1 = "ttir.select"(%arg0, %0) <{dim = 1: si32, begin = 0: si32, length = 4: si32, stride = -1: si32, operand_constraints = [#any_device_tile, #any_device_tile]}>  :
        (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}

// -----

#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @select_negative_invalid_begin(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tensor.empty() : tensor<4x4xf32>
    // CHECK: {{.*error.*Invalid begin index.*}}
    %1 = "ttir.select"(%arg0, %0) <{dim = 1: si32, begin = -3: si32, length = 4: si32, stride = 1: si32, operand_constraints = [#any_device_tile, #any_device_tile]}>  :
        (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}

// -----

#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @select_negative_invalid_begin_2(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tensor.empty() : tensor<4x4xf32>
    // CHECK: {{.*error.*Invalid begin index.*}}
    %1 = "ttir.select"(%arg0, %0) <{dim = 1: si32, begin = 4: si32, length = 4: si32, stride = 1: si32, operand_constraints = [#any_device_tile, #any_device_tile]}>  :
        (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}

// -----

#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @select_negative_invalid_length(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tensor.empty() : tensor<4x4xf32>
    // CHECK: {{.*error.*Invalid length.*}}
    %1 = "ttir.select"(%arg0, %0) <{dim = 1: si32, begin = 0: si32, length = 5: si32, stride = 1: si32, operand_constraints = [#any_device_tile, #any_device_tile]}>  :
        (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}

// -----

#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @select_negative_invalid_length_2(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tensor.empty() : tensor<4x4xf32>
    // CHECK: {{.*error.*Invalid length.*}}
    %1 = "ttir.select"(%arg0, %0) <{dim = 1: si32, begin = 0: si32, length = 0: si32, stride = 1: si32, operand_constraints = [#any_device_tile, #any_device_tile]}>  :
        (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}

// -----

#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @select_negative_invalid_length_3(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tensor.empty() : tensor<4x4xf32>
    // CHECK: {{.*error.*Invalid length.*}}
    %1 = "ttir.select"(%arg0, %0) <{dim = 1: si32, begin = 0: si32, length = 2: si32, stride = 1: si32, operand_constraints = [#any_device_tile, #any_device_tile]}>  :
        (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}

// -----

#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @select_negative_invalid_total_size(%arg0: tensor<4x2x64x48xf32>) -> tensor<4x2x4x48xf32> {
    %0 = tensor.empty() : tensor<4x2x4x48xf32>
    // CHECK: {{.*error.*Sum of all slices.*}}
    %1 = "ttir.select"( %arg0, %0) <{dim = 2: si32, begin = 0: si32, length = 4: si32, stride = 4: si32, operand_constraints = [#any_device_tile, #any_device_tile]}>  :
        (tensor<4x2x64x48xf32>, tensor<4x2x4x48xf32>) -> tensor<4x2x4x48xf32>
    return %1 : tensor<4x2x4x48xf32>
  }
}
