// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true" %s -o %t.mlir
// RUN: less %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: printf "\n\nbetween ttnn and modify signatures\n\n\n"
// RUN: ttmlir-opt --ttnn-modify-signatures-for-dylib --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: less %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp
// RUN: less %basename_t.cpp

func.func @mnist_fwd(%arg0: tensor<1x784xf32>, %arg1: tensor<1x10xf32>, %arg2: tensor<256x10xf32>, %arg3: tensor<1x256xf32>, %arg4: tensor<784x256xf32>) -> tensor<1x10xf32> {
  %0 = ttir.empty() : tensor<1x256xf32>
  %1 = "ttir.matmul"(%arg0, %arg4, %0) : (tensor<1x784xf32>, tensor<784x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32>
  %2 = ttir.empty() : tensor<1x256xf32>
  %3 = "ttir.add"(%1, %arg3, %2) : (tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32>
  %4 = ttir.empty() : tensor<1x256xf32>
  %5 = "ttir.relu"(%3, %4) : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32>
  %6 = ttir.empty() : tensor<1x10xf32>
  %7 = "ttir.matmul"(%5, %arg2, %6) : (tensor<1x256xf32>, tensor<256x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
  %8 = ttir.empty() : tensor<1x10xf32>
  %9 = "ttir.add"(%7, %arg1, %8) : (tensor<1x10xf32>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
  %10 = ttir.empty() : tensor<1x10xf32>
  %11 = "ttir.softmax"(%9, %10) <{dimension = 1 : si32}> : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
  return %11 : tensor<1x10xf32>
}
