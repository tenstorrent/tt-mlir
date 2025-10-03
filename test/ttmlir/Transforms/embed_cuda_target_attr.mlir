// RUN: ttmlir-opt --embed-cuda-target-attributes -o %t %s
// RUN: FileCheck %s --input-file=%t --check-prefix=CHECK-DEFAULT

// RUN: ttmlir-opt --embed-cuda-target-attributes="chip=sm_80 features=+ptx70 opt-level=3" -o %t2 %s
// RUN: FileCheck %s --input-file=%t2 --check-prefix=CHECK-CUSTOM

// CHECK-DEFAULT: module attributes {cuda.chip = "sm_50", cuda.features = "+ptx50", cuda.opt_level = 2 : i32}
// CHECK-CUSTOM: module attributes {cuda.chip = "sm_80", cuda.features = "+ptx70", cuda.opt_level = 3 : i32}
module {
  func.func @simple_kernel(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) {
    return
  }
}
