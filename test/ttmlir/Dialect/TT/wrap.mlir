// RUN: ttmlir-opt --ttcore-wrap-device-module -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @test(%arg0: i32) -> i32 {
    return %arg0 : i32
  }
}

// CHECK: module
// CHECK: ttcore.device_module {
// CHECK: module {
// CHECK: func.func @test
