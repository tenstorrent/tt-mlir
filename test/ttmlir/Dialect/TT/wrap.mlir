// RUN: ttmlir-opt %s --ttcore-wrap-device-module | FileCheck %s

module {
  func.func @test(%arg0: i32) -> i32 {
    return %arg0 : i32
  }
}

// CHECK: module
// CHECK: ttcore.device_module {
// CHECK: module {
// CHECK: func.func @test
