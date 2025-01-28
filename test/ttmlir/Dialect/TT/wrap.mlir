// RUN: ttmlir-opt %s --tt-wrap-device-module | FileCheck %s

module {
  func.func @test(%arg0: i32) -> i32 {
    return %arg0 : i32
  }
}

// CHECK: module
// CHECK: tt.device_module {
// CHECK: module {
// CHECK: func.func @test
