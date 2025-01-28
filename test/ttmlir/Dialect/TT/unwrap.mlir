// RUN: ttmlir-opt --tt-unwrap-device-module %s | FileCheck %s

module {
  tt.device_module {
    builtin.module {
      func.func @test(%arg0: i32) -> i32 {
        return %arg0 : i32
      }
    }
  }
}

// CHECK: module {
// CHECK: func.func @test
