// RUN: ttmlir-opt --convert-ttkernel-to-emitc %s | FileCheck %s
module {
  func.func @const() -> i32 {
    // CHECK: %[[ZERO:[0-9]+]] = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %0 = arith.constant 0 : i32
    // CHECK: %[[ONE:[0-9]+]] = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1  = arith.constant 1 : i32
    return %1 : i32
  }
}
