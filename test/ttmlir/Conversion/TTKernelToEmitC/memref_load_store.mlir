// RUN: ttmlir-opt --convert-ttkernel-to-emitc %s | FileCheck %s
module {
  func.func @load_store() -> i32 {
    // CHECK: %[[VAL:[0-9]+]] = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %val = arith.constant 1 : i32
    // CHECK: %[[ZERO:[0-9]+]] = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
    %i0  = arith.constant 0 : index
    // CHECK: %[[TWO:[0-9]+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1xi32>
    // CHECK: %[[THREE:[0-9]+]] = emitc.subscript %[[TWO]][%[[ZERO]]] : (!emitc.array<1xi32>, !emitc.size_t) -> !emitc.lvalue<i32>
    %acc = memref.alloca() : memref<1xi32>
    // CHECK: emitc.assign %[[VAL]] : i32 to %[[THREE]] : <i32>
    memref.store %val, %acc[%i0] : memref<1xi32>

    // CHECK: %[[FOUR:[0-9]+]] = emitc.subscript %[[TWO]][%[[ZERO]]] : (!emitc.array<1xi32>, !emitc.size_t) -> !emitc.lvalue<i32>
    // CHECK: %[[FIVE:[0-9]+]] = emitc.load %[[FOUR:[0-9]+]] : <i32>
    %res = memref.load %acc[%i0] : memref<1xi32>
    func.return %res : i32
  }
}
