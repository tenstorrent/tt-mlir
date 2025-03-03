// RUN: ttmlir-opt --convert-ttkernel-to-emitc %s | FileCheck %s
module {
  func.func @accumulate(%num_iter : index, %incr : i32) -> i32 {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : index
    %i0  = arith.constant 0 : index
    %acc = memref.alloca() : memref<1xi32>
    memref.store %zero, %acc[%i0] : memref<1xi32>
    scf.for %i = %i0 to %num_iter step %one {
      %current_sum = memref.load %acc[%i0] : memref<1xi32>
      %new_sum = arith.addi %current_sum, %incr : i32
      memref.store %new_sum, %acc[%i0] : memref<1xi32>
    }
    %res = memref.load %acc[%i0] : memref<1xi32>
    func.return %res : i32
  }
}
