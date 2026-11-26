// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-const-eval=false" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitpy-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t2.mlir
// RUN: FileCheck %s --input-file=%t.py

// A counted loop becomes `for _ in range(n)`, so it needs no counter variable
// and never evaluates its condition. Loop-carried values become named locals
// that the emitter reassigns at the end of the body.
//
// Each local starts out as a new handle on its init's buffer rather than as the
// init itself. The init keeps its own name after the loop, and as one object, a
// non-forced deallocation of the init would free a result that is still the
// init, as after zero iterations.
// CHECK-LABEL: def counted
// CHECK: carried_0 = ttnn.Tensor(
// CHECK: carried_1 = ttnn.Tensor(var_0)
// CHECK: for _ in range(4):
// CHECK-NOT: break
// CHECK: carried_0, carried_1 = ttnn_add
func.func @counted(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %i0 = "ttir.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  %limit = "ttir.constant"() <{value = dense<4> : tensor<i32>}> : () -> tensor<i32>
  %step = "ttir.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
  %r:2 = ttir.while inits(%i0, %arg0 : tensor<i32>, tensor<32x32xf32>)
                    captures(%limit, %step : tensor<i32>, tensor<i32>)
    cond {
    ^cond(%i: tensor<i32>, %acc: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>):
      %p = "ttir.lt"(%i, %l) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      ttir.yield %p : tensor<i1>
    } do {
    ^body(%i: tensor<i32>, %acc: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>):
      %next = "ttir.add"(%i, %s) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %acc2 = "ttir.add"(%acc, %acc) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
      ttir.yield %next, %acc2 : tensor<i32>, tensor<32x32xf32>
    } -> (tensor<i32>, tensor<32x32xf32>)
  return %r#1 : tensor<32x32xf32>
}

// A data-dependent loop runs the condition every iteration and reads the
// predicate back to host to decide whether to keep going. Python has no
// do-while, so the test is a break inside `while True`.
// CHECK-LABEL: def data_dependent
// CHECK: while True:
// CHECK: if {{.*}}.to_torch().item() == 0: break
// CHECK: carried_0, carried_1 = ttnn_add
func.func @data_dependent(%arg0: tensor<32x32xf32>, %arg1: tensor<i32>) -> tensor<32x32xf32> {
  %i0 = "ttir.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  %step = "ttir.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
  %r:2 = ttir.while inits(%i0, %arg0 : tensor<i32>, tensor<32x32xf32>)
                    captures(%arg1, %step : tensor<i32>, tensor<i32>)
    cond {
    ^cond(%i: tensor<i32>, %acc: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>):
      %p = "ttir.lt"(%i, %l) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      ttir.yield %p : tensor<i1>
    } do {
    ^body(%i: tensor<i32>, %acc: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>):
      %next = "ttir.add"(%i, %s) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %acc2 = "ttir.add"(%acc, %acc) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
      ttir.yield %next, %acc2 : tensor<i32>, tensor<32x32xf32>
    } -> (tensor<i32>, tensor<32x32xf32>)
  return %r#1 : tensor<32x32xf32>
}

// The body yields two of its arguments swapped. The carry-back has to be a
// single tuple assignment: assigning one at a time would overwrite a value the
// other half still has to read.
// CHECK-LABEL: def swap
// CHECK: for _ in range(2):
// CHECK: carried_0, carried_1, carried_2 = ttnn_add{{.*}}, carried_2, carried_1
func.func @swap(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>)
    -> (tensor<32x32xf32>, tensor<32x32xf32>) {
  %i0 = "ttir.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  %limit = "ttir.constant"() <{value = dense<2> : tensor<i32>}> : () -> tensor<i32>
  %step = "ttir.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
  %r:3 = ttir.while inits(%i0, %arg0, %arg1 : tensor<i32>, tensor<32x32xf32>, tensor<32x32xf32>)
                    captures(%limit, %step : tensor<i32>, tensor<i32>)
    cond {
    ^cond(%i: tensor<i32>, %a: tensor<32x32xf32>, %b: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>):
      %p = "ttir.lt"(%i, %l) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      ttir.yield %p : tensor<i1>
    } do {
    ^body(%i: tensor<i32>, %a: tensor<32x32xf32>, %b: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>):
      %next = "ttir.add"(%i, %s) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      ttir.yield %next, %b, %a : tensor<i32>, tensor<32x32xf32>, tensor<32x32xf32>
    } -> (tensor<i32>, tensor<32x32xf32>, tensor<32x32xf32>)
  return %r#1, %r#2 : tensor<32x32xf32>, tensor<32x32xf32>
}

// A capture the body yields is bound through a new handle, since it keeps its
// own name outside the loop, and so is a value the body yields twice after the
// first time. A value the body computes, or a carried one it moves between
// slots, is bound as is.
// CHECK-LABEL: def forwards
// CHECK: for _ in range(2):
// CHECK: carried_0, carried_1, carried_2, carried_3 = ttnn_add{{[_0-9]*}}, ttnn.Tensor(arg[1]), [[V:ttnn_multiply[_0-9]*]], ttnn.Tensor([[V]])
func.func @forwards(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>)
    -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) {
  %i0 = "ttir.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  %limit = "ttir.constant"() <{value = dense<2> : tensor<i32>}> : () -> tensor<i32>
  %step = "ttir.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
  %r:4 = ttir.while inits(%i0, %arg0, %arg0, %arg0 : tensor<i32>, tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>)
                    captures(%limit, %step, %arg1 : tensor<i32>, tensor<i32>, tensor<32x32xf32>)
    cond {
    ^cond(%i: tensor<i32>, %a: tensor<32x32xf32>, %b: tensor<32x32xf32>, %c: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>, %k: tensor<32x32xf32>):
      %p = "ttir.lt"(%i, %l) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      ttir.yield %p : tensor<i1>
    } do {
    ^body(%i: tensor<i32>, %a: tensor<32x32xf32>, %b: tensor<32x32xf32>, %c: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>, %k: tensor<32x32xf32>):
      %next = "ttir.add"(%i, %s) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %m = "ttir.multiply"(%b, %c) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
      ttir.yield %next, %k, %m, %m : tensor<i32>, tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>
    } -> (tensor<i32>, tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>)
  return %r#1, %r#2, %r#3 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>
}
