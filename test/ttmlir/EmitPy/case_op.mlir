// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-const-eval=false" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitpy-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t2.mlir
// RUN: FileCheck %s --input-file=%t.py

// The index is read back to host once, ahead of the chain, and the last branch
// is the bare `else`, so an index matching no branch selects it - the op's
// out-of-range rule.
// CHECK-LABEL: def three_branch
// CHECK: case_index = int({{.*}}.to_torch().item())
// CHECK: if case_index == 0:
// CHECK: branch_0 = ttnn_multiply
// CHECK: elif case_index == 1:
// CHECK: branch_0 = ttnn_subtract
// CHECK: else:
// CHECK: branch_0 = ttnn_add
// CHECK: return [branch_0]
func.func @three_branch(%arg0: tensor<32x32xf32>, %index: tensor<i32>) -> tensor<32x32xf32> {
  %two = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<32x32xf32>}> : () -> tensor<32x32xf32>
  %r = ttir.case index(%index : tensor<i32>) captures(%arg0, %two : tensor<32x32xf32>, tensor<32x32xf32>)
  branches {
  ^bb0(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>):
    %0 = "ttir.multiply"(%a, %b) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  }, {
  ^bb0(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>):
    %0 = "ttir.subtract"(%a, %b) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  }, {
  ^bb0(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>):
    %0 = "ttir.add"(%a, %b) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  } -> (tensor<32x32xf32>)
  return %r : tensor<32x32xf32>
}

// Two produced values are assigned as one tuple, matching the while loop's
// carry-back, so a branch that yields the same variables in a different order
// cannot clobber one it still has to read.
//
// A capture is bound through a new handle on its buffer rather than as itself:
// a plain assignment would make the result the capture's own object, and a
// non-forced deallocation of either name would then free both. A value the
// branch computes is its own, so it is bound as is.
// CHECK-LABEL: def swap
// CHECK: if case_index == 0:
// CHECK: branch_0, branch_1 = ttnn_add{{.*}}, ttnn.Tensor(var_0)
// CHECK: else:
// CHECK: branch_0, branch_1 = ttnn.Tensor(var_1), ttnn.Tensor(var_0)
func.func @swap(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %index: tensor<i32>) -> tensor<32x32xf32> {
  %r:2 = ttir.case index(%index : tensor<i32>) captures(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>)
  branches {
  ^bb0(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>):
    %0 = "ttir.add"(%a, %b) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0, %a : tensor<32x32xf32>, tensor<32x32xf32>
  }, {
  ^bb0(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>):
    ttir.yield %b, %a : tensor<32x32xf32>, tensor<32x32xf32>
  } -> (tensor<32x32xf32>, tensor<32x32xf32>)
  return %r#0 : tensor<32x32xf32>
}

// A single branch always runs, and a lone `else` is not Python, so it is
// emitted inline and the index is never read.
// CHECK-LABEL: def one_branch
// CHECK-NOT: else:
// CHECK: ttnn_add{{.*}} = ttnn.add(
// CHECK-NEXT: branch_0 = ttnn_add
// CHECK: return [branch_0]
func.func @one_branch(%arg0: tensor<32x32xf32>, %index: tensor<i32>) -> tensor<32x32xf32> {
  %r = ttir.case index(%index : tensor<i32>) captures(%arg0 : tensor<32x32xf32>)
  branches {
  ^bb0(%a: tensor<32x32xf32>):
    %0 = "ttir.add"(%a, %a) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  } -> (tensor<32x32xf32>)
  return %r : tensor<32x32xf32>
}

// A value yielded twice is two results, so the repeat is bound through a new
// handle too: as one object, the first result's deallocation would free the
// second. The first deallocation is then a no-op, and the second is forced to
// free the buffer the two share.
// CHECK-LABEL: def yields_twice
// CHECK: if case_index == 0:
// CHECK: branch_0, branch_1 = [[V:ttnn_add[_0-9]*]], ttnn.Tensor([[V]])
// CHECK: else:
// CHECK: branch_0, branch_1 = ttnn_multiply{{[_0-9]*}}, ttnn_subtract{{[_0-9]*}}
// CHECK: ttnn.deallocate(branch_0, False)
// CHECK: ttnn.deallocate(branch_1, True)
func.func @yields_twice(%arg0: tensor<32x32xf32>, %index: tensor<i32>) -> tensor<32x32xf32> {
  %r:2 = ttir.case index(%index : tensor<i32>) captures(%arg0 : tensor<32x32xf32>)
  branches {
  ^bb0(%a: tensor<32x32xf32>):
    %0 = "ttir.add"(%a, %a) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0, %0 : tensor<32x32xf32>, tensor<32x32xf32>
  }, {
  ^bb0(%a: tensor<32x32xf32>):
    %0 = "ttir.multiply"(%a, %a) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %1 = "ttir.subtract"(%a, %a) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0, %1 : tensor<32x32xf32>, tensor<32x32xf32>
  } -> (tensor<32x32xf32>, tensor<32x32xf32>)
  %e = "ttir.exp"(%r#0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %s = "ttir.add"(%e, %r#1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %s : tensor<32x32xf32>
}
