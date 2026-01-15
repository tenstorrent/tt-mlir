// RUN: ttmlir-opt --emit-calling-convention-wrappers -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {ttir.cpu_module} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @add(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> attributes {arg_ranks = [2, 2, 2], result_ranks = [2]} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.return %0 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
}

// CHECK: llvm.func @add_helper(%arg0: !llvm.ptr) -> !llvm.ptr
