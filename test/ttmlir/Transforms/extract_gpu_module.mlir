// RUN: ttmlir-opt --extract-gpu-modules -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    //CHECK-NOT: gpu.module
    //CHECK: module
    //CHECK-NOT: module
  gpu.module @matmul_kernel [#nvvm.target<features = "+ptx50">] {
      llvm.func @matmul()
      {
        llvm.return
      }

    }
  gpu.module @matmul_kernel_0 [#nvvm.target<features = "+ptx50">] {
      llvm.func @matmul()
      {
        llvm.return
      }
    }
}
