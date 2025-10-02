// RUN: ttmlir-opt --extract-gpu-modules -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {cuda.chip = "sm_50", cuda.features = "+ptx50", cuda.opt_level = 2 : i32, gpu.container_module} {
    //CHECK-NOT: gpu.module
    //CHECK: module
    //CHECK-NOT: module
  func.func @matmul(%arg0: memref<64x128xbf16>, %arg1: memref<128x96xbf16>) -> memref<64x96xbf16> {
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c96 = arith.constant 96 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<64x128xbf16> -> memref<bf16>, index, index, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [1, 64, 128], strides: [8192, 128, 1] : memref<bf16> to memref<1x64x128xbf16>
    %base_buffer_0, %offset_1, %sizes_2:2, %strides_3:2 = memref.extract_strided_metadata %arg1 : memref<128x96xbf16> -> memref<bf16>, index, index, index, index, index
    %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [0], sizes: [1, 128, 96], strides: [12288, 96, 1] : memref<bf16> to memref<1x128x96xbf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x64x96xbf16>
    gpu.launch_func  @matmul_kernel::@matmul_kernel blocks in (%c1, %c64, %c96) threads in (%c32, %c1, %c1)  args(%c32 : index, %c0 : index, %c1 : index, %c1 : index, %c1 : index, %c0 : index, %c64 : index, %c96 : index, %cst : bf16, %alloc : memref<1x64x96xbf16>)
    gpu.launch_func  @matmul_kernel_0::@matmul_kernel blocks in (%c1, %c64, %c96) threads in (%c32, %c1, %c1)  args(%c32 : index, %c0 : index, %c1 : index, %c1 : index, %c1 : index, %c0 : index, %c64 : index, %c96 : index, %reinterpret_cast : memref<1x64x128xbf16>, %reinterpret_cast_4 : memref<1x128x96xbf16>, %alloc : memref<1x64x96xbf16>, %c128 : index)
    %reinterpret_cast_5 = memref.reinterpret_cast %alloc to offset: [0], sizes: [64, 96], strides: [96, 1] : memref<1x64x96xbf16> to memref<64x96xbf16>
    return %reinterpret_cast_5 : memref<64x96xbf16>
  }
  gpu.module @matmul_kernel [#nvvm.target<features = "+ptx50">] {
    llvm.func @matmul_kernel(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: bf16, %arg9: !llvm.ptr) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %1 = llvm.insertvalue %arg9, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %2 = llvm.insertvalue %arg9, %1[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %3 = llvm.mlir.constant(0 : index) : i64
      %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %5 = llvm.mlir.constant(1 : index) : i64
      %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %7 = llvm.mlir.constant(6144 : index) : i64
      %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %9 = llvm.mlir.constant(64 : index) : i64
      %10 = llvm.insertvalue %9, %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %11 = llvm.mlir.constant(96 : index) : i64
      %12 = llvm.insertvalue %11, %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %13 = llvm.mlir.constant(96 : index) : i64
      %14 = llvm.insertvalue %13, %12[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %15 = llvm.mlir.constant(1 : index) : i64
      %16 = llvm.insertvalue %15, %14[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %17 = nvvm.read.ptx.sreg.ctaid.x : i32
      %18 = llvm.sext %17 : i32 to i64
      %19 = nvvm.read.ptx.sreg.ctaid.y : i32
      %20 = llvm.sext %19 : i32 to i64
      %21 = nvvm.read.ptx.sreg.ctaid.z : i32
      %22 = llvm.sext %21 : i32 to i64
      %23 = nvvm.read.ptx.sreg.tid.x : i32
      %24 = llvm.sext %23 : i32 to i64
      %25 = nvvm.read.ptx.sreg.tid.y : i32
      %26 = llvm.sext %25 : i32 to i64
      %27 = nvvm.read.ptx.sreg.tid.z : i32
      %28 = llvm.sext %27 : i32 to i64
      %29 = llvm.mul %18, %arg0 overflow<nsw> : i64
      %30 = llvm.add %29, %arg1 : i64
      %31 = llvm.mul %20, %arg2 overflow<nsw> : i64
      %32 = llvm.add %31, %arg1 : i64
      %33 = llvm.mul %22, %arg3 overflow<nsw> : i64
      %34 = llvm.add %33, %arg1 : i64
      %35 = llvm.mul %24, %arg4 overflow<nsw> : i64
      %36 = llvm.add %35, %arg5 : i64
      %37 = llvm.mul %26, %arg4 overflow<nsw> : i64
      %38 = llvm.add %37, %arg5 : i64
      %39 = llvm.mul %28, %arg4 overflow<nsw> : i64
      %40 = llvm.add %39, %arg5 : i64
      %41 = llvm.add %36, %30 : i64
      %42 = llvm.add %38, %32 : i64
      %43 = llvm.add %40, %34 : i64
      %44 = llvm.mul %36, %arg4 : i64
      %45 = llvm.add %44, %30 : i64
      %46 = llvm.icmp "ult" %45, %arg4 : i64
      %47 = llvm.mul %38, %arg4 : i64
      %48 = llvm.add %47, %32 : i64
      %49 = llvm.icmp "ult" %48, %arg6 : i64
      %50 = llvm.and %46, %49 : i1
      %51 = llvm.mul %40, %arg4 : i64
      %52 = llvm.add %51, %34 : i64
      %53 = llvm.icmp "ult" %52, %arg7 : i64
      %54 = llvm.and %50, %53 : i1
      llvm.cond_br %54, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %55 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %56 = llvm.mlir.constant(6144 : index) : i64
      %57 = llvm.mul %41, %56 overflow<nsw, nuw> : i64
      %58 = llvm.mlir.constant(96 : index) : i64
      %59 = llvm.mul %42, %58 overflow<nsw, nuw> : i64
      %60 = llvm.add %57, %59 overflow<nsw, nuw> : i64
      %61 = llvm.add %60, %43 overflow<nsw, nuw> : i64
      %62 = llvm.getelementptr inbounds|nuw %55[%61] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
      llvm.store %arg8, %62 : bf16, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    }
  }
  gpu.module @matmul_kernel_0 [#nvvm.target<features = "+ptx50">] {
    llvm.func @matmul_kernel(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: !llvm.ptr, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = builtin.unrealized_conversion_cast %arg11 : i64 to index
      %1 = builtin.unrealized_conversion_cast %arg4 : i64 to index
      %2 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %3 = llvm.insertvalue %arg10, %2[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %4 = llvm.insertvalue %arg10, %3[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %5 = llvm.mlir.constant(0 : index) : i64
      %6 = llvm.insertvalue %5, %4[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %7 = llvm.mlir.constant(1 : index) : i64
      %8 = llvm.insertvalue %7, %6[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %9 = llvm.mlir.constant(6144 : index) : i64
      %10 = llvm.insertvalue %9, %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %11 = llvm.mlir.constant(64 : index) : i64
      %12 = llvm.insertvalue %11, %10[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %13 = llvm.mlir.constant(96 : index) : i64
      %14 = llvm.insertvalue %13, %12[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %15 = llvm.mlir.constant(96 : index) : i64
      %16 = llvm.insertvalue %15, %14[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %17 = llvm.mlir.constant(1 : index) : i64
      %18 = llvm.insertvalue %17, %16[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %19 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %20 = llvm.insertvalue %arg9, %19[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %21 = llvm.insertvalue %arg9, %20[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %22 = llvm.mlir.constant(0 : index) : i64
      %23 = llvm.insertvalue %22, %21[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %24 = llvm.mlir.constant(1 : index) : i64
      %25 = llvm.insertvalue %24, %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %26 = llvm.mlir.constant(12288 : index) : i64
      %27 = llvm.insertvalue %26, %25[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %28 = llvm.mlir.constant(128 : index) : i64
      %29 = llvm.insertvalue %28, %27[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %30 = llvm.mlir.constant(96 : index) : i64
      %31 = llvm.insertvalue %30, %29[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %32 = llvm.mlir.constant(96 : index) : i64
      %33 = llvm.insertvalue %32, %31[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %34 = llvm.mlir.constant(1 : index) : i64
      %35 = llvm.insertvalue %34, %33[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %36 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %37 = llvm.insertvalue %arg8, %36[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %38 = llvm.insertvalue %arg8, %37[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %39 = llvm.mlir.constant(0 : index) : i64
      %40 = llvm.insertvalue %39, %38[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %41 = llvm.mlir.constant(1 : index) : i64
      %42 = llvm.insertvalue %41, %40[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %43 = llvm.mlir.constant(8192 : index) : i64
      %44 = llvm.insertvalue %43, %42[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %45 = llvm.mlir.constant(64 : index) : i64
      %46 = llvm.insertvalue %45, %44[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %47 = llvm.mlir.constant(128 : index) : i64
      %48 = llvm.insertvalue %47, %46[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %49 = llvm.mlir.constant(128 : index) : i64
      %50 = llvm.insertvalue %49, %48[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %51 = llvm.mlir.constant(1 : index) : i64
      %52 = llvm.insertvalue %51, %50[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %53 = nvvm.read.ptx.sreg.ctaid.x : i32
      %54 = llvm.sext %53 : i32 to i64
      %55 = nvvm.read.ptx.sreg.ctaid.y : i32
      %56 = llvm.sext %55 : i32 to i64
      %57 = nvvm.read.ptx.sreg.ctaid.z : i32
      %58 = llvm.sext %57 : i32 to i64
      %59 = nvvm.read.ptx.sreg.tid.x : i32
      %60 = llvm.sext %59 : i32 to i64
      %61 = nvvm.read.ptx.sreg.tid.y : i32
      %62 = llvm.sext %61 : i32 to i64
      %63 = nvvm.read.ptx.sreg.tid.z : i32
      %64 = llvm.sext %63 : i32 to i64
      %65 = llvm.mul %54, %arg0 overflow<nsw> : i64
      %66 = llvm.add %65, %arg1 : i64
      %67 = llvm.mul %56, %arg2 overflow<nsw> : i64
      %68 = llvm.add %67, %arg1 : i64
      %69 = llvm.mul %58, %arg3 overflow<nsw> : i64
      %70 = llvm.add %69, %arg1 : i64
      %71 = llvm.mul %60, %arg4 overflow<nsw> : i64
      %72 = llvm.add %71, %arg5 : i64
      %73 = llvm.mul %62, %arg4 overflow<nsw> : i64
      %74 = llvm.add %73, %arg5 : i64
      %75 = llvm.mul %64, %arg4 overflow<nsw> : i64
      %76 = llvm.add %75, %arg5 : i64
      %77 = llvm.add %72, %66 : i64
      %78 = llvm.add %74, %68 : i64
      %79 = llvm.add %76, %70 : i64
      %80 = llvm.mul %72, %arg4 : i64
      %81 = llvm.add %80, %66 : i64
      %82 = llvm.icmp "ult" %81, %arg4 : i64
      %83 = llvm.mul %74, %arg4 : i64
      %84 = llvm.add %83, %68 : i64
      %85 = llvm.icmp "ult" %84, %arg6 : i64
      %86 = llvm.and %82, %85 : i1
      %87 = llvm.mul %76, %arg4 : i64
      %88 = llvm.add %87, %70 : i64
      %89 = llvm.icmp "ult" %88, %arg7 : i64
      %90 = llvm.and %86, %89 : i1
      llvm.cond_br %90, ^bb1, ^bb5
    ^bb1:  // pred: ^bb0
      llvm.br ^bb2(%arg1 : i64)
    ^bb2(%91: i64):  // 2 preds: ^bb1, ^bb3
      %92 = builtin.unrealized_conversion_cast %91 : i64 to index
      %93 = arith.cmpi slt, %92, %0 : index
      llvm.cond_br %93, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      %94 = builtin.unrealized_conversion_cast %92 : index to i64
      %95 = llvm.extractvalue %52[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %96 = llvm.mlir.constant(8192 : index) : i64
      %97 = llvm.mul %77, %96 overflow<nsw, nuw> : i64
      %98 = llvm.mlir.constant(128 : index) : i64
      %99 = llvm.mul %78, %98 overflow<nsw, nuw> : i64
      %100 = llvm.add %97, %99 overflow<nsw, nuw> : i64
      %101 = llvm.add %100, %94 overflow<nsw, nuw> : i64
      %102 = llvm.getelementptr inbounds|nuw %95[%101] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
      %103 = llvm.load %102 : !llvm.ptr -> bf16
      %104 = llvm.extractvalue %35[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %105 = llvm.mlir.constant(12288 : index) : i64
      %106 = llvm.mul %77, %105 overflow<nsw, nuw> : i64
      %107 = llvm.mlir.constant(96 : index) : i64
      %108 = llvm.mul %94, %107 overflow<nsw, nuw> : i64
      %109 = llvm.add %106, %108 overflow<nsw, nuw> : i64
      %110 = llvm.add %109, %79 overflow<nsw, nuw> : i64
      %111 = llvm.getelementptr inbounds|nuw %104[%110] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
      %112 = llvm.load %111 : !llvm.ptr -> bf16
      %113 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %114 = llvm.mlir.constant(6144 : index) : i64
      %115 = llvm.mul %77, %114 overflow<nsw, nuw> : i64
      %116 = llvm.mlir.constant(96 : index) : i64
      %117 = llvm.mul %78, %116 overflow<nsw, nuw> : i64
      %118 = llvm.add %115, %117 overflow<nsw, nuw> : i64
      %119 = llvm.add %118, %79 overflow<nsw, nuw> : i64
      %120 = llvm.getelementptr inbounds|nuw %113[%119] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
      %121 = llvm.load %120 : !llvm.ptr -> bf16
      %122 = llvm.fmul %103, %112 : bf16
      %123 = llvm.fadd %121, %122 : bf16
      %124 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
      %125 = llvm.mlir.constant(6144 : index) : i64
      %126 = llvm.mul %77, %125 overflow<nsw, nuw> : i64
      %127 = llvm.mlir.constant(96 : index) : i64
      %128 = llvm.mul %78, %127 overflow<nsw, nuw> : i64
      %129 = llvm.add %126, %128 overflow<nsw, nuw> : i64
      %130 = llvm.add %129, %79 overflow<nsw, nuw> : i64
      %131 = llvm.getelementptr inbounds|nuw %124[%130] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
      llvm.store %123, %131 : bf16, !llvm.ptr
      %132 = arith.addi %92, %1 : index
      %133 = builtin.unrealized_conversion_cast %132 : index to i64
      llvm.br ^bb2(%133 : i64)
    ^bb4:  // pred: ^bb2
      llvm.br ^bb5
    ^bb5:  // 2 preds: ^bb0, ^bb4
      llvm.return
    }
  }
}
