module @MNISTLinear attributes {gpu.container_module} {
  llvm.func @forward_kernel_forward_kernel(%arg0: i64, %arg1: i64, %arg2: f32, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64) attributes {gpu.kernel, nvvm.kernel} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(512 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = nvvm.read.ptx.sreg.ctaid.x : i32
    %4 = llvm.sext %3 : i32 to i64
    %5 = nvvm.read.ptx.sreg.tid.x : i32
    %6 = llvm.sext %5 : i32 to i64
    %7 = llvm.add %arg0, %4 : i64
    %8 = llvm.add %arg1, %6 : i64
    llvm.br ^bb1(%0 : i64)
  ^bb1(%9: i64):  // 2 preds: ^bb0, ^bb2
    %10 = llvm.icmp "slt" %9, %1 : i64
    llvm.cond_br %10, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %11 = llvm.mul %7, %1 : i64
    %12 = llvm.mul %8, %1 : i64
    %13 = llvm.add %11, %12 : i64
    %14 = llvm.add %13, %9 : i64
    %15 = llvm.getelementptr %arg4[%14] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %arg2, %15 : f32, !llvm.ptr
    %16 = llvm.add %9, %2 : i64
    llvm.br ^bb1(%16 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @forward_kernel_0_forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: !llvm.ptr, %arg21: !llvm.ptr, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64) attributes {gpu.kernel, nvvm.kernel} {
    %0 = llvm.mlir.constant(401408 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(512 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(784 : index) : i64
    %5 = nvvm.read.ptx.sreg.ctaid.x : i32
    %6 = llvm.sext %5 : i32 to i64
    %7 = nvvm.read.ptx.sreg.tid.x : i32
    %8 = llvm.sext %7 : i32 to i64
    %9 = llvm.add %arg0, %6 : i64
    %10 = llvm.add %arg1, %8 : i64
    llvm.br ^bb1(%1 : i64)
  ^bb1(%11: i64):  // 2 preds: ^bb0, ^bb5
    %12 = llvm.icmp "slt" %11, %2 : i64
    llvm.cond_br %12, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%1 : i64)
  ^bb3(%13: i64):  // 2 preds: ^bb2, ^bb4
    %14 = llvm.icmp "slt" %13, %4 : i64
    llvm.cond_br %14, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %15 = llvm.mul %9, %4 : i64
    %16 = llvm.mul %10, %4 : i64
    %17 = llvm.add %15, %16 : i64
    %18 = llvm.add %17, %13 : i64
    %19 = llvm.getelementptr %arg3[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %20 = llvm.load %19 : !llvm.ptr -> f32
    %21 = llvm.mul %9, %0 : i64
    %22 = llvm.mul %13, %2 : i64
    %23 = llvm.add %21, %22 : i64
    %24 = llvm.add %23, %11 : i64
    %25 = llvm.getelementptr %arg12[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %26 = llvm.load %25 : !llvm.ptr -> f32
    %27 = llvm.mul %9, %2 : i64
    %28 = llvm.mul %10, %2 : i64
    %29 = llvm.add %27, %28 : i64
    %30 = llvm.add %29, %11 : i64
    %31 = llvm.getelementptr %arg21[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %32 = llvm.load %31 : !llvm.ptr -> f32
    %33 = llvm.fmul %20, %26 : f32
    %34 = llvm.fadd %32, %33 : f32
    %35 = llvm.mul %9, %2 : i64
    %36 = llvm.mul %10, %2 : i64
    %37 = llvm.add %35, %36 : i64
    %38 = llvm.add %37, %11 : i64
    %39 = llvm.getelementptr %arg21[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %34, %39 : f32, !llvm.ptr
    %40 = llvm.add %13, %3 : i64
    llvm.br ^bb3(%40 : i64)
  ^bb5:  // pred: ^bb3
    %41 = llvm.add %11, %3 : i64
    llvm.br ^bb1(%41 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
  llvm.func @forward_kernel_1_forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: !llvm.ptr, %arg17: !llvm.ptr, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64) attributes {gpu.kernel, nvvm.kernel} {
    %0 = llvm.mlir.constant(512 : index) : i64
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = llvm.sext %1 : i32 to i64
    %3 = nvvm.read.ptx.sreg.tid.x : i32
    %4 = llvm.sext %3 : i32 to i64
    %5 = llvm.add %arg0, %2 : i64
    %6 = llvm.add %arg1, %4 : i64
    %7 = llvm.mul %5, %0 : i64
    %8 = llvm.add %7, %6 : i64
    %9 = llvm.getelementptr %arg3[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %10 = llvm.load %9 : !llvm.ptr -> f32
    %11 = llvm.mul %5, %0 : i64
    %12 = llvm.add %11, %6 : i64
    %13 = llvm.getelementptr %arg10[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %14 = llvm.load %13 : !llvm.ptr -> f32
    %15 = llvm.fadd %10, %14 : f32
    %16 = llvm.mul %5, %0 : i64
    %17 = llvm.add %16, %6 : i64
    %18 = llvm.getelementptr %arg17[%17] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %15, %18 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func @forward_kernel_2_forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: f32, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64) attributes {gpu.kernel, nvvm.kernel} {
    %0 = llvm.mlir.constant(512 : index) : i64
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = llvm.sext %1 : i32 to i64
    %3 = nvvm.read.ptx.sreg.tid.x : i32
    %4 = llvm.sext %3 : i32 to i64
    %5 = llvm.add %arg0, %2 : i64
    %6 = llvm.add %arg1, %4 : i64
    %7 = llvm.mul %5, %0 : i64
    %8 = llvm.add %7, %6 : i64
    %9 = llvm.getelementptr %arg3[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %10 = llvm.load %9 : !llvm.ptr -> f32
    %11 = llvm.intr.maximum(%10, %arg9) : (f32, f32) -> f32
    %12 = llvm.mul %5, %0 : i64
    %13 = llvm.add %12, %6 : i64
    %14 = llvm.getelementptr %arg11[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %11, %14 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func @forward_kernel_3_forward_kernel(%arg0: i64, %arg1: i64, %arg2: f32, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64) attributes {gpu.kernel, nvvm.kernel} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(512 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = nvvm.read.ptx.sreg.ctaid.x : i32
    %4 = llvm.sext %3 : i32 to i64
    %5 = nvvm.read.ptx.sreg.tid.x : i32
    %6 = llvm.sext %5 : i32 to i64
    %7 = llvm.add %arg0, %4 : i64
    %8 = llvm.add %arg1, %6 : i64
    llvm.br ^bb1(%0 : i64)
  ^bb1(%9: i64):  // 2 preds: ^bb0, ^bb2
    %10 = llvm.icmp "slt" %9, %1 : i64
    llvm.cond_br %10, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %11 = llvm.mul %7, %1 : i64
    %12 = llvm.mul %8, %1 : i64
    %13 = llvm.add %11, %12 : i64
    %14 = llvm.add %13, %9 : i64
    %15 = llvm.getelementptr %arg4[%14] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %arg2, %15 : f32, !llvm.ptr
    %16 = llvm.add %9, %2 : i64
    llvm.br ^bb1(%16 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @forward_kernel_4_forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: !llvm.ptr, %arg21: !llvm.ptr, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64) attributes {gpu.kernel, nvvm.kernel} {
    %0 = llvm.mlir.constant(262144 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(512 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = nvvm.read.ptx.sreg.ctaid.x : i32
    %5 = llvm.sext %4 : i32 to i64
    %6 = nvvm.read.ptx.sreg.tid.x : i32
    %7 = llvm.sext %6 : i32 to i64
    %8 = llvm.add %arg0, %5 : i64
    %9 = llvm.add %arg1, %7 : i64
    llvm.br ^bb1(%1 : i64)
  ^bb1(%10: i64):  // 2 preds: ^bb0, ^bb5
    %11 = llvm.icmp "slt" %10, %2 : i64
    llvm.cond_br %11, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%1 : i64)
  ^bb3(%12: i64):  // 2 preds: ^bb2, ^bb4
    %13 = llvm.icmp "slt" %12, %2 : i64
    llvm.cond_br %13, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %14 = llvm.mul %8, %2 : i64
    %15 = llvm.mul %9, %2 : i64
    %16 = llvm.add %14, %15 : i64
    %17 = llvm.add %16, %12 : i64
    %18 = llvm.getelementptr %arg3[%17] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %19 = llvm.load %18 : !llvm.ptr -> f32
    %20 = llvm.mul %8, %0 : i64
    %21 = llvm.mul %12, %2 : i64
    %22 = llvm.add %20, %21 : i64
    %23 = llvm.add %22, %10 : i64
    %24 = llvm.getelementptr %arg12[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %25 = llvm.load %24 : !llvm.ptr -> f32
    %26 = llvm.mul %8, %2 : i64
    %27 = llvm.mul %9, %2 : i64
    %28 = llvm.add %26, %27 : i64
    %29 = llvm.add %28, %10 : i64
    %30 = llvm.getelementptr %arg21[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %31 = llvm.load %30 : !llvm.ptr -> f32
    %32 = llvm.fmul %19, %25 : f32
    %33 = llvm.fadd %31, %32 : f32
    %34 = llvm.mul %8, %2 : i64
    %35 = llvm.mul %9, %2 : i64
    %36 = llvm.add %34, %35 : i64
    %37 = llvm.add %36, %10 : i64
    %38 = llvm.getelementptr %arg21[%37] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %33, %38 : f32, !llvm.ptr
    %39 = llvm.add %12, %3 : i64
    llvm.br ^bb3(%39 : i64)
  ^bb5:  // pred: ^bb3
    %40 = llvm.add %10, %3 : i64
    llvm.br ^bb1(%40 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
  llvm.func @forward_kernel_5_forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: !llvm.ptr, %arg17: !llvm.ptr, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64) attributes {gpu.kernel, nvvm.kernel} {
    %0 = llvm.mlir.constant(512 : index) : i64
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = llvm.sext %1 : i32 to i64
    %3 = nvvm.read.ptx.sreg.tid.x : i32
    %4 = llvm.sext %3 : i32 to i64
    %5 = llvm.add %arg0, %2 : i64
    %6 = llvm.add %arg1, %4 : i64
    %7 = llvm.mul %5, %0 : i64
    %8 = llvm.add %7, %6 : i64
    %9 = llvm.getelementptr %arg3[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %10 = llvm.load %9 : !llvm.ptr -> f32
    %11 = llvm.mul %5, %0 : i64
    %12 = llvm.add %11, %6 : i64
    %13 = llvm.getelementptr %arg10[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %14 = llvm.load %13 : !llvm.ptr -> f32
    %15 = llvm.fadd %10, %14 : f32
    %16 = llvm.mul %5, %0 : i64
    %17 = llvm.add %16, %6 : i64
    %18 = llvm.getelementptr %arg17[%17] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %15, %18 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func @forward_kernel_6_forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: f32, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64) attributes {gpu.kernel, nvvm.kernel} {
    %0 = llvm.mlir.constant(512 : index) : i64
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = llvm.sext %1 : i32 to i64
    %3 = nvvm.read.ptx.sreg.tid.x : i32
    %4 = llvm.sext %3 : i32 to i64
    %5 = llvm.add %arg0, %2 : i64
    %6 = llvm.add %arg1, %4 : i64
    %7 = llvm.mul %5, %0 : i64
    %8 = llvm.add %7, %6 : i64
    %9 = llvm.getelementptr %arg3[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %10 = llvm.load %9 : !llvm.ptr -> f32
    %11 = llvm.intr.maximum(%10, %arg9) : (f32, f32) -> f32
    %12 = llvm.mul %5, %0 : i64
    %13 = llvm.add %12, %6 : i64
    %14 = llvm.getelementptr %arg11[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %11, %14 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func @forward_kernel_7_forward_kernel(%arg0: i64, %arg1: i64, %arg2: f32, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64) attributes {gpu.kernel, nvvm.kernel} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(10 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = nvvm.read.ptx.sreg.ctaid.x : i32
    %4 = llvm.sext %3 : i32 to i64
    %5 = nvvm.read.ptx.sreg.tid.x : i32
    %6 = llvm.sext %5 : i32 to i64
    %7 = llvm.add %arg0, %4 : i64
    %8 = llvm.add %arg1, %6 : i64
    llvm.br ^bb1(%0 : i64)
  ^bb1(%9: i64):  // 2 preds: ^bb0, ^bb2
    %10 = llvm.icmp "slt" %9, %1 : i64
    llvm.cond_br %10, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %11 = llvm.mul %7, %1 : i64
    %12 = llvm.mul %8, %1 : i64
    %13 = llvm.add %11, %12 : i64
    %14 = llvm.add %13, %9 : i64
    %15 = llvm.getelementptr %arg4[%14] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %arg2, %15 : f32, !llvm.ptr
    %16 = llvm.add %9, %2 : i64
    llvm.br ^bb1(%16 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @forward_kernel_8_forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: !llvm.ptr, %arg21: !llvm.ptr, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64) attributes {gpu.kernel, nvvm.kernel} {
    %0 = llvm.mlir.constant(5120 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(10 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(512 : index) : i64
    %5 = nvvm.read.ptx.sreg.ctaid.x : i32
    %6 = llvm.sext %5 : i32 to i64
    %7 = nvvm.read.ptx.sreg.tid.x : i32
    %8 = llvm.sext %7 : i32 to i64
    %9 = llvm.add %arg0, %6 : i64
    %10 = llvm.add %arg1, %8 : i64
    llvm.br ^bb1(%1 : i64)
  ^bb1(%11: i64):  // 2 preds: ^bb0, ^bb5
    %12 = llvm.icmp "slt" %11, %2 : i64
    llvm.cond_br %12, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%1 : i64)
  ^bb3(%13: i64):  // 2 preds: ^bb2, ^bb4
    %14 = llvm.icmp "slt" %13, %4 : i64
    llvm.cond_br %14, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %15 = llvm.mul %9, %4 : i64
    %16 = llvm.mul %10, %4 : i64
    %17 = llvm.add %15, %16 : i64
    %18 = llvm.add %17, %13 : i64
    %19 = llvm.getelementptr %arg3[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %20 = llvm.load %19 : !llvm.ptr -> f32
    %21 = llvm.mul %9, %0 : i64
    %22 = llvm.mul %13, %2 : i64
    %23 = llvm.add %21, %22 : i64
    %24 = llvm.add %23, %11 : i64
    %25 = llvm.getelementptr %arg12[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %26 = llvm.load %25 : !llvm.ptr -> f32
    %27 = llvm.mul %9, %2 : i64
    %28 = llvm.mul %10, %2 : i64
    %29 = llvm.add %27, %28 : i64
    %30 = llvm.add %29, %11 : i64
    %31 = llvm.getelementptr %arg21[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %32 = llvm.load %31 : !llvm.ptr -> f32
    %33 = llvm.fmul %20, %26 : f32
    %34 = llvm.fadd %32, %33 : f32
    %35 = llvm.mul %9, %2 : i64
    %36 = llvm.mul %10, %2 : i64
    %37 = llvm.add %35, %36 : i64
    %38 = llvm.add %37, %11 : i64
    %39 = llvm.getelementptr %arg21[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %34, %39 : f32, !llvm.ptr
    %40 = llvm.add %13, %3 : i64
    llvm.br ^bb3(%40 : i64)
  ^bb5:  // pred: ^bb3
    %41 = llvm.add %11, %3 : i64
    llvm.br ^bb1(%41 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
  llvm.func @forward_kernel_9_forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: !llvm.ptr, %arg17: !llvm.ptr, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64) attributes {gpu.kernel, nvvm.kernel} {
    %0 = llvm.mlir.constant(10 : index) : i64
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = llvm.sext %1 : i32 to i64
    %3 = nvvm.read.ptx.sreg.tid.x : i32
    %4 = llvm.sext %3 : i32 to i64
    %5 = llvm.add %arg0, %2 : i64
    %6 = llvm.add %arg1, %4 : i64
    %7 = llvm.mul %5, %0 : i64
    %8 = llvm.add %7, %6 : i64
    %9 = llvm.getelementptr %arg3[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %10 = llvm.load %9 : !llvm.ptr -> f32
    %11 = llvm.mul %5, %0 : i64
    %12 = llvm.add %11, %6 : i64
    %13 = llvm.getelementptr %arg10[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %14 = llvm.load %13 : !llvm.ptr -> f32
    %15 = llvm.fadd %10, %14 : f32
    %16 = llvm.mul %5, %0 : i64
    %17 = llvm.add %16, %6 : i64
    %18 = llvm.getelementptr %arg17[%17] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %15, %18 : f32, !llvm.ptr
    llvm.return
  }
  gpu.module @forward_kernel [#nvvm.target] {
    llvm.func @forward_kernel_forward_kernel(%arg0: i64, %arg1: i64, %arg2: f32, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.constant(0 : index) : i64
      %1 = llvm.mlir.constant(512 : index) : i64
      %2 = llvm.mlir.constant(1 : index) : i64
      %3 = nvvm.read.ptx.sreg.ctaid.x : i32
      %4 = llvm.sext %3 : i32 to i64
      %5 = nvvm.read.ptx.sreg.tid.x : i32
      %6 = llvm.sext %5 : i32 to i64
      %7 = llvm.add %arg0, %4 : i64
      %8 = llvm.add %arg1, %6 : i64
      llvm.br ^bb1(%0 : i64)
    ^bb1(%9: i64):  // 2 preds: ^bb0, ^bb2
      %10 = llvm.icmp "slt" %9, %1 : i64
      llvm.cond_br %10, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %11 = llvm.mul %7, %1 : i64
      %12 = llvm.mul %8, %1 : i64
      %13 = llvm.add %11, %12 : i64
      %14 = llvm.add %13, %9 : i64
      %15 = llvm.getelementptr %arg4[%14] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %arg2, %15 : f32, !llvm.ptr
      %16 = llvm.add %9, %2 : i64
      llvm.br ^bb1(%16 : i64)
    ^bb3:  // pred: ^bb1
      llvm.return
    }
  }
  gpu.module @forward_kernel_0 [#nvvm.target] {
    llvm.func @forward_kernel_0_forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: !llvm.ptr, %arg21: !llvm.ptr, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.constant(401408 : index) : i64
      %1 = llvm.mlir.constant(0 : index) : i64
      %2 = llvm.mlir.constant(512 : index) : i64
      %3 = llvm.mlir.constant(1 : index) : i64
      %4 = llvm.mlir.constant(784 : index) : i64
      %5 = nvvm.read.ptx.sreg.ctaid.x : i32
      %6 = llvm.sext %5 : i32 to i64
      %7 = nvvm.read.ptx.sreg.tid.x : i32
      %8 = llvm.sext %7 : i32 to i64
      %9 = llvm.add %arg0, %6 : i64
      %10 = llvm.add %arg1, %8 : i64
      llvm.br ^bb1(%1 : i64)
    ^bb1(%11: i64):  // 2 preds: ^bb0, ^bb5
      %12 = llvm.icmp "slt" %11, %2 : i64
      llvm.cond_br %12, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      llvm.br ^bb3(%1 : i64)
    ^bb3(%13: i64):  // 2 preds: ^bb2, ^bb4
      %14 = llvm.icmp "slt" %13, %4 : i64
      llvm.cond_br %14, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %15 = llvm.mul %9, %4 : i64
      %16 = llvm.mul %10, %4 : i64
      %17 = llvm.add %15, %16 : i64
      %18 = llvm.add %17, %13 : i64
      %19 = llvm.getelementptr %arg3[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %20 = llvm.load %19 : !llvm.ptr -> f32
      %21 = llvm.mul %9, %0 : i64
      %22 = llvm.mul %13, %2 : i64
      %23 = llvm.add %21, %22 : i64
      %24 = llvm.add %23, %11 : i64
      %25 = llvm.getelementptr %arg12[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %26 = llvm.load %25 : !llvm.ptr -> f32
      %27 = llvm.mul %9, %2 : i64
      %28 = llvm.mul %10, %2 : i64
      %29 = llvm.add %27, %28 : i64
      %30 = llvm.add %29, %11 : i64
      %31 = llvm.getelementptr %arg21[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %32 = llvm.load %31 : !llvm.ptr -> f32
      %33 = llvm.fmul %20, %26 : f32
      %34 = llvm.fadd %32, %33 : f32
      %35 = llvm.mul %9, %2 : i64
      %36 = llvm.mul %10, %2 : i64
      %37 = llvm.add %35, %36 : i64
      %38 = llvm.add %37, %11 : i64
      %39 = llvm.getelementptr %arg21[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %34, %39 : f32, !llvm.ptr
      %40 = llvm.add %13, %3 : i64
      llvm.br ^bb3(%40 : i64)
    ^bb5:  // pred: ^bb3
      %41 = llvm.add %11, %3 : i64
      llvm.br ^bb1(%41 : i64)
    ^bb6:  // pred: ^bb1
      llvm.return
    }
  }
  gpu.module @forward_kernel_1 [#nvvm.target] {
    llvm.func @forward_kernel_1_forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: !llvm.ptr, %arg17: !llvm.ptr, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.constant(512 : index) : i64
      %1 = nvvm.read.ptx.sreg.ctaid.x : i32
      %2 = llvm.sext %1 : i32 to i64
      %3 = nvvm.read.ptx.sreg.tid.x : i32
      %4 = llvm.sext %3 : i32 to i64
      %5 = llvm.add %arg0, %2 : i64
      %6 = llvm.add %arg1, %4 : i64
      %7 = llvm.mul %5, %0 : i64
      %8 = llvm.add %7, %6 : i64
      %9 = llvm.getelementptr %arg3[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %10 = llvm.load %9 : !llvm.ptr -> f32
      %11 = llvm.mul %5, %0 : i64
      %12 = llvm.add %11, %6 : i64
      %13 = llvm.getelementptr %arg10[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %14 = llvm.load %13 : !llvm.ptr -> f32
      %15 = llvm.fadd %10, %14 : f32
      %16 = llvm.mul %5, %0 : i64
      %17 = llvm.add %16, %6 : i64
      %18 = llvm.getelementptr %arg17[%17] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %15, %18 : f32, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @forward_kernel_2 [#nvvm.target] {
    llvm.func @forward_kernel_2_forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: f32, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.constant(512 : index) : i64
      %1 = nvvm.read.ptx.sreg.ctaid.x : i32
      %2 = llvm.sext %1 : i32 to i64
      %3 = nvvm.read.ptx.sreg.tid.x : i32
      %4 = llvm.sext %3 : i32 to i64
      %5 = llvm.add %arg0, %2 : i64
      %6 = llvm.add %arg1, %4 : i64
      %7 = llvm.mul %5, %0 : i64
      %8 = llvm.add %7, %6 : i64
      %9 = llvm.getelementptr %arg3[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %10 = llvm.load %9 : !llvm.ptr -> f32
      %11 = llvm.intr.maximum(%10, %arg9) : (f32, f32) -> f32
      %12 = llvm.mul %5, %0 : i64
      %13 = llvm.add %12, %6 : i64
      %14 = llvm.getelementptr %arg11[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %11, %14 : f32, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @forward_kernel_3 [#nvvm.target] {
    llvm.func @forward_kernel_3_forward_kernel(%arg0: i64, %arg1: i64, %arg2: f32, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.constant(0 : index) : i64
      %1 = llvm.mlir.constant(512 : index) : i64
      %2 = llvm.mlir.constant(1 : index) : i64
      %3 = nvvm.read.ptx.sreg.ctaid.x : i32
      %4 = llvm.sext %3 : i32 to i64
      %5 = nvvm.read.ptx.sreg.tid.x : i32
      %6 = llvm.sext %5 : i32 to i64
      %7 = llvm.add %arg0, %4 : i64
      %8 = llvm.add %arg1, %6 : i64
      llvm.br ^bb1(%0 : i64)
    ^bb1(%9: i64):  // 2 preds: ^bb0, ^bb2
      %10 = llvm.icmp "slt" %9, %1 : i64
      llvm.cond_br %10, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %11 = llvm.mul %7, %1 : i64
      %12 = llvm.mul %8, %1 : i64
      %13 = llvm.add %11, %12 : i64
      %14 = llvm.add %13, %9 : i64
      %15 = llvm.getelementptr %arg4[%14] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %arg2, %15 : f32, !llvm.ptr
      %16 = llvm.add %9, %2 : i64
      llvm.br ^bb1(%16 : i64)
    ^bb3:  // pred: ^bb1
      llvm.return
    }
  }
  gpu.module @forward_kernel_4 [#nvvm.target] {
    llvm.func @forward_kernel_4_forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: !llvm.ptr, %arg21: !llvm.ptr, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.constant(262144 : index) : i64
      %1 = llvm.mlir.constant(0 : index) : i64
      %2 = llvm.mlir.constant(512 : index) : i64
      %3 = llvm.mlir.constant(1 : index) : i64
      %4 = nvvm.read.ptx.sreg.ctaid.x : i32
      %5 = llvm.sext %4 : i32 to i64
      %6 = nvvm.read.ptx.sreg.tid.x : i32
      %7 = llvm.sext %6 : i32 to i64
      %8 = llvm.add %arg0, %5 : i64
      %9 = llvm.add %arg1, %7 : i64
      llvm.br ^bb1(%1 : i64)
    ^bb1(%10: i64):  // 2 preds: ^bb0, ^bb5
      %11 = llvm.icmp "slt" %10, %2 : i64
      llvm.cond_br %11, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      llvm.br ^bb3(%1 : i64)
    ^bb3(%12: i64):  // 2 preds: ^bb2, ^bb4
      %13 = llvm.icmp "slt" %12, %2 : i64
      llvm.cond_br %13, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %14 = llvm.mul %8, %2 : i64
      %15 = llvm.mul %9, %2 : i64
      %16 = llvm.add %14, %15 : i64
      %17 = llvm.add %16, %12 : i64
      %18 = llvm.getelementptr %arg3[%17] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %19 = llvm.load %18 : !llvm.ptr -> f32
      %20 = llvm.mul %8, %0 : i64
      %21 = llvm.mul %12, %2 : i64
      %22 = llvm.add %20, %21 : i64
      %23 = llvm.add %22, %10 : i64
      %24 = llvm.getelementptr %arg12[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %25 = llvm.load %24 : !llvm.ptr -> f32
      %26 = llvm.mul %8, %2 : i64
      %27 = llvm.mul %9, %2 : i64
      %28 = llvm.add %26, %27 : i64
      %29 = llvm.add %28, %10 : i64
      %30 = llvm.getelementptr %arg21[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %31 = llvm.load %30 : !llvm.ptr -> f32
      %32 = llvm.fmul %19, %25 : f32
      %33 = llvm.fadd %31, %32 : f32
      %34 = llvm.mul %8, %2 : i64
      %35 = llvm.mul %9, %2 : i64
      %36 = llvm.add %34, %35 : i64
      %37 = llvm.add %36, %10 : i64
      %38 = llvm.getelementptr %arg21[%37] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %33, %38 : f32, !llvm.ptr
      %39 = llvm.add %12, %3 : i64
      llvm.br ^bb3(%39 : i64)
    ^bb5:  // pred: ^bb3
      %40 = llvm.add %10, %3 : i64
      llvm.br ^bb1(%40 : i64)
    ^bb6:  // pred: ^bb1
      llvm.return
    }
  }
  gpu.module @forward_kernel_5 [#nvvm.target] {
    llvm.func @forward_kernel_5_forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: !llvm.ptr, %arg17: !llvm.ptr, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.constant(512 : index) : i64
      %1 = nvvm.read.ptx.sreg.ctaid.x : i32
      %2 = llvm.sext %1 : i32 to i64
      %3 = nvvm.read.ptx.sreg.tid.x : i32
      %4 = llvm.sext %3 : i32 to i64
      %5 = llvm.add %arg0, %2 : i64
      %6 = llvm.add %arg1, %4 : i64
      %7 = llvm.mul %5, %0 : i64
      %8 = llvm.add %7, %6 : i64
      %9 = llvm.getelementptr %arg3[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %10 = llvm.load %9 : !llvm.ptr -> f32
      %11 = llvm.mul %5, %0 : i64
      %12 = llvm.add %11, %6 : i64
      %13 = llvm.getelementptr %arg10[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %14 = llvm.load %13 : !llvm.ptr -> f32
      %15 = llvm.fadd %10, %14 : f32
      %16 = llvm.mul %5, %0 : i64
      %17 = llvm.add %16, %6 : i64
      %18 = llvm.getelementptr %arg17[%17] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %15, %18 : f32, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @forward_kernel_6 [#nvvm.target] {
    llvm.func @forward_kernel_6_forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: f32, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.constant(512 : index) : i64
      %1 = nvvm.read.ptx.sreg.ctaid.x : i32
      %2 = llvm.sext %1 : i32 to i64
      %3 = nvvm.read.ptx.sreg.tid.x : i32
      %4 = llvm.sext %3 : i32 to i64
      %5 = llvm.add %arg0, %2 : i64
      %6 = llvm.add %arg1, %4 : i64
      %7 = llvm.mul %5, %0 : i64
      %8 = llvm.add %7, %6 : i64
      %9 = llvm.getelementptr %arg3[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %10 = llvm.load %9 : !llvm.ptr -> f32
      %11 = llvm.intr.maximum(%10, %arg9) : (f32, f32) -> f32
      %12 = llvm.mul %5, %0 : i64
      %13 = llvm.add %12, %6 : i64
      %14 = llvm.getelementptr %arg11[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %11, %14 : f32, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @forward_kernel_7 [#nvvm.target] {
    llvm.func @forward_kernel_7_forward_kernel(%arg0: i64, %arg1: i64, %arg2: f32, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.constant(0 : index) : i64
      %1 = llvm.mlir.constant(10 : index) : i64
      %2 = llvm.mlir.constant(1 : index) : i64
      %3 = nvvm.read.ptx.sreg.ctaid.x : i32
      %4 = llvm.sext %3 : i32 to i64
      %5 = nvvm.read.ptx.sreg.tid.x : i32
      %6 = llvm.sext %5 : i32 to i64
      %7 = llvm.add %arg0, %4 : i64
      %8 = llvm.add %arg1, %6 : i64
      llvm.br ^bb1(%0 : i64)
    ^bb1(%9: i64):  // 2 preds: ^bb0, ^bb2
      %10 = llvm.icmp "slt" %9, %1 : i64
      llvm.cond_br %10, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %11 = llvm.mul %7, %1 : i64
      %12 = llvm.mul %8, %1 : i64
      %13 = llvm.add %11, %12 : i64
      %14 = llvm.add %13, %9 : i64
      %15 = llvm.getelementptr %arg4[%14] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %arg2, %15 : f32, !llvm.ptr
      %16 = llvm.add %9, %2 : i64
      llvm.br ^bb1(%16 : i64)
    ^bb3:  // pred: ^bb1
      llvm.return
    }
  }
  gpu.module @forward_kernel_8 [#nvvm.target] {
    llvm.func @forward_kernel_8_forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: !llvm.ptr, %arg21: !llvm.ptr, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.constant(5120 : index) : i64
      %1 = llvm.mlir.constant(0 : index) : i64
      %2 = llvm.mlir.constant(10 : index) : i64
      %3 = llvm.mlir.constant(1 : index) : i64
      %4 = llvm.mlir.constant(512 : index) : i64
      %5 = nvvm.read.ptx.sreg.ctaid.x : i32
      %6 = llvm.sext %5 : i32 to i64
      %7 = nvvm.read.ptx.sreg.tid.x : i32
      %8 = llvm.sext %7 : i32 to i64
      %9 = llvm.add %arg0, %6 : i64
      %10 = llvm.add %arg1, %8 : i64
      llvm.br ^bb1(%1 : i64)
    ^bb1(%11: i64):  // 2 preds: ^bb0, ^bb5
      %12 = llvm.icmp "slt" %11, %2 : i64
      llvm.cond_br %12, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      llvm.br ^bb3(%1 : i64)
    ^bb3(%13: i64):  // 2 preds: ^bb2, ^bb4
      %14 = llvm.icmp "slt" %13, %4 : i64
      llvm.cond_br %14, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %15 = llvm.mul %9, %4 : i64
      %16 = llvm.mul %10, %4 : i64
      %17 = llvm.add %15, %16 : i64
      %18 = llvm.add %17, %13 : i64
      %19 = llvm.getelementptr %arg3[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %20 = llvm.load %19 : !llvm.ptr -> f32
      %21 = llvm.mul %9, %0 : i64
      %22 = llvm.mul %13, %2 : i64
      %23 = llvm.add %21, %22 : i64
      %24 = llvm.add %23, %11 : i64
      %25 = llvm.getelementptr %arg12[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %26 = llvm.load %25 : !llvm.ptr -> f32
      %27 = llvm.mul %9, %2 : i64
      %28 = llvm.mul %10, %2 : i64
      %29 = llvm.add %27, %28 : i64
      %30 = llvm.add %29, %11 : i64
      %31 = llvm.getelementptr %arg21[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %32 = llvm.load %31 : !llvm.ptr -> f32
      %33 = llvm.fmul %20, %26 : f32
      %34 = llvm.fadd %32, %33 : f32
      %35 = llvm.mul %9, %2 : i64
      %36 = llvm.mul %10, %2 : i64
      %37 = llvm.add %35, %36 : i64
      %38 = llvm.add %37, %11 : i64
      %39 = llvm.getelementptr %arg21[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %34, %39 : f32, !llvm.ptr
      %40 = llvm.add %13, %3 : i64
      llvm.br ^bb3(%40 : i64)
    ^bb5:  // pred: ^bb3
      %41 = llvm.add %11, %3 : i64
      llvm.br ^bb1(%41 : i64)
    ^bb6:  // pred: ^bb1
      llvm.return
    }
  }
  gpu.module @forward_kernel_9 [#nvvm.target] {
    llvm.func @forward_kernel_9_forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: !llvm.ptr, %arg17: !llvm.ptr, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.constant(10 : index) : i64
      %1 = nvvm.read.ptx.sreg.ctaid.x : i32
      %2 = llvm.sext %1 : i32 to i64
      %3 = nvvm.read.ptx.sreg.tid.x : i32
      %4 = llvm.sext %3 : i32 to i64
      %5 = llvm.add %arg0, %2 : i64
      %6 = llvm.add %arg1, %4 : i64
      %7 = llvm.mul %5, %0 : i64
      %8 = llvm.add %7, %6 : i64
      %9 = llvm.getelementptr %arg3[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %10 = llvm.load %9 : !llvm.ptr -> f32
      %11 = llvm.mul %5, %0 : i64
      %12 = llvm.add %11, %6 : i64
      %13 = llvm.getelementptr %arg10[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %14 = llvm.load %13 : !llvm.ptr -> f32
      %15 = llvm.fadd %10, %14 : f32
      %16 = llvm.mul %5, %0 : i64
      %17 = llvm.add %16, %6 : i64
      %18 = llvm.getelementptr %arg17[%17] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %15, %18 : f32, !llvm.ptr
      llvm.return
    }
  }
}
