; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define ptx_kernel void @forward_kernel_forward_kernel(i64 %0, i64 %1, float %2, ptr %3, ptr %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11) {
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %14 = sext i32 %13 to i64
  %15 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %16 = sext i32 %15 to i64
  %17 = add i64 %0, %14
  %18 = add i64 %1, %16
  br label %19

19:                                               ; preds = %22, %12
  %20 = phi i64 [ %28, %22 ], [ 0, %12 ]
  %21 = icmp slt i64 %20, 512
  br i1 %21, label %22, label %29

22:                                               ; preds = %19
  %23 = mul i64 %17, 512
  %24 = mul i64 %18, 512
  %25 = add i64 %23, %24
  %26 = add i64 %25, %20
  %27 = getelementptr float, ptr %4, i64 %26
  store float %2, ptr %27, align 4
  %28 = add i64 %20, 1
  br label %19

29:                                               ; preds = %19
  ret void
}

define ptx_kernel void @forward_kernel_0_forward_kernel(i64 %0, i64 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr %11, ptr %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28) {
  %30 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %31 = sext i32 %30 to i64
  %32 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %33 = sext i32 %32 to i64
  %34 = add i64 %0, %31
  %35 = add i64 %1, %33
  br label %36

36:                                               ; preds = %70, %29
  %37 = phi i64 [ %71, %70 ], [ 0, %29 ]
  %38 = icmp slt i64 %37, 512
  br i1 %38, label %39, label %72

39:                                               ; preds = %36
  br label %40

40:                                               ; preds = %43, %39
  %41 = phi i64 [ %69, %43 ], [ 0, %39 ]
  %42 = icmp slt i64 %41, 784
  br i1 %42, label %43, label %70

43:                                               ; preds = %40
  %44 = mul i64 %34, 784
  %45 = mul i64 %35, 784
  %46 = add i64 %44, %45
  %47 = add i64 %46, %41
  %48 = getelementptr float, ptr %3, i64 %47
  %49 = load float, ptr %48, align 4
  %50 = mul i64 %34, 401408
  %51 = mul i64 %41, 512
  %52 = add i64 %50, %51
  %53 = add i64 %52, %37
  %54 = getelementptr float, ptr %12, i64 %53
  %55 = load float, ptr %54, align 4
  %56 = mul i64 %34, 512
  %57 = mul i64 %35, 512
  %58 = add i64 %56, %57
  %59 = add i64 %58, %37
  %60 = getelementptr float, ptr %21, i64 %59
  %61 = load float, ptr %60, align 4
  %62 = fmul float %49, %55
  %63 = fadd float %61, %62
  %64 = mul i64 %34, 512
  %65 = mul i64 %35, 512
  %66 = add i64 %64, %65
  %67 = add i64 %66, %37
  %68 = getelementptr float, ptr %21, i64 %67
  store float %63, ptr %68, align 4
  %69 = add i64 %41, 1
  br label %40

70:                                               ; preds = %40
  %71 = add i64 %37, 1
  br label %36

72:                                               ; preds = %36
  ret void
}

define ptx_kernel void @forward_kernel_1_forward_kernel(i64 %0, i64 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22) {
  %24 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %25 = sext i32 %24 to i64
  %26 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %27 = sext i32 %26 to i64
  %28 = add i64 %0, %25
  %29 = add i64 %1, %27
  %30 = mul i64 %28, 512
  %31 = add i64 %30, %29
  %32 = getelementptr float, ptr %3, i64 %31
  %33 = load float, ptr %32, align 4
  %34 = mul i64 %28, 512
  %35 = add i64 %34, %29
  %36 = getelementptr float, ptr %10, i64 %35
  %37 = load float, ptr %36, align 4
  %38 = fadd float %33, %37
  %39 = mul i64 %28, 512
  %40 = add i64 %39, %29
  %41 = getelementptr float, ptr %17, i64 %40
  store float %38, ptr %41, align 4
  ret void
}

define ptx_kernel void @forward_kernel_2_forward_kernel(i64 %0, i64 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, float %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16) {
  %18 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %19 = sext i32 %18 to i64
  %20 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %21 = sext i32 %20 to i64
  %22 = add i64 %0, %19
  %23 = add i64 %1, %21
  %24 = mul i64 %22, 512
  %25 = add i64 %24, %23
  %26 = getelementptr float, ptr %3, i64 %25
  %27 = load float, ptr %26, align 4
  %28 = call float @llvm.maximum.f32(float %27, float %9)
  %29 = mul i64 %22, 512
  %30 = add i64 %29, %23
  %31 = getelementptr float, ptr %11, i64 %30
  store float %28, ptr %31, align 4
  ret void
}

define ptx_kernel void @forward_kernel_3_forward_kernel(i64 %0, i64 %1, float %2, ptr %3, ptr %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11) {
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %14 = sext i32 %13 to i64
  %15 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %16 = sext i32 %15 to i64
  %17 = add i64 %0, %14
  %18 = add i64 %1, %16
  br label %19

19:                                               ; preds = %22, %12
  %20 = phi i64 [ %28, %22 ], [ 0, %12 ]
  %21 = icmp slt i64 %20, 512
  br i1 %21, label %22, label %29

22:                                               ; preds = %19
  %23 = mul i64 %17, 512
  %24 = mul i64 %18, 512
  %25 = add i64 %23, %24
  %26 = add i64 %25, %20
  %27 = getelementptr float, ptr %4, i64 %26
  store float %2, ptr %27, align 4
  %28 = add i64 %20, 1
  br label %19

29:                                               ; preds = %19
  ret void
}

define ptx_kernel void @forward_kernel_4_forward_kernel(i64 %0, i64 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr %11, ptr %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28) {
  %30 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %31 = sext i32 %30 to i64
  %32 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %33 = sext i32 %32 to i64
  %34 = add i64 %0, %31
  %35 = add i64 %1, %33
  br label %36

36:                                               ; preds = %70, %29
  %37 = phi i64 [ %71, %70 ], [ 0, %29 ]
  %38 = icmp slt i64 %37, 512
  br i1 %38, label %39, label %72

39:                                               ; preds = %36
  br label %40

40:                                               ; preds = %43, %39
  %41 = phi i64 [ %69, %43 ], [ 0, %39 ]
  %42 = icmp slt i64 %41, 512
  br i1 %42, label %43, label %70

43:                                               ; preds = %40
  %44 = mul i64 %34, 512
  %45 = mul i64 %35, 512
  %46 = add i64 %44, %45
  %47 = add i64 %46, %41
  %48 = getelementptr float, ptr %3, i64 %47
  %49 = load float, ptr %48, align 4
  %50 = mul i64 %34, 262144
  %51 = mul i64 %41, 512
  %52 = add i64 %50, %51
  %53 = add i64 %52, %37
  %54 = getelementptr float, ptr %12, i64 %53
  %55 = load float, ptr %54, align 4
  %56 = mul i64 %34, 512
  %57 = mul i64 %35, 512
  %58 = add i64 %56, %57
  %59 = add i64 %58, %37
  %60 = getelementptr float, ptr %21, i64 %59
  %61 = load float, ptr %60, align 4
  %62 = fmul float %49, %55
  %63 = fadd float %61, %62
  %64 = mul i64 %34, 512
  %65 = mul i64 %35, 512
  %66 = add i64 %64, %65
  %67 = add i64 %66, %37
  %68 = getelementptr float, ptr %21, i64 %67
  store float %63, ptr %68, align 4
  %69 = add i64 %41, 1
  br label %40

70:                                               ; preds = %40
  %71 = add i64 %37, 1
  br label %36

72:                                               ; preds = %36
  ret void
}

define ptx_kernel void @forward_kernel_5_forward_kernel(i64 %0, i64 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22) {
  %24 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %25 = sext i32 %24 to i64
  %26 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %27 = sext i32 %26 to i64
  %28 = add i64 %0, %25
  %29 = add i64 %1, %27
  %30 = mul i64 %28, 512
  %31 = add i64 %30, %29
  %32 = getelementptr float, ptr %3, i64 %31
  %33 = load float, ptr %32, align 4
  %34 = mul i64 %28, 512
  %35 = add i64 %34, %29
  %36 = getelementptr float, ptr %10, i64 %35
  %37 = load float, ptr %36, align 4
  %38 = fadd float %33, %37
  %39 = mul i64 %28, 512
  %40 = add i64 %39, %29
  %41 = getelementptr float, ptr %17, i64 %40
  store float %38, ptr %41, align 4
  ret void
}

define ptx_kernel void @forward_kernel_6_forward_kernel(i64 %0, i64 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, float %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16) {
  %18 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %19 = sext i32 %18 to i64
  %20 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %21 = sext i32 %20 to i64
  %22 = add i64 %0, %19
  %23 = add i64 %1, %21
  %24 = mul i64 %22, 512
  %25 = add i64 %24, %23
  %26 = getelementptr float, ptr %3, i64 %25
  %27 = load float, ptr %26, align 4
  %28 = call float @llvm.maximum.f32(float %27, float %9)
  %29 = mul i64 %22, 512
  %30 = add i64 %29, %23
  %31 = getelementptr float, ptr %11, i64 %30
  store float %28, ptr %31, align 4
  ret void
}

define ptx_kernel void @forward_kernel_7_forward_kernel(i64 %0, i64 %1, float %2, ptr %3, ptr %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11) {
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %14 = sext i32 %13 to i64
  %15 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %16 = sext i32 %15 to i64
  %17 = add i64 %0, %14
  %18 = add i64 %1, %16
  br label %19

19:                                               ; preds = %22, %12
  %20 = phi i64 [ %28, %22 ], [ 0, %12 ]
  %21 = icmp slt i64 %20, 10
  br i1 %21, label %22, label %29

22:                                               ; preds = %19
  %23 = mul i64 %17, 10
  %24 = mul i64 %18, 10
  %25 = add i64 %23, %24
  %26 = add i64 %25, %20
  %27 = getelementptr float, ptr %4, i64 %26
  store float %2, ptr %27, align 4
  %28 = add i64 %20, 1
  br label %19

29:                                               ; preds = %19
  ret void
}

define ptx_kernel void @forward_kernel_8_forward_kernel(i64 %0, i64 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr %11, ptr %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28) {
  %30 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %31 = sext i32 %30 to i64
  %32 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %33 = sext i32 %32 to i64
  %34 = add i64 %0, %31
  %35 = add i64 %1, %33
  br label %36

36:                                               ; preds = %70, %29
  %37 = phi i64 [ %71, %70 ], [ 0, %29 ]
  %38 = icmp slt i64 %37, 10
  br i1 %38, label %39, label %72

39:                                               ; preds = %36
  br label %40

40:                                               ; preds = %43, %39
  %41 = phi i64 [ %69, %43 ], [ 0, %39 ]
  %42 = icmp slt i64 %41, 512
  br i1 %42, label %43, label %70

43:                                               ; preds = %40
  %44 = mul i64 %34, 512
  %45 = mul i64 %35, 512
  %46 = add i64 %44, %45
  %47 = add i64 %46, %41
  %48 = getelementptr float, ptr %3, i64 %47
  %49 = load float, ptr %48, align 4
  %50 = mul i64 %34, 5120
  %51 = mul i64 %41, 10
  %52 = add i64 %50, %51
  %53 = add i64 %52, %37
  %54 = getelementptr float, ptr %12, i64 %53
  %55 = load float, ptr %54, align 4
  %56 = mul i64 %34, 10
  %57 = mul i64 %35, 10
  %58 = add i64 %56, %57
  %59 = add i64 %58, %37
  %60 = getelementptr float, ptr %21, i64 %59
  %61 = load float, ptr %60, align 4
  %62 = fmul float %49, %55
  %63 = fadd float %61, %62
  %64 = mul i64 %34, 10
  %65 = mul i64 %35, 10
  %66 = add i64 %64, %65
  %67 = add i64 %66, %37
  %68 = getelementptr float, ptr %21, i64 %67
  store float %63, ptr %68, align 4
  %69 = add i64 %41, 1
  br label %40

70:                                               ; preds = %40
  %71 = add i64 %37, 1
  br label %36

72:                                               ; preds = %36
  ret void
}

define ptx_kernel void @forward_kernel_9_forward_kernel(i64 %0, i64 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22) {
  %24 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %25 = sext i32 %24 to i64
  %26 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %27 = sext i32 %26 to i64
  %28 = add i64 %0, %25
  %29 = add i64 %1, %27
  %30 = mul i64 %28, 10
  %31 = add i64 %30, %29
  %32 = getelementptr float, ptr %3, i64 %31
  %33 = load float, ptr %32, align 4
  %34 = mul i64 %28, 10
  %35 = add i64 %34, %29
  %36 = getelementptr float, ptr %10, i64 %35
  %37 = load float, ptr %36, align 4
  %38 = fadd float %33, %37
  %39 = mul i64 %28, 10
  %40 = add i64 %39, %29
  %41 = getelementptr float, ptr %17, i64 %40
  store float %38, ptr %41, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
