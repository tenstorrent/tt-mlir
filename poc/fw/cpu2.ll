; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @cpu_hoisted_ttir_abs_713b937c(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13) {
  %15 = alloca float, i64 4000, align 64
  br label %16

16:                                               ; preds = %30, %14
  %17 = phi i64 [ %31, %30 ], [ 0, %14 ]
  %18 = icmp slt i64 %17, 4000
  br i1 %18, label %19, label %32

19:                                               ; preds = %16
  br label %20

20:                                               ; preds = %23, %19
  %21 = phi i64 [ %29, %23 ], [ 0, %19 ]
  %22 = icmp slt i64 %21, 1
  br i1 %22, label %23, label %30

23:                                               ; preds = %20
  %24 = add nuw nsw i64 %17, %21
  %25 = getelementptr inbounds nuw float, ptr %1, i64 %24
  %26 = load float, ptr %25, align 4
  %27 = call float @llvm.fabs.f32(float %26)
  %28 = getelementptr inbounds nuw float, ptr %15, i64 %24
  store float %27, ptr %28, align 4
  %29 = add i64 %21, 1
  br label %20

30:                                               ; preds = %20
  %31 = add i64 %17, 1
  br label %16

32:                                               ; preds = %16
  %33 = getelementptr float, ptr %8, i64 %9
  call void @llvm.memcpy.p0.p0.i64(ptr %33, ptr %15, i64 16000, i1 false)
  ret void
}

define void @cpu_hoisted_ttir_matmul_37790696(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20) {
  %22 = alloca float, i64 1, align 64
  br label %23

23:                                               ; preds = %41, %21
  %24 = phi i64 [ %42, %41 ], [ 0, %21 ]
  %25 = icmp slt i64 %24, 1
  br i1 %25, label %26, label %43

26:                                               ; preds = %23
  br label %27

27:                                               ; preds = %39, %26
  %28 = phi i64 [ %40, %39 ], [ 0, %26 ]
  %29 = icmp slt i64 %28, 1
  br i1 %29, label %30, label %41

30:                                               ; preds = %27
  br label %31

31:                                               ; preds = %34, %30
  %32 = phi i64 [ %38, %34 ], [ 0, %30 ]
  %33 = icmp slt i64 %32, 1
  br i1 %33, label %34, label %39

34:                                               ; preds = %31
  %35 = add nuw nsw i64 %24, %28
  %36 = add nuw nsw i64 %35, %32
  %37 = getelementptr inbounds nuw float, ptr %22, i64 %36
  store float 0.000000e+00, ptr %37, align 4
  %38 = add i64 %32, 1
  br label %31

39:                                               ; preds = %31
  %40 = add i64 %28, 1
  br label %27

41:                                               ; preds = %27
  %42 = add i64 %24, 1
  br label %23

43:                                               ; preds = %23
  br label %44

44:                                               ; preds = %81, %43
  %45 = phi i64 [ %82, %81 ], [ 0, %43 ]
  %46 = icmp slt i64 %45, 1
  br i1 %46, label %47, label %83

47:                                               ; preds = %44
  br label %48

48:                                               ; preds = %79, %47
  %49 = phi i64 [ %80, %79 ], [ 0, %47 ]
  %50 = icmp slt i64 %49, 1
  br i1 %50, label %51, label %81

51:                                               ; preds = %48
  br label %52

52:                                               ; preds = %77, %51
  %53 = phi i64 [ %78, %77 ], [ 0, %51 ]
  %54 = icmp slt i64 %53, 1
  br i1 %54, label %55, label %79

55:                                               ; preds = %52
  br label %56

56:                                               ; preds = %59, %55
  %57 = phi i64 [ %76, %59 ], [ 0, %55 ]
  %58 = icmp slt i64 %57, 4000
  br i1 %58, label %59, label %77

59:                                               ; preds = %56
  %60 = mul nuw nsw i64 %45, 4000
  %61 = mul nuw nsw i64 %49, 4000
  %62 = add nuw nsw i64 %60, %61
  %63 = add nuw nsw i64 %62, %57
  %64 = getelementptr inbounds nuw float, ptr %1, i64 %63
  %65 = load float, ptr %64, align 4
  %66 = add nuw nsw i64 %60, %57
  %67 = add nuw nsw i64 %66, %53
  %68 = getelementptr inbounds nuw float, ptr %8, i64 %67
  %69 = load float, ptr %68, align 4
  %70 = add nuw nsw i64 %45, %49
  %71 = add nuw nsw i64 %70, %53
  %72 = getelementptr inbounds nuw float, ptr %22, i64 %71
  %73 = load float, ptr %72, align 4
  %74 = fmul float %65, %69
  %75 = fadd float %73, %74
  store float %75, ptr %72, align 4
  %76 = add i64 %57, 1
  br label %56

77:                                               ; preds = %56
  %78 = add i64 %53, 1
  br label %52

79:                                               ; preds = %52
  %80 = add i64 %49, 1
  br label %48

81:                                               ; preds = %48
  %82 = add i64 %45, 1
  br label %44

83:                                               ; preds = %44
  %84 = getelementptr float, ptr %15, i64 %16
  call void @llvm.memcpy.p0.p0.i64(ptr %84, ptr %22, i64 4, i1 false)
  ret void
}

define void @cpu_hoisted_ttir_abs_713b937c_helper(ptr %0) {
  %2 = getelementptr inbounds ptr, ptr %0, i64 0
  %3 = load { ptr, ptr, i64, ptr }, ptr %2, align 8
  %4 = extractvalue { ptr, ptr, i64, ptr } %3, 0
  %5 = extractvalue { ptr, ptr, i64, ptr } %3, 1
  %6 = extractvalue { ptr, ptr, i64, ptr } %3, 2
  %7 = extractvalue { ptr, ptr, i64, ptr } %3, 3
  %8 = getelementptr ptr, ptr %7, i64 0
  %9 = load i64, ptr %8, align 4
  %10 = getelementptr ptr, ptr %7, i64 1
  %11 = load i64, ptr %10, align 4
  %12 = getelementptr ptr, ptr %7, i64 2
  %13 = load i64, ptr %12, align 4
  %14 = getelementptr ptr, ptr %7, i64 3
  %15 = load i64, ptr %14, align 4
  %16 = getelementptr inbounds ptr, ptr %0, i64 4
  %17 = load { ptr, ptr, i64, ptr }, ptr %16, align 8
  %18 = extractvalue { ptr, ptr, i64, ptr } %17, 0
  %19 = extractvalue { ptr, ptr, i64, ptr } %17, 1
  %20 = extractvalue { ptr, ptr, i64, ptr } %17, 2
  %21 = extractvalue { ptr, ptr, i64, ptr } %17, 3
  %22 = getelementptr ptr, ptr %21, i64 0
  %23 = load i64, ptr %22, align 4
  %24 = getelementptr ptr, ptr %21, i64 1
  %25 = load i64, ptr %24, align 4
  %26 = getelementptr ptr, ptr %21, i64 2
  %27 = load i64, ptr %26, align 4
  %28 = getelementptr ptr, ptr %21, i64 3
  %29 = load i64, ptr %28, align 4
  call void @cpu_hoisted_ttir_abs_713b937c(ptr %4, ptr %5, i64 %6, i64 %9, i64 %11, i64 %13, i64 %15, ptr %18, ptr %19, i64 %20, i64 %23, i64 %25, i64 %27, i64 %29)
  ret void
}

define void @cpu_hoisted_ttir_matmul_37790696_helper(ptr %0) {
  %2 = getelementptr inbounds ptr, ptr %0, i64 0
  %3 = load { ptr, ptr, i64, ptr }, ptr %2, align 8
  %4 = extractvalue { ptr, ptr, i64, ptr } %3, 0
  %5 = extractvalue { ptr, ptr, i64, ptr } %3, 1
  %6 = extractvalue { ptr, ptr, i64, ptr } %3, 2
  %7 = extractvalue { ptr, ptr, i64, ptr } %3, 3
  %8 = getelementptr ptr, ptr %7, i64 0
  %9 = load i64, ptr %8, align 4
  %10 = getelementptr ptr, ptr %7, i64 1
  %11 = load i64, ptr %10, align 4
  %12 = getelementptr ptr, ptr %7, i64 2
  %13 = load i64, ptr %12, align 4
  %14 = getelementptr ptr, ptr %7, i64 3
  %15 = load i64, ptr %14, align 4
  %16 = getelementptr inbounds ptr, ptr %0, i64 4
  %17 = load { ptr, ptr, i64, ptr }, ptr %16, align 8
  %18 = extractvalue { ptr, ptr, i64, ptr } %17, 0
  %19 = extractvalue { ptr, ptr, i64, ptr } %17, 1
  %20 = extractvalue { ptr, ptr, i64, ptr } %17, 2
  %21 = extractvalue { ptr, ptr, i64, ptr } %17, 3
  %22 = getelementptr ptr, ptr %21, i64 0
  %23 = load i64, ptr %22, align 4
  %24 = getelementptr ptr, ptr %21, i64 1
  %25 = load i64, ptr %24, align 4
  %26 = getelementptr ptr, ptr %21, i64 2
  %27 = load i64, ptr %26, align 4
  %28 = getelementptr ptr, ptr %21, i64 3
  %29 = load i64, ptr %28, align 4
  %30 = getelementptr inbounds ptr, ptr %0, i64 8
  %31 = load { ptr, ptr, i64, ptr }, ptr %30, align 8
  %32 = extractvalue { ptr, ptr, i64, ptr } %31, 0
  %33 = extractvalue { ptr, ptr, i64, ptr } %31, 1
  %34 = extractvalue { ptr, ptr, i64, ptr } %31, 2
  %35 = extractvalue { ptr, ptr, i64, ptr } %31, 3
  %36 = getelementptr ptr, ptr %35, i64 0
  %37 = load i64, ptr %36, align 4
  %38 = getelementptr ptr, ptr %35, i64 1
  %39 = load i64, ptr %38, align 4
  %40 = getelementptr ptr, ptr %35, i64 2
  %41 = load i64, ptr %40, align 4
  %42 = getelementptr ptr, ptr %35, i64 3
  %43 = load i64, ptr %42, align 4
  call void @cpu_hoisted_ttir_matmul_37790696(ptr %4, ptr %5, i64 %6, i64 %9, i64 %11, i64 %13, i64 %15, ptr %18, ptr %19, i64 %20, i64 %23, i64 %25, i64 %27, i64 %29, ptr %32, ptr %33, i64 %34, i64 %37, i64 %39, i64 %41, i64 %43)
  ret void
}

define void @x280_cpu_dispatch(i32 %0, ptr %1) section ".text.start" {
  %3 = icmp eq i32 %0, 0
  br i1 %3, label %6, label %4

4:                                                ; preds = %2
  %5 = icmp eq i32 %0, 1
  br i1 %5, label %7, label %8

6:                                                ; preds = %2
  call void @cpu_hoisted_ttir_abs_713b937c_helper(ptr %1)
  br label %8

7:                                                ; preds = %4
  call void @cpu_hoisted_ttir_matmul_37790696_helper(ptr %1)
  br label %8

8:                                                ; preds = %6, %7, %4
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #1

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
