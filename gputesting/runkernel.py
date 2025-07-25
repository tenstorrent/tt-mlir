# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import subprocess


def compile_mlir_to_ptx(gpu_module_str: str, chip_type="sm_80"):
    """Compiles MLIR module string to PTX code."""

    # Generate PTX from the GPU module
    ptx_result = subprocess.run(
        ["llc", "-march=nvptx64", f"-mcpu={chip_type}", "-"],
        input=gpu_module_str,
        capture_output=True,
        text=True,
    )

    if ptx_result.returncode != 0:
        print("Error generating PTX:")
        print(ptx_result.stderr)
        return None

    print("------------------\n")
    print(ptx_result.stdout)
    print("------------------\n")
    return ptx_result.stdout


SOURCE = """
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define ptx_kernel void @forward_kernel(i64 %0, i64 %1, float %2, ptr %3, ptr %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11) {
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

define ptx_kernel void @forward_kernel_1(i64 %0, i64 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr %11, ptr %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28) {
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

define ptx_kernel void @forward_kernel_2(i64 %0, i64 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22) {
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

define ptx_kernel void @forward_kernel_3(i64 %0, i64 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, float %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16) {
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

define ptx_kernel void @forward_kernel_4(i64 %0, i64 %1, float %2, ptr %3, ptr %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11) {
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

define ptx_kernel void @forward_kernel_5(i64 %0, i64 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr %11, ptr %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28) {
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

define ptx_kernel void @forward_kernel_6(i64 %0, i64 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22) {
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

define ptx_kernel void @forward_kernel_7(i64 %0, i64 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, float %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16) {
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

define ptx_kernel void @forward_kernel_8(i64 %0, i64 %1, float %2, ptr %3, ptr %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11) {
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

define ptx_kernel void @forward_kernel_9(i64 %0, i64 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr %11, ptr %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28) {
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

define ptx_kernel void @forward_kernel_10(i64 %0, i64 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22) {
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
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

"""

ptx_code = compile_mlir_to_ptx(SOURCE)


def run_relu_kernel():
    # Create sample input data
    size = 512
    input_data_1 = np.random.randn(size).astype(np.float32)
    input_data_2 = np.random.randn(size).astype(np.float32)
    output_data = np.zeros_like(input_data_1)

    # Allocate GPU memory
    input_gpu_1 = cuda.mem_alloc(input_data_1.nbytes)
    input_gpu_2 = cuda.mem_alloc(input_data_2.nbytes)
    output_gpu_1 = cuda.mem_alloc(output_data.nbytes)
    output_gpu_2 = cuda.mem_alloc(output_data.nbytes)

    # Copy input data to GPU
    cuda.memcpy_htod(input_gpu_1, input_data_1)
    cuda.memcpy_htod(input_gpu_2, input_data_2)

    # Load PTX module
    mod = cuda.module_from_buffer(ptx_code.encode())
    kernel = mod.get_function("forward_kernel_2")

    threshold = np.float32(0.0)  # Standard ReLU threshold
    array_size = np.uint64(size)
    zero = np.uint64(0)

    # Set up grid and block dimensions
    block_size = 256
    grid_size = (size + block_size - 1) // block_size

    # (i64 %0, i64 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22)
    # Launch kernel
    kernel(
        zero,
        zero,
        input_gpu_1,
        input_gpu_1,
        zero,
        zero,
        zero,
        zero,
        zero,
        input_gpu_2,
        input_gpu_2,
        zero,
        zero,
        zero,
        zero,
        zero,
        output_gpu_1,
        output_gpu_1,
        zero,
        zero,
        zero,
        zero,
        zero,
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
    )

    kernel2 = mod.get_function("forward_kernel_3")

    # (i64 %0, i64 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, float %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16) {
    kernel2(
        zero,
        zero,
        output_gpu_1,
        output_gpu_1,
        zero,
        zero,
        zero,
        zero,
        zero,
        threshold,
        output_gpu_2,
        output_gpu_2,
        zero,
        zero,
        zero,
        zero,
        zero,
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
    )

    # Copy result back to host
    cuda.memcpy_dtoh(output_data, output_gpu_2)

    # Clean up GPU memory
    input_gpu_1.free()
    input_gpu_2.free()
    output_gpu_1.free()
    output_gpu_2.free()

    return input_data_1, input_data_2, output_data


print("Running ReLU kernel example...")

# Run with PyCUDA
input_vals_1, input_vals_2, output_vals = run_relu_kernel()

print(f"Sample input values: {input_vals_1}")
print(f"Sample input values: {input_vals_2}")
print(f"Sample output values: {output_vals}")
