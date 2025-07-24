module @MNISTLinear attributes {gpu.container_module} {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_1x512xf32(dense<0.000000e+00> : tensor<1x512xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x array<512 x f32>>
  llvm.func @forward(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: !llvm.ptr, %arg20: !llvm.ptr, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: !llvm.ptr, %arg27: !llvm.ptr, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: !llvm.ptr, %arg32: !llvm.ptr, %arg33: i64, %arg34: i64, %arg35: i64, %arg36: i64, %arg37: i64, %arg38: !llvm.ptr, %arg39: !llvm.ptr, %arg40: i64, %arg41: i64, %arg42: i64) -> (!llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {ttir.name = "MNISTLinear_350.output_add_981"}) {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg38, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg39, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg40, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.insertvalue %arg41, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.insertvalue %arg42, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg31, %6[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.insertvalue %arg32, %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg33, %8[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.insertvalue %arg34, %9[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.insertvalue %arg36, %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.insertvalue %arg35, %11[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %arg37, %12[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.insertvalue %arg26, %14[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.insertvalue %arg27, %15[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.insertvalue %arg28, %16[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.insertvalue %arg29, %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.insertvalue %arg30, %18[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %21 = llvm.insertvalue %arg19, %20[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %22 = llvm.insertvalue %arg20, %21[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %23 = llvm.insertvalue %arg21, %22[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %24 = llvm.insertvalue %arg22, %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %25 = llvm.insertvalue %arg24, %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %26 = llvm.insertvalue %arg23, %25[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %27 = llvm.insertvalue %arg25, %26[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %28 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %29 = llvm.insertvalue %arg14, %28[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %30 = llvm.insertvalue %arg15, %29[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.insertvalue %arg16, %30[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %32 = llvm.insertvalue %arg17, %31[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %33 = llvm.insertvalue %arg18, %32[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %34 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %35 = llvm.insertvalue %arg7, %34[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %36 = llvm.insertvalue %arg8, %35[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %37 = llvm.insertvalue %arg9, %36[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %38 = llvm.insertvalue %arg10, %37[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %39 = llvm.insertvalue %arg12, %38[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %40 = llvm.insertvalue %arg11, %39[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %41 = llvm.insertvalue %arg13, %40[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %42 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %43 = llvm.insertvalue %arg0, %42[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %44 = llvm.insertvalue %arg1, %43[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %45 = llvm.insertvalue %arg2, %44[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.insertvalue %arg3, %45[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %47 = llvm.insertvalue %arg5, %46[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %48 = llvm.insertvalue %arg4, %47[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %49 = llvm.insertvalue %arg6, %48[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %50 = llvm.mlir.constant(10 : index) : i64
    %51 = llvm.mlir.constant(512 : index) : i64
    %52 = llvm.mlir.constant(1 : index) : i64
    %53 = llvm.mlir.constant(0 : index) : i64
    %54 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %55 = llvm.extractvalue %49[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %56 = llvm.extractvalue %49[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %57 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %58 = llvm.insertvalue %55, %57[0] : !llvm.struct<(ptr, ptr, i64)>
    %59 = llvm.insertvalue %56, %58[1] : !llvm.struct<(ptr, ptr, i64)>
    %60 = llvm.mlir.constant(0 : index) : i64
    %61 = llvm.insertvalue %60, %59[2] : !llvm.struct<(ptr, ptr, i64)>
    %62 = llvm.extractvalue %49[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %63 = llvm.extractvalue %49[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %64 = llvm.extractvalue %49[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %65 = llvm.extractvalue %49[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %66 = llvm.extractvalue %49[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %67 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %68 = llvm.extractvalue %61[0] : !llvm.struct<(ptr, ptr, i64)>
    %69 = llvm.extractvalue %61[1] : !llvm.struct<(ptr, ptr, i64)>
    %70 = llvm.insertvalue %68, %67[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %71 = llvm.insertvalue %69, %70[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %72 = llvm.mlir.constant(0 : index) : i64
    %73 = llvm.insertvalue %72, %71[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %74 = llvm.mlir.constant(1 : index) : i64
    %75 = llvm.insertvalue %74, %73[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %76 = llvm.mlir.constant(784 : index) : i64
    %77 = llvm.insertvalue %76, %75[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %78 = llvm.mlir.constant(1 : index) : i64
    %79 = llvm.insertvalue %78, %77[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %80 = llvm.mlir.constant(784 : index) : i64
    %81 = llvm.insertvalue %80, %79[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %82 = llvm.mlir.constant(784 : index) : i64
    %83 = llvm.insertvalue %82, %81[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %84 = llvm.mlir.constant(1 : index) : i64
    %85 = llvm.insertvalue %84, %83[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %86 = llvm.extractvalue %41[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %87 = llvm.extractvalue %41[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %88 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %89 = llvm.insertvalue %86, %88[0] : !llvm.struct<(ptr, ptr, i64)>
    %90 = llvm.insertvalue %87, %89[1] : !llvm.struct<(ptr, ptr, i64)>
    %91 = llvm.mlir.constant(0 : index) : i64
    %92 = llvm.insertvalue %91, %90[2] : !llvm.struct<(ptr, ptr, i64)>
    %93 = llvm.extractvalue %41[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %94 = llvm.extractvalue %41[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %95 = llvm.extractvalue %41[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %96 = llvm.extractvalue %41[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %97 = llvm.extractvalue %41[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %98 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %99 = llvm.extractvalue %92[0] : !llvm.struct<(ptr, ptr, i64)>
    %100 = llvm.extractvalue %92[1] : !llvm.struct<(ptr, ptr, i64)>
    %101 = llvm.insertvalue %99, %98[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %102 = llvm.insertvalue %100, %101[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %103 = llvm.mlir.constant(0 : index) : i64
    %104 = llvm.insertvalue %103, %102[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %105 = llvm.mlir.constant(1 : index) : i64
    %106 = llvm.insertvalue %105, %104[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %107 = llvm.mlir.constant(401408 : index) : i64
    %108 = llvm.insertvalue %107, %106[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %109 = llvm.mlir.constant(784 : index) : i64
    %110 = llvm.insertvalue %109, %108[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %111 = llvm.mlir.constant(512 : index) : i64
    %112 = llvm.insertvalue %111, %110[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %113 = llvm.mlir.constant(512 : index) : i64
    %114 = llvm.insertvalue %113, %112[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %115 = llvm.mlir.constant(1 : index) : i64
    %116 = llvm.insertvalue %115, %114[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %117 = llvm.mlir.constant(1 : index) : i64
    %118 = llvm.mlir.constant(1 : index) : i64
    %119 = llvm.mlir.constant(512 : index) : i64
    %120 = llvm.mlir.constant(1 : index) : i64
    %121 = llvm.mlir.constant(512 : index) : i64
    %122 = llvm.mlir.constant(512 : index) : i64
    %123 = llvm.mlir.zero : !llvm.ptr
    %124 = llvm.getelementptr %123[%122] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %125 = llvm.ptrtoint %124 : !llvm.ptr to i64
    %126 = llvm.mlir.constant(64 : index) : i64
    %127 = llvm.add %125, %126 : i64
    %128 = llvm.call @malloc(%127) : (i64) -> !llvm.ptr
    %129 = llvm.ptrtoint %128 : !llvm.ptr to i64
    %130 = llvm.mlir.constant(1 : index) : i64
    %131 = llvm.sub %126, %130 : i64
    %132 = llvm.add %129, %131 : i64
    %133 = llvm.urem %132, %126 : i64
    %134 = llvm.sub %132, %133 : i64
    %135 = llvm.inttoptr %134 : i64 to !llvm.ptr
    %136 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %137 = llvm.insertvalue %128, %136[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %138 = llvm.insertvalue %135, %137[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %139 = llvm.mlir.constant(0 : index) : i64
    %140 = llvm.insertvalue %139, %138[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %141 = llvm.insertvalue %117, %140[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %142 = llvm.insertvalue %118, %141[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %143 = llvm.insertvalue %119, %142[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %144 = llvm.insertvalue %121, %143[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %145 = llvm.insertvalue %119, %144[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %146 = llvm.insertvalue %120, %145[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %147 = llvm.extractvalue %146[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %148 = llvm.extractvalue %146[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %149 = llvm.extractvalue %146[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %150 = llvm.extractvalue %146[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %151 = llvm.extractvalue %146[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %152 = llvm.extractvalue %146[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %153 = llvm.extractvalue %146[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %154 = llvm.extractvalue %146[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %155 = llvm.extractvalue %146[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    gpu.launch_func  @forward_kernel::@forward_kernel blocks in (%52, %52, %52) threads in (%52, %52, %52) : i64 args(%53 : i64, %53 : i64, %54 : f32, %147 : !llvm.ptr, %148 : !llvm.ptr, %149 : i64, %150 : i64, %151 : i64, %152 : i64, %153 : i64, %154 : i64, %155 : i64)
    %156 = llvm.extractvalue %85[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %157 = llvm.extractvalue %85[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %158 = llvm.extractvalue %85[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %159 = llvm.extractvalue %85[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %160 = llvm.extractvalue %85[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %161 = llvm.extractvalue %85[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %162 = llvm.extractvalue %85[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %163 = llvm.extractvalue %85[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %164 = llvm.extractvalue %85[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %165 = llvm.extractvalue %116[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %166 = llvm.extractvalue %116[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %167 = llvm.extractvalue %116[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %168 = llvm.extractvalue %116[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %169 = llvm.extractvalue %116[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %170 = llvm.extractvalue %116[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %171 = llvm.extractvalue %116[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %172 = llvm.extractvalue %116[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %173 = llvm.extractvalue %116[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %174 = llvm.extractvalue %146[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %175 = llvm.extractvalue %146[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %176 = llvm.extractvalue %146[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %177 = llvm.extractvalue %146[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %178 = llvm.extractvalue %146[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %179 = llvm.extractvalue %146[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %180 = llvm.extractvalue %146[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %181 = llvm.extractvalue %146[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %182 = llvm.extractvalue %146[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    gpu.launch_func  @forward_kernel_0::@forward_kernel blocks in (%52, %52, %52) threads in (%52, %52, %52) : i64 args(%53 : i64, %53 : i64, %156 : !llvm.ptr, %157 : !llvm.ptr, %158 : i64, %159 : i64, %160 : i64, %161 : i64, %162 : i64, %163 : i64, %164 : i64, %165 : !llvm.ptr, %166 : !llvm.ptr, %167 : i64, %168 : i64, %169 : i64, %170 : i64, %171 : i64, %172 : i64, %173 : i64, %174 : !llvm.ptr, %175 : !llvm.ptr, %176 : i64, %177 : i64, %178 : i64, %179 : i64, %180 : i64, %181 : i64, %182 : i64)
    %183 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %184 = llvm.extractvalue %146[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %185 = llvm.extractvalue %146[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %186 = llvm.insertvalue %184, %183[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %187 = llvm.insertvalue %185, %186[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %188 = llvm.mlir.constant(0 : index) : i64
    %189 = llvm.insertvalue %188, %187[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %190 = llvm.mlir.constant(1 : index) : i64
    %191 = llvm.insertvalue %190, %189[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %192 = llvm.mlir.constant(512 : index) : i64
    %193 = llvm.insertvalue %192, %191[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %194 = llvm.mlir.constant(512 : index) : i64
    %195 = llvm.insertvalue %194, %193[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %196 = llvm.mlir.constant(1 : index) : i64
    %197 = llvm.insertvalue %196, %195[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %198 = llvm.extractvalue %33[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %199 = llvm.extractvalue %33[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %200 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %201 = llvm.insertvalue %198, %200[0] : !llvm.struct<(ptr, ptr, i64)>
    %202 = llvm.insertvalue %199, %201[1] : !llvm.struct<(ptr, ptr, i64)>
    %203 = llvm.mlir.constant(0 : index) : i64
    %204 = llvm.insertvalue %203, %202[2] : !llvm.struct<(ptr, ptr, i64)>
    %205 = llvm.extractvalue %33[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %206 = llvm.extractvalue %33[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %207 = llvm.extractvalue %33[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %208 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %209 = llvm.extractvalue %204[0] : !llvm.struct<(ptr, ptr, i64)>
    %210 = llvm.extractvalue %204[1] : !llvm.struct<(ptr, ptr, i64)>
    %211 = llvm.insertvalue %209, %208[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %212 = llvm.insertvalue %210, %211[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %213 = llvm.mlir.constant(0 : index) : i64
    %214 = llvm.insertvalue %213, %212[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %215 = llvm.mlir.constant(1 : index) : i64
    %216 = llvm.insertvalue %215, %214[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %217 = llvm.mlir.constant(512 : index) : i64
    %218 = llvm.insertvalue %217, %216[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %219 = llvm.mlir.constant(512 : index) : i64
    %220 = llvm.insertvalue %219, %218[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %221 = llvm.mlir.constant(1 : index) : i64
    %222 = llvm.insertvalue %221, %220[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %223 = llvm.mlir.constant(1 : index) : i64
    %224 = llvm.mlir.constant(512 : index) : i64
    %225 = llvm.mlir.constant(1 : index) : i64
    %226 = llvm.mlir.constant(512 : index) : i64
    %227 = llvm.mlir.zero : !llvm.ptr
    %228 = llvm.getelementptr %227[%226] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %229 = llvm.ptrtoint %228 : !llvm.ptr to i64
    %230 = llvm.mlir.constant(64 : index) : i64
    %231 = llvm.add %229, %230 : i64
    %232 = llvm.call @malloc(%231) : (i64) -> !llvm.ptr
    %233 = llvm.ptrtoint %232 : !llvm.ptr to i64
    %234 = llvm.mlir.constant(1 : index) : i64
    %235 = llvm.sub %230, %234 : i64
    %236 = llvm.add %233, %235 : i64
    %237 = llvm.urem %236, %230 : i64
    %238 = llvm.sub %236, %237 : i64
    %239 = llvm.inttoptr %238 : i64 to !llvm.ptr
    %240 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %241 = llvm.insertvalue %232, %240[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %242 = llvm.insertvalue %239, %241[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %243 = llvm.mlir.constant(0 : index) : i64
    %244 = llvm.insertvalue %243, %242[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %245 = llvm.insertvalue %223, %244[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %246 = llvm.insertvalue %224, %245[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %247 = llvm.insertvalue %224, %246[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %248 = llvm.insertvalue %225, %247[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %249 = llvm.extractvalue %197[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %250 = llvm.extractvalue %197[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %251 = llvm.extractvalue %197[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %252 = llvm.extractvalue %197[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %253 = llvm.extractvalue %197[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %254 = llvm.extractvalue %197[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %255 = llvm.extractvalue %197[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %256 = llvm.extractvalue %222[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %257 = llvm.extractvalue %222[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %258 = llvm.extractvalue %222[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %259 = llvm.extractvalue %222[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %260 = llvm.extractvalue %222[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %261 = llvm.extractvalue %222[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %262 = llvm.extractvalue %222[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %263 = llvm.extractvalue %248[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %264 = llvm.extractvalue %248[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %265 = llvm.extractvalue %248[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %266 = llvm.extractvalue %248[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %267 = llvm.extractvalue %248[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %268 = llvm.extractvalue %248[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %269 = llvm.extractvalue %248[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    gpu.launch_func  @forward_kernel_1::@forward_kernel blocks in (%52, %52, %52) threads in (%51, %52, %52) : i64 args(%53 : i64, %53 : i64, %249 : !llvm.ptr, %250 : !llvm.ptr, %251 : i64, %252 : i64, %253 : i64, %254 : i64, %255 : i64, %256 : !llvm.ptr, %257 : !llvm.ptr, %258 : i64, %259 : i64, %260 : i64, %261 : i64, %262 : i64, %263 : !llvm.ptr, %264 : !llvm.ptr, %265 : i64, %266 : i64, %267 : i64, %268 : i64, %269 : i64)
    %270 = llvm.mlir.constant(1 : index) : i64
    %271 = llvm.mlir.constant(512 : index) : i64
    %272 = llvm.mlir.constant(1 : index) : i64
    %273 = llvm.mlir.constant(512 : index) : i64
    %274 = llvm.mlir.zero : !llvm.ptr
    %275 = llvm.getelementptr %274[%273] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %276 = llvm.ptrtoint %275 : !llvm.ptr to i64
    %277 = llvm.mlir.constant(64 : index) : i64
    %278 = llvm.add %276, %277 : i64
    %279 = llvm.call @malloc(%278) : (i64) -> !llvm.ptr
    %280 = llvm.ptrtoint %279 : !llvm.ptr to i64
    %281 = llvm.mlir.constant(1 : index) : i64
    %282 = llvm.sub %277, %281 : i64
    %283 = llvm.add %280, %282 : i64
    %284 = llvm.urem %283, %277 : i64
    %285 = llvm.sub %283, %284 : i64
    %286 = llvm.inttoptr %285 : i64 to !llvm.ptr
    %287 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %288 = llvm.insertvalue %279, %287[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %289 = llvm.insertvalue %286, %288[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %290 = llvm.mlir.constant(0 : index) : i64
    %291 = llvm.insertvalue %290, %289[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %292 = llvm.insertvalue %270, %291[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %293 = llvm.insertvalue %271, %292[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %294 = llvm.insertvalue %271, %293[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %295 = llvm.insertvalue %272, %294[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %296 = llvm.extractvalue %248[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %297 = llvm.extractvalue %248[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %298 = llvm.extractvalue %248[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %299 = llvm.extractvalue %248[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %300 = llvm.extractvalue %248[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %301 = llvm.extractvalue %248[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %302 = llvm.extractvalue %248[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %303 = llvm.extractvalue %295[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %304 = llvm.extractvalue %295[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %305 = llvm.extractvalue %295[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %306 = llvm.extractvalue %295[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %307 = llvm.extractvalue %295[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %308 = llvm.extractvalue %295[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %309 = llvm.extractvalue %295[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    gpu.launch_func  @forward_kernel_2::@forward_kernel blocks in (%52, %52, %52) threads in (%51, %52, %52) : i64 args(%53 : i64, %53 : i64, %296 : !llvm.ptr, %297 : !llvm.ptr, %298 : i64, %299 : i64, %300 : i64, %301 : i64, %302 : i64, %54 : f32, %303 : !llvm.ptr, %304 : !llvm.ptr, %305 : i64, %306 : i64, %307 : i64, %308 : i64, %309 : i64)
    %310 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %311 = llvm.extractvalue %295[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %312 = llvm.extractvalue %295[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %313 = llvm.insertvalue %311, %310[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %314 = llvm.insertvalue %312, %313[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %315 = llvm.mlir.constant(0 : index) : i64
    %316 = llvm.insertvalue %315, %314[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %317 = llvm.mlir.constant(1 : index) : i64
    %318 = llvm.insertvalue %317, %316[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %319 = llvm.mlir.constant(512 : index) : i64
    %320 = llvm.insertvalue %319, %318[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %321 = llvm.mlir.constant(1 : index) : i64
    %322 = llvm.insertvalue %321, %320[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %323 = llvm.mlir.constant(512 : index) : i64
    %324 = llvm.insertvalue %323, %322[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %325 = llvm.mlir.constant(512 : index) : i64
    %326 = llvm.insertvalue %325, %324[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %327 = llvm.mlir.constant(1 : index) : i64
    %328 = llvm.insertvalue %327, %326[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %329 = llvm.extractvalue %27[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %330 = llvm.extractvalue %27[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %331 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %332 = llvm.insertvalue %329, %331[0] : !llvm.struct<(ptr, ptr, i64)>
    %333 = llvm.insertvalue %330, %332[1] : !llvm.struct<(ptr, ptr, i64)>
    %334 = llvm.mlir.constant(0 : index) : i64
    %335 = llvm.insertvalue %334, %333[2] : !llvm.struct<(ptr, ptr, i64)>
    %336 = llvm.extractvalue %27[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %337 = llvm.extractvalue %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %338 = llvm.extractvalue %27[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %339 = llvm.extractvalue %27[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %340 = llvm.extractvalue %27[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %341 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %342 = llvm.extractvalue %335[0] : !llvm.struct<(ptr, ptr, i64)>
    %343 = llvm.extractvalue %335[1] : !llvm.struct<(ptr, ptr, i64)>
    %344 = llvm.insertvalue %342, %341[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %345 = llvm.insertvalue %343, %344[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %346 = llvm.mlir.constant(0 : index) : i64
    %347 = llvm.insertvalue %346, %345[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %348 = llvm.mlir.constant(1 : index) : i64
    %349 = llvm.insertvalue %348, %347[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %350 = llvm.mlir.constant(262144 : index) : i64
    %351 = llvm.insertvalue %350, %349[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %352 = llvm.mlir.constant(512 : index) : i64
    %353 = llvm.insertvalue %352, %351[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %354 = llvm.mlir.constant(512 : index) : i64
    %355 = llvm.insertvalue %354, %353[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %356 = llvm.mlir.constant(512 : index) : i64
    %357 = llvm.insertvalue %356, %355[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %358 = llvm.mlir.constant(1 : index) : i64
    %359 = llvm.insertvalue %358, %357[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %360 = llvm.mlir.constant(1 : index) : i64
    %361 = llvm.mlir.constant(1 : index) : i64
    %362 = llvm.mlir.constant(512 : index) : i64
    %363 = llvm.mlir.constant(1 : index) : i64
    %364 = llvm.mlir.constant(512 : index) : i64
    %365 = llvm.mlir.constant(512 : index) : i64
    %366 = llvm.mlir.zero : !llvm.ptr
    %367 = llvm.getelementptr %366[%365] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %368 = llvm.ptrtoint %367 : !llvm.ptr to i64
    %369 = llvm.mlir.constant(64 : index) : i64
    %370 = llvm.add %368, %369 : i64
    %371 = llvm.call @malloc(%370) : (i64) -> !llvm.ptr
    %372 = llvm.ptrtoint %371 : !llvm.ptr to i64
    %373 = llvm.mlir.constant(1 : index) : i64
    %374 = llvm.sub %369, %373 : i64
    %375 = llvm.add %372, %374 : i64
    %376 = llvm.urem %375, %369 : i64
    %377 = llvm.sub %375, %376 : i64
    %378 = llvm.inttoptr %377 : i64 to !llvm.ptr
    %379 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %380 = llvm.insertvalue %371, %379[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %381 = llvm.insertvalue %378, %380[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %382 = llvm.mlir.constant(0 : index) : i64
    %383 = llvm.insertvalue %382, %381[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %384 = llvm.insertvalue %360, %383[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %385 = llvm.insertvalue %361, %384[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %386 = llvm.insertvalue %362, %385[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %387 = llvm.insertvalue %364, %386[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %388 = llvm.insertvalue %362, %387[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %389 = llvm.insertvalue %363, %388[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %390 = llvm.extractvalue %389[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %391 = llvm.extractvalue %389[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %392 = llvm.extractvalue %389[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %393 = llvm.extractvalue %389[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %394 = llvm.extractvalue %389[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %395 = llvm.extractvalue %389[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %396 = llvm.extractvalue %389[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %397 = llvm.extractvalue %389[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %398 = llvm.extractvalue %389[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    gpu.launch_func  @forward_kernel_3::@forward_kernel blocks in (%52, %52, %52) threads in (%52, %52, %52) : i64 args(%53 : i64, %53 : i64, %54 : f32, %390 : !llvm.ptr, %391 : !llvm.ptr, %392 : i64, %393 : i64, %394 : i64, %395 : i64, %396 : i64, %397 : i64, %398 : i64)
    %399 = llvm.extractvalue %328[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %400 = llvm.extractvalue %328[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %401 = llvm.extractvalue %328[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %402 = llvm.extractvalue %328[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %403 = llvm.extractvalue %328[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %404 = llvm.extractvalue %328[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %405 = llvm.extractvalue %328[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %406 = llvm.extractvalue %328[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %407 = llvm.extractvalue %328[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %408 = llvm.extractvalue %359[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %409 = llvm.extractvalue %359[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %410 = llvm.extractvalue %359[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %411 = llvm.extractvalue %359[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %412 = llvm.extractvalue %359[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %413 = llvm.extractvalue %359[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %414 = llvm.extractvalue %359[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %415 = llvm.extractvalue %359[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %416 = llvm.extractvalue %359[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %417 = llvm.extractvalue %389[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %418 = llvm.extractvalue %389[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %419 = llvm.extractvalue %389[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %420 = llvm.extractvalue %389[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %421 = llvm.extractvalue %389[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %422 = llvm.extractvalue %389[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %423 = llvm.extractvalue %389[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %424 = llvm.extractvalue %389[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %425 = llvm.extractvalue %389[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    gpu.launch_func  @forward_kernel_4::@forward_kernel blocks in (%52, %52, %52) threads in (%52, %52, %52) : i64 args(%53 : i64, %53 : i64, %399 : !llvm.ptr, %400 : !llvm.ptr, %401 : i64, %402 : i64, %403 : i64, %404 : i64, %405 : i64, %406 : i64, %407 : i64, %408 : !llvm.ptr, %409 : !llvm.ptr, %410 : i64, %411 : i64, %412 : i64, %413 : i64, %414 : i64, %415 : i64, %416 : i64, %417 : !llvm.ptr, %418 : !llvm.ptr, %419 : i64, %420 : i64, %421 : i64, %422 : i64, %423 : i64, %424 : i64, %425 : i64)
    %426 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %427 = llvm.extractvalue %389[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %428 = llvm.extractvalue %389[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %429 = llvm.insertvalue %427, %426[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %430 = llvm.insertvalue %428, %429[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %431 = llvm.mlir.constant(0 : index) : i64
    %432 = llvm.insertvalue %431, %430[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %433 = llvm.mlir.constant(1 : index) : i64
    %434 = llvm.insertvalue %433, %432[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %435 = llvm.mlir.constant(512 : index) : i64
    %436 = llvm.insertvalue %435, %434[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %437 = llvm.mlir.constant(512 : index) : i64
    %438 = llvm.insertvalue %437, %436[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %439 = llvm.mlir.constant(1 : index) : i64
    %440 = llvm.insertvalue %439, %438[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %441 = llvm.extractvalue %19[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %442 = llvm.extractvalue %19[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %443 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %444 = llvm.insertvalue %441, %443[0] : !llvm.struct<(ptr, ptr, i64)>
    %445 = llvm.insertvalue %442, %444[1] : !llvm.struct<(ptr, ptr, i64)>
    %446 = llvm.mlir.constant(0 : index) : i64
    %447 = llvm.insertvalue %446, %445[2] : !llvm.struct<(ptr, ptr, i64)>
    %448 = llvm.extractvalue %19[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %449 = llvm.extractvalue %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %450 = llvm.extractvalue %19[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %451 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %452 = llvm.extractvalue %447[0] : !llvm.struct<(ptr, ptr, i64)>
    %453 = llvm.extractvalue %447[1] : !llvm.struct<(ptr, ptr, i64)>
    %454 = llvm.insertvalue %452, %451[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %455 = llvm.insertvalue %453, %454[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %456 = llvm.mlir.constant(0 : index) : i64
    %457 = llvm.insertvalue %456, %455[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %458 = llvm.mlir.constant(1 : index) : i64
    %459 = llvm.insertvalue %458, %457[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %460 = llvm.mlir.constant(512 : index) : i64
    %461 = llvm.insertvalue %460, %459[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %462 = llvm.mlir.constant(512 : index) : i64
    %463 = llvm.insertvalue %462, %461[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %464 = llvm.mlir.constant(1 : index) : i64
    %465 = llvm.insertvalue %464, %463[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %466 = llvm.mlir.constant(1 : index) : i64
    %467 = llvm.mlir.constant(512 : index) : i64
    %468 = llvm.mlir.constant(1 : index) : i64
    %469 = llvm.mlir.constant(512 : index) : i64
    %470 = llvm.mlir.zero : !llvm.ptr
    %471 = llvm.getelementptr %470[%469] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %472 = llvm.ptrtoint %471 : !llvm.ptr to i64
    %473 = llvm.mlir.constant(64 : index) : i64
    %474 = llvm.add %472, %473 : i64
    %475 = llvm.call @malloc(%474) : (i64) -> !llvm.ptr
    %476 = llvm.ptrtoint %475 : !llvm.ptr to i64
    %477 = llvm.mlir.constant(1 : index) : i64
    %478 = llvm.sub %473, %477 : i64
    %479 = llvm.add %476, %478 : i64
    %480 = llvm.urem %479, %473 : i64
    %481 = llvm.sub %479, %480 : i64
    %482 = llvm.inttoptr %481 : i64 to !llvm.ptr
    %483 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %484 = llvm.insertvalue %475, %483[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %485 = llvm.insertvalue %482, %484[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %486 = llvm.mlir.constant(0 : index) : i64
    %487 = llvm.insertvalue %486, %485[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %488 = llvm.insertvalue %466, %487[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %489 = llvm.insertvalue %467, %488[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %490 = llvm.insertvalue %467, %489[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %491 = llvm.insertvalue %468, %490[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %492 = llvm.extractvalue %440[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %493 = llvm.extractvalue %440[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %494 = llvm.extractvalue %440[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %495 = llvm.extractvalue %440[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %496 = llvm.extractvalue %440[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %497 = llvm.extractvalue %440[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %498 = llvm.extractvalue %440[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %499 = llvm.extractvalue %465[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %500 = llvm.extractvalue %465[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %501 = llvm.extractvalue %465[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %502 = llvm.extractvalue %465[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %503 = llvm.extractvalue %465[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %504 = llvm.extractvalue %465[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %505 = llvm.extractvalue %465[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %506 = llvm.extractvalue %491[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %507 = llvm.extractvalue %491[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %508 = llvm.extractvalue %491[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %509 = llvm.extractvalue %491[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %510 = llvm.extractvalue %491[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %511 = llvm.extractvalue %491[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %512 = llvm.extractvalue %491[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    gpu.launch_func  @forward_kernel_5::@forward_kernel blocks in (%52, %52, %52) threads in (%51, %52, %52) : i64 args(%53 : i64, %53 : i64, %492 : !llvm.ptr, %493 : !llvm.ptr, %494 : i64, %495 : i64, %496 : i64, %497 : i64, %498 : i64, %499 : !llvm.ptr, %500 : !llvm.ptr, %501 : i64, %502 : i64, %503 : i64, %504 : i64, %505 : i64, %506 : !llvm.ptr, %507 : !llvm.ptr, %508 : i64, %509 : i64, %510 : i64, %511 : i64, %512 : i64)
    %513 = llvm.mlir.constant(1 : index) : i64
    %514 = llvm.mlir.constant(512 : index) : i64
    %515 = llvm.mlir.constant(1 : index) : i64
    %516 = llvm.mlir.constant(512 : index) : i64
    %517 = llvm.mlir.zero : !llvm.ptr
    %518 = llvm.getelementptr %517[%516] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %519 = llvm.ptrtoint %518 : !llvm.ptr to i64
    %520 = llvm.mlir.constant(64 : index) : i64
    %521 = llvm.add %519, %520 : i64
    %522 = llvm.call @malloc(%521) : (i64) -> !llvm.ptr
    %523 = llvm.ptrtoint %522 : !llvm.ptr to i64
    %524 = llvm.mlir.constant(1 : index) : i64
    %525 = llvm.sub %520, %524 : i64
    %526 = llvm.add %523, %525 : i64
    %527 = llvm.urem %526, %520 : i64
    %528 = llvm.sub %526, %527 : i64
    %529 = llvm.inttoptr %528 : i64 to !llvm.ptr
    %530 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %531 = llvm.insertvalue %522, %530[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %532 = llvm.insertvalue %529, %531[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %533 = llvm.mlir.constant(0 : index) : i64
    %534 = llvm.insertvalue %533, %532[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %535 = llvm.insertvalue %513, %534[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %536 = llvm.insertvalue %514, %535[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %537 = llvm.insertvalue %514, %536[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %538 = llvm.insertvalue %515, %537[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %539 = llvm.extractvalue %491[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %540 = llvm.extractvalue %491[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %541 = llvm.extractvalue %491[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %542 = llvm.extractvalue %491[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %543 = llvm.extractvalue %491[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %544 = llvm.extractvalue %491[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %545 = llvm.extractvalue %491[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %546 = llvm.extractvalue %538[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %547 = llvm.extractvalue %538[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %548 = llvm.extractvalue %538[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %549 = llvm.extractvalue %538[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %550 = llvm.extractvalue %538[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %551 = llvm.extractvalue %538[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %552 = llvm.extractvalue %538[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    gpu.launch_func  @forward_kernel_6::@forward_kernel blocks in (%52, %52, %52) threads in (%51, %52, %52) : i64 args(%53 : i64, %53 : i64, %539 : !llvm.ptr, %540 : !llvm.ptr, %541 : i64, %542 : i64, %543 : i64, %544 : i64, %545 : i64, %54 : f32, %546 : !llvm.ptr, %547 : !llvm.ptr, %548 : i64, %549 : i64, %550 : i64, %551 : i64, %552 : i64)
    %553 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %554 = llvm.extractvalue %538[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %555 = llvm.extractvalue %538[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %556 = llvm.insertvalue %554, %553[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %557 = llvm.insertvalue %555, %556[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %558 = llvm.mlir.constant(0 : index) : i64
    %559 = llvm.insertvalue %558, %557[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %560 = llvm.mlir.constant(1 : index) : i64
    %561 = llvm.insertvalue %560, %559[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %562 = llvm.mlir.constant(512 : index) : i64
    %563 = llvm.insertvalue %562, %561[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %564 = llvm.mlir.constant(1 : index) : i64
    %565 = llvm.insertvalue %564, %563[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %566 = llvm.mlir.constant(512 : index) : i64
    %567 = llvm.insertvalue %566, %565[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %568 = llvm.mlir.constant(512 : index) : i64
    %569 = llvm.insertvalue %568, %567[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %570 = llvm.mlir.constant(1 : index) : i64
    %571 = llvm.insertvalue %570, %569[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %572 = llvm.extractvalue %13[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %573 = llvm.extractvalue %13[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %574 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %575 = llvm.insertvalue %572, %574[0] : !llvm.struct<(ptr, ptr, i64)>
    %576 = llvm.insertvalue %573, %575[1] : !llvm.struct<(ptr, ptr, i64)>
    %577 = llvm.mlir.constant(0 : index) : i64
    %578 = llvm.insertvalue %577, %576[2] : !llvm.struct<(ptr, ptr, i64)>
    %579 = llvm.extractvalue %13[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %580 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %581 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %582 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %583 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %584 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %585 = llvm.extractvalue %578[0] : !llvm.struct<(ptr, ptr, i64)>
    %586 = llvm.extractvalue %578[1] : !llvm.struct<(ptr, ptr, i64)>
    %587 = llvm.insertvalue %585, %584[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %588 = llvm.insertvalue %586, %587[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %589 = llvm.mlir.constant(0 : index) : i64
    %590 = llvm.insertvalue %589, %588[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %591 = llvm.mlir.constant(1 : index) : i64
    %592 = llvm.insertvalue %591, %590[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %593 = llvm.mlir.constant(5120 : index) : i64
    %594 = llvm.insertvalue %593, %592[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %595 = llvm.mlir.constant(512 : index) : i64
    %596 = llvm.insertvalue %595, %594[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %597 = llvm.mlir.constant(10 : index) : i64
    %598 = llvm.insertvalue %597, %596[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %599 = llvm.mlir.constant(10 : index) : i64
    %600 = llvm.insertvalue %599, %598[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %601 = llvm.mlir.constant(1 : index) : i64
    %602 = llvm.insertvalue %601, %600[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %603 = llvm.mlir.constant(1 : index) : i64
    %604 = llvm.mlir.constant(1 : index) : i64
    %605 = llvm.mlir.constant(10 : index) : i64
    %606 = llvm.mlir.constant(1 : index) : i64
    %607 = llvm.mlir.constant(10 : index) : i64
    %608 = llvm.mlir.constant(10 : index) : i64
    %609 = llvm.mlir.zero : !llvm.ptr
    %610 = llvm.getelementptr %609[%608] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %611 = llvm.ptrtoint %610 : !llvm.ptr to i64
    %612 = llvm.mlir.constant(64 : index) : i64
    %613 = llvm.add %611, %612 : i64
    %614 = llvm.call @malloc(%613) : (i64) -> !llvm.ptr
    %615 = llvm.ptrtoint %614 : !llvm.ptr to i64
    %616 = llvm.mlir.constant(1 : index) : i64
    %617 = llvm.sub %612, %616 : i64
    %618 = llvm.add %615, %617 : i64
    %619 = llvm.urem %618, %612 : i64
    %620 = llvm.sub %618, %619 : i64
    %621 = llvm.inttoptr %620 : i64 to !llvm.ptr
    %622 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %623 = llvm.insertvalue %614, %622[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %624 = llvm.insertvalue %621, %623[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %625 = llvm.mlir.constant(0 : index) : i64
    %626 = llvm.insertvalue %625, %624[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %627 = llvm.insertvalue %603, %626[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %628 = llvm.insertvalue %604, %627[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %629 = llvm.insertvalue %605, %628[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %630 = llvm.insertvalue %607, %629[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %631 = llvm.insertvalue %605, %630[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %632 = llvm.insertvalue %606, %631[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %633 = llvm.extractvalue %632[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %634 = llvm.extractvalue %632[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %635 = llvm.extractvalue %632[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %636 = llvm.extractvalue %632[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %637 = llvm.extractvalue %632[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %638 = llvm.extractvalue %632[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %639 = llvm.extractvalue %632[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %640 = llvm.extractvalue %632[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %641 = llvm.extractvalue %632[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    gpu.launch_func  @forward_kernel_7::@forward_kernel blocks in (%52, %52, %52) threads in (%52, %52, %52) : i64 args(%53 : i64, %53 : i64, %54 : f32, %633 : !llvm.ptr, %634 : !llvm.ptr, %635 : i64, %636 : i64, %637 : i64, %638 : i64, %639 : i64, %640 : i64, %641 : i64)
    %642 = llvm.extractvalue %571[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %643 = llvm.extractvalue %571[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %644 = llvm.extractvalue %571[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %645 = llvm.extractvalue %571[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %646 = llvm.extractvalue %571[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %647 = llvm.extractvalue %571[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %648 = llvm.extractvalue %571[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %649 = llvm.extractvalue %571[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %650 = llvm.extractvalue %571[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %651 = llvm.extractvalue %602[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %652 = llvm.extractvalue %602[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %653 = llvm.extractvalue %602[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %654 = llvm.extractvalue %602[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %655 = llvm.extractvalue %602[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %656 = llvm.extractvalue %602[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %657 = llvm.extractvalue %602[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %658 = llvm.extractvalue %602[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %659 = llvm.extractvalue %602[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %660 = llvm.extractvalue %632[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %661 = llvm.extractvalue %632[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %662 = llvm.extractvalue %632[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %663 = llvm.extractvalue %632[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %664 = llvm.extractvalue %632[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %665 = llvm.extractvalue %632[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %666 = llvm.extractvalue %632[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %667 = llvm.extractvalue %632[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %668 = llvm.extractvalue %632[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    gpu.launch_func  @forward_kernel_8::@forward_kernel blocks in (%52, %52, %52) threads in (%52, %52, %52) : i64 args(%53 : i64, %53 : i64, %642 : !llvm.ptr, %643 : !llvm.ptr, %644 : i64, %645 : i64, %646 : i64, %647 : i64, %648 : i64, %649 : i64, %650 : i64, %651 : !llvm.ptr, %652 : !llvm.ptr, %653 : i64, %654 : i64, %655 : i64, %656 : i64, %657 : i64, %658 : i64, %659 : i64, %660 : !llvm.ptr, %661 : !llvm.ptr, %662 : i64, %663 : i64, %664 : i64, %665 : i64, %666 : i64, %667 : i64, %668 : i64)
    %669 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %670 = llvm.extractvalue %632[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %671 = llvm.extractvalue %632[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %672 = llvm.insertvalue %670, %669[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %673 = llvm.insertvalue %671, %672[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %674 = llvm.mlir.constant(0 : index) : i64
    %675 = llvm.insertvalue %674, %673[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %676 = llvm.mlir.constant(1 : index) : i64
    %677 = llvm.insertvalue %676, %675[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %678 = llvm.mlir.constant(10 : index) : i64
    %679 = llvm.insertvalue %678, %677[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %680 = llvm.mlir.constant(10 : index) : i64
    %681 = llvm.insertvalue %680, %679[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %682 = llvm.mlir.constant(1 : index) : i64
    %683 = llvm.insertvalue %682, %681[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %684 = llvm.extractvalue %5[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %685 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %686 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %687 = llvm.insertvalue %684, %686[0] : !llvm.struct<(ptr, ptr, i64)>
    %688 = llvm.insertvalue %685, %687[1] : !llvm.struct<(ptr, ptr, i64)>
    %689 = llvm.mlir.constant(0 : index) : i64
    %690 = llvm.insertvalue %689, %688[2] : !llvm.struct<(ptr, ptr, i64)>
    %691 = llvm.extractvalue %5[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %692 = llvm.extractvalue %5[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %693 = llvm.extractvalue %5[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %694 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %695 = llvm.extractvalue %690[0] : !llvm.struct<(ptr, ptr, i64)>
    %696 = llvm.extractvalue %690[1] : !llvm.struct<(ptr, ptr, i64)>
    %697 = llvm.insertvalue %695, %694[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %698 = llvm.insertvalue %696, %697[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %699 = llvm.mlir.constant(0 : index) : i64
    %700 = llvm.insertvalue %699, %698[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %701 = llvm.mlir.constant(1 : index) : i64
    %702 = llvm.insertvalue %701, %700[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %703 = llvm.mlir.constant(10 : index) : i64
    %704 = llvm.insertvalue %703, %702[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %705 = llvm.mlir.constant(10 : index) : i64
    %706 = llvm.insertvalue %705, %704[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %707 = llvm.mlir.constant(1 : index) : i64
    %708 = llvm.insertvalue %707, %706[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %709 = llvm.mlir.constant(1 : index) : i64
    %710 = llvm.mlir.constant(10 : index) : i64
    %711 = llvm.mlir.constant(1 : index) : i64
    %712 = llvm.mlir.constant(10 : index) : i64
    %713 = llvm.mlir.zero : !llvm.ptr
    %714 = llvm.getelementptr %713[%712] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %715 = llvm.ptrtoint %714 : !llvm.ptr to i64
    %716 = llvm.mlir.constant(64 : index) : i64
    %717 = llvm.add %715, %716 : i64
    %718 = llvm.call @malloc(%717) : (i64) -> !llvm.ptr
    %719 = llvm.ptrtoint %718 : !llvm.ptr to i64
    %720 = llvm.mlir.constant(1 : index) : i64
    %721 = llvm.sub %716, %720 : i64
    %722 = llvm.add %719, %721 : i64
    %723 = llvm.urem %722, %716 : i64
    %724 = llvm.sub %722, %723 : i64
    %725 = llvm.inttoptr %724 : i64 to !llvm.ptr
    %726 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %727 = llvm.insertvalue %718, %726[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %728 = llvm.insertvalue %725, %727[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %729 = llvm.mlir.constant(0 : index) : i64
    %730 = llvm.insertvalue %729, %728[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %731 = llvm.insertvalue %709, %730[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %732 = llvm.insertvalue %710, %731[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %733 = llvm.insertvalue %710, %732[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %734 = llvm.insertvalue %711, %733[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %735 = llvm.extractvalue %683[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %736 = llvm.extractvalue %683[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %737 = llvm.extractvalue %683[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %738 = llvm.extractvalue %683[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %739 = llvm.extractvalue %683[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %740 = llvm.extractvalue %683[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %741 = llvm.extractvalue %683[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %742 = llvm.extractvalue %708[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %743 = llvm.extractvalue %708[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %744 = llvm.extractvalue %708[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %745 = llvm.extractvalue %708[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %746 = llvm.extractvalue %708[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %747 = llvm.extractvalue %708[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %748 = llvm.extractvalue %708[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %749 = llvm.extractvalue %734[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %750 = llvm.extractvalue %734[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %751 = llvm.extractvalue %734[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %752 = llvm.extractvalue %734[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %753 = llvm.extractvalue %734[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %754 = llvm.extractvalue %734[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %755 = llvm.extractvalue %734[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    gpu.launch_func  @forward_kernel_9::@forward_kernel blocks in (%52, %52, %52) threads in (%50, %52, %52) : i64 args(%53 : i64, %53 : i64, %735 : !llvm.ptr, %736 : !llvm.ptr, %737 : i64, %738 : i64, %739 : i64, %740 : i64, %741 : i64, %742 : !llvm.ptr, %743 : !llvm.ptr, %744 : i64, %745 : i64, %746 : i64, %747 : i64, %748 : i64, %749 : !llvm.ptr, %750 : !llvm.ptr, %751 : i64, %752 : i64, %753 : i64, %754 : i64, %755 : i64)
    %756 = llvm.extractvalue %146[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    llvm.call @free(%756) : (!llvm.ptr) -> ()
    %757 = llvm.extractvalue %248[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @free(%757) : (!llvm.ptr) -> ()
    %758 = llvm.extractvalue %295[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @free(%758) : (!llvm.ptr) -> ()
    %759 = llvm.extractvalue %389[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    llvm.call @free(%759) : (!llvm.ptr) -> ()
    %760 = llvm.extractvalue %491[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @free(%760) : (!llvm.ptr) -> ()
    %761 = llvm.extractvalue %538[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @free(%761) : (!llvm.ptr) -> ()
    %762 = llvm.extractvalue %632[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    llvm.call @free(%762) : (!llvm.ptr) -> ()
    llvm.return %734 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
  gpu.module @forward_kernel [#nvvm.target] {
    llvm.func @forward_kernel(%arg0: i64, %arg1: i64, %arg2: f32, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64) attributes {gpu.kernel, nvvm.kernel} {
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
    llvm.func @forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: !llvm.ptr, %arg21: !llvm.ptr, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64) attributes {gpu.kernel, nvvm.kernel} {
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
    llvm.func @forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: !llvm.ptr, %arg17: !llvm.ptr, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64) attributes {gpu.kernel, nvvm.kernel} {
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
    llvm.func @forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: f32, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64) attributes {gpu.kernel, nvvm.kernel} {
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
    llvm.func @forward_kernel(%arg0: i64, %arg1: i64, %arg2: f32, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64) attributes {gpu.kernel, nvvm.kernel} {
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
    llvm.func @forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: !llvm.ptr, %arg21: !llvm.ptr, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64) attributes {gpu.kernel, nvvm.kernel} {
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
    llvm.func @forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: !llvm.ptr, %arg17: !llvm.ptr, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64) attributes {gpu.kernel, nvvm.kernel} {
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
    llvm.func @forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: f32, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64) attributes {gpu.kernel, nvvm.kernel} {
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
    llvm.func @forward_kernel(%arg0: i64, %arg1: i64, %arg2: f32, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64) attributes {gpu.kernel, nvvm.kernel} {
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
    llvm.func @forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: !llvm.ptr, %arg21: !llvm.ptr, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64) attributes {gpu.kernel, nvvm.kernel} {
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
    llvm.func @forward_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: !llvm.ptr, %arg17: !llvm.ptr, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64) attributes {gpu.kernel, nvvm.kernel} {
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
