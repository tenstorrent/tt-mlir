// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @jit__normal attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <["default_updated"=1, "default"=1]>
  func.func public @main(%arg0: tensor<2xui32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<50xf32> {jax.result_info = "result", ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<-0.99999994> : tensor<f32>
    %cst_1 = stablehlo.constant dense<1.41421354> : tensor<f32>
    %cst_2 = stablehlo.constant dense<5.000000e+00> : tensor<50xf32>
    %cst_3 = stablehlo.constant dense<2.500000e+00> : tensor<50xf32>
    %cst_4 = stablehlo.constant dense<3.000000e+00> : tensor<50xf32>
    %cst_5 = stablehlo.constant dense<2.81022636E-8> : tensor<50xf32>
    %cst_6 = stablehlo.constant dense<-2.00214257E-4> : tensor<50xf32>
    %cst_7 = stablehlo.constant dense<3.43273939E-7> : tensor<50xf32>
    %cst_8 = stablehlo.constant dense<1.00950558E-4> : tensor<50xf32>
    %cst_9 = stablehlo.constant dense<-3.5233877E-6> : tensor<50xf32>
    %cst_10 = stablehlo.constant dense<0.00134934322> : tensor<50xf32>
    %cst_11 = stablehlo.constant dense<-4.39150654E-6> : tensor<50xf32>
    %cst_12 = stablehlo.constant dense<-0.00367342844> : tensor<50xf32>
    %cst_13 = stablehlo.constant dense<2.1858087E-4> : tensor<50xf32>
    %cst_14 = stablehlo.constant dense<0.00573950773> : tensor<50xf32>
    %cst_15 = stablehlo.constant dense<-0.00125372503> : tensor<50xf32>
    %cst_16 = stablehlo.constant dense<-0.0076224613> : tensor<50xf32>
    %cst_17 = stablehlo.constant dense<-0.00417768164> : tensor<50xf32>
    %cst_18 = stablehlo.constant dense<0.00943887047> : tensor<50xf32>
    %cst_19 = stablehlo.constant dense<0.246640727> : tensor<50xf32>
    %cst_20 = stablehlo.constant dense<1.00167406> : tensor<50xf32>
    %cst_21 = stablehlo.constant dense<1.50140941> : tensor<50xf32>
    %cst_22 = stablehlo.constant dense<2.83297682> : tensor<50xf32>
    %cst_23 = stablehlo.constant dense<1.000000e+00> : tensor<50xf32>
    %cst_24 = stablehlo.constant dense<0x7F800000> : tensor<50xf32>
    // CHECK: ttir.rand
    // CHECK-NOT: stablehlo.bitcast_convert
    // CHECK-SAME: <{dtype = f32, high = 1.000000e+00 : f32, low = -0.99999994 : f32, seed = 0 : ui32, size = [50 : i32]}>
    %0 = stablehlo.composite "tenstorrent.uniform" %arg0, %cst_0, %cst {composite_attributes = {dtype = f32, shape = [50]}, decomposition = @tenstorrent.uniform} : (tensor<2xui32>, tensor<f32>, tensor<f32>) -> tensor<50xf32>
    %1 = stablehlo.negate %0 : tensor<50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<50xf32>
    %3 = stablehlo.log_plus_one %2 : tensor<50xf32>
    %4 = stablehlo.negate %3 : tensor<50xf32>
    %5 = stablehlo.compare  LT, %4, %cst_2 : (tensor<50xf32>, tensor<50xf32>) -> tensor<50xi1>
    %6 = stablehlo.subtract %4, %cst_3 : tensor<50xf32>
    %7 = stablehlo.sqrt %4 : tensor<50xf32>
    %8 = stablehlo.subtract %7, %cst_4 : tensor<50xf32>
    %9 = stablehlo.select %5, %6, %8 : tensor<50xi1>, tensor<50xf32>
    %10 = stablehlo.select %5, %cst_5, %cst_6 : tensor<50xi1>, tensor<50xf32>
    %11 = stablehlo.select %5, %cst_7, %cst_8 : tensor<50xi1>, tensor<50xf32>
    %12 = stablehlo.multiply %10, %9 : tensor<50xf32>
    %13 = stablehlo.add %11, %12 : tensor<50xf32>
    %14 = stablehlo.select %5, %cst_9, %cst_10 : tensor<50xi1>, tensor<50xf32>
    %15 = stablehlo.multiply %13, %9 : tensor<50xf32>
    %16 = stablehlo.add %14, %15 : tensor<50xf32>
    %17 = stablehlo.select %5, %cst_11, %cst_12 : tensor<50xi1>, tensor<50xf32>
    %18 = stablehlo.multiply %16, %9 : tensor<50xf32>
    %19 = stablehlo.add %17, %18 : tensor<50xf32>
    %20 = stablehlo.select %5, %cst_13, %cst_14 : tensor<50xi1>, tensor<50xf32>
    %21 = stablehlo.multiply %19, %9 : tensor<50xf32>
    %22 = stablehlo.add %20, %21 : tensor<50xf32>
    %23 = stablehlo.select %5, %cst_15, %cst_16 : tensor<50xi1>, tensor<50xf32>
    %24 = stablehlo.multiply %22, %9 : tensor<50xf32>
    %25 = stablehlo.add %23, %24 : tensor<50xf32>
    %26 = stablehlo.select %5, %cst_17, %cst_18 : tensor<50xi1>, tensor<50xf32>
    %27 = stablehlo.multiply %25, %9 : tensor<50xf32>
    %28 = stablehlo.add %26, %27 : tensor<50xf32>
    %29 = stablehlo.select %5, %cst_19, %cst_20 : tensor<50xi1>, tensor<50xf32>
    %30 = stablehlo.multiply %28, %9 : tensor<50xf32>
    %31 = stablehlo.add %29, %30 : tensor<50xf32>
    %32 = stablehlo.select %5, %cst_21, %cst_22 : tensor<50xi1>, tensor<50xf32>
    %33 = stablehlo.multiply %31, %9 : tensor<50xf32>
    %34 = stablehlo.add %32, %33 : tensor<50xf32>
    %35 = stablehlo.multiply %34, %0 : tensor<50xf32>
    %36 = stablehlo.abs %0 : tensor<50xf32>
    %37 = stablehlo.compare  EQ, %36, %cst_23 : (tensor<50xf32>, tensor<50xf32>) -> tensor<50xi1>
    %38 = stablehlo.multiply %0, %cst_24 : tensor<50xf32>
    %39 = stablehlo.select %37, %38, %35 : tensor<50xi1>, tensor<50xf32>
    %40 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50xf32>
    %41 = stablehlo.multiply %40, %39 : tensor<50xf32>
    return %41 : tensor<50xf32>
  }
  func.func private @tenstorrent.uniform(%arg0: tensor<2xui32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, %arg1: tensor<f32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, %arg2: tensor<f32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<50xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %c = stablehlo.constant dense<466688986> : tensor<ui32>
    %c_0 = stablehlo.constant dense<13> : tensor<ui32>
    %c_1 = stablehlo.constant dense<19> : tensor<ui32>
    %c_2 = stablehlo.constant dense<15> : tensor<ui32>
    %c_3 = stablehlo.constant dense<17> : tensor<ui32>
    %c_4 = stablehlo.constant dense<26> : tensor<ui32>
    %c_5 = stablehlo.constant dense<6> : tensor<ui32>
    %c_6 = stablehlo.constant dense<1> : tensor<ui32>
    %c_7 = stablehlo.constant dense<29> : tensor<ui32>
    %c_8 = stablehlo.constant dense<3> : tensor<ui32>
    %c_9 = stablehlo.constant dense<16> : tensor<ui32>
    %c_10 = stablehlo.constant dense<24> : tensor<ui32>
    %c_11 = stablehlo.constant dense<8> : tensor<ui32>
    %c_12 = stablehlo.constant dense<2> : tensor<ui32>
    %c_13 = stablehlo.constant dense<4> : tensor<ui32>
    %c_14 = stablehlo.constant dense<5> : tensor<ui32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c_15 = stablehlo.constant dense<1065353216> : tensor<ui32>
    %c_16 = stablehlo.constant dense<9> : tensor<ui32>
    %c_17 = stablehlo.constant dense<32> : tensor<ui64>
    %c_18 = stablehlo.constant dense<1> : tensor<ui64>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %2 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %3 = stablehlo.reshape %2 : (tensor<1xui32>) -> tensor<ui32>
    %4 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %5 = stablehlo.reshape %4 : (tensor<1xui32>) -> tensor<ui32>
    %6 = stablehlo.iota dim = 0 : tensor<50xui64>
    %7 = stablehlo.broadcast_in_dim %c_18, dims = [] : (tensor<ui64>) -> tensor<50xui64>
    %8 = stablehlo.multiply %7, %6 : tensor<50xui64>
    %9 = stablehlo.broadcast_in_dim %c_17, dims = [] : (tensor<ui64>) -> tensor<50xui64>
    %10 = stablehlo.shift_right_logical %8, %9 : tensor<50xui64>
    %11 = stablehlo.convert %8 : (tensor<50xui64>) -> tensor<50xui32>
    %12 = stablehlo.convert %10 : (tensor<50xui64>) -> tensor<50xui32>
    %13 = stablehlo.xor %3, %5 : tensor<ui32>
    %14 = stablehlo.xor %13, %c : tensor<ui32>
    %15 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %16 = stablehlo.add %12, %15 : tensor<50xui32>
    %17 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %18 = stablehlo.add %11, %17 : tensor<50xui32>
    %19 = stablehlo.add %16, %18 : tensor<50xui32>
    %20 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %21 = stablehlo.shift_left %18, %20 : tensor<50xui32>
    %22 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %23 = stablehlo.shift_right_logical %18, %22 : tensor<50xui32>
    %24 = stablehlo.or %21, %23 : tensor<50xui32>
    %25 = stablehlo.xor %19, %24 : tensor<50xui32>
    %26 = stablehlo.add %19, %25 : tensor<50xui32>
    %27 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %28 = stablehlo.shift_left %25, %27 : tensor<50xui32>
    %29 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %30 = stablehlo.shift_right_logical %25, %29 : tensor<50xui32>
    %31 = stablehlo.or %28, %30 : tensor<50xui32>
    %32 = stablehlo.xor %26, %31 : tensor<50xui32>
    %33 = stablehlo.add %26, %32 : tensor<50xui32>
    %34 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %35 = stablehlo.shift_left %32, %34 : tensor<50xui32>
    %36 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %37 = stablehlo.shift_right_logical %32, %36 : tensor<50xui32>
    %38 = stablehlo.or %35, %37 : tensor<50xui32>
    %39 = stablehlo.xor %33, %38 : tensor<50xui32>
    %40 = stablehlo.add %33, %39 : tensor<50xui32>
    %41 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %42 = stablehlo.shift_left %39, %41 : tensor<50xui32>
    %43 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %44 = stablehlo.shift_right_logical %39, %43 : tensor<50xui32>
    %45 = stablehlo.or %42, %44 : tensor<50xui32>
    %46 = stablehlo.xor %40, %45 : tensor<50xui32>
    %47 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %48 = stablehlo.add %40, %47 : tensor<50xui32>
    %49 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %50 = stablehlo.add %46, %49 : tensor<50xui32>
    %51 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %52 = stablehlo.add %50, %51 : tensor<50xui32>
    %53 = stablehlo.add %48, %52 : tensor<50xui32>
    %54 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %55 = stablehlo.shift_left %52, %54 : tensor<50xui32>
    %56 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %57 = stablehlo.shift_right_logical %52, %56 : tensor<50xui32>
    %58 = stablehlo.or %55, %57 : tensor<50xui32>
    %59 = stablehlo.xor %53, %58 : tensor<50xui32>
    %60 = stablehlo.add %53, %59 : tensor<50xui32>
    %61 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %62 = stablehlo.shift_left %59, %61 : tensor<50xui32>
    %63 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %64 = stablehlo.shift_right_logical %59, %63 : tensor<50xui32>
    %65 = stablehlo.or %62, %64 : tensor<50xui32>
    %66 = stablehlo.xor %60, %65 : tensor<50xui32>
    %67 = stablehlo.add %60, %66 : tensor<50xui32>
    %68 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %69 = stablehlo.shift_left %66, %68 : tensor<50xui32>
    %70 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %71 = stablehlo.shift_right_logical %66, %70 : tensor<50xui32>
    %72 = stablehlo.or %69, %71 : tensor<50xui32>
    %73 = stablehlo.xor %67, %72 : tensor<50xui32>
    %74 = stablehlo.add %67, %73 : tensor<50xui32>
    %75 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %76 = stablehlo.shift_left %73, %75 : tensor<50xui32>
    %77 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %78 = stablehlo.shift_right_logical %73, %77 : tensor<50xui32>
    %79 = stablehlo.or %76, %78 : tensor<50xui32>
    %80 = stablehlo.xor %74, %79 : tensor<50xui32>
    %81 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %82 = stablehlo.add %74, %81 : tensor<50xui32>
    %83 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %84 = stablehlo.add %80, %83 : tensor<50xui32>
    %85 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %86 = stablehlo.add %84, %85 : tensor<50xui32>
    %87 = stablehlo.add %82, %86 : tensor<50xui32>
    %88 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %89 = stablehlo.shift_left %86, %88 : tensor<50xui32>
    %90 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %91 = stablehlo.shift_right_logical %86, %90 : tensor<50xui32>
    %92 = stablehlo.or %89, %91 : tensor<50xui32>
    %93 = stablehlo.xor %87, %92 : tensor<50xui32>
    %94 = stablehlo.add %87, %93 : tensor<50xui32>
    %95 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %96 = stablehlo.shift_left %93, %95 : tensor<50xui32>
    %97 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %98 = stablehlo.shift_right_logical %93, %97 : tensor<50xui32>
    %99 = stablehlo.or %96, %98 : tensor<50xui32>
    %100 = stablehlo.xor %94, %99 : tensor<50xui32>
    %101 = stablehlo.add %94, %100 : tensor<50xui32>
    %102 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %103 = stablehlo.shift_left %100, %102 : tensor<50xui32>
    %104 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %105 = stablehlo.shift_right_logical %100, %104 : tensor<50xui32>
    %106 = stablehlo.or %103, %105 : tensor<50xui32>
    %107 = stablehlo.xor %101, %106 : tensor<50xui32>
    %108 = stablehlo.add %101, %107 : tensor<50xui32>
    %109 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %110 = stablehlo.shift_left %107, %109 : tensor<50xui32>
    %111 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %112 = stablehlo.shift_right_logical %107, %111 : tensor<50xui32>
    %113 = stablehlo.or %110, %112 : tensor<50xui32>
    %114 = stablehlo.xor %108, %113 : tensor<50xui32>
    %115 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %116 = stablehlo.add %108, %115 : tensor<50xui32>
    %117 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %118 = stablehlo.add %114, %117 : tensor<50xui32>
    %119 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %120 = stablehlo.add %118, %119 : tensor<50xui32>
    %121 = stablehlo.add %116, %120 : tensor<50xui32>
    %122 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %123 = stablehlo.shift_left %120, %122 : tensor<50xui32>
    %124 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %125 = stablehlo.shift_right_logical %120, %124 : tensor<50xui32>
    %126 = stablehlo.or %123, %125 : tensor<50xui32>
    %127 = stablehlo.xor %121, %126 : tensor<50xui32>
    %128 = stablehlo.add %121, %127 : tensor<50xui32>
    %129 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %130 = stablehlo.shift_left %127, %129 : tensor<50xui32>
    %131 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %132 = stablehlo.shift_right_logical %127, %131 : tensor<50xui32>
    %133 = stablehlo.or %130, %132 : tensor<50xui32>
    %134 = stablehlo.xor %128, %133 : tensor<50xui32>
    %135 = stablehlo.add %128, %134 : tensor<50xui32>
    %136 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %137 = stablehlo.shift_left %134, %136 : tensor<50xui32>
    %138 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %139 = stablehlo.shift_right_logical %134, %138 : tensor<50xui32>
    %140 = stablehlo.or %137, %139 : tensor<50xui32>
    %141 = stablehlo.xor %135, %140 : tensor<50xui32>
    %142 = stablehlo.add %135, %141 : tensor<50xui32>
    %143 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %144 = stablehlo.shift_left %141, %143 : tensor<50xui32>
    %145 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %146 = stablehlo.shift_right_logical %141, %145 : tensor<50xui32>
    %147 = stablehlo.or %144, %146 : tensor<50xui32>
    %148 = stablehlo.xor %142, %147 : tensor<50xui32>
    %149 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %150 = stablehlo.add %142, %149 : tensor<50xui32>
    %151 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %152 = stablehlo.add %148, %151 : tensor<50xui32>
    %153 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %154 = stablehlo.add %152, %153 : tensor<50xui32>
    %155 = stablehlo.add %150, %154 : tensor<50xui32>
    %156 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %157 = stablehlo.shift_left %154, %156 : tensor<50xui32>
    %158 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %159 = stablehlo.shift_right_logical %154, %158 : tensor<50xui32>
    %160 = stablehlo.or %157, %159 : tensor<50xui32>
    %161 = stablehlo.xor %155, %160 : tensor<50xui32>
    %162 = stablehlo.add %155, %161 : tensor<50xui32>
    %163 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %164 = stablehlo.shift_left %161, %163 : tensor<50xui32>
    %165 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %166 = stablehlo.shift_right_logical %161, %165 : tensor<50xui32>
    %167 = stablehlo.or %164, %166 : tensor<50xui32>
    %168 = stablehlo.xor %162, %167 : tensor<50xui32>
    %169 = stablehlo.add %162, %168 : tensor<50xui32>
    %170 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %171 = stablehlo.shift_left %168, %170 : tensor<50xui32>
    %172 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %173 = stablehlo.shift_right_logical %168, %172 : tensor<50xui32>
    %174 = stablehlo.or %171, %173 : tensor<50xui32>
    %175 = stablehlo.xor %169, %174 : tensor<50xui32>
    %176 = stablehlo.add %169, %175 : tensor<50xui32>
    %177 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %178 = stablehlo.shift_left %175, %177 : tensor<50xui32>
    %179 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %180 = stablehlo.shift_right_logical %175, %179 : tensor<50xui32>
    %181 = stablehlo.or %178, %180 : tensor<50xui32>
    %182 = stablehlo.xor %176, %181 : tensor<50xui32>
    %183 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %184 = stablehlo.add %176, %183 : tensor<50xui32>
    %185 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %186 = stablehlo.add %182, %185 : tensor<50xui32>
    %187 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %188 = stablehlo.add %186, %187 : tensor<50xui32>
    %189 = stablehlo.xor %184, %188 : tensor<50xui32>
    %190 = stablehlo.broadcast_in_dim %c_16, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %191 = stablehlo.shift_right_logical %189, %190 : tensor<50xui32>
    %192 = stablehlo.broadcast_in_dim %c_15, dims = [] : (tensor<ui32>) -> tensor<50xui32>
    %193 = stablehlo.or %191, %192 : tensor<50xui32>
    %194 = stablehlo.bitcast_convert %193 : (tensor<50xui32>) -> tensor<50xf32>
    %195 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<50xf32>
    %196 = stablehlo.subtract %194, %195 : tensor<50xf32>
    %197 = stablehlo.subtract %1, %0 : tensor<1xf32>
    %198 = stablehlo.broadcast_in_dim %197, dims = [0] : (tensor<1xf32>) -> tensor<50xf32>
    %199 = stablehlo.multiply %196, %198 : tensor<50xf32>
    %200 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1xf32>) -> tensor<50xf32>
    %201 = stablehlo.add %199, %200 : tensor<50xf32>
    %202 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1xf32>) -> tensor<50xf32>
    %203 = stablehlo.maximum %202, %201 : tensor<50xf32>
    return %203 : tensor<50xf32>
  }
}
