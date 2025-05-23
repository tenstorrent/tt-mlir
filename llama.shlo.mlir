module {
  func.func @main(%arg0: tensor<3072xbf16>, %arg1: tensor<3072xbf16>, %arg2: tensor<3072xbf16>, %arg3: tensor<3072xbf16>, %arg4: tensor<3072xbf16>, %arg5: tensor<128256x3072xbf16>, %arg6: tensor<64xi64>, %arg7: tensor<15x64xbf16>, %arg8: tensor<1x64x1xf32>, %arg9: tensor<3072x3072xbf16>, %arg10: tensor<3072x1024xbf16>, %arg11: tensor<3072x1024xbf16>, %arg12: tensor<3072x3072xbf16>, %arg13: tensor<3072x8192xbf16>, %arg14: tensor<3072x8192xbf16>, %arg15: tensor<8192x3072xbf16>, %arg16: tensor<3072x3072xbf16>, %arg17: tensor<3072x1024xbf16>, %arg18: tensor<3072x1024xbf16>, %arg19: tensor<3072x3072xbf16>, %arg20: tensor<3072x8192xbf16>, %arg21: tensor<3072x8192xbf16>, %arg22: tensor<8192x3072xbf16>, %arg23: tensor<3072x128256xbf16>, %arg24: tensor<3072x3072xbf16>, %arg25: tensor<1024x3072xbf16>, %arg26: tensor<1024x3072xbf16>, %arg27: tensor<3072x3072xbf16>, %arg28: tensor<8192x3072xbf16>, %arg29: tensor<8192x3072xbf16>, %arg30: tensor<3072x8192xbf16>, %arg31: tensor<3072x3072xbf16>, %arg32: tensor<1024x3072xbf16>, %arg33: tensor<1024x3072xbf16>, %arg34: tensor<3072x3072xbf16>, %arg35: tensor<8192x3072xbf16>, %arg36: tensor<8192x3072xbf16>, %arg37: tensor<3072x8192xbf16>, %arg38: tensor<128256x3072xbf16>, %arg39: tensor<1x8x64x128xbf16>, %arg40: tensor<1x8x64x128xbf16>, %arg41: tensor<1x8x64x128xbf16>, %arg42: tensor<1x8x64x128xbf16>, %arg43: tensor<64xf32>, %arg44: tensor<1x15xi64>, %arg45: tensor<15xi64>) -> (tensor<1x15x128256xbf16>, tensor<1x8x64x128xbf16>, tensor<1x8x64x128xbf16>, tensor<1x8x64x128xbf16>, tensor<1x8x64x128xbf16>) {
    %c = stablehlo.constant dense<0> : tensor<1xi64>
    %c_0 = stablehlo.constant dense<1> : tensor<1xi64>
    %c_1 = stablehlo.constant dense<0> : tensor<8xi64>
    %c_2 = stablehlo.constant dense<1> : tensor<8xi64>
    %c_3 = stablehlo.constant dense<8> : tensor<i64>
    %c_4 = stablehlo.constant dense<0> : tensor<128xi64>
    %c_5 = stablehlo.constant dense<1> : tensor<128xi64>
    %c_6 = stablehlo.constant dense<128> : tensor<i64>
    %c_7 = stablehlo.constant dense<1> : tensor<i64>
    %c_8 = stablehlo.constant dense<0> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_9 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c_10 = stablehlo.constant dense<false> : tensor<i1>
    %cst_11 = arith.constant dense<1.000000e+00> : tensor<1xf64>
    %cst_12 = arith.constant dense<2> : tensor<1xi64>
    %cst_13 = arith.constant dense<3072> : tensor<1xi64>
    %cst_14 = arith.constant dense<1.000000e-05> : tensor<1xf64>
    %cst_15 = arith.constant dense<0.29730177875068026> : tensor<1xf64>
    %cst_16 = arith.constant dense<0xFFF0000000000000> : tensor<1xf64>
    %0 = "stablehlo.gather"(%arg38, %arg44) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 3072>}> : (tensor<128256x3072xbf16>, tensor<1x15xi64>) -> tensor<1x15x3072xbf16>
    %1 = stablehlo.convert %0 : tensor<1x15x3072xbf16>
    %2 = stablehlo.reshape %arg45 : (tensor<15xi64>) -> tensor<1x15xi64>
    %3 = stablehlo.reshape %arg45 : (tensor<15xi64>) -> tensor<15x1xi64>
    %4 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<64xi64>) -> tensor<15x64xi64>
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<15x1xi64>) -> tensor<15x64xi64>
    %6 = stablehlo.compare  GT, %4, %5,  SIGNED : (tensor<15x64xi64>, tensor<15x64xi64>) -> tensor<15x64xi1>
    %7 = stablehlo.convert %6 : (tensor<15x64xi1>) -> tensor<15x64xbf16>
    %8 = stablehlo.multiply %arg7, %7 : tensor<15x64xbf16>
    %9 = stablehlo.reshape %2 : (tensor<1x15xi64>) -> tensor<1x1x15xi64>
    %10 = stablehlo.convert %9 : (tensor<1x1x15xi64>) -> tensor<1x1x15xf32>
    %11 = stablehlo.reshape %10 : (tensor<1x1x15xf32>) -> tensor<1x1x15xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<1x1x15xf32>) -> tensor<1x1x15xf32>
    %13 = stablehlo.dot_general %arg8, %12, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x64x1xf32>, tensor<1x1x15xf32>) -> tensor<1x64x15xf32>
    %14 = stablehlo.reshape %13 : (tensor<1x64x15xf32>) -> tensor<1x64x15xf32>
    %15 = stablehlo.transpose %14, dims = [0, 2, 1] : (tensor<1x64x15xf32>) -> tensor<1x15x64xf32>
    %16 = stablehlo.concatenate %15, %15, dim = 2 : (tensor<1x15x64xf32>, tensor<1x15x64xf32>) -> tensor<1x15x128xf32>
    %17 = stablehlo.cosine %16 : tensor<1x15x128xf32>
    %18 = stablehlo.sine %16 : tensor<1x15x128xf32>
    %19 = stablehlo.convert %cst_11 : (tensor<1xf64>) -> tensor<1xf32>
    %20 = stablehlo.reshape %19 : (tensor<1xf32>) -> tensor<f32>
    %21 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<1x15x128xf32>) -> tensor<1x15x128xf32>
    %22 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<1x15x128xf32>
    %23 = stablehlo.multiply %21, %22 : tensor<1x15x128xf32>
    %24 = stablehlo.broadcast_in_dim %18, dims = [0, 1, 2] : (tensor<1x15x128xf32>) -> tensor<1x15x128xf32>
    %25 = stablehlo.multiply %24, %22 : tensor<1x15x128xf32>
    %26 = stablehlo.convert %23 : (tensor<1x15x128xf32>) -> tensor<1x15x128xbf16>
    %27 = stablehlo.convert %25 : (tensor<1x15x128xf32>) -> tensor<1x15x128xbf16>
    %28 = stablehlo.convert %1 : (tensor<1x15x3072xbf16>) -> tensor<1x15x3072xf32>
    %29 = stablehlo.convert %cst_12 : (tensor<1xi64>) -> tensor<1xf32>
    %30 = stablehlo.reshape %29 : (tensor<1xf32>) -> tensor<f32>
    %31 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2] : (tensor<1x15x3072xf32>) -> tensor<1x15x3072xf32>
    %32 = stablehlo.broadcast_in_dim %30, dims = [] : (tensor<f32>) -> tensor<1x15x3072xf32>
    %33 = stablehlo.power %31, %32 : tensor<1x15x3072xf32>
    %34 = stablehlo.reduce(%33 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x15x3072xf32>, tensor<f32>) -> tensor<1x15xf32>
    %35 = stablehlo.reshape %34 : (tensor<1x15xf32>) -> tensor<1x15x1xf32>
    %36 = stablehlo.convert %cst_13 : (tensor<1xi64>) -> tensor<1xf32>
    %37 = stablehlo.reshape %36 : (tensor<1xf32>) -> tensor<f32>
    %38 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2] : (tensor<1x15x1xf32>) -> tensor<1x15x1xf32>
    %39 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f32>) -> tensor<1x15x1xf32>
    %40 = stablehlo.divide %38, %39 : tensor<1x15x1xf32>
    %41 = stablehlo.convert %cst_14 : (tensor<1xf64>) -> tensor<1xf32>
    %42 = stablehlo.reshape %41 : (tensor<1xf32>) -> tensor<f32>
    %43 = stablehlo.broadcast_in_dim %40, dims = [0, 1, 2] : (tensor<1x15x1xf32>) -> tensor<1x15x1xf32>
    %44 = stablehlo.broadcast_in_dim %42, dims = [] : (tensor<f32>) -> tensor<1x15x1xf32>
    %45 = stablehlo.add %43, %44 : tensor<1x15x1xf32>
    %46 = stablehlo.rsqrt %45 : tensor<1x15x1xf32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2] : (tensor<1x15x1xf32>) -> tensor<1x15x3072xf32>
    %48 = stablehlo.multiply %31, %47 : tensor<1x15x3072xf32>
    %49 = stablehlo.convert %48 : (tensor<1x15x3072xf32>) -> tensor<1x15x3072xbf16>
    %50 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<3072xbf16>) -> tensor<1x15x3072xbf16>
    %51 = stablehlo.broadcast_in_dim %49, dims = [0, 1, 2] : (tensor<1x15x3072xbf16>) -> tensor<1x15x3072xbf16>
    %52 = stablehlo.multiply %50, %51 : tensor<1x15x3072xbf16>
    %53 = stablehlo.reshape %52 : (tensor<1x15x3072xbf16>) -> tensor<15x3072xbf16>
    %54 = stablehlo.dot_general %53, %arg9, contracting_dims = [1] x [0] : (tensor<15x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<15x3072xbf16>
    %55 = stablehlo.reshape %54 : (tensor<15x3072xbf16>) -> tensor<1x15x3072xbf16>
    %56 = stablehlo.dot_general %53, %arg10, contracting_dims = [1] x [0] : (tensor<15x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<15x1024xbf16>
    %57 = stablehlo.reshape %56 : (tensor<15x1024xbf16>) -> tensor<1x15x1024xbf16>
    %58 = stablehlo.dot_general %53, %arg11, contracting_dims = [1] x [0] : (tensor<15x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<15x1024xbf16>
    %59 = stablehlo.reshape %58 : (tensor<15x1024xbf16>) -> tensor<1x15x1024xbf16>
    %60 = stablehlo.reshape %55 : (tensor<1x15x3072xbf16>) -> tensor<1x15x24x128xbf16>
    %61 = stablehlo.transpose %60, dims = [0, 2, 1, 3] : (tensor<1x15x24x128xbf16>) -> tensor<1x24x15x128xbf16>
    %62 = stablehlo.reshape %57 : (tensor<1x15x1024xbf16>) -> tensor<1x15x8x128xbf16>
    %63 = stablehlo.transpose %62, dims = [0, 2, 1, 3] : (tensor<1x15x8x128xbf16>) -> tensor<1x8x15x128xbf16>
    %64 = stablehlo.reshape %59 : (tensor<1x15x1024xbf16>) -> tensor<1x15x8x128xbf16>
    %65 = stablehlo.transpose %64, dims = [0, 2, 1, 3] : (tensor<1x15x8x128xbf16>) -> tensor<1x8x15x128xbf16>
    %66 = stablehlo.reshape %26 : (tensor<1x15x128xbf16>) -> tensor<1x1x15x128xbf16>
    %67 = stablehlo.reshape %27 : (tensor<1x15x128xbf16>) -> tensor<1x1x15x128xbf16>
    %68 = stablehlo.broadcast_in_dim %61, dims = [0, 1, 2, 3] : (tensor<1x24x15x128xbf16>) -> tensor<1x24x15x128xbf16>
    %69 = stablehlo.broadcast_in_dim %66, dims = [0, 1, 2, 3] : (tensor<1x1x15x128xbf16>) -> tensor<1x24x15x128xbf16>
    %70 = stablehlo.multiply %68, %69 : tensor<1x24x15x128xbf16>
    %71 = stablehlo.slice %61 [0:1, 0:24, 0:15, 0:64] : (tensor<1x24x15x128xbf16>) -> tensor<1x24x15x64xbf16>
    %72 = stablehlo.slice %61 [0:1, 0:24, 0:15, 64:128] : (tensor<1x24x15x128xbf16>) -> tensor<1x24x15x64xbf16>
    %73 = stablehlo.negate %72 : tensor<1x24x15x64xbf16>
    %74 = stablehlo.concatenate %73, %71, dim = 3 : (tensor<1x24x15x64xbf16>, tensor<1x24x15x64xbf16>) -> tensor<1x24x15x128xbf16>
    %75 = stablehlo.broadcast_in_dim %74, dims = [0, 1, 2, 3] : (tensor<1x24x15x128xbf16>) -> tensor<1x24x15x128xbf16>
    %76 = stablehlo.broadcast_in_dim %67, dims = [0, 1, 2, 3] : (tensor<1x1x15x128xbf16>) -> tensor<1x24x15x128xbf16>
    %77 = stablehlo.multiply %75, %76 : tensor<1x24x15x128xbf16>
    %78 = stablehlo.add %70, %77 : tensor<1x24x15x128xbf16>
    %79 = stablehlo.broadcast_in_dim %63, dims = [0, 1, 2, 3] : (tensor<1x8x15x128xbf16>) -> tensor<1x8x15x128xbf16>
    %80 = stablehlo.broadcast_in_dim %66, dims = [0, 1, 2, 3] : (tensor<1x1x15x128xbf16>) -> tensor<1x8x15x128xbf16>
    %81 = stablehlo.multiply %79, %80 : tensor<1x8x15x128xbf16>
    %82 = stablehlo.slice %63 [0:1, 0:8, 0:15, 0:64] : (tensor<1x8x15x128xbf16>) -> tensor<1x8x15x64xbf16>
    %83 = stablehlo.slice %63 [0:1, 0:8, 0:15, 64:128] : (tensor<1x8x15x128xbf16>) -> tensor<1x8x15x64xbf16>
    %84 = stablehlo.negate %83 : tensor<1x8x15x64xbf16>
    %85 = stablehlo.concatenate %84, %82, dim = 3 : (tensor<1x8x15x64xbf16>, tensor<1x8x15x64xbf16>) -> tensor<1x8x15x128xbf16>
    %86 = stablehlo.broadcast_in_dim %85, dims = [0, 1, 2, 3] : (tensor<1x8x15x128xbf16>) -> tensor<1x8x15x128xbf16>
    %87 = stablehlo.broadcast_in_dim %67, dims = [0, 1, 2, 3] : (tensor<1x1x15x128xbf16>) -> tensor<1x8x15x128xbf16>
    %88 = stablehlo.multiply %86, %87 : tensor<1x8x15x128xbf16>
    %89 = stablehlo.add %81, %88 : tensor<1x8x15x128xbf16>
    %90 = stablehlo.convert %c_6 : (tensor<i64>) -> tensor<f64>
    %91 = stablehlo.convert %c_7 : (tensor<i64>) -> tensor<f64>
    %92 = stablehlo.divide %90, %91 : tensor<f64>
    %93 = stablehlo.ceil %92 : tensor<f64>
    %94 = stablehlo.convert %93 : (tensor<f64>) -> tensor<i64>
    %95 = stablehlo.reshape %94 : (tensor<i64>) -> tensor<1xi64>
    %96 = stablehlo.dynamic_iota %95, dim = 0 : (tensor<1xi64>) -> tensor<128xi64>
    %97 = stablehlo.broadcast_in_dim %96, dims = [0] : (tensor<128xi64>) -> tensor<128xi64>
    %98 = stablehlo.multiply %97, %c_5 : tensor<128xi64>
    %99 = stablehlo.broadcast_in_dim %98, dims = [0] : (tensor<128xi64>) -> tensor<128xi64>
    %100 = stablehlo.add %99, %c_4 : tensor<128xi64>
    %101 = stablehlo.reshape %arg45 : (tensor<15xi64>) -> tensor<15x1xi64>
    %102 = stablehlo.convert %c_3 : (tensor<i64>) -> tensor<f64>
    %103 = stablehlo.divide %102, %91 : tensor<f64>
    %104 = stablehlo.ceil %103 : tensor<f64>
    %105 = stablehlo.convert %104 : (tensor<f64>) -> tensor<i64>
    %106 = stablehlo.reshape %105 : (tensor<i64>) -> tensor<1xi64>
    %107 = stablehlo.dynamic_iota %106, dim = 0 : (tensor<1xi64>) -> tensor<8xi64>
    %108 = stablehlo.broadcast_in_dim %107, dims = [0] : (tensor<8xi64>) -> tensor<8xi64>
    %109 = stablehlo.multiply %108, %c_2 : tensor<8xi64>
    %110 = stablehlo.broadcast_in_dim %109, dims = [0] : (tensor<8xi64>) -> tensor<8xi64>
    %111 = stablehlo.add %110, %c_1 : tensor<8xi64>
    %112 = stablehlo.reshape %111 : (tensor<8xi64>) -> tensor<8x1xi64>
    %113 = stablehlo.reshape %112 : (tensor<8x1xi64>) -> tensor<8x1x1xi64>
    %114 = stablehlo.convert %c_7 : (tensor<i64>) -> tensor<f64>
    %115 = stablehlo.divide %114, %91 : tensor<f64>
    %116 = stablehlo.ceil %115 : tensor<f64>
    %117 = stablehlo.convert %116 : (tensor<f64>) -> tensor<i64>
    %118 = stablehlo.reshape %117 : (tensor<i64>) -> tensor<1xi64>
    %119 = stablehlo.dynamic_iota %118, dim = 0 : (tensor<1xi64>) -> tensor<1xi64>
    %120 = stablehlo.broadcast_in_dim %119, dims = [0] : (tensor<1xi64>) -> tensor<1xi64>
    %121 = stablehlo.multiply %120, %c_0 : tensor<1xi64>
    %122 = stablehlo.broadcast_in_dim %121, dims = [0] : (tensor<1xi64>) -> tensor<1xi64>
    %123 = stablehlo.add %122, %c : tensor<1xi64>
    %124 = stablehlo.reshape %123 : (tensor<1xi64>) -> tensor<1x1xi64>
    %125 = stablehlo.reshape %124 : (tensor<1x1xi64>) -> tensor<1x1x1xi64>
    %126 = stablehlo.reshape %125 : (tensor<1x1x1xi64>) -> tensor<1x1x1x1xi64>
    %127 = stablehlo.broadcast_in_dim %126, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xi64>) -> tensor<1x8x15x128xi64>
    %128 = stablehlo.reshape %127 : (tensor<1x8x15x128xi64>) -> tensor<1x8x15x128x1xi64>
    %129 = stablehlo.broadcast_in_dim %113, dims = [1, 2, 3] : (tensor<8x1x1xi64>) -> tensor<1x8x15x128xi64>
    %130 = stablehlo.reshape %129 : (tensor<1x8x15x128xi64>) -> tensor<1x8x15x128x1xi64>
    %131 = stablehlo.broadcast_in_dim %101, dims = [2, 3] : (tensor<15x1xi64>) -> tensor<1x8x15x128xi64>
    %132 = stablehlo.reshape %131 : (tensor<1x8x15x128xi64>) -> tensor<1x8x15x128x1xi64>
    %133 = stablehlo.broadcast_in_dim %100, dims = [3] : (tensor<128xi64>) -> tensor<1x8x15x128xi64>
    %134 = stablehlo.reshape %133 : (tensor<1x8x15x128xi64>) -> tensor<1x8x15x128x1xi64>
    %135 = stablehlo.concatenate %128, %130, %132, %134, dim = 4 : (tensor<1x8x15x128x1xi64>, tensor<1x8x15x128x1xi64>, tensor<1x8x15x128x1xi64>, tensor<1x8x15x128x1xi64>) -> tensor<1x8x15x128x4xi64>
    %136 = "stablehlo.scatter"(%arg39, %135, %89) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1, 2, 3], scatter_dims_to_operand_dims = [0, 1, 2, 3], index_vector_dim = 4>, unique_indices = false}> ({
    ^bb0(%arg46: tensor<bf16>, %arg47: tensor<bf16>):
      stablehlo.return %arg47 : tensor<bf16>
    }) : (tensor<1x8x64x128xbf16>, tensor<1x8x15x128x4xi64>, tensor<1x8x15x128xbf16>) -> tensor<1x8x64x128xbf16>
    %137 = "stablehlo.scatter"(%arg40, %135, %65) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1, 2, 3], scatter_dims_to_operand_dims = [0, 1, 2, 3], index_vector_dim = 4>, unique_indices = false}> ({
    ^bb0(%arg46: tensor<bf16>, %arg47: tensor<bf16>):
      stablehlo.return %arg47 : tensor<bf16>
    }) : (tensor<1x8x64x128xbf16>, tensor<1x8x15x128x4xi64>, tensor<1x8x15x128xbf16>) -> tensor<1x8x64x128xbf16>
    %138 = stablehlo.reshape %136 : (tensor<1x8x64x128xbf16>) -> tensor<1x8x1x64x128xbf16>
    %139 = stablehlo.broadcast_in_dim %138, dims = [0, 1, 2, 3, 4] : (tensor<1x8x1x64x128xbf16>) -> tensor<1x8x3x64x128xbf16>
    %140 = stablehlo.reshape %139 : (tensor<1x8x3x64x128xbf16>) -> tensor<1x24x64x128xbf16>
    %141 = stablehlo.reshape %137 : (tensor<1x8x64x128xbf16>) -> tensor<1x8x1x64x128xbf16>
    %142 = stablehlo.broadcast_in_dim %141, dims = [0, 1, 2, 3, 4] : (tensor<1x8x1x64x128xbf16>) -> tensor<1x8x3x64x128xbf16>
    %143 = stablehlo.reshape %142 : (tensor<1x8x3x64x128xbf16>) -> tensor<1x24x64x128xbf16>
    %144 = stablehlo.reshape %8 : (tensor<15x64xbf16>) -> tensor<1x15x64xbf16>
    %145 = stablehlo.reshape %144 : (tensor<1x15x64xbf16>) -> tensor<1x1x15x64xbf16>
    %146 = stablehlo.convert %78 : (tensor<1x24x15x128xbf16>) -> tensor<1x24x15x128xf32>
    %147 = stablehlo.convert %140 : (tensor<1x24x64x128xbf16>) -> tensor<1x24x64x128xf32>
    %148 = stablehlo.convert %143 : (tensor<1x24x64x128xbf16>) -> tensor<1x24x64x128xf32>
    %149 = stablehlo.convert %cst_15 : (tensor<1xf64>) -> tensor<1xf32>
    %150 = stablehlo.reshape %149 : (tensor<1xf32>) -> tensor<f32>
    %151 = stablehlo.broadcast_in_dim %146, dims = [0, 1, 2, 3] : (tensor<1x24x15x128xf32>) -> tensor<1x24x15x128xf32>
    %152 = stablehlo.broadcast_in_dim %150, dims = [] : (tensor<f32>) -> tensor<1x24x15x128xf32>
    %153 = stablehlo.multiply %151, %152 : tensor<1x24x15x128xf32>
    %154 = stablehlo.transpose %147, dims = [0, 1, 3, 2] : (tensor<1x24x64x128xf32>) -> tensor<1x24x128x64xf32>
    %155 = stablehlo.broadcast_in_dim %154, dims = [0, 1, 2, 3] : (tensor<1x24x128x64xf32>) -> tensor<1x24x128x64xf32>
    %156 = stablehlo.broadcast_in_dim %150, dims = [] : (tensor<f32>) -> tensor<1x24x128x64xf32>
    %157 = stablehlo.multiply %155, %156 : tensor<1x24x128x64xf32>
    %158 = stablehlo.reshape %153 : (tensor<1x24x15x128xf32>) -> tensor<24x15x128xf32>
    %159 = stablehlo.reshape %157 : (tensor<1x24x128x64xf32>) -> tensor<24x128x64xf32>
    %160 = stablehlo.broadcast_in_dim %159, dims = [0, 1, 2] : (tensor<24x128x64xf32>) -> tensor<24x128x64xf32>
    %161 = stablehlo.dot_general %158, %160, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<24x15x128xf32>, tensor<24x128x64xf32>) -> tensor<24x15x64xf32>
    %162 = stablehlo.reshape %161 : (tensor<24x15x64xf32>) -> tensor<1x24x15x64xf32>
    %163 = stablehlo.convert %145 : (tensor<1x1x15x64xbf16>) -> tensor<1x1x15x64xf32>
    %164 = stablehlo.broadcast_in_dim %162, dims = [0, 1, 2, 3] : (tensor<1x24x15x64xf32>) -> tensor<1x24x15x64xf32>
    %165 = stablehlo.broadcast_in_dim %163, dims = [0, 1, 2, 3] : (tensor<1x1x15x64xf32>) -> tensor<1x24x15x64xf32>
    %166 = stablehlo.add %164, %165 : tensor<1x24x15x64xf32>
    %167 = stablehlo.reduce(%166 init: %cst_9) applies stablehlo.maximum across dimensions = [3] : (tensor<1x24x15x64xf32>, tensor<f32>) -> tensor<1x24x15xf32>
    %168 = stablehlo.reshape %167 : (tensor<1x24x15xf32>) -> tensor<1x24x15x1xf32>
    %169 = stablehlo.broadcast_in_dim %166, dims = [0, 1, 2, 3] : (tensor<1x24x15x64xf32>) -> tensor<1x24x15x64xf32>
    %170 = stablehlo.broadcast_in_dim %168, dims = [0, 1, 2, 3] : (tensor<1x24x15x1xf32>) -> tensor<1x24x15x64xf32>
    %171 = stablehlo.subtract %169, %170 : tensor<1x24x15x64xf32>
    %172 = stablehlo.exponential %171 : tensor<1x24x15x64xf32>
    %173 = stablehlo.reduce(%172 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x24x15x64xf32>, tensor<f32>) -> tensor<1x24x15xf32>
    %174 = stablehlo.reshape %173 : (tensor<1x24x15xf32>) -> tensor<1x24x15x1xf32>
    %175 = stablehlo.broadcast_in_dim %172, dims = [0, 1, 2, 3] : (tensor<1x24x15x64xf32>) -> tensor<1x24x15x64xf32>
    %176 = stablehlo.broadcast_in_dim %174, dims = [0, 1, 2, 3] : (tensor<1x24x15x1xf32>) -> tensor<1x24x15x64xf32>
    %177 = stablehlo.divide %175, %176 : tensor<1x24x15x64xf32>
    %178 = stablehlo.convert %cst_16 : tensor<1xf64>
    %179 = stablehlo.reshape %178 : (tensor<1xf64>) -> tensor<f64>
    %180 = stablehlo.convert %179 : (tensor<f64>) -> tensor<f32>
    %181 = stablehlo.broadcast_in_dim %180, dims = [] : (tensor<f32>) -> tensor<1x24x15x64xf32>
    %182 = stablehlo.compare  EQ, %169, %181,  FLOAT : (tensor<1x24x15x64xf32>, tensor<1x24x15x64xf32>) -> tensor<1x24x15x64xi1>
    %183 = stablehlo.not %182 : tensor<1x24x15x64xi1>
    %184 = stablehlo.reduce(%183 init: %c_10) applies stablehlo.or across dimensions = [3] : (tensor<1x24x15x64xi1>, tensor<i1>) -> tensor<1x24x15xi1>
    %185 = stablehlo.reshape %184 : (tensor<1x24x15xi1>) -> tensor<1x24x15x1xi1>
    %186 = stablehlo.not %185 : tensor<1x24x15x1xi1>
    %187 = stablehlo.convert %c_8 : (tensor<i64>) -> tensor<f32>
    %188 = stablehlo.broadcast_in_dim %187, dims = [] : (tensor<f32>) -> tensor<1x24x15x64xf32>
    %189 = stablehlo.broadcast_in_dim %186, dims = [0, 1, 2, 3] : (tensor<1x24x15x1xi1>) -> tensor<1x24x15x64xi1>
    %190 = stablehlo.broadcast_in_dim %188, dims = [0, 1, 2, 3] : (tensor<1x24x15x64xf32>) -> tensor<1x24x15x64xf32>
    %191 = stablehlo.broadcast_in_dim %177, dims = [0, 1, 2, 3] : (tensor<1x24x15x64xf32>) -> tensor<1x24x15x64xf32>
    %192 = stablehlo.select %189, %190, %191 : tensor<1x24x15x64xi1>, tensor<1x24x15x64xf32>
    %193 = stablehlo.reshape %192 : (tensor<1x24x15x64xf32>) -> tensor<24x15x64xf32>
    %194 = stablehlo.reshape %148 : (tensor<1x24x64x128xf32>) -> tensor<24x64x128xf32>
    %195 = stablehlo.broadcast_in_dim %194, dims = [0, 1, 2] : (tensor<24x64x128xf32>) -> tensor<24x64x128xf32>
    %196 = stablehlo.dot_general %193, %195, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<24x15x64xf32>, tensor<24x64x128xf32>) -> tensor<24x15x128xf32>
    %197 = stablehlo.reshape %196 : (tensor<24x15x128xf32>) -> tensor<1x24x15x128xf32>
    %198 = stablehlo.convert %197 : (tensor<1x24x15x128xf32>) -> tensor<1x24x15x128xbf16>
    %199 = stablehlo.transpose %198, dims = [2, 0, 1, 3] : (tensor<1x24x15x128xbf16>) -> tensor<15x1x24x128xbf16>
    %200 = stablehlo.transpose %199, dims = [1, 2, 0, 3] : (tensor<15x1x24x128xbf16>) -> tensor<1x24x15x128xbf16>
    %201 = stablehlo.transpose %200, dims = [0, 2, 1, 3] : (tensor<1x24x15x128xbf16>) -> tensor<1x15x24x128xbf16>
    %202 = stablehlo.reshape %201 : (tensor<1x15x24x128xbf16>) -> tensor<1x15x3072xbf16>
    %203 = stablehlo.reshape %202 : (tensor<1x15x3072xbf16>) -> tensor<15x3072xbf16>
    %204 = stablehlo.dot_general %203, %arg12, contracting_dims = [1] x [0] : (tensor<15x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<15x3072xbf16>
    %205 = stablehlo.reshape %204 : (tensor<15x3072xbf16>) -> tensor<1x15x3072xbf16>
    %206 = stablehlo.add %1, %205 : tensor<1x15x3072xbf16>
    %207 = stablehlo.convert %206 : (tensor<1x15x3072xbf16>) -> tensor<1x15x3072xf32>
    %208 = stablehlo.broadcast_in_dim %207, dims = [0, 1, 2] : (tensor<1x15x3072xf32>) -> tensor<1x15x3072xf32>
    %209 = stablehlo.power %208, %32 : tensor<1x15x3072xf32>
    %210 = stablehlo.reduce(%209 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x15x3072xf32>, tensor<f32>) -> tensor<1x15xf32>
    %211 = stablehlo.reshape %210 : (tensor<1x15xf32>) -> tensor<1x15x1xf32>
    %212 = stablehlo.broadcast_in_dim %211, dims = [0, 1, 2] : (tensor<1x15x1xf32>) -> tensor<1x15x1xf32>
    %213 = stablehlo.divide %212, %39 : tensor<1x15x1xf32>
    %214 = stablehlo.broadcast_in_dim %213, dims = [0, 1, 2] : (tensor<1x15x1xf32>) -> tensor<1x15x1xf32>
    %215 = stablehlo.add %214, %44 : tensor<1x15x1xf32>
    %216 = stablehlo.rsqrt %215 : tensor<1x15x1xf32>
    %217 = stablehlo.broadcast_in_dim %216, dims = [0, 1, 2] : (tensor<1x15x1xf32>) -> tensor<1x15x3072xf32>
    %218 = stablehlo.multiply %208, %217 : tensor<1x15x3072xf32>
    %219 = stablehlo.convert %218 : (tensor<1x15x3072xf32>) -> tensor<1x15x3072xbf16>
    %220 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<3072xbf16>) -> tensor<1x15x3072xbf16>
    %221 = stablehlo.broadcast_in_dim %219, dims = [0, 1, 2] : (tensor<1x15x3072xbf16>) -> tensor<1x15x3072xbf16>
    %222 = stablehlo.multiply %220, %221 : tensor<1x15x3072xbf16>
    %223 = stablehlo.reshape %222 : (tensor<1x15x3072xbf16>) -> tensor<15x3072xbf16>
    %224 = stablehlo.dot_general %223, %arg13, contracting_dims = [1] x [0] : (tensor<15x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<15x8192xbf16>
    %225 = stablehlo.reshape %224 : (tensor<15x8192xbf16>) -> tensor<1x15x8192xbf16>
    %226 = stablehlo.convert %225 : (tensor<1x15x8192xbf16>) -> tensor<1x15x8192xf32>
    %227 = stablehlo.logistic %226 : tensor<1x15x8192xf32>
    %228 = stablehlo.multiply %226, %227 : tensor<1x15x8192xf32>
    %229 = stablehlo.convert %228 : (tensor<1x15x8192xf32>) -> tensor<1x15x8192xbf16>
    %230 = stablehlo.dot_general %223, %arg14, contracting_dims = [1] x [0] : (tensor<15x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<15x8192xbf16>
    %231 = stablehlo.reshape %230 : (tensor<15x8192xbf16>) -> tensor<1x15x8192xbf16>
    %232 = stablehlo.multiply %229, %231 : tensor<1x15x8192xbf16>
    %233 = stablehlo.reshape %232 : (tensor<1x15x8192xbf16>) -> tensor<15x8192xbf16>
    %234 = stablehlo.dot_general %233, %arg15, contracting_dims = [1] x [0] : (tensor<15x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<15x3072xbf16>
    %235 = stablehlo.reshape %234 : (tensor<15x3072xbf16>) -> tensor<1x15x3072xbf16>
    %236 = stablehlo.add %206, %235 : tensor<1x15x3072xbf16>
    %237 = stablehlo.convert %236 : (tensor<1x15x3072xbf16>) -> tensor<1x15x3072xf32>
    %238 = stablehlo.broadcast_in_dim %237, dims = [0, 1, 2] : (tensor<1x15x3072xf32>) -> tensor<1x15x3072xf32>
    %239 = stablehlo.power %238, %32 : tensor<1x15x3072xf32>
    %240 = stablehlo.reduce(%239 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x15x3072xf32>, tensor<f32>) -> tensor<1x15xf32>
    %241 = stablehlo.reshape %240 : (tensor<1x15xf32>) -> tensor<1x15x1xf32>
    %242 = stablehlo.broadcast_in_dim %241, dims = [0, 1, 2] : (tensor<1x15x1xf32>) -> tensor<1x15x1xf32>
    %243 = stablehlo.divide %242, %39 : tensor<1x15x1xf32>
    %244 = stablehlo.broadcast_in_dim %243, dims = [0, 1, 2] : (tensor<1x15x1xf32>) -> tensor<1x15x1xf32>
    %245 = stablehlo.add %244, %44 : tensor<1x15x1xf32>
    %246 = stablehlo.rsqrt %245 : tensor<1x15x1xf32>
    %247 = stablehlo.broadcast_in_dim %246, dims = [0, 1, 2] : (tensor<1x15x1xf32>) -> tensor<1x15x3072xf32>
    %248 = stablehlo.multiply %238, %247 : tensor<1x15x3072xf32>
    %249 = stablehlo.convert %248 : (tensor<1x15x3072xf32>) -> tensor<1x15x3072xbf16>
    %250 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<3072xbf16>) -> tensor<1x15x3072xbf16>
    %251 = stablehlo.broadcast_in_dim %249, dims = [0, 1, 2] : (tensor<1x15x3072xbf16>) -> tensor<1x15x3072xbf16>
    %252 = stablehlo.multiply %250, %251 : tensor<1x15x3072xbf16>
    %253 = stablehlo.reshape %252 : (tensor<1x15x3072xbf16>) -> tensor<15x3072xbf16>
    %254 = stablehlo.dot_general %253, %arg16, contracting_dims = [1] x [0] : (tensor<15x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<15x3072xbf16>
    %255 = stablehlo.reshape %254 : (tensor<15x3072xbf16>) -> tensor<1x15x3072xbf16>
    %256 = stablehlo.dot_general %253, %arg17, contracting_dims = [1] x [0] : (tensor<15x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<15x1024xbf16>
    %257 = stablehlo.reshape %256 : (tensor<15x1024xbf16>) -> tensor<1x15x1024xbf16>
    %258 = stablehlo.dot_general %253, %arg18, contracting_dims = [1] x [0] : (tensor<15x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<15x1024xbf16>
    %259 = stablehlo.reshape %258 : (tensor<15x1024xbf16>) -> tensor<1x15x1024xbf16>
    %260 = stablehlo.reshape %255 : (tensor<1x15x3072xbf16>) -> tensor<1x15x24x128xbf16>
    %261 = stablehlo.transpose %260, dims = [0, 2, 1, 3] : (tensor<1x15x24x128xbf16>) -> tensor<1x24x15x128xbf16>
    %262 = stablehlo.reshape %257 : (tensor<1x15x1024xbf16>) -> tensor<1x15x8x128xbf16>
    %263 = stablehlo.transpose %262, dims = [0, 2, 1, 3] : (tensor<1x15x8x128xbf16>) -> tensor<1x8x15x128xbf16>
    %264 = stablehlo.reshape %259 : (tensor<1x15x1024xbf16>) -> tensor<1x15x8x128xbf16>
    %265 = stablehlo.transpose %264, dims = [0, 2, 1, 3] : (tensor<1x15x8x128xbf16>) -> tensor<1x8x15x128xbf16>
    %266 = stablehlo.broadcast_in_dim %261, dims = [0, 1, 2, 3] : (tensor<1x24x15x128xbf16>) -> tensor<1x24x15x128xbf16>
    %267 = stablehlo.multiply %266, %69 : tensor<1x24x15x128xbf16>
    %268 = stablehlo.slice %261 [0:1, 0:24, 0:15, 0:64] : (tensor<1x24x15x128xbf16>) -> tensor<1x24x15x64xbf16>
    %269 = stablehlo.slice %261 [0:1, 0:24, 0:15, 64:128] : (tensor<1x24x15x128xbf16>) -> tensor<1x24x15x64xbf16>
    %270 = stablehlo.negate %269 : tensor<1x24x15x64xbf16>
    %271 = stablehlo.concatenate %270, %268, dim = 3 : (tensor<1x24x15x64xbf16>, tensor<1x24x15x64xbf16>) -> tensor<1x24x15x128xbf16>
    %272 = stablehlo.broadcast_in_dim %271, dims = [0, 1, 2, 3] : (tensor<1x24x15x128xbf16>) -> tensor<1x24x15x128xbf16>
    %273 = stablehlo.multiply %272, %76 : tensor<1x24x15x128xbf16>
    %274 = stablehlo.add %267, %273 : tensor<1x24x15x128xbf16>
    %275 = stablehlo.broadcast_in_dim %263, dims = [0, 1, 2, 3] : (tensor<1x8x15x128xbf16>) -> tensor<1x8x15x128xbf16>
    %276 = stablehlo.multiply %275, %80 : tensor<1x8x15x128xbf16>
    %277 = stablehlo.slice %263 [0:1, 0:8, 0:15, 0:64] : (tensor<1x8x15x128xbf16>) -> tensor<1x8x15x64xbf16>
    %278 = stablehlo.slice %263 [0:1, 0:8, 0:15, 64:128] : (tensor<1x8x15x128xbf16>) -> tensor<1x8x15x64xbf16>
    %279 = stablehlo.negate %278 : tensor<1x8x15x64xbf16>
    %280 = stablehlo.concatenate %279, %277, dim = 3 : (tensor<1x8x15x64xbf16>, tensor<1x8x15x64xbf16>) -> tensor<1x8x15x128xbf16>
    %281 = stablehlo.broadcast_in_dim %280, dims = [0, 1, 2, 3] : (tensor<1x8x15x128xbf16>) -> tensor<1x8x15x128xbf16>
    %282 = stablehlo.multiply %281, %87 : tensor<1x8x15x128xbf16>
    %283 = stablehlo.add %276, %282 : tensor<1x8x15x128xbf16>
    %284 = "stablehlo.scatter"(%arg41, %135, %283) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1, 2, 3], scatter_dims_to_operand_dims = [0, 1, 2, 3], index_vector_dim = 4>, unique_indices = false}> ({
    ^bb0(%arg46: tensor<bf16>, %arg47: tensor<bf16>):
      stablehlo.return %arg47 : tensor<bf16>
    }) : (tensor<1x8x64x128xbf16>, tensor<1x8x15x128x4xi64>, tensor<1x8x15x128xbf16>) -> tensor<1x8x64x128xbf16>
    %285 = "stablehlo.scatter"(%arg42, %135, %265) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1, 2, 3], scatter_dims_to_operand_dims = [0, 1, 2, 3], index_vector_dim = 4>, unique_indices = false}> ({
    ^bb0(%arg46: tensor<bf16>, %arg47: tensor<bf16>):
      stablehlo.return %arg47 : tensor<bf16>
    }) : (tensor<1x8x64x128xbf16>, tensor<1x8x15x128x4xi64>, tensor<1x8x15x128xbf16>) -> tensor<1x8x64x128xbf16>
    %286 = stablehlo.reshape %284 : (tensor<1x8x64x128xbf16>) -> tensor<1x8x1x64x128xbf16>
    %287 = stablehlo.broadcast_in_dim %286, dims = [0, 1, 2, 3, 4] : (tensor<1x8x1x64x128xbf16>) -> tensor<1x8x3x64x128xbf16>
    %288 = stablehlo.reshape %287 : (tensor<1x8x3x64x128xbf16>) -> tensor<1x24x64x128xbf16>
    %289 = stablehlo.reshape %285 : (tensor<1x8x64x128xbf16>) -> tensor<1x8x1x64x128xbf16>
    %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1, 2, 3, 4] : (tensor<1x8x1x64x128xbf16>) -> tensor<1x8x3x64x128xbf16>
    %291 = stablehlo.reshape %290 : (tensor<1x8x3x64x128xbf16>) -> tensor<1x24x64x128xbf16>
    %292 = stablehlo.convert %274 : (tensor<1x24x15x128xbf16>) -> tensor<1x24x15x128xf32>
    %293 = stablehlo.convert %288 : (tensor<1x24x64x128xbf16>) -> tensor<1x24x64x128xf32>
    %294 = stablehlo.convert %291 : (tensor<1x24x64x128xbf16>) -> tensor<1x24x64x128xf32>
    %295 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2, 3] : (tensor<1x24x15x128xf32>) -> tensor<1x24x15x128xf32>
    %296 = stablehlo.multiply %295, %152 : tensor<1x24x15x128xf32>
    %297 = stablehlo.transpose %293, dims = [0, 1, 3, 2] : (tensor<1x24x64x128xf32>) -> tensor<1x24x128x64xf32>
    %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2, 3] : (tensor<1x24x128x64xf32>) -> tensor<1x24x128x64xf32>
    %299 = stablehlo.multiply %298, %156 : tensor<1x24x128x64xf32>
    %300 = stablehlo.reshape %296 : (tensor<1x24x15x128xf32>) -> tensor<24x15x128xf32>
    %301 = stablehlo.reshape %299 : (tensor<1x24x128x64xf32>) -> tensor<24x128x64xf32>
    %302 = stablehlo.broadcast_in_dim %301, dims = [0, 1, 2] : (tensor<24x128x64xf32>) -> tensor<24x128x64xf32>
    %303 = stablehlo.dot_general %300, %302, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<24x15x128xf32>, tensor<24x128x64xf32>) -> tensor<24x15x64xf32>
    %304 = stablehlo.reshape %303 : (tensor<24x15x64xf32>) -> tensor<1x24x15x64xf32>
    %305 = stablehlo.broadcast_in_dim %304, dims = [0, 1, 2, 3] : (tensor<1x24x15x64xf32>) -> tensor<1x24x15x64xf32>
    %306 = stablehlo.add %305, %165 : tensor<1x24x15x64xf32>
    %307 = stablehlo.reduce(%306 init: %cst_9) applies stablehlo.maximum across dimensions = [3] : (tensor<1x24x15x64xf32>, tensor<f32>) -> tensor<1x24x15xf32>
    %308 = stablehlo.reshape %307 : (tensor<1x24x15xf32>) -> tensor<1x24x15x1xf32>
    %309 = stablehlo.broadcast_in_dim %306, dims = [0, 1, 2, 3] : (tensor<1x24x15x64xf32>) -> tensor<1x24x15x64xf32>
    %310 = stablehlo.broadcast_in_dim %308, dims = [0, 1, 2, 3] : (tensor<1x24x15x1xf32>) -> tensor<1x24x15x64xf32>
    %311 = stablehlo.subtract %309, %310 : tensor<1x24x15x64xf32>
    %312 = stablehlo.exponential %311 : tensor<1x24x15x64xf32>
    %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x24x15x64xf32>, tensor<f32>) -> tensor<1x24x15xf32>
    %314 = stablehlo.reshape %313 : (tensor<1x24x15xf32>) -> tensor<1x24x15x1xf32>
    %315 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2, 3] : (tensor<1x24x15x64xf32>) -> tensor<1x24x15x64xf32>
    %316 = stablehlo.broadcast_in_dim %314, dims = [0, 1, 2, 3] : (tensor<1x24x15x1xf32>) -> tensor<1x24x15x64xf32>
    %317 = stablehlo.divide %315, %316 : tensor<1x24x15x64xf32>
    %318 = stablehlo.compare  EQ, %309, %181,  FLOAT : (tensor<1x24x15x64xf32>, tensor<1x24x15x64xf32>) -> tensor<1x24x15x64xi1>
    %319 = stablehlo.not %318 : tensor<1x24x15x64xi1>
    %320 = stablehlo.reduce(%319 init: %c_10) applies stablehlo.or across dimensions = [3] : (tensor<1x24x15x64xi1>, tensor<i1>) -> tensor<1x24x15xi1>
    %321 = stablehlo.reshape %320 : (tensor<1x24x15xi1>) -> tensor<1x24x15x1xi1>
    %322 = stablehlo.not %321 : tensor<1x24x15x1xi1>
    %323 = stablehlo.broadcast_in_dim %322, dims = [0, 1, 2, 3] : (tensor<1x24x15x1xi1>) -> tensor<1x24x15x64xi1>
    %324 = stablehlo.broadcast_in_dim %317, dims = [0, 1, 2, 3] : (tensor<1x24x15x64xf32>) -> tensor<1x24x15x64xf32>
    %325 = stablehlo.select %323, %190, %324 : tensor<1x24x15x64xi1>, tensor<1x24x15x64xf32>
    %326 = stablehlo.reshape %325 : (tensor<1x24x15x64xf32>) -> tensor<24x15x64xf32>
    %327 = stablehlo.reshape %294 : (tensor<1x24x64x128xf32>) -> tensor<24x64x128xf32>
    %328 = stablehlo.broadcast_in_dim %327, dims = [0, 1, 2] : (tensor<24x64x128xf32>) -> tensor<24x64x128xf32>
    %329 = stablehlo.dot_general %326, %328, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<24x15x64xf32>, tensor<24x64x128xf32>) -> tensor<24x15x128xf32>
    %330 = stablehlo.reshape %329 : (tensor<24x15x128xf32>) -> tensor<1x24x15x128xf32>
    %331 = stablehlo.convert %330 : (tensor<1x24x15x128xf32>) -> tensor<1x24x15x128xbf16>
    %332 = stablehlo.transpose %331, dims = [2, 0, 1, 3] : (tensor<1x24x15x128xbf16>) -> tensor<15x1x24x128xbf16>
    %333 = stablehlo.transpose %332, dims = [1, 2, 0, 3] : (tensor<15x1x24x128xbf16>) -> tensor<1x24x15x128xbf16>
    %334 = stablehlo.transpose %333, dims = [0, 2, 1, 3] : (tensor<1x24x15x128xbf16>) -> tensor<1x15x24x128xbf16>
    %335 = stablehlo.reshape %334 : (tensor<1x15x24x128xbf16>) -> tensor<1x15x3072xbf16>
    %336 = stablehlo.reshape %335 : (tensor<1x15x3072xbf16>) -> tensor<15x3072xbf16>
    %337 = stablehlo.dot_general %336, %arg19, contracting_dims = [1] x [0] : (tensor<15x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<15x3072xbf16>
    %338 = stablehlo.reshape %337 : (tensor<15x3072xbf16>) -> tensor<1x15x3072xbf16>
    %339 = stablehlo.add %236, %338 : tensor<1x15x3072xbf16>
    %340 = stablehlo.convert %339 : (tensor<1x15x3072xbf16>) -> tensor<1x15x3072xf32>
    %341 = stablehlo.broadcast_in_dim %340, dims = [0, 1, 2] : (tensor<1x15x3072xf32>) -> tensor<1x15x3072xf32>
    %342 = stablehlo.power %341, %32 : tensor<1x15x3072xf32>
    %343 = stablehlo.reduce(%342 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x15x3072xf32>, tensor<f32>) -> tensor<1x15xf32>
    %344 = stablehlo.reshape %343 : (tensor<1x15xf32>) -> tensor<1x15x1xf32>
    %345 = stablehlo.broadcast_in_dim %344, dims = [0, 1, 2] : (tensor<1x15x1xf32>) -> tensor<1x15x1xf32>
    %346 = stablehlo.divide %345, %39 : tensor<1x15x1xf32>
    %347 = stablehlo.broadcast_in_dim %346, dims = [0, 1, 2] : (tensor<1x15x1xf32>) -> tensor<1x15x1xf32>
    %348 = stablehlo.add %347, %44 : tensor<1x15x1xf32>
    %349 = stablehlo.rsqrt %348 : tensor<1x15x1xf32>
    %350 = stablehlo.broadcast_in_dim %349, dims = [0, 1, 2] : (tensor<1x15x1xf32>) -> tensor<1x15x3072xf32>
    %351 = stablehlo.multiply %341, %350 : tensor<1x15x3072xf32>
    %352 = stablehlo.convert %351 : (tensor<1x15x3072xf32>) -> tensor<1x15x3072xbf16>
    %353 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<3072xbf16>) -> tensor<1x15x3072xbf16>
    %354 = stablehlo.broadcast_in_dim %352, dims = [0, 1, 2] : (tensor<1x15x3072xbf16>) -> tensor<1x15x3072xbf16>
    %355 = stablehlo.multiply %353, %354 : tensor<1x15x3072xbf16>
    %356 = stablehlo.reshape %355 : (tensor<1x15x3072xbf16>) -> tensor<15x3072xbf16>
    %357 = stablehlo.dot_general %356, %arg20, contracting_dims = [1] x [0] : (tensor<15x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<15x8192xbf16>
    %358 = stablehlo.reshape %357 : (tensor<15x8192xbf16>) -> tensor<1x15x8192xbf16>
    %359 = stablehlo.convert %358 : (tensor<1x15x8192xbf16>) -> tensor<1x15x8192xf32>
    %360 = stablehlo.logistic %359 : tensor<1x15x8192xf32>
    %361 = stablehlo.multiply %359, %360 : tensor<1x15x8192xf32>
    %362 = stablehlo.convert %361 : (tensor<1x15x8192xf32>) -> tensor<1x15x8192xbf16>
    %363 = stablehlo.dot_general %356, %arg21, contracting_dims = [1] x [0] : (tensor<15x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<15x8192xbf16>
    %364 = stablehlo.reshape %363 : (tensor<15x8192xbf16>) -> tensor<1x15x8192xbf16>
    %365 = stablehlo.multiply %362, %364 : tensor<1x15x8192xbf16>
    %366 = stablehlo.reshape %365 : (tensor<1x15x8192xbf16>) -> tensor<15x8192xbf16>
    %367 = stablehlo.dot_general %366, %arg22, contracting_dims = [1] x [0] : (tensor<15x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<15x3072xbf16>
    %368 = stablehlo.reshape %367 : (tensor<15x3072xbf16>) -> tensor<1x15x3072xbf16>
    %369 = stablehlo.add %339, %368 : tensor<1x15x3072xbf16>
    %370 = stablehlo.convert %369 : (tensor<1x15x3072xbf16>) -> tensor<1x15x3072xf32>
    %371 = stablehlo.broadcast_in_dim %370, dims = [0, 1, 2] : (tensor<1x15x3072xf32>) -> tensor<1x15x3072xf32>
    %372 = stablehlo.power %371, %32 : tensor<1x15x3072xf32>
    %373 = stablehlo.reduce(%372 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x15x3072xf32>, tensor<f32>) -> tensor<1x15xf32>
    %374 = stablehlo.reshape %373 : (tensor<1x15xf32>) -> tensor<1x15x1xf32>
    %375 = stablehlo.broadcast_in_dim %374, dims = [0, 1, 2] : (tensor<1x15x1xf32>) -> tensor<1x15x1xf32>
    %376 = stablehlo.divide %375, %39 : tensor<1x15x1xf32>
    %377 = stablehlo.broadcast_in_dim %376, dims = [0, 1, 2] : (tensor<1x15x1xf32>) -> tensor<1x15x1xf32>
    %378 = stablehlo.add %377, %44 : tensor<1x15x1xf32>
    %379 = stablehlo.rsqrt %378 : tensor<1x15x1xf32>
    %380 = stablehlo.broadcast_in_dim %379, dims = [0, 1, 2] : (tensor<1x15x1xf32>) -> tensor<1x15x3072xf32>
    %381 = stablehlo.multiply %371, %380 : tensor<1x15x3072xf32>
    %382 = stablehlo.convert %381 : (tensor<1x15x3072xf32>) -> tensor<1x15x3072xbf16>
    %383 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<3072xbf16>) -> tensor<1x15x3072xbf16>
    %384 = stablehlo.broadcast_in_dim %382, dims = [0, 1, 2] : (tensor<1x15x3072xbf16>) -> tensor<1x15x3072xbf16>
    %385 = stablehlo.multiply %383, %384 : tensor<1x15x3072xbf16>
    %386 = stablehlo.reshape %385 : (tensor<1x15x3072xbf16>) -> tensor<15x3072xbf16>
    %387 = stablehlo.dot_general %386, %arg23, contracting_dims = [1] x [0] : (tensor<15x3072xbf16>, tensor<3072x128256xbf16>) -> tensor<15x128256xbf16>
    %388 = stablehlo.reshape %387 : (tensor<15x128256xbf16>) -> tensor<1x15x128256xbf16>
    return %388, %136, %137, %284, %285 : tensor<1x15x128256xbf16>, tensor<1x8x64x128xbf16>, tensor<1x8x64x128xbf16>, tensor<1x8x64x128xbf16>, tensor<1x8x64x128xbf16>
  }
}