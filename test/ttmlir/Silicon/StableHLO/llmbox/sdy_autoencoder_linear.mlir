// REQUIRES: stablehlo
// RUN: ttmlir-opt --automatic-sharding-pipeline="mesh-shape=1,8" --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,8" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  sdy.mesh @mesh = <["model"=1, "batch"=8]>

  func.func @main(%arg0: tensor<784x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128x64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64x12xf32>, %arg5: tensor<12xf32>, %arg6: tensor<12x3xf32>, %arg7: tensor<3xf32>, %arg8: tensor<3x12xf32>, %arg9: tensor<12xf32>, %arg10: tensor<12x64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<64x128xf32>, %arg13: tensor<128xf32>, %arg14: tensor<128x784xf32>, %arg15: tensor<784xf32>, %arg16: tensor<128x784xbf16>, %arg17: tensor<128xbf16>, %arg18: tensor<64x128xbf16>, %arg19: tensor<64xbf16>, %arg20: tensor<12x64xbf16>, %arg21: tensor<12xbf16>, %arg22: tensor<3x12xbf16>, %arg23: tensor<3xbf16>, %arg24: tensor<12x3xbf16>, %arg25: tensor<12xbf16>, %arg26: tensor<64x12xbf16>, %arg27: tensor<64xbf16>, %arg28: tensor<128x64xbf16>, %arg29: tensor<128xbf16>, %arg30: tensor<784x128xbf16>, %arg31: tensor<784xbf16>, %arg32: tensor<32x1x1x784xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}) -> tensor<32x1x1x784xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<32x1x1x128xbf16>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<32x1x1x64xbf16>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<32x1x1x12xbf16>
    %cst_2 = arith.constant dense<1> : tensor<1xi64>
    %0 = stablehlo.reshape %arg32 : (tensor<32x1x1x784xbf16>) -> tensor<32x784xbf16>
    %1 = stablehlo.convert %0 : (tensor<32x784xbf16>) -> tensor<32x784xf32>
    %2 = stablehlo.dot_general %1, %arg0, contracting_dims = [1] x [0] : (tensor<32x784xf32>, tensor<784x128xf32>) -> tensor<32x128xf32>
    %3 = stablehlo.convert %cst_2 : (tensor<1xi64>) -> tensor<1xf32>
    %4 = stablehlo.reshape %3 : (tensor<1xf32>) -> tensor<f32>
    %5 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %6 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<32x128xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<32x128xf32>
    %10 = stablehlo.add %8, %9 : tensor<32x128xf32>
    %11 = stablehlo.convert %10 : (tensor<32x128xf32>) -> tensor<32x128xbf16>
    %12 = stablehlo.reshape %11 : (tensor<32x128xbf16>) -> tensor<32x1x1x128xbf16>
    %13 = stablehlo.maximum %12, %cst : tensor<32x1x1x128xbf16>
    %14 = stablehlo.reshape %13 : (tensor<32x1x1x128xbf16>) -> tensor<32x128xbf16>
    %15 = stablehlo.convert %14 : (tensor<32x128xbf16>) -> tensor<32x128xf32>
    %16 = stablehlo.dot_general %15, %arg2, contracting_dims = [1] x [0] : (tensor<32x128xf32>, tensor<128x64xf32>) -> tensor<32x64xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64xf32>
    %18 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<32x64xf32>
    %19 = stablehlo.multiply %17, %18 : tensor<32x64xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64xf32>
    %21 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<32x64xf32>
    %22 = stablehlo.add %20, %21 : tensor<32x64xf32>
    %23 = stablehlo.convert %22 : (tensor<32x64xf32>) -> tensor<32x64xbf16>
    %24 = stablehlo.reshape %23 : (tensor<32x64xbf16>) -> tensor<32x1x1x64xbf16>
    %25 = stablehlo.maximum %24, %cst_0 : tensor<32x1x1x64xbf16>
    %26 = stablehlo.reshape %25 : (tensor<32x1x1x64xbf16>) -> tensor<32x64xbf16>
    %27 = stablehlo.convert %26 : (tensor<32x64xbf16>) -> tensor<32x64xf32>
    %28 = stablehlo.dot_general %27, %arg4, contracting_dims = [1] x [0] : (tensor<32x64xf32>, tensor<64x12xf32>) -> tensor<32x12xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1] : (tensor<32x12xf32>) -> tensor<32x12xf32>
    %30 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<32x12xf32>
    %31 = stablehlo.multiply %29, %30 : tensor<32x12xf32>
    %32 = stablehlo.broadcast_in_dim %31, dims = [0, 1] : (tensor<32x12xf32>) -> tensor<32x12xf32>
    %33 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<12xf32>) -> tensor<32x12xf32>
    %34 = stablehlo.add %32, %33 : tensor<32x12xf32>
    %35 = stablehlo.convert %34 : (tensor<32x12xf32>) -> tensor<32x12xbf16>
    %36 = stablehlo.reshape %35 : (tensor<32x12xbf16>) -> tensor<32x1x1x12xbf16>
    %37 = stablehlo.maximum %36, %cst_1 : tensor<32x1x1x12xbf16>
    %38 = stablehlo.reshape %37 : (tensor<32x1x1x12xbf16>) -> tensor<32x12xbf16>
    %39 = stablehlo.convert %38 : (tensor<32x12xbf16>) -> tensor<32x12xf32>
    %40 = stablehlo.dot_general %39, %arg6, contracting_dims = [1] x [0] : (tensor<32x12xf32>, tensor<12x3xf32>) -> tensor<32x3xf32>
    %41 = stablehlo.broadcast_in_dim %40, dims = [0, 1] : (tensor<32x3xf32>) -> tensor<32x3xf32>
    %42 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<32x3xf32>
    %43 = stablehlo.multiply %41, %42 : tensor<32x3xf32>
    %44 = stablehlo.broadcast_in_dim %43, dims = [0, 1] : (tensor<32x3xf32>) -> tensor<32x3xf32>
    %45 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<3xf32>) -> tensor<32x3xf32>
    %46 = stablehlo.add %44, %45 : tensor<32x3xf32>
    %47 = stablehlo.convert %46 : (tensor<32x3xf32>) -> tensor<32x3xbf16>
    %48 = stablehlo.reshape %47 : (tensor<32x3xbf16>) -> tensor<32x1x1x3xbf16>
    %49 = stablehlo.reshape %48 : (tensor<32x1x1x3xbf16>) -> tensor<32x3xbf16>
    %50 = stablehlo.convert %49 : (tensor<32x3xbf16>) -> tensor<32x3xf32>
    %51 = stablehlo.dot_general %50, %arg8, contracting_dims = [1] x [0] : (tensor<32x3xf32>, tensor<3x12xf32>) -> tensor<32x12xf32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [0, 1] : (tensor<32x12xf32>) -> tensor<32x12xf32>
    %53 = stablehlo.multiply %52, %30 : tensor<32x12xf32>
    %54 = stablehlo.broadcast_in_dim %53, dims = [0, 1] : (tensor<32x12xf32>) -> tensor<32x12xf32>
    %55 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<12xf32>) -> tensor<32x12xf32>
    %56 = stablehlo.add %54, %55 : tensor<32x12xf32>
    %57 = stablehlo.convert %56 : (tensor<32x12xf32>) -> tensor<32x12xbf16>
    %58 = stablehlo.reshape %57 : (tensor<32x12xbf16>) -> tensor<32x1x1x12xbf16>
    %59 = stablehlo.maximum %58, %cst_1 : tensor<32x1x1x12xbf16>
    %60 = stablehlo.reshape %59 : (tensor<32x1x1x12xbf16>) -> tensor<32x12xbf16>
    %61 = stablehlo.convert %60 : (tensor<32x12xbf16>) -> tensor<32x12xf32>
    %62 = stablehlo.dot_general %61, %arg10, contracting_dims = [1] x [0] : (tensor<32x12xf32>, tensor<12x64xf32>) -> tensor<32x64xf32>
    %63 = stablehlo.broadcast_in_dim %62, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64xf32>
    %64 = stablehlo.multiply %63, %18 : tensor<32x64xf32>
    %65 = stablehlo.broadcast_in_dim %64, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64xf32>
    %66 = stablehlo.broadcast_in_dim %arg11, dims = [1] : (tensor<64xf32>) -> tensor<32x64xf32>
    %67 = stablehlo.add %65, %66 : tensor<32x64xf32>
    %68 = stablehlo.convert %67 : (tensor<32x64xf32>) -> tensor<32x64xbf16>
    %69 = stablehlo.reshape %68 : (tensor<32x64xbf16>) -> tensor<32x1x1x64xbf16>
    %70 = stablehlo.maximum %69, %cst_0 : tensor<32x1x1x64xbf16>
    %71 = stablehlo.reshape %70 : (tensor<32x1x1x64xbf16>) -> tensor<32x64xbf16>
    %72 = stablehlo.convert %71 : (tensor<32x64xbf16>) -> tensor<32x64xf32>
    %73 = stablehlo.dot_general %72, %arg12, contracting_dims = [1] x [0] : (tensor<32x64xf32>, tensor<64x128xf32>) -> tensor<32x128xf32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %75 = stablehlo.multiply %74, %6 : tensor<32x128xf32>
    %76 = stablehlo.broadcast_in_dim %75, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %77 = stablehlo.broadcast_in_dim %arg13, dims = [1] : (tensor<128xf32>) -> tensor<32x128xf32>
    %78 = stablehlo.add %76, %77 : tensor<32x128xf32>
    %79 = stablehlo.convert %78 : (tensor<32x128xf32>) -> tensor<32x128xbf16>
    %80 = stablehlo.reshape %79 : (tensor<32x128xbf16>) -> tensor<32x1x1x128xbf16>
    %81 = stablehlo.maximum %80, %cst : tensor<32x1x1x128xbf16>
    %82 = stablehlo.reshape %81 : (tensor<32x1x1x128xbf16>) -> tensor<32x128xbf16>
    %83 = stablehlo.convert %82 : (tensor<32x128xbf16>) -> tensor<32x128xf32>
    %84 = stablehlo.dot_general %83, %arg14, contracting_dims = [1] x [0] : (tensor<32x128xf32>, tensor<128x784xf32>) -> tensor<32x784xf32>
    %85 = stablehlo.broadcast_in_dim %84, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784xf32>
    %86 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<32x784xf32>
    %87 = stablehlo.multiply %85, %86 : tensor<32x784xf32>
    %88 = stablehlo.broadcast_in_dim %87, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784xf32>
    %89 = stablehlo.broadcast_in_dim %arg15, dims = [1] : (tensor<784xf32>) -> tensor<32x784xf32>
    %90 = stablehlo.add %88, %89 : tensor<32x784xf32>
    %91 = stablehlo.convert %90 : (tensor<32x784xf32>) -> tensor<32x784xbf16>
    %92 = stablehlo.reshape %91 : (tensor<32x784xbf16>) -> tensor<32x1x1x784xbf16>
    return %92 : tensor<32x1x1x784xbf16>
  }
}
