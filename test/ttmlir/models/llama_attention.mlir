func.func @test_llama_attention(%arg0: tensor<1x12x3200xf32>, %arg1: tensor<1x1x12x12xf32>, %arg2: tensor<1x12xf32>, %arg3: tensor<1x50x1xf32>, %arg4: tensor<1x32x50x100xf32>, %arg5: tensor<1x1xf32>, %arg6: tensor<1x32x50x100xf32>, %arg7: tensor<1x32x50x100xf32>, %arg8: tensor<1x1xf32>, %arg9: tensor<1x32x50x100xf32>, %arg10: tensor<1x1xf32>, %arg11: tensor<3200x3200xf32>, %arg12: tensor<3200x3200xf32>, %arg13: tensor<3200x3200xf32>, %arg14: tensor<3200x3200xf32>) -> tensor<1x12x3200xf32> {
  %0 = "ttir.squeeze"(%arg0) <{dim = 0 : si32}> : (tensor<1x12x3200xf32>) -> tensor<12x3200xf32>
  %1 = "ttir.matmul"(%0, %arg11) : (tensor<12x3200xf32>, tensor<3200x3200xf32>) -> tensor<12x3200xf32>
  %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 12 : i32, 32 : i32, 100 : i32]}> : (tensor<12x3200xf32>) -> tensor<1x12x32x100xf32>
  %3 = "ttir.transpose"(%2) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x12x32x100xf32>) -> tensor<1x32x12x100xf32>
  %4 = "ttir.unsqueeze"(%arg2) <{dim = 1 : si32}> : (tensor<1x12xf32>) -> tensor<1x1x12xf32>
  %5 = "ttir.matmul"(%arg3, %4) : (tensor<1x50x1xf32>, tensor<1x1x12xf32>) -> tensor<1x50x12xf32>
  %6 = "ttir.transpose"(%5) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x50x12xf32>) -> tensor<1x12x50xf32>
  %7 = "ttir.concat"(%6, %6) <{dim = -1 : si32}> : (tensor<1x12x50xf32>, tensor<1x12x50xf32>) -> tensor<1x12x100xf32>
  %8 = "ttir.cos"(%7) : (tensor<1x12x100xf32>) -> tensor<1x12x100xf32>
  %9 = "ttir.unsqueeze"(%8) <{dim = 1 : si32}> : (tensor<1x12x100xf32>) -> tensor<1x1x12x100xf32>
  %10 = "ttir.multiply"(%3, %9) : (tensor<1x32x12x100xf32>, tensor<1x1x12x100xf32>) -> tensor<1x32x12x100xf32>
  %11 = "ttir.transpose"(%3) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x12x100xf32>) -> tensor<1x32x100x12xf32>
  %12 = "ttir.matmul"(%arg4, %11) : (tensor<1x32x50x100xf32>, tensor<1x32x100x12xf32>) -> tensor<1x32x50x12xf32>
  %13 = "ttir.transpose"(%12) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x50x12xf32>) -> tensor<1x32x12x50xf32>
  %14 = "ttir.multiply"(%13, %arg5) : (tensor<1x32x12x50xf32>, tensor<1x1xf32>) -> tensor<1x32x12x50xf32>
  %15 = "ttir.transpose"(%3) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x12x100xf32>) -> tensor<1x32x100x12xf32>
  %16 = "ttir.matmul"(%arg6, %15) : (tensor<1x32x50x100xf32>, tensor<1x32x100x12xf32>) -> tensor<1x32x50x12xf32>
  %17 = "ttir.transpose"(%16) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x50x12xf32>) -> tensor<1x32x12x50xf32>
  %18 = "ttir.concat"(%14, %17) <{dim = -1 : si32}> : (tensor<1x32x12x50xf32>, tensor<1x32x12x50xf32>) -> tensor<1x32x12x100xf32>
  %19 = "ttir.sin"(%7) : (tensor<1x12x100xf32>) -> tensor<1x12x100xf32>
  %20 = "ttir.unsqueeze"(%19) <{dim = 1 : si32}> : (tensor<1x12x100xf32>) -> tensor<1x1x12x100xf32>
  %21 = "ttir.multiply"(%18, %20) : (tensor<1x32x12x100xf32>, tensor<1x1x12x100xf32>) -> tensor<1x32x12x100xf32>
  %22 = "ttir.add"(%10, %21) : (tensor<1x32x12x100xf32>, tensor<1x32x12x100xf32>) -> tensor<1x32x12x100xf32>
  %23 = "ttir.squeeze"(%22) <{dim = 0 : si32}> : (tensor<1x32x12x100xf32>) -> tensor<32x12x100xf32>
  %24 = "ttir.matmul"(%0, %arg12) : (tensor<12x3200xf32>, tensor<3200x3200xf32>) -> tensor<12x3200xf32>
  %25 = "ttir.reshape"(%24) <{shape = [1 : i32, 12 : i32, 32 : i32, 100 : i32]}> : (tensor<12x3200xf32>) -> tensor<1x12x32x100xf32>
  %26 = "ttir.transpose"(%25) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x12x32x100xf32>) -> tensor<1x32x12x100xf32>
  %27 = "ttir.multiply"(%26, %9) : (tensor<1x32x12x100xf32>, tensor<1x1x12x100xf32>) -> tensor<1x32x12x100xf32>
  %28 = "ttir.transpose"(%26) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x12x100xf32>) -> tensor<1x32x100x12xf32>
  %29 = "ttir.matmul"(%arg7, %28) : (tensor<1x32x50x100xf32>, tensor<1x32x100x12xf32>) -> tensor<1x32x50x12xf32>
  %30 = "ttir.transpose"(%29) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x50x12xf32>) -> tensor<1x32x12x50xf32>
  %31 = "ttir.multiply"(%30, %arg8) : (tensor<1x32x12x50xf32>, tensor<1x1xf32>) -> tensor<1x32x12x50xf32>
  %32 = "ttir.transpose"(%26) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x12x100xf32>) -> tensor<1x32x100x12xf32>
  %33 = "ttir.matmul"(%arg9, %32) : (tensor<1x32x50x100xf32>, tensor<1x32x100x12xf32>) -> tensor<1x32x50x12xf32>
  %34 = "ttir.transpose"(%33) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x50x12xf32>) -> tensor<1x32x12x50xf32>
  %35 = "ttir.concat"(%31, %34) <{dim = -1 : si32}> : (tensor<1x32x12x50xf32>, tensor<1x32x12x50xf32>) -> tensor<1x32x12x100xf32>
  %36 = "ttir.multiply"(%35, %20) : (tensor<1x32x12x100xf32>, tensor<1x1x12x100xf32>) -> tensor<1x32x12x100xf32>
  %37 = "ttir.add"(%27, %36) : (tensor<1x32x12x100xf32>, tensor<1x32x12x100xf32>) -> tensor<1x32x12x100xf32>
  %38 = "ttir.squeeze"(%37) <{dim = 0 : si32}> : (tensor<1x32x12x100xf32>) -> tensor<32x12x100xf32>
  %39 = "ttir.transpose"(%38) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x12x100xf32>) -> tensor<32x100x12xf32>
  %40 = "ttir.matmul"(%23, %39) : (tensor<32x12x100xf32>, tensor<32x100x12xf32>) -> tensor<32x12x12xf32>
  %41 = "ttir.unsqueeze"(%40) <{dim = 0 : si32}> : (tensor<32x12x12xf32>) -> tensor<1x32x12x12xf32>
  %42 = "ttir.multiply"(%41, %arg10) : (tensor<1x32x12x12xf32>, tensor<1x1xf32>) -> tensor<1x32x12x12xf32>
  %43 = "ttir.add"(%42, %arg1) : (tensor<1x32x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x32x12x12xf32>
  %44 = "ttir.softmax"(%43) <{dimension = -1 : si32}> : (tensor<1x32x12x12xf32>) -> tensor<1x32x12x12xf32>
  %45 = "ttir.squeeze"(%44) <{dim = 0 : si32}> : (tensor<1x32x12x12xf32>) -> tensor<32x12x12xf32>
  %46 = "ttir.matmul"(%0, %arg13) : (tensor<12x3200xf32>, tensor<3200x3200xf32>) -> tensor<12x3200xf32>
  %47 = "ttir.reshape"(%46) <{shape = [1 : i32, 12 : i32, 32 : i32, 100 : i32]}> : (tensor<12x3200xf32>) -> tensor<1x12x32x100xf32>
  %48 = "ttir.transpose"(%47) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x12x32x100xf32>) -> tensor<1x32x12x100xf32>
  %49 = "ttir.transpose"(%48) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x12x100xf32>) -> tensor<1x32x100x12xf32>
  %50 = "ttir.squeeze"(%49) <{dim = 0 : si32}> : (tensor<1x32x100x12xf32>) -> tensor<32x100x12xf32>
  %51 = "ttir.transpose"(%50) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x100x12xf32>) -> tensor<32x12x100xf32>
  %52 = "ttir.matmul"(%45, %51) : (tensor<32x12x12xf32>, tensor<32x12x100xf32>) -> tensor<32x12x100xf32>
  %53 = "ttir.unsqueeze"(%52) <{dim = 0 : si32}> : (tensor<32x12x100xf32>) -> tensor<1x32x12x100xf32>
  %54 = "ttir.transpose"(%53) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x12x100xf32>) -> tensor<1x12x32x100xf32>
  %55 = "ttir.reshape"(%54) <{shape = [12 : i32, 3200 : i32]}> : (tensor<1x12x32x100xf32>) -> tensor<12x3200xf32>
  %56 = "ttir.matmul"(%55, %arg14) : (tensor<12x3200xf32>, tensor<3200x3200xf32>) -> tensor<12x3200xf32>
  %57 = "ttir.unsqueeze"(%56) <{dim = 0 : si32}> : (tensor<12x3200xf32>) -> tensor<1x12x3200xf32>
  return %57 : tensor<1x12x3200xf32>
}
