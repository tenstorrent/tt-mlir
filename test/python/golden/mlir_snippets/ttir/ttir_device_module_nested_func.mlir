module {
  ttcore.device_module {
    builtin.module {
      func.func @my_modela(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
        %0 = "ttir.sigmoid"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
        %1 = call @nested_func(%0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
        return %1 : tensor<32x32xf32>
      }
      func.func nested @nested_func(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
        %0 = "ttir.sigmoid"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
        return %0 : tensor<32x32xf32>
      }
      func.func @my_modelb(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
        %0 = "ttir.sigmoid"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
        return %0 : tensor<32x32xf32>
      }
    }
  }
}
