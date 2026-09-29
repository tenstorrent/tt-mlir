module {
  func.func @while_add_one(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %i0 = "ttir.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %limit = "ttir.constant"() <{value = dense<4> : tensor<i32>}> : () -> tensor<i32>
    %step = "ttir.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    %one = "ttir.full"() <{fill_value = 1.000000e+00 : f32, shape = array<i32: 32, 32>}> : () -> tensor<32x32xf32>
    %result:2 = ttir.while inits(%i0, %arg0 : tensor<i32>, tensor<32x32xf32>)
                           captures(%limit, %step, %one : tensor<i32>, tensor<i32>, tensor<32x32xf32>)
                           {trip_count = 4 : i64}
      cond {
      ^cond(%i: tensor<i32>, %acc: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>, %ones: tensor<32x32xf32>):
        %predicate = "ttir.lt"(%i, %l) : (tensor<i32>, tensor<i32>) -> tensor<i1>
        ttir.yield %predicate : tensor<i1>
      } do {
      ^body(%i: tensor<i32>, %acc: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>, %ones: tensor<32x32xf32>):
        %next = "ttir.add"(%i, %s) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        %next_acc = "ttir.add"(%acc, %ones) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
        ttir.yield %next, %next_acc : tensor<i32>, tensor<32x32xf32>
      } -> (tensor<i32>, tensor<32x32xf32>)
    return %result#1 : tensor<32x32xf32>
  }
}
