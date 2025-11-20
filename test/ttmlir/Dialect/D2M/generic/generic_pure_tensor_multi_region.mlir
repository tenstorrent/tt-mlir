module {
  func.func @pure_tensor_multiple_regions(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = d2m.empty() : tensor<64x128xf32>
    %1 = d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<64x128xf32>)
        outs(%0 : tensor<64x128xf32>)  {
    ^datamovement0(%cb0: !d2m.cb<tensor<64x128xf32>>, %cb1: !d2m.cb<tensor<64x128xf32>>, %cb2: !d2m.cb<tensor<64x128xf32>>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
      %c0 = arith.constant 0 : index
      %2 = d2m.wait %cb0 : <tensor<64x128xf32>> -> tensor<64x128xf32>
      %3 = d2m.reserve %cb2 : <tensor<64x128xf32>> -> tensor<64x128xf32>
      %extracted = tensor.extract %2[%c0, %c0] : tensor<64x128xf32>
      %inserted = tensor.insert %extracted into %3[%c0, %c0] : tensor<64x128xf32>
      d2m.yield
    }, {
    ^datamovement1(%cb0: !d2m.cb<tensor<64x128xf32>>, %cb1: !d2m.cb<tensor<64x128xf32>>, %cb2: !d2m.cb<tensor<64x128xf32>>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
      %c0 = arith.constant 0 : index
      %2 = d2m.wait %cb1 : <tensor<64x128xf32>> -> tensor<64x128xf32>
      %3 = d2m.reserve %cb2 : <tensor<64x128xf32>> -> tensor<64x128xf32>
      %extracted = tensor.extract %2[%c0, %c0] : tensor<64x128xf32>
      %inserted = tensor.insert %extracted into %3[%c0, %c0] : tensor<64x128xf32>
      d2m.yield
    }, {
    ^compute0(%cb0: !d2m.cb<tensor<64x128xf32>>, %cb1: !d2m.cb<tensor<64x128xf32>>, %cb2: !d2m.cb<tensor<64x128xf32>>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
      %c0 = arith.constant 0 : index
      %2 = d2m.wait %cb0 : <tensor<64x128xf32>> -> tensor<64x128xf32>
      %3 = d2m.wait %cb1 : <tensor<64x128xf32>> -> tensor<64x128xf32>
      %4 = d2m.reserve %cb2 : <tensor<64x128xf32>> -> tensor<64x128xf32>
      %extracted = tensor.extract %2[%c0, %c0] : tensor<64x128xf32>
      %extracted_0 = tensor.extract %3[%c0, %c0] : tensor<64x128xf32>
      %5 = arith.addf %extracted, %extracted_0 : f32
      %inserted = tensor.insert %5 into %4[%c0, %c0] : tensor<64x128xf32>
      d2m.yield
    } : tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
