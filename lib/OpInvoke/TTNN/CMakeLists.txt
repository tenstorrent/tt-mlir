add_library(TTNNOpInvokeLib STATIC
  utils/utils.cpp
  Conv/Conv2dOp.cpp
  Conv/Conv3dOp.cpp
  Conv/ConvTranspose2dOp.cpp
  Conv/PrepareConv2dBiasOp.cpp
  Conv/PrepareConv2dWeightsOp.cpp
  Conv/PrepareConv3dWeightsOp.cpp
  Conv/PrepareConvTranspose2dBiasOp.cpp
  Conv/PrepareConvTranspose2dWeightsOp.cpp
  DataMovement/AssignOp.cpp
  DataMovement/ConcatOp.cpp
  DataMovement/GatherOp.cpp
  DataMovement/PadOp.cpp
  DataMovement/PermuteOp.cpp
  DataMovement/RepeatInterleaveOp.cpp
  DataMovement/RepeatOp.cpp
  DataMovement/ReshapeOp.cpp
  DataMovement/ScatterOp.cpp
  DataMovement/SliceOp.cpp
  DataMovement/SortOp.cpp
  DataMovement/TransposeOp.cpp
  DataMovement/WriteTensorOp.cpp
  Eltwise/Binary/EltwiseBinaryCompositeOp.cpp
  Eltwise/Binary/EltwiseBinaryOp.cpp
  Eltwise/Quantization/EltwiseQuantizationOp.cpp
  Eltwise/Ternary/EltwiseTernaryOp.cpp
  Eltwise/Unary/EltwiseUnaryCompositeOp.cpp
  Eltwise/Unary/EltwiseUnaryOp.cpp
  Embedding/EmbeddingBackwardOp.cpp
  Experimental/DropoutOp.cpp
  Experimental/ExperimentalEltwiseBinaryBackwardOp.cpp
  KVCache/FillCacheOp.cpp
  KVCache/PagedFillCacheOp.cpp
  KVCache/PagedUpdateCacheOp.cpp
  Matmul/MatmulOp.cpp
  Normalization/BatchNormOp.cpp
  Normalization/DistributedRMSNormOp.cpp
  Normalization/GroupNormOp.cpp
  Normalization/LayerNormOp.cpp
  Normalization/LayerNormPostAllGatherOp.cpp
  Normalization/LayerNormPreAllGatherOp.cpp
  Normalization/RMSNormOp.cpp
  Normalization/RMSNormPreAllGatherOp.cpp
  Normalization/SoftmaxOp.cpp
  Reduction/ArgMaxOp.cpp
  Reduction/CumSumOp.cpp
  Reduction/ProdOp.cpp
  Reduction/ReductionOp.cpp
  Reduction/TopKOp.cpp
  Reduction/TopKRouterGptOp.cpp
  Transformer/ConcatenateHeadsOp.cpp
  Transformer/NLPConcatHeadsDecodeOp.cpp
  Transformer/NLPConcatHeadsOp.cpp
  Transformer/NLPCreateQKVHeadsDecodeOp.cpp
  Transformer/PagedFlashMultiLatentAttentionDecodeOp.cpp
  Transformer/PagedScaledDotProductAttentionDecodeOp.cpp
  Transformer/RotaryEmbeddingLlamaOp.cpp
  Transformer/RotaryEmbeddingOp.cpp
  Transformer/ScaledDotProductAttentionDecodeOp.cpp
  Transformer/ScaledDotProductAttentionOp.cpp
  Transformer/SplitQueryKeyValueAndSplitHeadsOp.cpp
)

set_property(TARGET TTNNOpInvokeLib PROPERTY CXX_STANDARD 20)

target_include_directories(TTNNOpInvokeLib PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/ttmlir/OpInvoke>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common>
)

target_include_directories(TTNNOpInvokeLib SYSTEM PUBLIC
  "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>"
)

target_link_libraries(TTNNOpInvokeLib PRIVATE LLVMSupport)

add_dependencies(TTNNOpInvokeLib tt-metal)
