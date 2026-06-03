add_library(TTNNOpInvokeLib STATIC
  utils/utils.cpp
  Conv/Conv2dOp.cpp
  Conv/Conv3dOp.cpp
  Conv/ConvTranspose2dOp.cpp
  Conv/PrepareConv2dBiasOp.cpp
  Conv/PrepareConv2dWeightsOp.cpp
  Conv/PrepareConvTranspose2dBiasOp.cpp
  Conv/PrepareConvTranspose2dWeightsOp.cpp
  Eltwise/Binary/EltwiseBinaryCompositeOp.cpp
  Eltwise/Binary/EltwiseBinaryOp.cpp
  Eltwise/Quantization/EltwiseQuantizationOp.cpp
  Eltwise/Ternary/EltwiseTernaryOp.cpp
  Eltwise/Unary/EltwiseUnaryCompositeOp.cpp
  Eltwise/Unary/EltwiseUnaryOp.cpp
  Matmul/MatmulOp.cpp
  Transformer/ConcatenateHeadsOp.cpp
  Transformer/NlpConcatHeadsDecodeOp.cpp
  Transformer/NlpConcatHeadsOp.cpp
  Transformer/NlpCreateQKVHeadsDecodeOp.cpp
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
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/runtime/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/runtime/include>
)

target_include_directories(TTNNOpInvokeLib SYSTEM PUBLIC
  "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>"
)

target_link_libraries(TTNNOpInvokeLib PRIVATE LLVMSupport)

add_dependencies(TTNNOpInvokeLib tt-metal)
