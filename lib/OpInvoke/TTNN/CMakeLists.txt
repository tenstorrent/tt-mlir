add_library(TTNNOpInvokeLib STATIC
  utils/utils.cpp
  conv/conv2dOp.cpp
  conv/conv3dOp.cpp
  conv/convTranspose2dOp.cpp
  conv/prepareConv2dBiasOp.cpp
  conv/prepareConv2dWeightsOp.cpp
  conv/prepareConvTranspose2dBiasOp.cpp
  conv/prepareConvTranspose2dWeightsOp.cpp
  eltwise/binary/eltwiseBinaryCompositeOp.cpp
  eltwise/binary/eltwiseBinaryOp.cpp
  eltwise/quantization/eltwiseQuantizationOp.cpp
  eltwise/ternary/eltwiseTernaryOp.cpp
  eltwise/unary/eltwiseUnaryCompositeOp.cpp
  eltwise/unary/eltwiseUnaryOp.cpp
  matmul/matmulOp.cpp
  transformer/concatenateHeadsOp.cpp
  transformer/nlpConcatHeadsOp.cpp
  transformer/nlpCreateQKVHeadsDecodeOp.cpp
  transformer/pagedFlashMultiLatentAttentionDecodeOp.cpp
  transformer/pagedScaledDotProductAttentionDecodeOp.cpp
  transformer/rotaryEmbeddingLlamaOp.cpp
  transformer/rotaryEmbeddingOp.cpp
  transformer/scaledDotProductAttentionDecodeOp.cpp
  transformer/scaledDotProductAttentionOp.cpp
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
