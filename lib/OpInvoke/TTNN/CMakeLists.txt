add_library(TTNNOpInvokeLib STATIC
  utils/utils.cpp
  conv/conv2dOp.cpp
  conv/convTranspose2dOp.cpp
  eltwise/binary/eltwiseBinaryCompositeOp.cpp
  eltwise/binary/eltwiseBinaryOp.cpp
  eltwise/quantization/eltwiseQuantizationOp.cpp
  eltwise/ternary/eltwiseTernaryOp.cpp
  eltwise/unary/eltwiseUnaryCompositeOp.cpp
  eltwise/unary/eltwiseUnaryOp.cpp
  matmul/matmulOp.cpp
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
