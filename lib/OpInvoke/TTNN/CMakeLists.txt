add_library(TTNNOpInvokeLib STATIC
  utils/utils.cpp
  Conv/Conv2dOp.cpp
  Eltwise/Binary/EltwiseBinaryCompositeOp.cpp
  Eltwise/Binary/EltwiseBinaryOp.cpp
  Eltwise/Unary/EltwiseUnaryCompositeOp.cpp
  Eltwise/Unary/EltwiseUnaryOp.cpp
  Matmul/MatmulOp.cpp
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
