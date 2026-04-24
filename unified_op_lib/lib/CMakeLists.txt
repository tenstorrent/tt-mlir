add_library(TTUnifiedOpLib STATIC
  utils/utils.cpp
  conv/unifiedConv2dOp.cpp
  conv/unifiedPrepareConv2dBiasOp.cpp
  conv/unifiedPrepareConvTranspose2dBiasOp.cpp
  conv/unifiedPrepareConvTranspose2dWeightsOp.cpp
  eltwise/binary/unifiedEltwiseBinaryCompositeOp.cpp
  eltwise/binary/unifiedEltwiseBinaryOp.cpp
  eltwise/ternary/unifiedEltwiseTernaryOp.cpp
  eltwise/unary/unifiedEltwiseUnaryCompositeOp.cpp
  eltwise/unary/unifiedEltwiseUnaryOp.cpp
  eltwise/quantization/unifiedEltwiseQuantizationOp.cpp
  matmul/unifiedMatmulOp.cpp
)

set_property(TARGET TTUnifiedOpLib PROPERTY CXX_STANDARD 20)

target_include_directories(TTUnifiedOpLib PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/unified_op_lib/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common>
)

target_include_directories(TTUnifiedOpLib PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/runtime/include
)

target_include_directories(TTUnifiedOpLib SYSTEM PUBLIC
  "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>"
)

add_dependencies(TTUnifiedOpLib tt-metal)
