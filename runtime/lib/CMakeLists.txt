add_library(TTRuntimeBinary STATIC binary/binary.cpp)
target_include_directories(TTRuntimeBinary PUBLIC ${PROJECT_SOURCE_DIR}/runtime/include ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common)
add_dependencies(TTRuntimeBinary FBS_GENERATION)
