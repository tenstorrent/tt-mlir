add_library(TTRuntimeSysDesc STATIC system_desc.cpp)
set_property(TARGET TTRuntimeSysDesc PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeSysDesc
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeSysDesc PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
add_dependencies(TTRuntimeSysDesc tt-metal FBS_GENERATION)

add_library(TTRuntimeDebug STATIC debug.cpp)
target_include_directories(TTRuntimeDebug
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
)
