find_package(Python3 COMPONENTS Development Interpreter REQUIRED)
find_package(pybind11 REQUIRED)

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
set_property(TARGET TTRuntimeDebug PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeDebug
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${Python3_INCLUDE_DIRS}
)
target_link_libraries(TTRuntimeDebug PUBLIC $ENV{TTMLIR_PYTHON_VERSION} pybind11::pybind11 pybind11::embed)

add_library(TTRuntimeWorkarounds STATIC workarounds.cpp)
set_property(TARGET TTRuntimeWorkarounds PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeWorkarounds
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
)
