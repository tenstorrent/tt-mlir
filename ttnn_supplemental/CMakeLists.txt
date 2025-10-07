find_package(pybind11 CONFIG REQUIRED)

add_library(ttnn_supplemental MODULE ops.cpp pybind_ops.cpp)
set_property(TARGET ttnn_supplemental PROPERTY CXX_STANDARD 20)
target_include_directories(ttnn_supplemental SYSTEM PRIVATE ${TTMETAL_INCLUDE_DIRS})
target_link_libraries(ttnn_supplemental PRIVATE pybind11::module TTNN_LIBRARY)
add_dependencies(ttnn_supplemental tt-metal TTNN_LIBRARY)
set_target_properties(ttnn_supplemental PROPERTIES PREFIX "" SUFFIX ".so")