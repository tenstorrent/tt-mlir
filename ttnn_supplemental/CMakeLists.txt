# ttnn_supplemental lib
#
# Contains supplemental APIs for TT-NN library, like ttnn::mesh_shard, etc.

add_library(ttnn_supplemental MODULE supplemental.cpp pybind_supplemental.cpp)

find_package(pybind11 CONFIG REQUIRED)

set_property(TARGET ttnn_supplemental PROPERTY CXX_STANDARD 20)

target_include_directories(ttnn_supplemental SYSTEM PRIVATE ${TTMETAL_INCLUDE_DIRS})

target_link_libraries(ttnn_supplemental PRIVATE pybind11::module TTNN_LIBRARY)

add_dependencies(ttnn_supplemental TTNN_LIBRARY)

# The name of the lib will be ttnn_supplemental.so
#set_target_properties(ttnn_supplemental PROPERTIES PREFIX "" SUFFIX ".so")
