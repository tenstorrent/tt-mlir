# ONNX Runtime plugin execution provider (libtt_crank_ort.so).
add_library(tt_crank_ort SHARED
    ort.cpp
    graph.cpp
    options.cpp
    tensor.cpp
    builder.cpp
    partition.cpp
    compute.cpp
)

set_target_properties(tt_crank_ort PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_include_directories(tt_crank_ort
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/onnx>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

target_compile_features(tt_crank_ort PRIVATE cxx_std_20)

target_link_libraries(tt_crank_ort
    PRIVATE
        tt_crank::tt_crank
        tt_crank_common
        tt_crank_warnings
        tt_crank_sanitizers
)

# Wheel: goes in tt_crank/lib with the other native libs. Only the installed copy
# gets this RPATH; build-tree binaries keep CMake's build RPATH.
set_target_properties(tt_crank_ort PROPERTIES INSTALL_RPATH "$ORIGIN")
