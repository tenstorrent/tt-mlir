if (!TT_RUNTIME_ENABLE_TTNN)
    message(WARNING "Building TTNN Dialect without TTNN runtime support")
endif()


# C++20 required for tt-metal and tt-nn
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_definitions(MLIRTTNNDialect TT_RUNTIME_ENABLE_TTNN) # todo: wrap around macos build

add_mlir_dialect_library(MLIRTTNNDialect
        TTNNDialect.cpp
        TTNNOps.cpp
        TTNNOpsTypes.cpp
        TTNNOpsBackendInterfaces.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTNNOpsIncGen
        MLIRTTOpsIncGen

        LINK_LIBS PUBLIC
        TTMLIRTTNNUtils
        MLIRSCFToEmitC
        MLIRLinalgDialect
        MLIRMLProgramDialect
        )

# Include TTMETAL and TTNN headers
target_include_directories(MLIRTTNNDialect PRIVATE ${TTMETAL_INCLUDE_DIRS})

# Link against TTMETAL_LIBRARY and TTNN_LIBRARY
target_link_libraries(MLIRTTNNDialect PRIVATE TTMETAL_LIBRARY TTNN_LIBRARY)

# Disable warnings treated as errors that we are hitting in the metal headers included at TTNNOpsBackendInterfaces.cpp
set_source_files_properties(TTNNOpsBackendInterfaces.cpp PROPERTIES COMPILE_FLAGS 
    "-Wno-error=unused-local-typedef -Wno-error=unneeded-internal-declaration -Wno-error=unused-function -Wno-error=undefined-inline -Wno-error=unknown-warning-option -Wno-error=unused-variable -Wno-error=sign-compare -Wno-error=ignored-qualifiers -Wno-error=gnu-zero-variadic-macro-arguments  -Wno-error=ignored-qualifiers  -Wno-error=covered-switch-default -Wno-error=vla-extension -Wno-error=reorder-ctor -Wno-error=ast-qual -Wno-error=ctad-maybe-unsupported -Wno-error=deprecated-this-capture -Wno-error=suggest-override -Wno-error=cast-qual -Wno-error=gnu-anonymous-struct -Wno-error=nested-anon-types -Wno-error=non-virtual-dtor -Wno-error=mismatched-tags -Wno-error=deprecated-volatile -Wno-error=deprecated-volatile -Wno-error=implicit-fallthrough -Wno-error=logical-op-parentheses")