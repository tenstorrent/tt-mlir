if (TTMLIR_ENABLE_OPMODEL)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_mlir_unittest(TestOpModelInterface
TestOpModelInterface.cpp
)

target_compile_options(TestOpModelInterface
    PRIVATE
    -fno-rtti
)

target_include_directories(TestOpModelInterface
    PUBLIC
    ${PROJECT_SOURCE_DIR}/lib/OpModel/TTNN/
    ${PROJECT_SOURCE_DIR}/test/unittests/OpModel/TTNN/
)

target_link_libraries(TestOpModelInterface
    PRIVATE
    TTNNOpModelLib
    MLIRTTCoreDialect
    MLIRTTNNDialect
    MLIRTTTransforms
)

endif()
