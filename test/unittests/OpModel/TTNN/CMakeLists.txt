# Create shared YamlParamLoader library target for both Lib and Op tests
if(TTMLIR_ENABLE_OPMODEL)
    add_library(YamlParamLoader STATIC YamlParamLoader.cpp YamlParamLoader.h)

    target_include_directories(
        YamlParamLoader
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/lib/OpModel/TTNN/
    )

    target_link_libraries(
        YamlParamLoader
        PUBLIC yaml-cpp MLIRTTCoreDialect MLIRTTIRDialect MLIRTTNNDialect
    )

    target_compile_options(YamlParamLoader PRIVATE -fno-rtti)

    set_target_properties(
        YamlParamLoader
        PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
    )
endif()

add_subdirectory(Conversion)
add_subdirectory(Lib)
add_subdirectory(Op)
