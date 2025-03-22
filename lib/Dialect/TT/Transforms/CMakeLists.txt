add_mlir_dialect_library(MLIRTTTransforms
        TTModuleWrap.cpp
        TTRegisterDevice.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir
        )

target_link_libraries(MLIRTTTransforms PUBLIC coverage_config)
