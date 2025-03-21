add_mlir_dialect_library(MLIRTTUtils
        PopulateArgumentTypes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTOpsIncGen
        )

target_link_libraries(MLIRTTUtils PUBLIC coverage_config)
