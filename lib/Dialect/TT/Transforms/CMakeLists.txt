add_mlir_dialect_library(MLIRTTTransforms
        TTModuleWrap.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir
        )
