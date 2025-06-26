add_mlir_dialect_library(MLIRTTTransforms
        TTCoreModuleWrap.cpp
        TTCoreRegisterDevice.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir
        )
