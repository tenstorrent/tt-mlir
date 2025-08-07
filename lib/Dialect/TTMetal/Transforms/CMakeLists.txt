add_mlir_dialect_library(MLIRTTMetalTransforms
        GenerateDimAlignedHostIO.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTMetalOpsIncGen
        MLIRTTCoreOpsIncGen
        MLIRSupport
        )
