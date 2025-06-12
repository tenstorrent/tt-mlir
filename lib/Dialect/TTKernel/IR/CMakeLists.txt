add_mlir_dialect_library(MLIRTTKernelDialect
        TTKernelDialect.cpp
        TTKernelOps.cpp
        TTKernelOpsTypes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTKernelOpsIncGen
        MLIRTTCoreOpsIncGen

        LINK_LIBS PUBLIC
        MLIRTTMetalDialect
        MLIRTTCoreDialect
        )
