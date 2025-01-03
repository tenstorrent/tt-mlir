add_mlir_dialect_library(TTMLIRTTKernelDialect
        TTKernelDialect.cpp
        TTKernelOps.cpp
        TTKernelOpsTypes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTKernelOpsIncGen
        MLIRTTOpsIncGen

        LINK_LIBS PUBLIC
        TTMLIRTTMetalDialect
        )
