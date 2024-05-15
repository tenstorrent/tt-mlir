add_mlir_dialect_library(MLIRTTMetalTransforms
        Passes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTMetalOpsIncGen
        MLIRTTMetalPassesIncGen
        MLIRTTOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
