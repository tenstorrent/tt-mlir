add_mlir_dialect_library(MLIRTTMetalTransforms
        ApplyHostMemrefCallingConvention.cpp

        DEPENDS
        MLIRTTMetalOpsIncGen
        MLIRTTMetalPassesIncGen
        )
