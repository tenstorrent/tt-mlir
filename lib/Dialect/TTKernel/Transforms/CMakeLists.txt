add_mlir_dialect_library(MLIRTTKernelTransforms
        ControlDstSection.cpp

        DEPENDS
        MLIRTTKernelOpsIncGen
        MLIRTTKernelPassesIncGen
        MLIRTTCoreOpsIncGen
        )
