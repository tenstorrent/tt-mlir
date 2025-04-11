add_mlir_dialect_library(MLIRTTKernelTransforms
        ControlDstSection.cpp
        EraseFunctionArgs.cpp

        DEPENDS
        MLIRTTKernelOpsIncGen
        MLIRTTKernelPassesIncGen
        MLIRTTOpsIncGen
        )
