add_mlir_dialect_library(MLIRTTKernelTransforms
        ControlDstSection.cpp
        InsertDeviceZoneScopes.cpp

        DEPENDS
        MLIRTTKernelOpsIncGen
        MLIRTTKernelPassesIncGen
        MLIRTTCoreOpsIncGen
        )
