add_mlir_dialect_library(MLIRTTKernelTransforms
        ControlDstSection.cpp
        InsertDeviceZoneScopes.cpp
        HoistInits.cpp

        DEPENDS
        MLIRTTKernelOpsIncGen
        MLIRTTKernelPassesIncGen
        MLIRTTCoreOpsIncGen
        )
