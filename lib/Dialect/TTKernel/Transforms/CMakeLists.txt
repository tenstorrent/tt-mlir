add_mlir_dialect_library(MLIRTTKernelTransforms
        ControlDstSection.cpp
        DedupInits.cpp
        HoistInits.cpp
        InsertDeviceZoneScopes.cpp

        DEPENDS
        MLIRTTKernelOpsIncGen
        MLIRTTKernelPassesIncGen
        MLIRTTCoreOpsIncGen
        )
