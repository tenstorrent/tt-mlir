add_mlir_extension_library(TTMLIRTransforms
        CollapseParallelLoops.cpp
        ConstEvalHoist.cpp
        EmbedCudaTargetAttributes.cpp
        EnableDPSForHoistedFuncs.cpp
        ExtractGPUModules.cpp
        RemoveReturnValues.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTCoreOpsIncGen
        )
