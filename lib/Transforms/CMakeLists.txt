add_mlir_extension_library(TTMLIRTransforms
        ConstEvalHoist.cpp
        ReenableLostDPS.cpp
        RemoveReturnValues.cpp
        WrapAffineFor.cpp

        EmbedCudaTargetAttributes.cpp
        ExtractGPUModules.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTCoreOpsIncGen
        )
