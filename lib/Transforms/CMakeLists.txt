add_mlir_extension_library(TTMLIRTransforms
        ConstEvalHoist.cpp
        EmbedCudaTargetAttributes.cpp
        ReenableLostDPS.cpp
        RemoveReturnValues.cpp
        ExtractGPUModules.cpp
        WrapAffineFor.cpp
        ExtractGPUModules.cpp
        WrapAffineFor.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTCoreOpsIncGen
        )
