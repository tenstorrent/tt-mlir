add_mlir_extension_library(TTMLIRTransforms
        CollapseParallelLoops.cpp
        ConstEvalHoist.cpp
        ReenableLostDPS.cpp
        RemoveReturnValues.cpp
        WrapAffineFor.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTCoreOpsIncGen
        )
