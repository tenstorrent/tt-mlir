add_mlir_extension_library(TTMLIRTransforms
        ConstEvalHoist.cpp
        ReenableLostDPS.cpp
        RemoveReturnValues.cpp
<<<<<<< HEAD
        WrapAffineFor.cpp
=======
>>>>>>> f4793b162 (Cuda runtime.)
        ExtractGPUModules.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTCoreOpsIncGen
        )
