add_mlir_extension_library(TTMLIRTransforms
        ConstEvalHoist.cpp
        EmbedCudaTargetAttributes.cpp
        ExtractGPUModules.cpp
        ReenableLostDPS.cpp
        RemoveReturnValues.cpp
        WrapAffineFor.cpp
<<<<<<< HEAD

=======
=======
>>>>>>> f4793b162 (Cuda runtime.)
        ExtractGPUModules.cpp
        WrapAffineFor.cpp
>>>>>>> 4fa39775e (Fixing translation to NVVM with 1D tensors.)

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTCoreOpsIncGen
        )
