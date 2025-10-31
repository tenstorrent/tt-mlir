add_mlir_dialect_library(MLIRTTTransforms
        OptimizationBarrierFold.cpp
        TTCoreModuleWrap.cpp
        TTCoreOneShotBufferize.cpp
        TTCoreRegisterDevice.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        LINK_LIBS PUBLIC
        MLIRBufferizationTransforms
        )
