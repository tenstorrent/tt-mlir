add_mlir_dialect_library(MLIRTTIRDialect
        TTIRDialect.cpp
        TTIROps.cpp
        TTIROpsTypes.cpp
        TTIRTraits.cpp
        TTIRTMFusionPatterns.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        TTIROpsInterfacesIncGen
        MLIRTTIRPassesIncGen
        MLIRTTCoreOpsIncGen

        LINK_LIBS PUBLIC
        MLIRTTCoreDialect
        MLIRBufferizationTransforms
        TTMLIRTTIRUtils
        )
