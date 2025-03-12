add_mlir_dialect_library(MLIRTTIRDialect
        TTIRDialect.cpp
        TTIROps.cpp
        TTIROpsTypes.cpp
        TTIROpsInterfaces.cpp
        TTIRTraits.cpp
        TTIRGenericRegionOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        TTIRGenericRegionOpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTOpsIncGen

        LINK_LIBS PUBLIC
        MLIRBufferizationTransforms
        )
