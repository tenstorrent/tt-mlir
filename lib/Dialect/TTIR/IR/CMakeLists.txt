add_mlir_dialect_library(MLIRTTIRDialect
        TTIRDialect.cpp
        TTIROps.cpp
        TTIROpsInterfaces.cpp
        TTIRTraits.cpp
        TTIRTileOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        TTIRTileOpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTOpsIncGen
        )
