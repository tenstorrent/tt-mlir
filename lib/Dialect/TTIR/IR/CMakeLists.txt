add_mlir_dialect_library(TTMLIRTTIRDialect
        TTIRDialect.cpp
        TTIROps.cpp
        TTIROpsInterfaces.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTOpsIncGen
        )
