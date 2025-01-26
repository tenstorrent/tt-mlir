# add_subdirectory(Canonicalization)
add_mlir_dialect_library(MLIRTTIRDialect
        TTIRDialect.cpp
        TTIROps.cpp
        TTIROpsInterfaces.cpp
        TTIRTraits.cpp
        # Canonicalization/TTIRCanonicalization.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTOpsIncGen
        )
