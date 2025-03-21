add_mlir_dialect_library(MLIRTTMetalDialect
        TTMetalDialect.cpp
        TTMetalOps.cpp
        TTMetalOpsTypes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTMetalOpsIncGen
        MLIRTTOpsIncGen
        )

target_link_libraries(MLIRTTMetalDialect PUBLIC coverage_config)
