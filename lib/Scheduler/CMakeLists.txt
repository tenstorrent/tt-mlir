add_mlir_library(MLIRScheduler
    Scheduler.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ttmlir/Scheduler

    LINK_LIBS PUBLIC
      MLIRTTIRDialect
      MLIRIR
      MLIRPass
)

target_link_libraries(MLIRScheduler PUBLIC coverage_config)
