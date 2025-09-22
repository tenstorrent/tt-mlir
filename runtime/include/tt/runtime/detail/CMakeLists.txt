add_subdirectory(distributed)

add_custom_target(RUNTIME_FBS_GENERATION DEPENDS DISTRIBUTED_FBS)

# Install detail test headers
if (TTMLIR_ENABLE_RUNTIME_TESTS)
  install(
    FILES
      test/ttnn/utils.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tt/runtime/detail/test/ttnn
    COMPONENT TTNNStandalone
  )
endif()
