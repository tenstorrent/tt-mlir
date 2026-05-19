add_runtime_gtest(test_tensor_serialization)
add_runtime_gtest(test_tensor_cache_lifetime)
add_runtime_gtest(test_unsafe_borrowed_host_tensor)

# Race reproducer for tt::tt_metal::GraphTracker. Needs OpModel libs in
# addition to the runtime libs that add_runtime_gtest already wires up.
if (TTMLIR_ENABLE_OPMODEL)
  add_runtime_gtest(test_graph_tracker_race)
  target_compile_options(test_graph_tracker_race PRIVATE -fno-rtti)
  target_include_directories(test_graph_tracker_race PRIVATE
    ${PROJECT_SOURCE_DIR}/test/unittests
    ${PROJECT_SOURCE_DIR}/test/unittests/OpModel/TTNN
    ${PROJECT_SOURCE_DIR}/lib/OpModel/TTNN
  )
  target_link_libraries(test_graph_tracker_race PRIVATE
    TTNNOpModelLib
    MLIRTTCoreDialect
    MLIRTTNNDialect
    MLIRTTTransforms
  )
endif()
