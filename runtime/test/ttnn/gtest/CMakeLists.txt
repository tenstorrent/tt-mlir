if(DISTRIBUTED_RUNTIME_ENABLED)
  add_runtime_gtest(test_runtime_worker)
endif()

add_runtime_gtest(test_tensor_serialization)
