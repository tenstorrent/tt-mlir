add_custom_target(MLIRUnitTests)
set_target_properties(MLIRUnitTests PROPERTIES FOLDER "MLIR Tests")

# unit tests are compiled against gtest included with LLVM
include_directories(SYSTEM ${TTMLIR_TOOLCHAIN_DIR}/include/llvm-gtest ${TTMLIR_TOOLCHAIN_DIR}/include/llvm-gmock)

function(add_mlir_unittest test_dirname)
  add_unittest(MLIRUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(lib) # test util lib

add_subdirectory(Allocation)
add_subdirectory(Support)
add_subdirectory(TestScheduler)
add_subdirectory(Optimizer)
add_subdirectory(OpModel)
add_subdirectory(TTNNToEmitC)
add_subdirectory(Validation)
