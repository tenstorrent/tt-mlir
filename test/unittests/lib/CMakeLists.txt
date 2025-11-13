
# TestUtils lib provides extensions to gtest functionality.

if (NOT GTest_FOUND)
  message(WARNING "GTest not found")
  return()
endif()

set(TESTLIB TTMLIRTestUtils)
add_library(${TESTLIB})

# TestUtils lib sources.

target_sources(${TESTLIB}
  PRIVATE
    Utils.cpp
)

# TestUtils lib includes.

target_include_directories(${TESTLIB}
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../
)

target_link_libraries(${TESTLIB} GTest::gtest GTest::gtest_main)
