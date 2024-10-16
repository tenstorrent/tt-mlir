get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
find_package(jsoncpp REQUIRED)
set(LIBS ${dialect_libs} ${conversion_libs} MLIROptLib MLIRTargetCpp TTMLIRStatic jsoncpp_lib)

set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/ttmlir-noc-analyzer.cpp ${CMAKE_CURRENT_SOURCE_DIR}/noc-analytical-model.cpp)
set(HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/noc-analytical-model.hpp)

add_executable(ttmlir-noc-analyzer ${HEADERS} ${SOURCES})

llvm_update_compile_flags(ttmlir-noc-analyzer)
target_link_libraries(ttmlir-noc-analyzer PRIVATE ${LIBS})

mlir_check_all_link_libraries(ttmlir-noc-analyzer)

install(TARGETS ttmlir-noc-analyzer DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Test EXCLUDE_FROM_ALL)
