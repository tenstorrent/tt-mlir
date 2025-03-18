find_program(FLATBUFFERS_COMPILER flatc)

function(build_flatbuffers sources target)
# Query deps if they exist
set(deps)
if(ARGC GREATER 2)
  foreach(target_dep IN LISTS ARGV2)
    get_property(target_sources TARGET ${target_dep} PROPERTY SOURCES)
    list(APPEND deps ${target_sources})
  endforeach()
endif()

set(FBS_GEN_OUTPUTS)

foreach(FILE ${sources})
  get_filename_component(BASE_NAME ${FILE} NAME_WE)
  add_custom_command(OUTPUT
    "${CMAKE_CURRENT_BINARY_DIR}/${BASE_NAME}_generated.h"
    "${CMAKE_CURRENT_BINARY_DIR}/${BASE_NAME}_bfbs_generated.h"
    COMMAND ${FLATBUFFERS_COMPILER}
    ARGS -I ${PROJECT_SOURCE_DIR}/include/ttmlir/Target
    ARGS --bfbs-gen-embed
    ARGS --cpp --cpp-std c++17
    ARGS --scoped-enums --warnings-as-errors
    ARGS -o "${CMAKE_CURRENT_BINARY_DIR}/" "${FILE}"
    DEPENDS ${FILE} ${deps} ${sources}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  list(APPEND FBS_GEN_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/${BASE_NAME}_generated.h)
  list(APPEND FBS_GEN_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/${BASE_NAME}_bfbs_generated.h)
endforeach()
add_library(${target} INTERFACE ${FBS_GEN_OUTPUTS})
endfunction()
