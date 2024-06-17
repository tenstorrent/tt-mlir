find_program(FLATBUFFERS_COMPILER flatc)

function(fbs_generate_cpp)
foreach(FILE ${FBS_GEN_SOURCES})
  get_filename_component(BASE_NAME ${FILE} NAME_WE)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${BASE_NAME}_generated.h"
    COMMAND ${FLATBUFFERS_COMPILER}
    ARGS --cpp --cpp-std c++17
    ARGS --scoped-enums --warnings-as-errors
    ARGS -o "${CMAKE_CURRENT_BINARY_DIR}/" "${FILE}"
    DEPENDS ${FILE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  list(APPEND FBS_GEN_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/${BASE_NAME}_generated.h)
endforeach()
endfunction()
