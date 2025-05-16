# Function to generate C++ header files from input header files using xxd
function(generate_llk_header INPUT_FILE OUTPUT_FILE VARIABLE_NAME)
  # Create the output directory if it doesn't exist
  get_filename_component(OUTPUT_DIR ${OUTPUT_FILE} DIRECTORY)
  file(MAKE_DIRECTORY ${OUTPUT_DIR})

  add_custom_command(
    OUTPUT ${OUTPUT_FILE}
    COMMAND bash -c "echo \"// Auto-generated from ${INPUT_FILE} - Do not edit directly\" > ${OUTPUT_FILE}"
    COMMAND bash -c "echo \"const unsigned char ${VARIABLE_NAME}[] = {\" >> ${OUTPUT_FILE}"
    # Use xxd to convert the file content to a hex array
    COMMAND bash -c "xxd -i < ${INPUT_FILE} >> ${OUTPUT_FILE}"
    COMMAND bash -c "echo \"}; const unsigned int ${VARIABLE_NAME}_len = sizeof(${VARIABLE_NAME});\" >> ${OUTPUT_FILE}"
    DEPENDS ${INPUT_FILE}
    COMMENT "Generating llk header for ${INPUT_FILE}"
    VERBATIM
  )
endfunction()
