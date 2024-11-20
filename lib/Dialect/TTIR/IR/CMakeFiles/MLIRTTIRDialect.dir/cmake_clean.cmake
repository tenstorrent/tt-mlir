file(REMOVE_RECURSE
  "../../../libMLIRTTIRDialect.a"
  "../../../libMLIRTTIRDialect.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIRTTIRDialect.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
