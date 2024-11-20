file(REMOVE_RECURSE
  "../../../libMLIRTTNNDialect.a"
  "../../../libMLIRTTNNDialect.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIRTTNNDialect.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
