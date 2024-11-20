file(REMOVE_RECURSE
  "../../libTTNNTargetFlatbuffer.a"
  "../../libTTNNTargetFlatbuffer.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/TTNNTargetFlatbuffer.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
