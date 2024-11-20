file(REMOVE_RECURSE
  "../../libTTLLVMToDynamicLib.a"
  "../../libTTLLVMToDynamicLib.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/TTLLVMToDynamicLib.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
