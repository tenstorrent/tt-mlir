# Simple for loop versions of mapping commits

# for tt-metal to tt-mlir
alias metal_version_map='f(){ for h in $(git -C third_party/tt-metal/src/tt-metal log --reverse --format="%h" "$1"); do sed -i -E "s/set\\(TT_METAL_VERSION \\\"[a-zA-Z0-9]+\\\"\\)/set(TT_METAL_VERSION \\\"$h\\\")/" third_party/CMakeLists.txt; git add third_party/CMakeLists.txt; git commit --no-verify -m "Set TT_METAL_VERSION to $h"; done; }; f'
# Usage: metal_version_map <tt-metal-log-range>
# Example: metal_version_map abc..def

# for tt-mlir to tt-torch/tt-xla
alias mlir_version_map='f(){ for h in $(git -C third_party/tt-mlir/src/tt-mlir log --reverse --format="%h" "$1"); do sed -i -E "s/set\\(TT_MLIR_VERSION \\\"[a-zA-Z0-9]+\\\"\\)/set(TT_MLIR_VERSION \\\"$h\\\")/" third_party/CMakeLists.txt; git add third_party/CMakeLists.txt; git commit --no-verify -m "Set TT_MLIR_VERSION to $h"; done; }; f'
# Usage: mlir_version_map <tt-mlir-log-range>
# Example: mlir_version_map abc..def


# for tt-mlir to tt-forge-fe
alias mlir_submodule_map='f(){ for h in $(git -C third_party/tt-mlir log --reverse --format="%h" "$1"); do git -C third_party/tt-mlir checkout $h; git add third_party/tt-mlir; git commit --no-verify -m "Set TT_MLIR_VERSION to $h"; done; }; f'
# Usage: mlir_submodule_map <tt-mlir-log-range>
# Example: mlir_submodule_map abc..def
