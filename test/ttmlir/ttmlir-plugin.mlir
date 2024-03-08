// RUN: mlir-opt %s --load-dialect-plugin=%ttmlir_libs/TTMLIRPlugin%shlibext --pass-pipeline="builtin.module(ttmlir-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @ttmlir_types(%arg0: !ttmlir.custom<"10">)
  func.func @ttmlir_types(%arg0: !ttmlir.custom<"10">) {
    return
  }
}
