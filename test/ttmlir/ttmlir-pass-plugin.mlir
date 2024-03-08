// RUN: mlir-opt %s --load-pass-plugin=%ttmlir_libs/TTMLIRPlugin%shlibext --pass-pipeline="builtin.module(ttmlir-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @abar()
  func.func @abar() {
    return
  }
}
