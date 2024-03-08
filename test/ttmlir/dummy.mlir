// RUN: ttmlir-opt %s | ttmlir-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = ttmlir.foo %{{.*}} : i32
        %res = ttmlir.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @ttmlir_types(%arg0: !ttmlir.custom<"10">)
    func.func @ttmlir_types(%arg0: !ttmlir.custom<"10">) {
        return
    }
}
