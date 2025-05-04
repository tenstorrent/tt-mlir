// RUN: ttmlir-opt --convert-ttkernel-to-emitc --form-expressions %s | ttmlir-translate --ttkernel-to-cpp | FileCheck %s

func.func @kernel_main() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %c42_i32 = arith.constant 42 : i32
    // CHECK: ((DPRINT << "Hello world, ") << v1) << "!\n";
    ttkernel.dprint("Hello world, {}!\\n", %c42_i32) : (i32)
    func.return
}
