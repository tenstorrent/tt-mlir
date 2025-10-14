// RUN: ttmlir-opt --convert-ttkernel-to-emitc --form-expressions -o %t %s
// RUN: ttmlir-translate --ttkernel-to-cpp -o %t.cpp %t
// RUN: FileCheck %s --input-file=%t.cpp
#l1_ = #ttcore.memory_space<l1>
func.func @kernel_main() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %c42_i32 = arith.constant 42 : i32
    // CHECK: ttmlir::dprint("Hello world, ", v1, "!\n");
    ttkernel.dprint("Hello world, {}!\\n", %c42_i32) : (i32) -> ()
    %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<memref<32x32xf32, #l1_>>
    // CHECK: ttmlir::CBPrinter
    ttkernel.dprint("{}", %cb) : (!ttkernel.cb<memref<32x32xf32, #l1_>>) -> ()
    func.return
}
