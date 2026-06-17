// RUN: ttmlir-opt -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK-LABEL: func.func @test_l1_addr_ptr_default
func.func @test_l1_addr_ptr_default() -> () {
  %temp = arith.constant 262400 : i32
  // CHECK: (i32) -> !ttkernel.l1_addr_ptr
  %ptr = ttkernel.reinterpret_cast(%temp) : (i32) -> !ttkernel.l1_addr_ptr
  return
}

// CHECK-LABEL: func.func @test_l1_addr_ptr_8
func.func @test_l1_addr_ptr_8() -> () {
  %temp = arith.constant 262400 : i32
  // CHECK: (i32) -> !ttkernel.l1_addr_ptr<8>
  %ptr = ttkernel.reinterpret_cast(%temp) : (i32) -> !ttkernel.l1_addr_ptr<8>
  return
}

// CHECK-LABEL: func.func @test_l1_addr_ptr_16
func.func @test_l1_addr_ptr_16() -> () {
  %temp = arith.constant 262400 : i32
  // CHECK: (i32) -> !ttkernel.l1_addr_ptr<16>
  %ptr = ttkernel.reinterpret_cast(%temp) : (i32) -> !ttkernel.l1_addr_ptr<16>
  return
}

// CHECK-LABEL: func.func @test_load_from_l1_i32
func.func @test_load_from_l1_i32() -> () {
  %temp = arith.constant 262400 : i32
  %ptr = ttkernel.reinterpret_cast(%temp) : (i32) -> !ttkernel.l1_addr_ptr
  %offset = arith.constant 0 : i32
  // CHECK: ttkernel.load_from_l1(%{{.*}}, %{{.*}}) : (!ttkernel.l1_addr_ptr, i32) -> i32
  %val = ttkernel.load_from_l1(%ptr, %offset) : (!ttkernel.l1_addr_ptr, i32) -> i32
  return
}

// CHECK-LABEL: func.func @test_load_from_l1_i16
func.func @test_load_from_l1_i16() -> () {
  %temp = arith.constant 262400 : i32
  %ptr = ttkernel.reinterpret_cast(%temp) : (i32) -> !ttkernel.l1_addr_ptr<16>
  %offset = arith.constant 0 : i32
  // CHECK: ttkernel.load_from_l1(%{{.*}}, %{{.*}}) : (!ttkernel.l1_addr_ptr<16>, i32) -> i16
  %val = ttkernel.load_from_l1(%ptr, %offset) : (!ttkernel.l1_addr_ptr<16>, i32) -> i16
  return
}

// CHECK-LABEL: func.func @test_load_from_l1_i8
func.func @test_load_from_l1_i8() -> () {
  %temp = arith.constant 262400 : i32
  %ptr = ttkernel.reinterpret_cast(%temp) : (i32) -> !ttkernel.l1_addr_ptr<8>
  %offset = arith.constant 0 : i32
  // CHECK: ttkernel.load_from_l1(%{{.*}}, %{{.*}}) : (!ttkernel.l1_addr_ptr<8>, i32) -> i8
  %val = ttkernel.load_from_l1(%ptr, %offset) : (!ttkernel.l1_addr_ptr<8>, i32) -> i8
  return
}

// CHECK-LABEL: func.func @test_store_to_l1_i32
func.func @test_store_to_l1_i32() -> () {
  %temp = arith.constant 262400 : i32
  %ptr = ttkernel.reinterpret_cast(%temp) : (i32) -> !ttkernel.l1_addr_ptr
  %offset = arith.constant 0 : i32
  %val = arith.constant 42 : i32
  // CHECK: ttkernel.store_to_l1(%{{.*}}, %{{.*}}, %{{.*}}) : (i32, !ttkernel.l1_addr_ptr, i32) -> ()
  ttkernel.store_to_l1(%val, %ptr, %offset) : (i32, !ttkernel.l1_addr_ptr, i32) -> ()
  return
}

// CHECK-LABEL: func.func @test_store_to_l1_i16
func.func @test_store_to_l1_i16() -> () {
  %temp = arith.constant 262400 : i32
  %ptr = ttkernel.reinterpret_cast(%temp) : (i32) -> !ttkernel.l1_addr_ptr<16>
  %offset = arith.constant 0 : i32
  %val = arith.constant 42 : i16
  // CHECK: ttkernel.store_to_l1(%{{.*}}, %{{.*}}, %{{.*}}) : (i16, !ttkernel.l1_addr_ptr<16>, i32) -> ()
  ttkernel.store_to_l1(%val, %ptr, %offset) : (i16, !ttkernel.l1_addr_ptr<16>, i32) -> ()
  return
}

// CHECK-LABEL: func.func @test_store_to_l1_i8
func.func @test_store_to_l1_i8() -> () {
  %temp = arith.constant 262400 : i32
  %ptr = ttkernel.reinterpret_cast(%temp) : (i32) -> !ttkernel.l1_addr_ptr<8>
  %offset = arith.constant 0 : i32
  %val = arith.constant 42 : i8
  // CHECK: ttkernel.store_to_l1(%{{.*}}, %{{.*}}, %{{.*}}) : (i8, !ttkernel.l1_addr_ptr<8>, i32) -> ()
  ttkernel.store_to_l1(%val, %ptr, %offset) : (i8, !ttkernel.l1_addr_ptr<8>, i32) -> ()
  return
}
