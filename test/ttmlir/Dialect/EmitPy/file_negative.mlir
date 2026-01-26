// RUN: not ttmlir-opt %s -split-input-file 2>&1 | FileCheck %s

// Test: FileOp without id (parser interprets "{" as attribute dict)
module {
  // CHECK: error: expected '}' in attribute dictionary
  emitpy.file {
    func.func @test() { return }
  }
}

// -----

// Test: FileOp without region
module {
  // CHECK: error: expected '{' to begin a region
  emitpy.file "main"
}

// -----

// Test: FileOp with empty region (no block) (violates SizedRegion constraint)
module {
  // CHECK: error: 'emitpy.file' op region #0 ('bodyRegion') failed to verify constraint: region with 1 blocks
  "emitpy.file"() ({
  }) {id = "main"} : () -> ()
}

// -----

// Test: FileOp with multiple regions (should fail due to SizedRegion<1>)
module {
  // CHECK: error: 'emitpy.file' op requires one region
  "emitpy.file"() ({
    func.func @test1() { return }
  }, {
    func.func @test2() { return }
  }) {id = "main"} : () -> ()
}

// -----

// Test: FileOp with region arguments (violates NoRegionArguments trait)
module {
  // CHECK: error: 'emitpy.file' op region should have no arguments
  "emitpy.file"() ({
  ^bb0(%arg0: !emitpy.opaque<"int">):
    func.func @test() { return }
  }) {id = "main"} : () -> ()
}

// -----

// Test: FileOp with results (should have no results)
module {
  // CHECK: error: 'emitpy.file' op requires zero results
  %0 = "emitpy.file"() ({
    func.func @test() { return }
  }) {id = "main"} : () -> !emitpy.opaque<"int">
}

// -----

// Test: FileOp with operands (should have no operands)
module {
  %arg = arith.constant 0 : i64
  // CHECK: error: 'emitpy.file' op requires zero operands
  "emitpy.file"(%arg) ({
    func.func @test() { return }
  }) {id = "main"} : (i64) -> ()
}

// -----

// Test: FileOp with invalid id type (not a string)
module {
  // CHECK: error: 'emitpy.file' op attribute 'id' failed to satisfy constraint: An Attribute containing a string
  "emitpy.file"() ({
    func.func @test() { return }
  }) {id = 123} : () -> ()
}

// -----

// Test: FileOp with missing id attribute
module {
  // CHECK: error: 'emitpy.file' op requires attribute 'id'
  "emitpy.file"() ({
    func.func @test() { return }
  }) : () -> ()
}

// -----

// Test: FileOp with wrong attribute name
module {
  // CHECK: error: 'emitpy.file' op requires attribute 'id'
  "emitpy.file"() ({
    func.func @test() { return }
  }) {name = "main"} : () -> ()
}
