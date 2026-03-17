// RUN: ttmlir-opt %s --split-input-file --verify-diagnostics

// ===== SetKeyValueOp =====

// Test: set_key_value must have at least one value
func.func @set_key_value_no_values(%dict: !ttcore.dict) -> () {
  // expected-error @+1 {{'ttcore.set_key_value' op requires at least one value}}
  "ttcore.set_key_value"(%dict) <{key = "empty"}> : (!ttcore.dict) -> ()
  return
}

// -----

// ===== GetKeyValueOp =====

// Test: get_key_value must have at least one result
func.func @get_key_value_no_results(%dict: !ttcore.dict) -> () {
  // expected-error @+1 {{'ttcore.get_key_value' op requires at least one result}}
  ttcore.get_key_value %dict["key"] : (!ttcore.dict) -> ()
  return
}
