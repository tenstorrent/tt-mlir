

# 1.0 StableHLO Pipeline

## 1.1 Passes
1. (inliner) inliner 
2. (apply-tt-argument-annotations)
3. (apply-argument-shard-status)
4. (apply-shardy-argument-annotations)
  - if manual computation op exists, graph is solved. remove all sdy annotations from arguments
  - if not solved graph
    - partially solved graph: if sdy annotations exists (arguments, graph ops, sharding constraint) do nothing
    - if no sdy annotations exists
      - if arguments are annotated with tt.arguments, convert into sdy for batch parallel
      - if arguments are not annotated with tt.arguments, perform automatic analysis for batch parallel
5. (apply-sharding-constraints) apply sharding constraints
6. (aggressive-propagation) sdy propagation
7. (sharding-constraint-to-reshard) sharding constraint to reshards
8. (insert-explicit-reshards) insert explicit reshards
9. (wrap-under-manual-computation) wrap all global ops under manual computation op
10. (reshard-to-collectives) reshard to collectives
11. (update-global-to-local-shapes) update global shapes to local shapes
12. (close-shardings) close shardings
13. (canonicalizer) canonicalizer

After all passes have run, we have a completely solved graph with all ops under a manual computation op and arguments are annotated with whether they are pre-sharded or not. No sdy annotations exist (except for sdy.mesh)

## 1.2 Explanation of passes
### 1.2.1 (inliner)
### 1.2.2. (apply-tt-argument-annotations)
User provides a list of what the argument types are (input, parameter, constant). This information has to come from the frontends. They will be applied to argument attribute dictionary.
### 1.2.3. (apply-argument-shard-status)
Analyze which arguments are pre-sharded and annotate them with attribute (sharded/unsharded). In jax and torch xla, the way you can tell if an argument is pre-sharded in the compiler is if it has a sdy.sharding annotation attached to it.
### 1.2.4. (apply-shardy-argument-annotations)
Apply sdy annotations to arguments. At this stage, we have three types of sdy graphs: solved, partially solved or no analysis has been performend.
a. If manual computation op exists, graph is solved. Remove all sdy annotations from arguments. 
b. If sdy annotations exists (arguments, graph ops, sharding constraint), graph is partially solved. Do nothing.
c. If no sdy annotations exists, no analysis has been performend (this is likely a single chip graph that needs to be automatically parallelized).
    - If arguments are annotated with tt.arguments, convert into sdy for batch parallelization
    - If arguments are not annotated with tt.arguments, perform automatic analysis for batch parallelization
### 1.2.5. (apply-sharding-constraints): EXTERNAL SHARDY PASS
This pass will detect sharding mismatches and apply requires sdy.sharding_constraints into the graph.
### 1.2.6. (aggressive-propagation): EXTERNAL SHARDY PASS
This pass will propagate all shardy shardings using stablehlo op propogation algorithms written by shardy.
### 1.2.7. (sharding-constraint-to-reshard): EXTERNAL SHARDY PASS
This pass will convert all user defined and compiler inserted sharding constraints to reshards
### 1.2.8. (insert-explicit-reshards): EXTERNAL SHARDY PASS
After sharding propagation has occured, there may be situations where shardings don't match. This pass inserts explicit reshards into the graph.
### 1.2.9. (wrap-under-manual-computation)
a. If manual computation op exists, graph is solved. Do nothing.
b. Otherwise, migrate all ops under a manual computation op. At this stage, all tensor shapes are still global shapes. From its definition, a manual computation op defines local tensor shapes for each axis listed in it's manual_axes attribute dictionary.
We do not update the manual_axes of this manual computation op just yet, since we need to insert collectives and that requires global shapes. We need to wrap everything under a manual computation op to allow conversion into ttcore dialect.
### 1.2.10. (reshard-to-collectives): EXTERNAL SHARDY PASS
This pass converts all reshards into a shardy collective operation.
### 1.2.11. (update-global-to-local-shapes)
a. If manual computation op exists, graph is solved. Do nothing.
b. Add the appropriate manual axes to manual computation op and cut all the global tensor shapes to local tensor shapes according to their sdy annotation.
### 1.2.12. (close-shardings): EXTERNAL SHARDY PASS
Close all sdy annotations. This will remove any replicated axis that are still open for sharding.
### 1.2.13. (canonicalizer)

# 2.0 Types of shardy graphs from JAX + Torch XLA
## 2.1 Case 1
Graph was compiled for single chip. We either want to run it on a single chip or parallelize across multi chips.
arguments: no sdy annotations
graph ops: no sdy annotations
manual computation op: false
input is not sharded
output is not sharded

example
'''
func.func public @abs(%arg0: tensor<32x48x24x32xf32>) -> tensor<32x48x24x32xf32> {
  %0 = stablehlo.abs %arg0 : tensor<32x48x24x32xf32>
  return %0 : tensor<32x48x24x32xf32>
}
'''

1. (inliner): no change
2. (apply-tt-argument-annotations)
'''
func.func public @abs(%arg0: tensor<32x48x24x32xf32> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<32x48x24x32xf32> {
  %0 = stablehlo.abs %arg0 : tensor<32x48x24x32xf32>
  return %0 : tensor<32x48x24x32xf32>
}
'''
3. (apply-argument-shard-status)
'''
func.func public @abs(%arg0: tensor<32x48x24x32xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> tensor<32x48x24x32xf32> {
  %0 = stablehlo.abs %arg0 : tensor<32x48x24x32xf32>
  return %0 : tensor<32x48x24x32xf32>
}
'''
4. (apply-shardy-argument-annotations)
This is assuming argument dictionary is provided. If it is not provided, perform automatic analysis to determine how to shard the inputs.
Also, we need to pass the system desc at this stage to determine whether to compile for single chip or multi chip.
Single chip graphs will have 1x1 mesh, multi chip graphs will have 1xn.
'''
sdy.mesh @mesh = <["model"=1, "batch"=2]>
func.func public @abs(%arg0: tensor<32x48x24x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> tensor<32x48x24x32xf32> {
  %0 = stablehlo.abs %arg0 : tensor<32x48x24x32xf32>
  return %0 : tensor<32x48x24x32xf32>
}
'''
5. (apply-sharding-constraints): no change
6. (aggressive-propagation)
'''
sdy.mesh @mesh = <["model"=1, "batch"=2]>
func.func public @abs(%arg0: tensor<32x48x24x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<32x48x24x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch", ?}, {?}, {?}, {?}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>} : tensor<32x48x24x32xf32>
  return %0 : tensor<32x48x24x32xf32>
}
'''
7. (sharding-constraint-to-reshard): no change
8. (insert-explicit-reshards): no change
9. (wrap-under-manual-computation)
'''
sdy.mesh @mesh = <["model"=1, "batch"=2]>
func.func public @abs(%arg0: tensor<32x48x24x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<32x48x24x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch", ?}, {?}, {?}, {?}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>] out_shardings=[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>] manual_axes={} (%arg1: tensor<32x48x24x32xf32>) {
    %1 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>} : tensor<32x48x24x32xf32>
    sdy.return %1 : tensor<32x48x24x32xf32>
  } : (tensor<32x48x24x32xf32>) -> tensor<32x48x24x32xf32>
  return %0 : tensor<32x48x24x32xf32>
}
'''
10. (reshard-to-collectives): no change
11. (update-global-to-local-shapes)
'''
sdy.mesh @mesh = <["model"=1, "batch"=2]>
func.func public @abs(%arg0: tensor<32x48x24x32xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> tensor<32x48x24x32xf32> {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>] out_shardings=[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>] manual_axes={"model", "batch"} (%arg1: tensor<16x48x24x32xf32>) {
    %1 = stablehlo.abs %arg1 : tensor<16x48x24x32xf32>
    sdy.return %1 : tensor<16x48x24x32xf32>
  } : (tensor<32x48x24x32xf32>) -> tensor<32x48x24x32xf32>
  return %0 : tensor<32x48x24x32xf32>
}
'''
12. (close-shardings)
'''
sdy.mesh @mesh = <["model"=1, "batch"=2]>
func.func public @abs(%arg0: tensor<32x48x24x32xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> tensor<32x48x24x32xf32> {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>] manual_axes={"model", "batch"} (%arg1: tensor<16x48x24x32xf32>) {
    %1 = stablehlo.abs %arg1 : tensor<16x48x24x32xf32>
    sdy.return %1 : tensor<16x48x24x32xf32>
  } : (tensor<32x48x24x32xf32>) -> tensor<32x48x24x32xf32>
  return %0 : tensor<32x48x24x32xf32>
}
'''
13. (canonicalizer): no change

## 2.2 Case 2
This is case where graph is already solved.
arguments: no sdy annotations
graph ops: no sdy annotations
manual computation op: true
input is not sharded
output is not sharded

example
'''
sdy.mesh @mesh = <["x"=2, "y"=4]>
func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"x"}, {}, {"y"}]>] out_shardings=[<@mesh, [{}, {"x"}, {}, {"y"}]>] manual_axes={"y", "x"} (%arg1: tensor<1x512x128x256xf32>) {
    %1 = stablehlo.negate %arg1 : tensor<1x512x128x256xf32>
    sdy.return %1 : tensor<1x512x128x256xf32>
  } : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
  return %0 : tensor<1x1024x128x1024xf32>
}
'''

1. (inliner): no change
2. (apply-tt-argument-annotations)
'''
sdy.mesh @mesh = <["x"=2, "y"=4]>
func.func public @main(%arg0: tensor<1x1024x128x1024xf32> {ttcore.argument_type = #ttcore.argument_type<input>}) -> (tensor<1x1024x128x1024xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"x"}, {}, {"y"}]>] out_shardings=[<@mesh, [{}, {"x"}, {}, {"y"}]>] manual_axes={"y", "x"} (%arg1: tensor<1x512x128x256xf32>) {
    %1 = stablehlo.negate %arg1 : tensor<1x512x128x256xf32>
    sdy.return %1 : tensor<1x512x128x256xf32>
  } : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
  return %0 : tensor<1x1024x128x1024xf32>
}
'''
3. (apply-argument-shard-status)
'''
sdy.mesh @mesh = <["x"=2, "y"=4]>
func.func public @main(%arg0: tensor<1x1024x128x1024xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<1x1024x128x1024xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"x"}, {}, {"y"}]>] out_shardings=[<@mesh, [{}, {"x"}, {}, {"y"}]>] manual_axes={"y", "x"} (%arg1: tensor<1x512x128x256xf32>) {
    %1 = stablehlo.negate %arg1 : tensor<1x512x128x256xf32>
    sdy.return %1 : tensor<1x512x128x256xf32>
  } : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
  return %0 : tensor<1x1024x128x1024xf32>
}
'''
4-13. no change

## 2.3 Case 3
This is the case where the graph is already solved and the inputs are pre-sharded by jax or torch_xla.
arguments: yes sdy annotations
graph ops: no sdy annotations
manual computation op: true
inputs are pre-sharded
outputs will remain sharded

example
'''
sdy.mesh @mesh = <["x"=2, "y"=4]>
func.func public @main(%arg0: tensor<8192x784xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<784x16384xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8192x16384xf32> {jax.result_info = ""}) {
  %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"y"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x", "y"} (%arg2: tensor<4096x196xf32>, %arg3: tensor<196x16384xf32>) {
    %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4096x196xf32>, tensor<196x16384xf32>) -> tensor<4096x16384xf32>
    %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
      %3 = stablehlo.add %arg4, %arg5 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<4096x16384xf32>) -> tensor<4096x16384xf32>
    sdy.return %2 : tensor<4096x16384xf32>
  } : (tensor<8192x784xf32>, tensor<784x16384xf32>) -> tensor<8192x16384xf32>
  return %0 : tensor<8192x16384xf32>
}
'''

1. (inliner): no change
2. (apply-tt-argument-annotations)
'''
sdy.mesh @mesh = <["x"=2, "y"=4]>
func.func public @main(%arg0: tensor<8192x784xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>, ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<784x16384xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>, ttcore.argument_type = #ttcore.argument_type<parameter>}) -> (tensor<8192x16384xf32> {jax.result_info = ""}) {
  %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"y"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x", "y"} (%arg2: tensor<4096x196xf32>, %arg3: tensor<196x16384xf32>) {
    %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4096x196xf32>, tensor<196x16384xf32>) -> tensor<4096x16384xf32>
    %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
      %3 = stablehlo.add %arg4, %arg5 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<4096x16384xf32>) -> tensor<4096x16384xf32>
    sdy.return %2 : tensor<4096x16384xf32>
  } : (tensor<8192x784xf32>, tensor<784x16384xf32>) -> tensor<8192x16384xf32>
  return %0 : tensor<8192x16384xf32>
}
'''
3. (apply-argument-shard-status)
'''
sdy.mesh @mesh = <["x"=2, "y"=4]>
func.func public @main(%arg0: tensor<8192x784xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<sharded>}, %arg1: tensor<784x16384xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>, ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<sharded>}) -> (tensor<8192x16384xf32> {jax.result_info = ""}) {
  %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"y"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x", "y"} (%arg2: tensor<4096x196xf32>, %arg3: tensor<196x16384xf32>) {
    %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4096x196xf32>, tensor<196x16384xf32>) -> tensor<4096x16384xf32>
    %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
      %3 = stablehlo.add %arg4, %arg5 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<4096x16384xf32>) -> tensor<4096x16384xf32>
    sdy.return %2 : tensor<4096x16384xf32>
  } : (tensor<8192x784xf32>, tensor<784x16384xf32>) -> tensor<8192x16384xf32>
  return %0 : tensor<8192x16384xf32>
}
'''
4-13. no change

## 2.4 Case 4
This is the case where the graph is partially solved, and the inputs are pre-sharded. Typically the case from torch_xla/torch ax.
arguments: yes sdy annotations
graph ops: no sdy annotations
manual computation op: false
inputs are pre-sharded
outputs will remain sharded

example
'''
sdy.mesh @mesh = <["x"=1, "batch"=8]>
func.func public @main(%arg0: tensor<1024x2x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}) -> (tensor<2048x1024xf32> {jax.result_info = ""}) {
  %0 = stablehlo.reshape %arg0 : (tensor<1024x2x32x32xf32>) -> tensor<2048x1024xf32>
  return %0 : tensor<2048x1024xf32>
}
'''

1. (inliner): no change
2. (apply-tt-argument-annotations): no change (we could provide them but just for example sake, we didn't)
3. (apply-argument-shard-status)
'''
sdy.mesh @mesh = <["x"=1, "batch"=8]>
func.func public @main(%arg0: tensor<1024x2x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>, ttcore.shard_status = #ttcore.shard_status<sharded>}) -> (tensor<2048x1024xf32> {jax.result_info = ""}) {
  %0 = stablehlo.reshape %arg0 : (tensor<1024x2x32x32xf32>) -> tensor<2048x1024xf32>
  return %0 : tensor<2048x1024xf32>
}
'''
4. (apply-shardy-argument-annotations): no change
5. (apply-sharding-constraints): no change
6. (aggressive-propagation)
'''
sdy.mesh @mesh = <["x"=1, "batch"=8]>
func.func public @main(%arg0: tensor<1024x2x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>, ttcore.shard_status = #ttcore.shard_status<sharded>}) -> (tensor<2048x1024xf32> {jax.result_info = "", sdy.sharding = #sdy.sharding<@mesh, [{"batch", ?}, {?}]>}) {
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}]>]>} : (tensor<1024x2x32x32xf32>) -> tensor<2048x1024xf32>
  return %0 : tensor<2048x1024xf32>
}
'''
7. (sharding-constraint-to-reshard): no change
8. (insert-explicit-reshards): no change
9. (wrap-under-manual-computation)
'''
sdy.mesh @mesh = <["x"=1, "batch"=8]>
func.func public @main(%arg0: tensor<1024x2x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>, ttcore.shard_status = #ttcore.shard_status<sharded>}) -> (tensor<2048x1024xf32> {jax.result_info = "", sdy.sharding = #sdy.sharding<@mesh, [{"batch", ?}, {?}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>] out_shardings=[<@mesh, [{"batch", ?}, {?}]>] manual_axes={} (%arg1: tensor<1024x2x32x32xf32>) {
    %1 = stablehlo.reshape %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}]>]>} : (tensor<1024x2x32x32xf32>) -> tensor<2048x1024xf32>
    sdy.return %1 : tensor<2048x1024xf32>
  } : (tensor<1024x2x32x32xf32>) -> tensor<2048x1024xf32>
  return %0 : tensor<2048x1024xf32>
}
'''
10. (reshard-to-collectives): no change
11. (update-global-to-local-shapes)
'''
sdy.mesh @mesh = <["x"=1, "batch"=8]>
func.func public @main(%arg0: tensor<1024x2x32x32xf32> {ttcore.shard_status = #ttcore.shard_status<sharded>}) -> (tensor<2048x1024xf32> {jax.result_info = ""}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>] out_shardings=[<@mesh, [{"batch", ?}, {?}]>] manual_axes={"x", "batch"} (%arg1: tensor<128x2x32x32xf32>) {
    %1 = stablehlo.reshape %arg1 : (tensor<128x2x32x32xf32>) -> tensor<256x1024xf32>
    sdy.return %1 : tensor<256x1024xf32>
  } : (tensor<1024x2x32x32xf32>) -> tensor<2048x1024xf32>
  return %0 : tensor<2048x1024xf32>
}
'''
12. (close-shardings)
'''
sdy.mesh @mesh = <["x"=1, "batch"=8]>
func.func public @main(%arg0: tensor<1024x2x32x32xf32> {ttcore.shard_status = #ttcore.shard_status<sharded>}) -> (tensor<2048x1024xf32> {jax.result_info = ""}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}]>] manual_axes={"x", "batch"} (%arg1: tensor<128x2x32x32xf32>) {
    %1 = stablehlo.reshape %arg1 : (tensor<128x2x32x32xf32>) -> tensor<256x1024xf32>
    sdy.return %1 : tensor<256x1024xf32>
  } : (tensor<1024x2x32x32xf32>) -> tensor<2048x1024xf32>
  return %0 : tensor<2048x1024xf32>
}
'''
13. (canonicalizer): no change

## 2.5 Case 5
This is the case where the graph is partially solved, and the inputs are pre-sharded. Typically the case from torch_xla/torch ax. There is no fundamental change in strategy from case 4.
arguments: yes sdy annotations
graph ops: yes sdy annotations
manual computation op: false
inputs are pre-sharded
outputs will remain sharded

example
'''
sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
func.func @main(%arg0: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, %arg1: tensor<8192x784xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {"_axis_1"}]>}) -> (tensor<16384x784xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {"_axis_1"}]>}) {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0"}, {"_axis_1"}]>]>} : (tensor<f32>) -> tensor<8192x784xf32>
  %1 = stablehlo.add %arg1, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0"}, {"_axis_1"}]>]>} : tensor<8192x784xf32>
  %2 = "stablehlo.all_gather"(%1) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, use_global_device_ids}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0"}, {"_axis_1"}]>]>} : (tensor<8192x784xf32>) -> tensor<16384x784xf32>
  return %2 : tensor<16384x784xf32>
}
'''

## 2.6 Case 6
There can be other situations where the graph is partially solved, at various steps of the sharding solver stage, in which case the above passes should take care of. For example, we walk through a sharding constraint example which will exercise the external shardy passes.

example
'''
sdy.mesh @mesh = <["x"=1, "y"=2]>
func.func public @main(%arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {}]> : tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}
'''

1. (inliner): no change
2. (apply-tt-argument-annotations): no change (doesn't matter if we include it for this example or not)
3. (apply-argument-shard-status)
'''
sdy.mesh @mesh = <["x"=1, "y"=2]>
func.func public @main(%arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>, ttcore.shard_status = #ttcore.shard_status<sharded>}) -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {}]> : tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}
'''
4. (apply-shardy-argument-annotations): no change
5. (apply-sharding-constraints): no change
6. (aggressive-propagation): no change
7. (sharding-constraint-to-reshard)
'''
sdy.mesh @mesh = <["x"=1, "y"=2]>
func.func public @main(%arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>, ttcore.shard_status = #ttcore.shard_status<sharded>}) -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = sdy.reshard %arg0 <@mesh, [{}, {}]> : tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}
'''
8. (insert-explicit-reshards)
'''
sdy.mesh @mesh = <["x"=1, "y"=2]>
func.func public @main(%arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>, ttcore.shard_status = #ttcore.shard_status<sharded>}) -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = sdy.reshard %arg0 <@mesh, [{}, {}]> : tensor<32x32xf32>
  %1 = sdy.reshard %0 <@mesh, [{"y"}, {"x"}]> : tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}
'''
9. (wrap-under-manual-computation)
'''
sdy.mesh @mesh = <["x"=1, "y"=2]>
func.func public @main(%arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>, ttcore.shard_status = #ttcore.shard_status<sharded>}) -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {"y"}]>] out_shardings=[<@mesh, [{"y"}, {"x"}]>] manual_axes={} (%arg1: tensor<32x32xf32>) {
    %1 = sdy.reshard %arg1 <@mesh, [{}, {}]> : tensor<32x32xf32>
    %2 = sdy.reshard %1 <@mesh, [{"y"}, {"x"}]> : tensor<32x32xf32>
    sdy.return %2 : tensor<32x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}
'''
10. (reshard-to-collectives)
'''
sdy.mesh @mesh = <["x"=1, "y"=2]>
func.func public @main(%arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>, ttcore.shard_status = #ttcore.shard_status<sharded>}) -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {"y"}]>] out_shardings=[<@mesh, [{"y"}, {"x"}]>] manual_axes={} (%arg1: tensor<32x32xf32>) {
    %1 = sdy.all_gather [{"x"}, {"y"}] %arg1 out_sharding=<@mesh, [{}, {}]> : tensor<32x32xf32>
    %2 = sdy.all_slice [{"y"}, {}] %1 out_sharding=<@mesh, [{"y"}, {}]> : tensor<32x32xf32>
    sdy.return %2 : tensor<32x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}
'''
11. (update-global-to-local-shapes)
We don't have conversion for sdy.all_slice yet but sdy.all_gather will convert into stablehlo.all_gather.
12. (close-shardings): no change
13. (canonicalizer): no change

# 3.0 ttcore enums
## 3.1 TTCore_MeshShardDirection
'''
- TTCore_MeshShardDirection_FullToShard
- TTCore_MeshShardDirection_ShardToFull
'''

## 3.2 TTCore_MeshShardType
'''
- TTCore_MeshShardType_Identity: input and output tensors are pre-sharded and no sharding is required
- TTCore_MeshShardType_Replicate: all devices have replicated tensor data
- TTCore_MeshShardType_Devices: all devices have sharded data
'''

## 3.3 TTCore_ArgumentType
'''
- TTCore_ArgumentType_Input
- TTCore_ArgumentType_Parameter
- TTCore_ArgumentType_Constant
- TTCore_ArgumentType_Default
'''

## 3.4 TTCore_ReduceType
'''
- TTCore_ReduceType_Sum
- TTCore_ReduceType_Mean
- TTCore_ReduceType_Max
- TTCore_ReduceType_Min
- TTCore_ReduceType_Std
- TTCore_ReduceType_Var 
'''

## 3.5 TTCore_ShardStatus
'''
- TTCore_ShardStatusType_Sharded
- TTCore_ShardStatusType_Unsharded
'''

# 4.0 ttcore types
## 4.1 TTCore_MeshAttr
'''
- string: name
- array: shape
'''

## 4.2 TTCore_MeshesAttr
'''
- array: TTCore_MeshAttr
'''

## 4.3 TTCore_MeshShardDirectionAttr
'''
- TTCore_MeshShardDirection
'''

## 4.4 TTCore_MeshShardTypeAttr
'''
- TTCore_MeshShardType
'''

## 4.5 TTCore_ReduceTypeAttr
'''
- TTCore_ReduceType
'''

## 4.6 TTCore_TensorMeshAttr
'''
tensor encoding which declares what mesh a tensor lives on
- TTCore_MeshAttr
'''

## 4.7 TTCore_ShardStatusAttr
'''
- TTCore_ShardStatus
'''

##4.8 TTCore_ArgumentTypeAttr
'''
- TTCore_ArgumentType
'''

# 5.0 ttir ops
## 5.1 TTIR_MeshShardOp
'''
- ranked_tensor: input
- ranked_tensor: output
- TTCore_MeshShardTypeAttr
- TTCore_MeshShardDirectionAttr
- array: shard_shape
- array: shard_dims
'''

## 5.2 TTIR_AllGatherOp
'''
- ranked_tensor: input
- ranked_tensor: output
- i32: all_gather_dim
- ui32: cluster_axis
'''

## 5.3 TTIR_AllReduceOp
'''
- ranked_tensor: input
- ranked_tensor: output
- TTCore_ReduceTypeAttr
- ui32: cluster_axis
'''

## 5.4 TTIR_ReduceScatterOp
'''
- ranked_tensor: input
- ranked_tensor: output
- TTCore_ReduceTypeAttr
- i32: scatter_dim
- ui32: cluster_axis
'''

## 5.5 TTIR_CollectivePermuteOp
'''
- ranked_tensor: input
- ranked_tensor: output
- array: source_target_pairs
'''

# 6.0 ttnn ops
## 6.1 TTNN_MeshShardOp
'''
- ranked_tensor: input
- TTNN_Device
- TTCore_MeshShardDirectionAttr
- TTCore_MeshShardTypeAttr
- array: shard_shape
- array: shard_dims
'''

## 6.2 TTNN_AllGatherOp
'''
- ranked_tensor: input
- TTNN_Device
- i32: all_gather_dim
- ui32: cluster_axis
- ui32: num_links
'''

## 6.3 TTNN_ReduceScatterOp
'''
- ranked_tensor: input
- TTNN_Device
- TTCore_ReduceTypeAttr
- i32: scatter_dim
- ui32: cluster_axis
- ui32: num_links
'''

## 6.4 TTNN_AllReduceOp
'''
- ranked_tensor: input
- TTNN_Device
- TTCore_ReduceTypeAttr
- ui32: cluster_axis
- ui32: num_links
'''

## 6.5 TTNN_CollectivePermuteOp
'''
- ranked_tensor: input
- TTNN_Device
- array: source_target_pairs
'''
