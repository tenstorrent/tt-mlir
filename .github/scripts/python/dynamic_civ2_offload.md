# Dynamic CIv2 Offload

It is possible to configure offloading from CIv1 (still default) to CIv2 by setting the repository variable `CIV2_OFFLOAD_CONFIG` to a JSON offload configuration.

## JSON Offload Configuration

The JSON file contains an array of rules. Each rule defines:
- **`runs-on`**: Specifies the machine label to offload to CIv2
- **Conditional logic**: Either unconditional scope or conditional configs

There are two configuration approaches:

1. **Unconditional**: Define only a `scope` field to always offload the specified scope to CIv2
2. **Conditional**: Define a `configs` (or `config`) array with `at` conditions and corresponding `scope` fields

### At Condition

The `at` field defines UTC hours when the rule applies. Supported formats:

- `<hour>`: Specific hour (e.g., `14` for 2 PM UTC)
- `-<hour>`: From midnight to specified hour inclusive (e.g., `-8` for midnight to 8 AM)
- `<hour>-`: From specified hour to midnight inclusive (e.g., `16-` for 4 PM to midnight)
- `<hour>-<hour>`: Between specified hours inclusive (e.g., `9-17` for 9 AM to 5 PM)
- `default`: [See default rule](#default-rule)

### Scope

Scope defines how many machines from the specified label to offload to CIv2:

```
[[-][all|half|random]][+/-<number>]
```

**Behavior**:
- Tasks are sorted by duration (ascending: shortest first)
- Positive numbers select shortest duration tasks
- Negative numbers select longest duration tasks
- `all`: All tasks
- `half`: Half of tasks (randomly rounds up/down for odd numbers)
- `random`: Random number from 0 to total tasks
- Numbers can be added (`+`) or subtracted (`-`) from the base scope, or
- Single number can be specified as scope

**Examples**:
- `-all+2`: Offload all jobs except the two shortest
- `-random`: Offload a random number of longest jobs
- `1`: Offload the shortest job
- `half+1`: Offload half plus one shortest jobs

### Default Rule

Setting `"at": "default"` or `"default": true` means the rule applies only if no previous rules in the `configs` array resulted in offloading.

### Break Processing

Setting `"break": true` stops processing subsequent rules in the `configs` array if the current rule results in offloading.

## Example JSON configuration

```json
[
 { "runs-on": "llmbox", "scope": "-all+1" },
 { "runs-on": "p150", "config": [ { "at": "15-20", "scope": "random" } ] },
 { "runs-on": "n300", "config": [ {"at": "15-20", "scope": "-half" } ] },
 { "runs-on": "n150", "config": [ {"at": "15-20", "scope": "-half" } ] }
]
```

This will:
- Offload all except one llmbox
- At peak hours offload random number of p150 jobs and longest half of n150 and n300 jobs.
