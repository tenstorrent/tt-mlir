# Duration-Based Task Grouping Script

This Python script takes a JSON file containing tasks with durations and groups them so that each group's total duration is as close as possible to a specified target time.

## Usage

```bash
python group_by_duration.py input.json target_time [options]
```

### Arguments
- `input.json`: JSON file containing an array of tasks with "name" and "duration" fields
- `target_time`: Target duration for each group (float)

### Options
- `--output`, `-o`: Save results to a JSON file instead of printing to stdout
- `--algorithm`, `-a`: Choose algorithm (`first_fit` or `best_fit`, default: `best_fit`)
- `--optimize`: Apply post-processing optimization to improve grouping
- `--quiet`, `-q`: Suppress detailed output

## Input Format

The input JSON file should contain an array of objects, each with `name` and `duration` fields:

```json
[
    {"name": "task1", "duration": 10.5},
    {"name": "task2", "duration": 8.2},
    {"name": "task3", "duration": 15.7}
]
```

## Output Format

The script outputs groups in JSON format:

```json
[
    {
        "group_id": 1,
        "total_duration": 18.7,
        "task_count": 2,
        "tasks": [
            {"name": "task3", "duration": 15.7},
            {"name": "task2", "duration": 8.2}
        ]
    }
]
```

## Examples

### Basic usage
```bash
python group_by_duration.py sample_tasks.json 20.0
```

### Save to file with optimization
```bash
python group_by_duration.py sample_tasks.json 25.0 --output results.json --optimize
```

### Quiet mode with first-fit algorithm
```bash
python group_by_duration.py sample_tasks.json 30.0 --algorithm first_fit --quiet
```

## Algorithms

- **Best Fit Decreasing**: Sorts tasks by duration (largest first) and places each task in the group with the least remaining capacity that can still fit it
- **First Fit Decreasing**: Sorts tasks by duration (largest first) and places each task in the first group that can accommodate it

The optimization option applies post-processing to try moving tasks between groups to achieve better balance.

## Sample Data

Use `sample_tasks.json` as a test input file containing 15 sample tasks with various durations.
