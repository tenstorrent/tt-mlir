#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Script to split tasks with durations into groups where each group's total duration
is as close as possible to a target time.

Usage:
    python group_by_duration.py input.json target_time [--output output.json]

Input JSON format:
[
    {"name": "task1", "duration": 10},
    {"name": "task2", "duration": 15},
    ...
]
"""

import json
import argparse
import sys
from typing import List, Dict, Tuple
from dataclasses import dataclass
from copy import deepcopy
import hashlib


@dataclass
class Task:
    name: str
    duration: float

    def __repr__(self):
        return f"Task({self.name}, {self.duration})"


@dataclass
class Group:
    tasks: List[Task]
    total_duration: float = 0.0

    def __post_init__(self):
        if not hasattr(self, "total_duration") or self.total_duration == 0.0:
            self.total_duration = sum(task.duration for task in self.tasks)

    def add_task(self, task: Task):
        self.tasks.append(task)
        self.total_duration += task.duration

    def remove_task(self, task: Task):
        if task in self.tasks:
            self.tasks.remove(task)
            self.total_duration -= task.duration

    def __repr__(self):
        return f"Group(tasks={len(self.tasks)}, total={self.total_duration:.2f})"


def calculate_hash(data: str) -> str:
    data_hash = hashlib.sha256(data.encode()).hexdigest()
    print(f"Data hash: {data_hash}")
    return data_hash


def load_tasks(filename: str) -> List[Task]:
    """Load tasks from JSON file."""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain an array of objects")

        tasks = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} must be an object")
            if "name" not in item or "duration" not in item:
                raise ValueError(f"Item {i} must have 'name' and 'duration' fields")

            tasks.append(Task(name=item["name"], duration=float(item["duration"])))

        return tasks

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{filename}': {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def first_fit_decreasing(tasks: List[Task], target_time: float) -> List[Group]:
    """
    First Fit Decreasing algorithm for bin packing.
    Sort tasks by duration (descending) and place each task in the first group
    that has enough remaining capacity.
    """
    # Sort tasks by duration in descending order
    sorted_tasks = sorted(tasks, key=lambda x: x.duration, reverse=True)
    groups = []

    for task in sorted_tasks:
        # Find the first group that can accommodate this task
        placed = False
        for group in groups:
            if group.total_duration + task.duration <= target_time:
                group.add_task(task)
                placed = True
                break

        # If no existing group can accommodate, create a new one
        if not placed:
            groups.append(Group(tasks=[task], total_duration=task.duration))

    return groups


def best_fit_decreasing(tasks: List[Task], target_time: float) -> List[Group]:
    """
    Best Fit Decreasing algorithm for bin packing.
    Sort tasks by duration (descending) and place each task in the group
    with the least remaining capacity that can still accommodate it.
    """
    # Sort tasks by duration in descending order
    sorted_tasks = sorted(tasks, key=lambda x: x.duration, reverse=True)
    groups = []

    for task in sorted_tasks:
        # Find the best fitting group (least remaining capacity that can fit the task)
        best_group = None
        best_remaining = float("inf")

        for group in groups:
            remaining = target_time - group.total_duration
            if remaining >= task.duration and remaining < best_remaining:
                best_group = group
                best_remaining = remaining

        if best_group:
            best_group.add_task(task)
        else:
            # Create new group if no existing group can accommodate
            groups.append(Group(tasks=[task], total_duration=task.duration))

    return groups


def optimize_groups(
    groups: List[Group], target_time: float, iterations: int = 100
) -> List[Group]:
    """
    Post-processing optimization: try to move tasks between groups to improve balance.
    """
    best_groups = deepcopy(groups)
    best_variance = calculate_variance(groups, target_time)

    for _ in range(iterations):
        # Try moving tasks between groups
        improved = False
        for i, source_group in enumerate(groups):
            if not source_group.tasks:
                continue

            for task in source_group.tasks[
                :
            ]:  # Copy list to avoid modification during iteration
                for j, target_group in enumerate(groups):
                    if i == j:
                        continue

                    # Calculate new totals if we move the task
                    new_source_total = source_group.total_duration - task.duration
                    new_target_total = target_group.total_duration + task.duration

                    # Only move if it improves the balance and doesn't exceed target
                    if (
                        new_target_total <= target_time
                        and abs(new_target_total - target_time)
                        < abs(target_group.total_duration - target_time)
                        and abs(new_source_total - target_time)
                        <= abs(source_group.total_duration - target_time)
                    ):

                        source_group.remove_task(task)
                        target_group.add_task(task)
                        improved = True
                        break

            if improved:
                break

        if not improved:
            break

        # Check if this configuration is better
        current_variance = calculate_variance(groups, target_time)
        if current_variance < best_variance:
            best_groups = deepcopy(groups)
            best_variance = current_variance

    return best_groups


def calculate_variance(groups: List[Group], target_time: float) -> float:
    """Calculate variance of group totals from target time."""
    if not groups:
        return 0.0

    deviations = [(group.total_duration - target_time) ** 2 for group in groups]
    return sum(deviations) / len(deviations)


def print_summary(groups: List[Group], target_time: float):
    """Print summary of the grouping results."""
    total_tasks = sum(len(group.tasks) for group in groups)
    total_duration = sum(group.total_duration for group in groups)

    print(f"\n=== Grouping Summary ===")
    print(f"Total tasks: {total_tasks}")
    print(f"Total duration: {total_duration:.2f}")
    print(f"Target time per group: {target_time:.2f}")
    print(f"Number of groups: {len(groups)}")
    print(f"Average group duration: {total_duration / len(groups):.2f}")

    # Calculate efficiency metrics
    variance = calculate_variance(groups, target_time)
    max_deviation = max(abs(group.total_duration - target_time) for group in groups)

    print(f"Variance from target: {variance:.2f}")
    print(f"Max deviation from target: {max_deviation:.2f}")

    print(f"\n=== Groups ===")
    for i, group in enumerate(groups, 1):
        deviation = group.total_duration - target_time
        deviation_pct = (deviation / target_time) * 100 if target_time > 0 else 0
        print(
            f"Group {i}: {len(group.tasks)} tasks, duration={group.total_duration:.2f} "
            f"(deviation: {deviation:+.2f}, {deviation_pct:+.1f}%)"
        )

        # Show tasks in group (limit to first few if many)
        task_names = [task.name for task in group.tasks]
        if len(task_names) <= 5:
            print(f"  Tasks: {', '.join(task_names)}")
        else:
            print(
                f"  Tasks: {', '.join(task_names[:5])}, ... and {len(task_names)-5} more"
            )


def save_results(groups: List[Group], filename: str):
    """Save grouping results to JSON file."""
    result = []
    for i, group in enumerate(groups, 1):
        group_data = {
            "group_id": i,
            "total_duration": group.total_duration,
            "task_count": len(group.tasks),
            "tasks": [
                {"name": task.name, "duration": task.duration} for task in group.tasks
            ],
        }
        result.append(group_data)

    with open(filename, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Group tasks by duration to match target time"
    )
    parser.add_argument(
        "input_file", help="JSON file containing tasks with name and duration"
    )
    parser.add_argument(
        "target_time", type=float, help="Target duration for each group"
    )
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    parser.add_argument(
        "--algorithm",
        "-a",
        choices=["first_fit", "best_fit"],
        default="best_fit",
        help="Algorithm to use (default: best_fit)",
    )
    parser.add_argument(
        "--optimize", action="store_true", help="Apply post-processing optimization"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress detailed output"
    )

    args = parser.parse_args()

    # Load tasks
    tasks = load_tasks(args.input_file)

    if not tasks:
        print("Error: No tasks found in input file")
        sys.exit(1)

    if not args.quiet:
        print(f"Loaded {len(tasks)} tasks from {args.input_file}")
        print(f"Target duration per group: {args.target_time}")

    # Group tasks
    if args.algorithm == "first_fit":
        groups = first_fit_decreasing(tasks, args.target_time)
    else:
        groups = best_fit_decreasing(tasks, args.target_time)

    # Apply optimization if requested
    if args.optimize:
        if not args.quiet:
            print("Applying optimization...")
        groups = optimize_groups(groups, args.target_time)

    # Remove empty groups
    groups = [group for group in groups if group.tasks]

    # Print results
    if not args.quiet:
        print_summary(groups, args.target_time)

    # Save results if output file specified
    if args.output:
        save_results(groups, args.output)
    else:
        # Print JSON to stdout if no output file
        result = []
        for i, group in enumerate(groups, 1):
            group_data = {
                "group_id": i,
                "total_duration": group.total_duration,
                "task_count": len(group.tasks),
                "tasks": [
                    {"name": task.name, "duration": task.duration}
                    for task in group.tasks
                ],
            }
            result.append(group_data)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
