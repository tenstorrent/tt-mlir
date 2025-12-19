#!/usr/bin/env python3
import csv
import sys

def format_nanoseconds(ns):
    return f"{round(ns / 1e3, 3)} us"

def load_csv_to_dicts(filepath):
    data = []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(dict(row))

    return data

d = load_csv_to_dicts(sys.argv[1])
fast_dispatch_overhead = 8000
s = 0
for i in d:
    print(i["DEVICE KERNEL DURATION [ns]"])
    s += int(i["DEVICE KERNEL DURATION [ns]"])

print("total", s, format_nanoseconds(s))
