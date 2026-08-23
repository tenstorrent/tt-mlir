#!/usr/bin/env bash
# Regenerate the report: prose + generated tables, in reading order.
set -eu
PY=/opt/ttmlir-toolchain/venv/bin/python
$PY ttnn_role_diff.py --root raw --graph 1 --out roles_g1_decode.csv >/dev/null
$PY ttnn_role_diff.py --root raw --graph 1 --base after --new patched --out roles_g1_guard.csv >/dev/null
$PY ttnn_role_diff.py --root raw --graph 0 --out roles_g0_prefill.csv >/dev/null
$PY join_report.py >/dev/null
$PY make_report.py > report_tables.md
$PY device_report.py --dir fleet --out device_matmuls.csv > device_tables.md
$PY guard_report.py > guard_tables.md
cat report_head.md \
    report_device_head.md device_tables.md \
    guard_tables.md \
    report_tables.md \
    report_tail.md > n150-ds-matmul-ab.md
$PY render_html.py >/dev/null
echo "wrote n150-ds-matmul-ab.md ($(wc -l < n150-ds-matmul-ab.md) lines) + html"
