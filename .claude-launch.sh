#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
exec claude "$(cat INSTRUCTION.md)"
