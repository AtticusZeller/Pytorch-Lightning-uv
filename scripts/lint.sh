#!/usr/bin/env bash

set -e
set -x

mypy src                # type check
ruff check src          # linter
ruff format src --check # formatter
