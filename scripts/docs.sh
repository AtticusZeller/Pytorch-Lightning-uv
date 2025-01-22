#!/usr/bin/env bash

#  run `uv sync --group docs`` first
if [ "$1" == "dev" ]; then
    uv run mkdocs serve
elif [ "$1" == "deploy" ]; then
    uv run mkdocs gh-deploy --force
elif [ "$1" == "build" ]; then
    uv run mkdocs build
else
    echo "usage: $0 {dev|deploy|build}"
    exit 1
fi
