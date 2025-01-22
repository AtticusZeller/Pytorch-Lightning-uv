#!/bin/sh -e
set -x

ruff check src tests --fix
ruff format src tests
