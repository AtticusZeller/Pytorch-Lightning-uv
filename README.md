# Python Template with UV

<div align="center">

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![Versions](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20-green.svg)](https://github.com/atticuszz/python-uv)

[![Test](https://github.com/atticuszz/python-uv/actions/workflows/main.yml/badge.svg)](https://github.com/atticuszz/python-uv/actions/workflows/main.yml)
[![Coverage](https://codecov.io/gh/Atticuszz/python-uv/branch/main/graph/badge.svg?token=YOUR_TOKEN)](https://github.com/atticuszz/python-uv/actions/workflows/main.yml)

<!-- [![Docker](https://github.com/atticuszz/python-uv/actions/workflows/docker.yml/badge.svg)](https://github.com/atticuszz/python-uv/actions/workflows/docker.yml) -->

</div>

## Feature

### CI/CD

1. publish your package to pypi
2. test matrix
3. mkdocs-material

mkdocs deps

```bash
uv sync --group docs
```

### pre-commit

1. basic `pre-commit-hooks`
2. `codespell check`
3. `ruff-pre-commit`

### Lint and Format with Ruff

```toml
[tool.ruff]
# cover and extend the default config in https://docs.astral.sh/ruff/configuration/
extend-exclude = [""]
target-version = "py310"

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "ARG001", # unused arguments in functions
]
ignore = [
    "E501",   # line too long, handled by black
    "B008",   # do not perform function calls in argument defaults
    "W191",   # indentation contains tabs
    "B904",   # Allow raising exceptions without from e, for HTTPException
    "COM819", # Trailing comma prohibited
    "D100",   # Missing docstring in public module(file)
    "D104",   # Missing docstring in public package
    "D203",   # 1 blank line required before class docstring
    "E201",   # Whitespace after '('
    "E202",   # Whitespace before ')'
    "E203",   # Whitespace before ':'
    "E221",   # Multiple spaces before operator
    "E241",   # Multiple spaces after ','
    "E251",   # Unexpected spaces around keyword / parameter equals
    "W291",   # Trailing whitespace
    "W293",   # Blank line contains whitespace
]

isort = { combine-as-imports = true , split-on-trailing-comma = false }

# Avoid trying to fix flake8-bugbear (`B`) violations.
unfixable = ["B"]

[tool.ruff.format]
docstring-code-format = true
skip-magic-trailing-comma = true
```

### Mypy and Pytest

```toml
[tool.pytest.ini_options]
# Set additional command line options for pytest:
# -r: show extra test summary info
# X: show extra info on xfailed tests
# s: don't capture stdout (allow print statements)
# --strict-config: any warnings about configuration are treated as errors
# --strict-markers: treat unregistered markers as errors
addopts = "-rXs --strict-config --strict-markers"
xfail_strict = true         # Treat tests that are marked as xfail but pass as test failures
filterwarnings = ["error"]  # Treat all warnings as errors

[tool.coverage.report]
fail_under = 100
show_missing = true
skip_covered = true
```
todo：codecov ci and replace coverage with it

### git-cliff

We follow a specific format for commit messages to maintain a clear and organized project history.
with `git-cliff` default config

- `feat:` New features or enhancements
- `fix:` Bug fixes
- `doc:` Documentation updates
- `perf:` Performance improvements
- `refactor:` Code refactoring without adding features or fixing bugs
- `style:` Code style changes (formatting, missing semi-colons, etc.)
- `test:` Adding or modifying tests
- `chore:` Routine tasks, maintenance, or tooling changes
- `revert:` Reverting a previous commit
