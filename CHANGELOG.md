# Changelog

All notable changes to this project will be documented in this file.

## 0.0.42 - 2025-01-13

### 🚀 Features

- Add dockerfile

### 🐛 Bug Fixes

- Update remote repo user
- Author info

### ⚙️ Miscellaneous Tasks

- Update pre-commit
- *(CI)* Update astral-sh/setup-uv
- Update lock

## 0.0.40 - 2024-11-30

### 📚 Documentation

- Add docs.sh to simplify dev with mkdocs

### CI

- Remove publish to pypi

## 0.0.39 - 2024-11-30

### 🐛 Bug Fixes

- Import python_uv. in tests

### 📚 Documentation

- Update how to install mkdocs  deps
- Update obsidian config

### ⚙️ Miscellaneous Tasks

- *(clean)* Remove classifiers in toml

## 0.0.38 - 2024-11-05

### ⚙️ Miscellaneous Tasks

- *(CHANGELOG)* Remove user after PR number

## 0.0.37 - 2024-10-29

### 🐛 Bug Fixes

- Move docs deps from project.optional-dependencies to docs group
- *(VCS)* Add git-cliff.remote.github to pyproject.toml instead of with remote repo options in cmd

### 📚 Documentation

- Update obsidian plugin

### ⚙️ Miscellaneous Tasks

- *(remove)* Duplicate publish.yml
- Add .ruff_cache/* to `.gitignore`
- Add py.typed in python_uv
- Replace deprecated tool.uv dev-dependencies with dependency-groups
- Pre-commit run
- Remove .obsidian in root dir

## 0.0.36 - 2024-10-27

### 🐛 Bug Fixes

- *(CI)* Uv run git-cliff

## 0.0.35 - 2024-10-27

### 🐛 Bug Fixes

- Test bump

## 0.0.34 - 2024-10-27

### 🐛 Bug Fixes

- Test bump in ci

## 0.0.33 - 2024-10-27

### 🐛 Bug Fixes

- *(CI)* Exclude all hidden folder and file on pre-commit

## 0.0.32 - 2024-10-27

### 🚀 Features

- Package python with uv

### 🐛 Bug Fixes

- *(CI)* Add PAT token to git cliff

### 🚜 Refactor

- Remove src/* -> src/python_uv/* as a package

### ⚙️ Miscellaneous Tasks

- Update vscode plugin python import config

## 0.0.30 - 2024-10-26

### ⚙️ Miscellaneous Tasks

- *(Merge)* Pull request #2 from Atticuszz:test-PR in #2

## 0.0.29 - 2024-10-26

### ⚙️ Miscellaneous Tasks

- Remove commit_preprocessors on issue number

## 0.0.28 - 2024-10-26

### 🚀 Features

- *(CHANGELOG)* Add pr number and author name after commit message via set remote repo for git-cliff

## 0.0.27 - 2024-10-26

### 🐛 Bug Fixes

- Exclude hidden folder and file in pre-commit

### 📚 Documentation

- *(VCS)* Update CHANGELOG.md with link

### ⚙️ Miscellaneous Tasks

- *(VCS)* Add bump version shell
- Fix typo in README.md
- Clean config for git-cliff
- *(CHANGELOG)* Remove "[]" in version number
- *(CHANGELOG)* Remove "[]" in version number and filter duplicate commit message
- Remove duplicate config on git-cliff

## 0.0.23 - 2024-10-26

### 🐛 Bug Fixes

- *(CI)* Body->body_path context in release ci

## 0.0.22 - 2024-10-26

### 🐛 Bug Fixes

- *(CI)* Release without CHANGES.md

## 0.0.21 - 2024-10-26

### 🐛 Bug Fixes

- *(CI)* Release ci can not read latest changes

## 0.0.20 - 2024-10-26

### ⚙️ Miscellaneous Tasks

- Add notes for tool.bumpversion.allow_dirty=true

## 0.0.19 - 2024-10-25

### 🐛 Bug Fixes

- Pyproject.toml

## 0.0.18 - 2024-10-25

### 🐛 Bug Fixes

- Pyproject.toml version

## 0.0.17 - 2024-10-25

### 🐛 Bug Fixes

- Bumpverison files search name
- CHANGELOG.md bump replace result

## 0.0.16 - 2024-10-25

### 🐛 Bug Fixes

- Version in pyproject.toml

### ⚙️ Miscellaneous Tasks

- Add bump version to CHANGELOG.md

## 0.0.15 - 2024-10-25

### 🐛 Bug Fixes

- Bumpversion files

## 0.0.14 - 2024-10-25

### 🐛 Bug Fixes

- Bumpversion.files

## 0.0.13 - 2024-10-25

### ⚙️ Miscellaneous Tasks

- Add bumpversion files

## 0.0.12 - 2024-10-25

### 🐛 Bug Fixes

- Test bump

## 0.0.11 - 2024-10-25

### 🐛 Bug Fixes

- Test

## 0.0.10 - 2024-10-25

### 🐛 Bug Fixes

- *(vcs)* Remove changelog header
- *(CI)* Release-note is not total CHANGELOG.md
- *(CI)* Release only with tags branches
- *(CI)* Release on tags branches
- Test git-changelog
- *(dep)* Remove git-changelog

### ⚙️ Miscellaneous Tasks

- Add git-cliff
- Run pre-commit
- *(dep)* Update changelog
- Update pyproject.toml
- *(CI)* Remove release condition

### Bump

- *(CI)* Fix release body path file
- Fix ci if release conditions
- 0.0.9->0.0.10

## 0.0.9 - 2024-10-23

### 🐛 Bug Fixes

- *(pyproject.toml)* Bug show
- *(pyproject.toml)* Add local dev commitizen

### 📚 Documentation

- Test  docs

### ⚡ Performance

- Test pre

### Bump

- Version 0.0.8 → 0.0.9

## 0.0.8 - 2024-10-23

### 🐛 Bug Fixes

- *(.vscode)* Enable python package import plugin

### Bump

- Version 0.0.7 → 0.0.8

## 0.0.7 - 2024-10-22

### 🐛 Bug Fixes

- *(main.yml)* Release condition is the commit message with bump:

### Bump

- Version 0.0.6 → 0.0.7

## 0.0.6 - 2024-10-22

### 🐛 Bug Fixes

- *(main.ci)* Failed to get tag to release

### 📚 Documentation

- Update obsidian plugins

### Bump

- Version 0.0.5 → 0.0.6

## 0.0.5 - 2024-10-20

### 🐛 Bug Fixes

- *(pyproject.toml)* Test fix
- *(pyproject.toml)* Add all change_type to changelog_pattern
- *(pyproject.toml)* Change_type_map
- Pyproject.toml

### 📚 Documentation

- *(mkdocs.yml)* Update site name and copyright

### Bump

- Version 0.0.4 → 0.0.5

## 0.0.4 - 2024-10-19

### 🐛 Bug Fixes

- *(.obsidian)* Remove obsidian-git plugin

### 📚 Documentation

- *(docs/)* Remove git plugin of .obsidian and update all
- *(README.md)* Update

### 🎨 Styling

- *(.obsidian)* Pre-commit run

### ⚙️ Miscellaneous Tasks

- *(docs.yml,-main.yml)* 1. update ci uses as @main instead of @v_number, 2. add bump and release

### Bump

- Version 0.0.3 → 0.0.4

## 0.0.3 - 2024-10-19

### 🐛 Bug Fixes

- *(pre-commit-config.yaml)* Allow codespell to write changes

### 📚 Documentation

- *(mkdocs.yml)* Remove navigation.expand

### Bump

- Version 0.0.2 → 0.0.3

## 0.0.2 - 2024-10-19

### 🐛 Bug Fixes

- Test commitizen
- *(CHANGELOG.md)* Test run commitizen

### 📚 Documentation

- Initial commit

### Bump

- Version 0.0.1 → 0.0.2

<!-- generated by git-cliff -->
