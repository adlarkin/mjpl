# mjpl

[![Build](https://img.shields.io/github/actions/workflow/status/adlarkin/mjpl/ci.yml)](https://github.com/adlarkin/mjpl/actions)
[![Coverage Status](https://coveralls.io/repos/github/adlarkin/mjpl/badge.svg)](https://coveralls.io/github/adlarkin/mjpl?branch=main)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/adlarkin/mjpl/main.svg)](https://results.pre-commit.ci/latest/github/adlarkin/mjpl/main)
[![PyPI - Version](https://img.shields.io/pypi/v/mjpl)](https://pypi.org/project/mjpl/)

MuJoCo motion planning library.

> [!Note]
> This project is under active development.
> APIs may change, and features are still a work in progress.

Features:
- Joint-space planning via bi-directional RRT, with support for constraints
- Cartesian-space planning
- Interfaces for constraints, inverse kinematics, and trajectory generation

Limitations:
- This library is designed for manipulator models that are composed of hinge/slide joints, and will not work with models that have ball/free joints.

## Installation

```
uv add mjpl
# or
pip install mjpl
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and builds.
For local development, clone the repository and sync the developer dependencies:
```
uv sync --extra dev
```

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.
Unit tests are written via the [unittest](https://docs.python.org/3/library/unittest.html) framework.

To run the linter, formatter, and tests locally:
```bash
# Check for linter errors
uv run ruff check .
# Apply lint fixes
uv run ruff check --fix .

# Check for format errors
uv run ruff format --diff .
# Apply format fixes
uv run ruff format .

# Run unit tests
uv run python -m unittest -v
```

[Pre-commit](https://pre-commit.com/) hooks are also available which run the linter, formatter, and unit tests.
To trigger the hooks automatically on commit, install the pre-commit hooks:
```
uv run pre-commit install
```

To trigger the hooks manually:
```
uv run pre-commit run --all-files
```

To bypass installed pre-commit hooks on commit:
```
git commit --no-verify -m "your message"
```

## Acknowledgements:

Thank you Sebastian Castro for the guidance and support that has been offered throughout the early stages of this project.
If you find this library useful or interesting, consider checking out Sebastian's [pyroboplan](https://github.com/sea-bass/pyroboplan), which offers similar features via [Pinocchio](https://github.com/stack-of-tasks/pinocchio)!
