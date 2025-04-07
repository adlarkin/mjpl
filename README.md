# mjpl

MuJoCo motion planning library.

## Installation

Clone this repository and run the following from the repository root (`venv` is recommended):

```
pip3 install -e .
```

## Getting started

See the [examples](./examples) folder.

## Development

For local development, install the developer dependencies:
```
pip3 install -e ".[dev]"
```

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.
Unit tests are written via the [unittest](https://docs.python.org/3/library/unittest.html) framework.

To run the linter, formatter, and tests locally:
```bash
# Check for linter errors
ruff check .
# Apply lint fixes
ruff check --fix .

# Check for format errors
ruff format --diff .
# Apply format fixes
ruff format .

# Run unit tests
python3 -m unittest -v
```

[Pre-commit](https://pre-commit.com/) hooks are also available which run the linter, formatter, and unit tests.
To trigger the hooks automatically on commit, install the pre-commit hooks:
```
pre-commit install
```

To trigger the hooks manually:
```
pre-commit run --all-files
```

To bypass installed pre-commit hooks on commit:
```
git commit --no-verify -m "your message"
```

## Other notes

This project is under active development.
APIs may change, and features are still a work in progress.
The [issues](https://github.com/adlarkin/mjpl/issues) labeled `enhancement` track upcoming features to be developed.
