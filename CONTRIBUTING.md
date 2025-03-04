# Contributing

## Setup

1. [Fork the repository on GitHub](https://github.com/gerlero/foamlib/fork)

1. Clone your fork locally

```bash
git clone https://github.com/<your_username>/foamlib.git
```

2. Install the project in editable mode in a virtual environment

```bash
cd foamlib
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Contributing changes via a pull request

1. Create a new branch for your changes

```bash
git checkout -b my-new-feature
```

2. Make your changes

3. Test your changes (see below for instructions)

4. Commit your changes

```bash
git add .
git commit -m "Add some feature"
```

5. Push your changes to your fork

```bash
git push origin my-new-feature
```

6. [Open a pull request on GitHub](https://github.com/gerlero/foamlib/compare)


## Checks

The following checks will be run by the CI pipeline, so it is recommended to run them locally before opening a pull request.

### Testing

Run the tests with:

```bash
pytest
```

### Type checking

Type check the code with:

```bash
mypy
```

### Linting

Lint the code with:

```bash
ruff check
```

### Formatting

Format the code with:

```bash
ruff format
```

### Documentation

Generate the documentation locally with:

```bash
cd docs
make html
```
