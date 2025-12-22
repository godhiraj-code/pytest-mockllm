# Contributing to pytest-mockllm

First off, thank you for considering contributing to pytest-mockllm! 🎉

## How Can I Contribute?

### 🐛 Reporting Bugs

1. **Check existing issues** to see if the bug has already been reported
2. **Create a new issue** with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version, OS, and package versions

### 💡 Suggesting Features

1. **Open a discussion** first for major features
2. **Create an issue** with the `enhancement` label
3. Explain the use case and why it would benefit others

### 🔧 Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add/update tests
5. Run the test suite: `pytest`
6. Run linting: `ruff check src/`
7. Commit with clear messages: `git commit -m "feat: add amazing feature"`
8. Push and create a PR

## Development Setup

```bash
# Clone your fork
git clone https://github.com/godhiraj-code/pytest-mockllm.git
cd pytest-mockllm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev,all]"

# Install pre-commit hooks
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pytest_mockllm --cov-report=html

# Run specific test file
pytest tests/test_openai.py -v

# Run specific test
pytest tests/test_openai.py::TestOpenAIMock::test_basic_response -v
```

## Code Style

We use:
- **Ruff** for linting and formatting
- **MyPy** for type checking
- **Black-compatible** formatting via Ruff

```bash
# Check linting
ruff check src/

# Auto-fix issues
ruff check src/ --fix

# Type checking
mypy src/
```

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding/updating tests
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

## Adding a New Provider

1. Create `src/pytest_mockllm/providers/newprovider.py`
2. Inherit from `MockLLM` base class
3. Implement required methods:
   - `__enter__` / `__exit__`
   - `_raise_provider_error`
   - Provider-specific response builders
4. Add to `providers/__init__.py`
5. Create fixture in `fixtures.py`
6. Add tests in `tests/test_newprovider.py`
7. Update README with examples

## Questions?

Feel free to open an issue or start a discussion!
