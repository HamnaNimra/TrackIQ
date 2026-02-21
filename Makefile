.PHONY: help install install-test install-dev test test-verbose test-coverage test-fast test-specific clean lint format regression-example

help:
	@echo "TrackIQ Development Commands"
	@echo "================================"
	@echo ""
	@echo "Installation:"
	@echo "  make install          - Install package"
	@echo "  make install-test     - Install with test dependencies"
	@echo "  make install-dev      - Install with dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - Run all tests"
	@echo "  make test-verbose     - Run tests with verbose output"
	@echo "  make test-coverage    - Run tests with coverage report"
	@echo "  make test-fast        - Run tests without capturing output"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             - Run linting (flake8)"
	@echo "  make format           - Format code (black, isort)"
	@echo ""
	@echo "Examples:"
	@echo "  make regression-example - Run regression detection example"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            - Remove test artifacts and cache"

install:
	pip install -e .

install-test:
	pip install -e ".[test]"

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -q

test-verbose:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=autoperfpy --cov=trackiq_core --cov=minicluster --cov=trackiq_compare --cov-report=html --cov-report=term-missing --cov-report=xml
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

test-fast:
	pytest tests/ -s

test-specific:
	@echo "Usage: make test-specific FILE=tests/test_core.py"
	@echo "       make test-specific CLASS=tests/test_core.py::TestRegressionDetector"
	@echo "       make test-specific FUNC=tests/test_core.py::TestRegressionDetector::test_save_and_load_baseline"
	ifdef FILE
		pytest $(FILE) -v
	endif
	ifdef CLASS
		pytest $(CLASS) -v
	endif
	ifdef FUNC
		pytest $(FUNC) -v
	endif

lint:
	flake8 autoperfpy trackiq_core minicluster trackiq_compare tests --max-line-length=120 --ignore=E501,W503 || true

format:
	black autoperfpy trackiq_core minicluster trackiq_compare tests
	isort autoperfpy trackiq_core minicluster trackiq_compare tests

regression-example:
	python examples/regression_detection_example.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .tox
	rm -rf .trackiq/baselines
	@echo "✅ Cleaned up test artifacts"
