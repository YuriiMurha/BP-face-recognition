.PHONY: setup install run evaluate lint clean test

# Python executable
PYTHON = python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install uv
	uv sync

install:
	uv sync

run:
	$(PYTHON) src/main.py

evaluate:
	$(PYTHON) src/evaluation/evaluate_methods.py

lint:
	nox -s lint

type-check:
	nox -s type_check

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .nox

test:
	nox -s tests
