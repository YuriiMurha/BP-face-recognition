.PHONY: setup install run evaluate lint clean test

# Python executable
PYTHON = uv run python
PYTHONPATH = src

setup:
	uv sync

install:
	uv sync

run:
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) src/bp_face_recognition/main.py

evaluate:
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) src/bp_face_recognition/evaluation/evaluate_methods.py

lint:
	uv run nox -s lint

type-check:
	uv run nox -s type_check

clean:
	if exist .nox rmdir /s /q .nox
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

test:
	uv run nox -s tests
