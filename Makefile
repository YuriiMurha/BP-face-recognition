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

evaluate-recognition:
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) src/bp_face_recognition/evaluation/evaluate_recognition.py $(dataset)

pipeline:
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) src/scripts/update_pipeline.py

init-dataset:
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) src/scripts/init_dataset.py $(name) $(args)

register:
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) src/scripts/register_person.py $(name) $(dir) --db $(db)

train:
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) src/bp_face_recognition/models/train.py --dataset seccam_2 --epochs 10

lint:
	uv run nox -s lint

type-check:
	uv run nox -s type_check

clean:
	if exist .nox rmdir /s /q .nox
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

test:
	uv run nox -s tests
