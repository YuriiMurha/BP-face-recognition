.PHONY: setup install run evaluate lint clean test quantize test-quantization test-mediapipe test-all benchmark train train-comparison validate-training

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
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) experiments/train.py $(args)

train-comparison:
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) scripts/run_training_comparison.py

validate-training:
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) scripts/validate_training_setup.py

sample-uncertain:
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) src/scripts/active_learning_sampler.py --input $(input) $(if $(lower),--lower $(lower),) $(if $(upper),--upper $(upper),)

train:
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) src/bp_face_recognition/vision/training/production_trainer.py $(args)

lint:
	uv run nox -s lint

type-check:
	uv run nox -s type_check

clean:
	if exist .nox rmdir /s /q .nox
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

test:
	uv run nox -s tests

test-quantization:
	uv run nox -s test_quantization

test-mediapipe:
	uv run nox -s test_mediapipe

test-integration:
	uv run nox -s test_integration

test-all:
	uv run nox -s test_quantization test_mediapipe test_integration tests

benchmark-tests:
	uv run python -m pytest src/bp_face_recognition/tests/unit/test_quantization.py src/bp_face_recognition/tests/integration/test_mediapipe_real.py

quantize:
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) src/scripts/quantize_model.py --model "$(model)" --type "$(type)" $(if $(dataset),--dataset $(dataset),) $(if $(output),--output $(output),)

benchmark:
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) src/benchmark_quantization_mediapipe.py

benchmark-model:
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) scripts/benchmark_models.py --action test

benchmark-report:
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) scripts/benchmark_models.py --action report

benchmark-compare:
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) scripts/benchmark_models.py --action compare
