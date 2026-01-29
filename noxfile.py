import nox

# Use uv for faster venv creation and package installation
nox.options.default_venv_backend = "uv"


@nox.session
def tests(session):
    """Run full test suite."""
    session.install("-e", ".[dev]")
    session.run("pytest", "src/bp_face_recognition/tests")


@nox.session
def test_quantization(session):
    """Run quantization tests."""
    session.install("-e", ".[dev]")
    session.run(
        "pytest", "src/bp_face_recognition/tests/unit/test_quantization.py", "-v"
    )


@nox.session
def test_mediapipe(session):
    """Run MediaPipe performance tests."""
    session.install("-e", ".[dev]")
    session.run(
        "pytest",
        "src/bp_face_recognition/tests/unit/detectors/test_mediapipe_performance.py",
        "-v",
    )


@nox.session
def test_integration(session):
    """Run integration tests for quantization and MediaPipe."""
    session.install("-e", ".[dev]")
    session.run(
        "pytest",
        "src/bp_face_recognition/tests/integration/test_quantization_mediapipe.py",
        "-v",
    )


@nox.session
def test_quick(session):
    """Run quick subset of tests for CI."""
    session.install("-e", ".[dev]")
    session.run(
        "pytest",
        "src/bp_face_recognition/tests/unit/test_quantization.py::TestModelQuantizer::test_quantizer_initialization",
        "src/bp_face_recognition/tests/unit/test_quantization.py::TestModelQuantizer::test_float16_quantization",
        "src/bp_face_recognition/tests/unit/test_quantization.py::TestQuantizationIntegration::test_tflite_recognizer_loads_quantized_model",
        "-v",
    )


@nox.session
def lint(session):
    """Run linting."""
    session.install("ruff")
    session.run("ruff", "check", ".")


@nox.session
def type_check(session):
    """Run type checking."""
    session.install("-e", ".[dev]")
    # Run mypy on package source
    session.run("mypy", "src/bp_face_recognition")
