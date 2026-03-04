import nox

# Use uv for faster venv creation and package installation
nox.options.default_venv_backend = "uv"


@nox.session
def tests(session):
    """Run full test suite."""
    session.install("-e", ".[dev]")
    # Install setuptools in the venv to fix face_recognition import issue
    session.run("uv", "pip", "install", "setuptools", "face-recognition-models")
    session.run("pytest", "src/bp_face_recognition/tests")


@nox.session
def test_quantization(session):
    """Run quantization tests."""
    session.install("-e", ".[dev]")
    session.run("uv", "pip", "install", "setuptools", "face-recognition-models")
    session.run(
        "pytest", "src/bp_face_recognition/tests/unit/test_quantization.py", "-v"
    )


@nox.session
def test_mediapipe(session):
    """Run MediaPipe performance tests."""
    session.install("-e", ".[dev]")
    session.run("uv", "pip", "install", "setuptools", "face-recognition-models")
    session.run(
        "pytest",
        "src/bp_face_recognition/tests/unit/detectors/test_mediapipe_performance.py",
        "-v",
    )


@nox.session
def test_integration(session):
    """Run integration tests for quantization and MediaPipe."""
    session.install("-e", ".[dev]")
    session.run("uv", "pip", "install", "setuptools", "face-recognition-models")
    session.run(
        "pytest",
        "src/bp_face_recognition/tests/integration/test_quantization_mediapipe.py",
        "-v",
    )


@nox.session
def test_config(session):
    """Run config system tests."""
    session.install("-e", ".[dev]")
    session.run("pytest", "src/bp_face_recognition/tests/unit/test_config.py", "-v")


@nox.session
def test_preprocessing(session):
    """Run preprocessing tests."""
    session.install("-e", ".[dev]")
    session.run("uv", "pip", "install", "setuptools", "face-recognition-models")
    session.run(
        "pytest", "src/bp_face_recognition/tests/unit/test_preprocessing.py", "-v"
    )


@nox.session
def test_training(session):
    """Run training pipeline tests."""
    session.install("-e", ".[dev]")
    session.run("pytest", "src/bp_face_recognition/tests/unit/test_training.py", "-v")


@nox.session
def test_camera(session):
    """Run camera source tests."""
    session.install("-e", ".[dev]")
    session.run(
        "pytest", "src/bp_face_recognition/tests/unit/test_camera_source.py", "-v"
    )


@nox.session
def test_camera_integration(session):
    """Run camera integration tests (requires camera hardware)."""
    session.install("-e", ".[dev]")
    session.run(
        "pytest",
        "src/bp_face_recognition/tests/integration/test_camera_stream.py::TestCameraIntegrationCI",
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
