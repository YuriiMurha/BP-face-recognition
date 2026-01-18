import nox

# Use uv for faster venv creation and package installation
nox.options.default_venv_backend = "uv"

@nox.session
def tests(session):
    """Run the test suite."""
    session.install("-e", ".[dev]")
    session.run("pytest", "src/bp_face_recognition/tests")

@nox.session
def lint(session):
    """Run linting."""
    session.install("ruff")
    session.run("ruff", "check", ".")

@nox.session
def type_check(session):
    """Run type checking."""
    session.install("-e", ".[dev]")
    # Run mypy on the package source
    session.run("mypy", "src/bp_face_recognition")
