# TODO

## Priority 1: Git Repository Recovery
- [ ] **Backup**: Manually backup the entire project folder.
- [ ] **Reset**: Run `git reset --mixed origin/main` to undo bloated commits.
- [ ] **Verify Ignore**: Ensure `.gitignore` correctly excludes `data/datasets` and `wheels/`.
- [ ] **Re-Stage**: Selectively add code and config files (`src/`, `pyproject.toml`, etc.).
- [ ] **Commit**: Create a clean commit without large binaries.
- [ ] **Push**: Sync with remote (`git pull --rebase` then `git push`).

## Priority 2: Environment Verification
- [ ] **Resolve dlib**: Investigate `dlib` build failure on Windows.

## Priority 3: Static Analysis & Type Safety
- [ ] **Run Initial Check**: Run `nox -s type_check` to identify baseline errors.
- [ ] **Configure Mypy**: Create configuration (in `pyproject.toml`) to handle missing imports (e.g., `cv2`, `tensorflow`, `mtcnn`).
- [ ] **Fix Type Errors**:
    - [ ] `src/config/`
    - [ ] `src/models/`
    - [ ] `src/database/`
    - [ ] `src/evaluation/`

## Priority 4: Functionality Verification (Smoke Tests) [LATER]
- [ ] **Data Pipeline**: Run `src/data/augmentation.py` to verify path resolution and augmentation logic.
- [ ] **Dataset Loading**: Run `src/models/dataset_loader.py` to ensure datasets load correctly from new paths.
- [ ] **Main Application**: Run `src/main.py` to verify camera access, detector initialization, and database connection.

## Priority 5: Thesis Integration [LATER]
- [ ] **Evaluation Scripts**: Ensure `src/evaluation/evaluate_methods.py` runs and saves results.
- [ ] **Assets**: Verify plots are saved to `assets/plots` for LaTeX integration.