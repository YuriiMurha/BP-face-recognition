# Progress Log

## 10-Oct-24

- [x] Mention OpenCV `CascadeClassifier`
- [x] Find other libraries and solutions
  - [Copilot search](https://copilot.microsoft.com/sl/eC5JSOibCkS)
  - [Comparison_of_deep_learning_software](https://en.wikipedia.org/wiki/Comparison_of_deep_learning_software)
- [x] Theorethical part - Face detection methods (CNN etc.)

---

## 17-Oct-24

- [x] Dataset chapter ✅ 2024-11-07
  - [x] vytvorenie svojho datasetu ✅ 2024-11-07
  - [ ] ~~použitije custom datasetu pre rôzne metody~~

---

## 07-Nov-24

- [x] vyhodnotenie existujucich riešeni na datasetoch
  - [x] napisanie kodu pre viacero metod
    - `face-recognition`
    - OpenCV Haar Cascades
    - Dlib Histogram of Gradients
    - Keras FaceNet
  - [x] graf alebo tabulka s porovnanim presnosti rozpoznavania tvari poznych reiseni
- [x] doplniť sec cam dataset obrazkami študentov

---

## 12-Dec-24

- [x] Manually label new dataset ✅ 2024-12-12
- [x] Commit code & thesis text to repository

---

## 25-Mar-25

- [X] Evaluate seccam_2
<!-- - [ ] FaceNet -->
- Implement deep learning:
  1. [x] Create NN
  2. [x] Train NN
  3. [x] Test NN

## End of April

pouzit jednu najlepšiu metodu na detekciu, potom vlasnty model na klasifikaciu kazdeho najdeneho človeka, porovnavať s každym človekom v datasete, ak je to novy clovek, tak ho pridame do datasetu

- [x] dat teoreticku cast do LaTeXu
- [x] dopisať teoreticku časť a poslať na review
  - [x] Abstraknty netechnicky "Uvod" s popisom problemu ktory rieši tato praca, motivacia
  - [x] Pridať potom priklady komerčneho použitia ku prehľadu existujucich riešeni a metod

<!-- - [ ] Diagramy porovnavania nemusia byt rozdelene podla subsetov, spomenuť to v texte -->

- [x] čas rozpoznavania pridať ako taubľku

---

## May
- Assets
  - [x] Create and insert system architecture diagram (for thesis and documentation)
  - [x] Add codebase workflow diagram (showing data flow between modules)
  - [x] Add sequence diagram for main application loop
  - [x] Add example screenshots for annotation and augmentation steps
  - [x] Summarize database schema and add example queries to documentation

- [x] dopisať prakticku časť a poslať na review
  - Write about the code in python notebooks
    - [x] Preprocessing, augmentation
    - [x] Deep Learning

- Write documentation
  - [x] User
  - [x] System

---

## Finish

- [x] Rename pictures in _/Files_ folder
- [x] Refactor text in LaTeX

## 10-Jan-26

- [x] Updated `.gemini/.geminiignore` with comprehensive ignore patterns to improve context efficiency.

## 12-Jan-26

- [x] Restructured project folders (moved `data`, `notebooks`, `assets` to root; created `src/models`, `src/utils`, `src/evaluation`, `src/scripts`, `src/database`).
- [x] Updated imports in `src/main.py` and `src/scripts/script.py`.
- [x] Created `IMPLEMENTATION_CONTEXT.md` to document the new project structure and components.
- [x] Refined `.gemini/GEMINI.md` guidelines to include rules for context maintenance and task logging.
- [x] Updated `README.md` to reflect the new project structure and correct file paths.
- [x] Fixed file paths in `src/utils/crop_faces.py`, `src/evaluation/evaluate_methods.py`, `src/evaluation/testing.py`, `notebooks/DeepLearning.ipynb`, and `notebooks/Preprocessing.ipynb` to account for the new project structure.
- [x] Modernized project setup:
    -   Created `pyproject.toml` for `uv`/pip dependency management.
    -   Added `src/config/settings.py` using `pydantic-settings` to replace hardcoded paths.
    -   Created `src/models/interfaces.py` with `FaceDetector` abstract base class.
    -   Refactored all detection methods (`haar_cascade.py`, `dlib_hog.py`, `mtcnn_detector.py`) to implement the new interface.
    -   Decoupled `FaceTracker` and updated `FaceDatabase` and `main.py` to use `settings`.
    -   Introduced a `Makefile` for automated setup, execution, and evaluation.
    -   Updated `evaluate_methods.py` to leverage the new architecture.
- [x] Integrated `tqdm` progress bars in `evaluate_methods.py` and `crop_faces.py` for better UX.
- [x] Reconstructed `src/evaluation/split_datasets.py` into a functional, runnable script with proper logging.

## 13-Jan-26

- [x] Transitioned environment management to `uv` and generated `uv.lock`.
- [x] Implemented `nox` for automated testing, linting, and type checking (`noxfile.py`).
- [x] Refactored core logic from Jupyter notebooks into modular scripts:
    -   `src/data/augmentation.py` (from `Preprocessing.ipynb`)
    -   `src/models/dataset_loader.py` (from `DeepLearning.ipynb`)
- [x] Added unit tests for `Settings`, `FaceDatabase`, and `MTCNNDetector` in `src/tests/`.
- [x] Updated `src/models/methods/mtcnn_detector.py` to support `detect_with_confidence` and integrated it into `FaceTracker`.
- [x] Fixed linting issues across the codebase to ensure `nox -s lint` passes.
- [x] Updated `Makefile` to utilize `uv` and `nox`.
- [x] Documented notebook migration in `notebooks/README.md`.
- [x] **Committed Changes (Local)**:
    -   *chore*: Environment modernization (uv, nox, settings).
    -   *refactor*: Project restructuring and notebook migration.
    -   *docs*: Updated project documentation.
    -   *Note*: Push pending due to remote Dependabot updates.

> Note: Evaluate augmented subsets, not raw images
