# Jupyter Notebooks

These notebooks were used for initial exploration, data visualization, and prototyping.

**Note:** The core logic from these notebooks has been refactored into modular Python scripts in the `src/` directory to support better maintainability, testing, and automation.

## Migration Mapping

- **Preprocessing.ipynb** -> `src/data/augmentation.py` (Data augmentation) & `src/utils/crop_faces.py` (Cropping)
- **DeepLearning.ipynb** -> `src/models/dataset_loader.py` (Dataset loading) & `src/models/model.py` (Model definition/training logic)
- **FaceDetection.ipynb** -> `src/evaluation/detection_methods.py` & `src/models/methods/mtcnn_detector.py`

Please refer to the scripts in `src/` for the production-ready code. These notebooks remain for reference and experimental visualization.
