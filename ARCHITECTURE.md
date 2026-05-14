# Architecture

## System Overview

Two recognition paradigms, sharing the same detection infrastructure:

```
                        ┌──────────────────┐
                        │   Camera Frame   │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  Face Detection  │  MediaPipe / MTCNN / Haar / Dlib
                        │  (50% resolution)│
                        └────────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
           ┌────────▼────────┐     ┌──────────▼──────────┐
           │    OPEN-SET     │     │     CLOSED-SET      │
           │  (main.py)      │     │ (closed_set_main.py)│
           └────────┬────────┘     └──────────┬──────────┘
                    │                         │
           ┌────────▼────────┐     ┌──────────▼──────────┐
           │ get_embedding() │     │    recognize()      │
           │ → 512D vector   │     │ → argmax(softmax)   │
           └────────┬────────┘     │ → (identity, conf)  │
                    │              └──────────┬──────────┘
           ┌────────▼────────┐                │
           │ DatabaseService │     ┌──────────▼──────────┐
           │ cosine matching │     │  Confidence check   │
           │ vs registered   │     │  ≥ threshold → ID   │
           │ embeddings      │     │  < threshold → Unkn │
           └────────┬────────┘     └──────────┬──────────┘
                    │                         │
                    └────────────┬─────────────┘
                                │
                       ┌────────▼─────────┐
                       │  Display Result  │
                       │  (OpenCV window) │
                       └──────────────────┘
```

## Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│  Entry Points                                       │
│  main.py (open-set)  │  closed_set_main.py          │
├─────────────────────────────────────────────────────┤
│  Service Layer                                      │
│  PipelineService     │  ClosedSetPipelineService    │
│  DatabaseService     │                              │
├─────────────────────────────────────────────────────┤
│  Vision Core                                        │
│  FaceTracker  │  FaceDetector (ABC)                 │
│               │  FaceRecognizer (ABC)               │
├─────────────────────────────────────────────────────┤
│  Plugin System                                      │
│  ModelRegistry (models.yaml) → RecognizerFactory    │
├──────────────────┬──────────────────────────────────┤
│  Detection       │  Recognition                     │
│  MediaPipe       │  FinetunedRecognizer (TL/PU/TLoss)│
│  MTCNN           │  FaceNetKerasRecognizer          │
│  Haar Cascade    │  DlibRecognizer                  │
│  Dlib HOG        │  KerasMetricRecognizer           │
│  face_recognition│  TFLiteRecognizer                │
├──────────────────┴──────────────────────────────────┤
│  Training                                           │
│  classifier/trainer.py  (softmax, closed-set)       │
│  metric/trainer.py      (triplet loss, open-set)    │
│  finetune/              (FaceNet: TL, PU, TLoss)    │
├─────────────────────────────────────────────────────┤
│  Preprocessing                                      │
│  crop_faces.py → split_lfw.py → augmentation.py     │
├─────────────────────────────────────────────────────┤
│  Config & Settings                                  │
│  config/models.yaml  │  config/settings.py (Pydantic)│
└─────────────────────────────────────────────────────┘
```

## Key Data Flows

### Open-Set Recognition
1. `AttendanceApp.run()` captures camera frame
2. `PipelineService.process_image()` calls `FaceTracker.track_faces()`
3. `FaceTracker` detects at 50% resolution, extracts embeddings at full resolution
4. `DatabaseService.recognize_face()` compares embedding against registered faces via cosine similarity
5. Returns `(identity, confidence)` if similarity ≥ threshold, else "Unknown"

### Closed-Set Recognition
1. `ClosedSetApp.run()` captures camera frame
2. `ClosedSetPipelineService.process_image()` detects faces at 50% resolution
3. For each face, calls `FinetunedRecognizer.recognize(face_crop)` directly
4. Model outputs softmax probabilities over 14 classes → argmax → identity
5. Returns identity if confidence ≥ threshold, else "Unknown"

### Registration (Open-Set only)
1. `register_from_camera.py` captures 10+ face samples
2. `PipelineService.register_person()` extracts embeddings per sample
3. `DatabaseService.register_person()` stores embeddings in `data/faces.csv`

## Plugin System

All models defined in `config/models.yaml`, loaded dynamically:

```yaml
recognizers:
  facenet_pu:
    class: "vision.recognition.finetuned_recognizer.FinetunedRecognizer"
    model_file: "bp_face_recognition/models/finetuned/facenet_progressive_v1.0.keras"
```

`ModelRegistry` resolves class paths → dynamic import → instantiation.
`RecognizerFactory` wraps registry with legacy name fallback.

## FaceNet Model Architecture

```
Input (160×160×3)
  ↓
FaceNet InceptionResNetV1 (pretrained, 512D output)
  ↓
Dense(256, ReLU)
  ↓
Dropout(0.5)
  ↓
Dense(14, Softmax) → Class probabilities
```

Three fine-tuning strategies:
- **TL** (Transfer Learning): Frozen base, train head only → 92.84%, 4 min
- **PU** (Progressive Unfreezing): 4-phase gradual unfreezing → **99.15%**, 50 min
- **TLoss** (Triplet Loss): Metric learning with margin → 94.63%, 90 min

## Benchmark Summary

### Detection (19 surveillance frames, 26 GT boxes, IoU≥0.5)
| Method | F1 | Precision | Recall | Mean IoU | FPS |
|--------|----|-----------|--------|----------|-----|
| **MTCNN** | **0.706** | 0.720 | 0.692 | 0.693 | 3.5 |
| MediaPipe | 0.250 | 0.667 | 0.154 | 0.720 | 238.3 |
| Haar Cascade | 0.105 | 0.167 | 0.077 | 0.774 | 25.5 |
| Dlib HOG | 0.056 | 0.100 | 0.038 | 0.580 | 6.0 |

### Recognition (1,062 test samples, 14 classes)
| Model | Accuracy | F1 | Size |
|-------|----------|------|------|
| FaceNet TL | 92.84% | 0.928 | 93 MB |
| **FaceNet PU** | **99.15%** | **0.991** | 272 MB |
| FaceNet TLoss | 94.63% | 0.946 | 270 MB |
| EfficientNetB0 | 100%* | 1.0 | 24 MB → 9 MB (quantized) |

\* Different dataset (seccam_2, 15 classes)

### Embedding Quality (512D FaceNet backbone)
| Metric | TL | PU | TLoss |
|---|---|---|---|
| Intra-class L2 | 0.651 | 0.575 | **0.337** |
| Inter-class L2 | 0.866 | 1.092 | **1.254** |
| Silhouette (cos) | 0.111 | **0.320** | 0.170 |
| Separation ratio | 1.330 | 1.901 | **3.724** |

Triplet Loss produces the most geometrically separated embeddings; PU wins classification accuracy.

Run `make thesis-benchmark`, `make detection-eval`, `make embedding-quality`, `make training-curves` to regenerate. Results saved to `results/`.

## File Organization

```
src/bp_face_recognition/
├── main.py                      # Open-set entry point
├── closed_set_main.py           # Closed-set entry point
├── config/settings.py           # Pydantic settings (ROOT_DIR, etc.)
├── services/
│   ├── pipeline_service.py      # Open-set orchestration
│   ├── closed_set_pipeline_service.py  # Closed-set orchestration
│   └── database_service.py      # Embedding storage/matching
├── vision/
│   ├── interfaces.py            # FaceDetector, FaceRecognizer ABCs
│   ├── registry.py              # Config-driven model loading
│   ├── factory.py               # RecognizerFactory (legacy compat)
│   ├── core/face_tracker.py     # Detection + recognition integration
│   ├── detection/               # 5 detector implementations
│   ├── recognition/             # 5+ recognizer implementations
│   └── training/                # 3 training paradigms
├── preprocessing/               # crop → split → augment pipeline
└── models/finetuned/            # Trained model files + dataset_info.json
```
