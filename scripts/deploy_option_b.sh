#!/bin/bash
# Deploy Option B Model to Production
# This script deploys the best performing FaceNet model (99.15% accuracy)

set -e  # Exit on error

echo "==================================="
echo "Option B Production Deployment"
echo "==================================="

# Configuration
MODEL_NAME="facenet_progressive_v1.0"
MODEL_PATH="src/bp_face_recognition/models/finetuned/${MODEL_NAME}.keras"
DEPLOY_DIR="production_models"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo ""
echo "Step 1: Verifying Model..."
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
echo "✓ Model found: $MODEL_PATH ($MODEL_SIZE)"

echo ""
echo "Step 2: Creating Deployment Directory..."
mkdir -p "$DEPLOY_DIR"
cp "$MODEL_PATH" "$DEPLOY_DIR/${MODEL_NAME}_${TIMESTAMP}.keras"
echo "✓ Model copied to $DEPLOY_DIR/"

echo ""
echo "Step 3: Creating Model Info File..."
cat > "$DEPLOY_DIR/model_info_${TIMESTAMP}.json" << EOF
{
  "model_name": "FaceNet Progressive Unfreezing (Option B)",
  "deployment_timestamp": "$TIMESTAMP",
  "source_path": "$MODEL_PATH",
  "accuracy": 99.15,
  "validation_accuracy": 99.53,
  "training_time": "50 minutes",
  "epochs": 19,
  "model_size_mb": 272,
  "trainable_parameters": "23.6M",
  "phases": [
    {"phase": 1, "layers": "Head only", "lr": "1e-3", "epochs": 5},
    {"phase": 2, "layers": "Top 20%", "lr": "1e-5", "epochs": 5},
    {"phase": 3, "layers": "Top 40%", "lr": "5e-6", "epochs": 5},
    {"phase": 4, "layers": "100%", "lr": "1e-6", "epochs": 4}
  ],
  "input_shape": [160, 160, 3],
  "output_classes": 14,
  "recommended": true
}
EOF
echo "✓ Model info saved"

echo ""
echo "Step 4: Creating Python Deployment Script..."
cat > "$DEPLOY_DIR/deploy_model_${TIMESTAMP}.py" << 'PYEOF'
"""
FaceNet Progressive Model - Production Deployment Script
Model: FaceNet Progressive Unfreezing (Option B)
Accuracy: 99.15%
"""

import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class FaceNetProgressiveRecognizer:
    """
    Production-ready recognizer using FaceNet Progressive model.
    Accuracy: 99.15%
    """
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = Path(__file__).parent / "facenet_progressive_v1.0_*.keras"
            # Find latest model
            import glob
            models = glob.glob(str(model_path))
            if not models:
                raise FileNotFoundError("No model found in deployment directory")
            model_path = models[0]
        
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = self._load_class_names()
        
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Input shape: {self.model.input_shape}")
        print(f"✓ Output classes: {len(self.class_names)}")
    
    def _load_class_names(self):
        """Load class names from training info"""
        # These should match your dataset classes
        return [
            "Stranger_1", "Stranger_10", "Stranger_11", "Stranger_12",
            "Stranger_14", "Stranger_2", "Stranger_3", "Stranger_4",
            "Stranger_5", "Stranger_6", "Stranger_7", "Stranger_8",
            "Stranger_9", "Yurii"
        ]
    
    def preprocess(self, img):
        """
        Preprocess image for FaceNet.
        
        Args:
            img: PIL Image or path to image
            
        Returns:
            Preprocessed image array
        """
        if isinstance(img, (str, Path)):
            img = image.load_img(img, target_size=(160, 160))
        
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = (img_array - 0.5) * 2  # Scale to [-1, 1] (FaceNet standard)
        return np.expand_dims(img_array, axis=0)
    
    def predict(self, img, return_confidence=True):
        """
        Predict identity from image.
        
        Args:
            img: PIL Image or path to image
            return_confidence: Whether to return confidence score
            
        Returns:
            Identity name, or (identity, confidence) if return_confidence=True
        """
        img = self.preprocess(img)
        predictions = self.model.predict(img, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        identity = self.class_names[class_idx]
        
        if return_confidence:
            return identity, confidence
        return identity
    
    def predict_batch(self, img_paths):
        """
        Predict identities for multiple images.
        
        Args:
            img_paths: List of image paths
            
        Returns:
            List of (identity, confidence) tuples
        """
        results = []
        for path in img_paths:
            identity, confidence = self.predict(path)
            results.append((identity, confidence))
        return results
    
    def benchmark(self, img_paths):
        """
        Benchmark inference speed.
        
        Args:
            img_paths: List of image paths
            
        Returns:
            Dictionary with timing statistics
        """
        import time
        
        # Warmup
        if img_paths:
            self.predict(img_paths[0])
        
        # Benchmark
        times = []
        for path in img_paths[:100]:  # Limit to 100 images
            start = time.time()
            self.predict(path)
            times.append(time.time() - start)
        
        return {
            "avg_time_ms": np.mean(times) * 1000,
            "min_time_ms": np.min(times) * 1000,
            "max_time_ms": np.max(times) * 1000,
            "std_time_ms": np.std(times) * 1000,
            "fps": 1.0 / np.mean(times)
        }


def main():
    """Example usage"""
    print("FaceNet Progressive Model - Production Deployment")
    print("=" * 60)
    
    # Initialize
    recognizer = FaceNetProgressiveRecognizer()
    
    # Example prediction
    # identity, confidence = recognizer.predict("path/to/face.jpg")
    # print(f"Identity: {identity}, Confidence: {confidence:.2%}")
    
    print("\n✓ Deployment ready!")
    print("Use: recognizer = FaceNetProgressiveRecognizer()")
    print("     identity, conf = recognizer.predict('image.jpg')")


if __name__ == "__main__":
    main()
PYEOF
chmod +x "$DEPLOY_DIR/deploy_model_${TIMESTAMP}.py"
echo "✓ Deployment script created"

echo ""
echo "Step 5: Creating Quick Start Guide..."
cat > "$DEPLOY_DIR/README_DEPLOYMENT_${TIMESTAMP}.md" << 'EOF'
# FaceNet Progressive Model - Production Deployment

**Model**: FaceNet Progressive Unfreezing (Option B)  
**Accuracy**: 99.15%  
**Deployment Date**: $(date)

## Quick Start

```python
from deploy_model_TIMESTAMP import FaceNetProgressiveRecognizer

# Initialize
recognizer = FaceNetProgressiveRecognizer()

# Predict
identity, confidence = recognizer.predict('face_image.jpg')
print(f"Identity: {identity}, Confidence: {confidence:.2%}")
```

## Files Included

- `facenet_progressive_v1.0_*.keras` - Trained model (272 MB)
- `model_info_*.json` - Model metadata
- `deploy_model_*.py` - Python deployment script
- `README_DEPLOYMENT_*.md` - This file

## Performance

- **Accuracy**: 99.15% on test set
- **Inference Time**: ~50-100ms (CPU), ~10-20ms (GPU)
- **Model Size**: 272 MB (can be quantized to ~68 MB)

## Model Details

- **Architecture**: FaceNet (InceptionResNetV1)
- **Training**: 4-phase progressive unfreezing
- **Input Size**: 160x160 RGB
- **Output**: 14 classes

## Support

For issues or questions, refer to the main project documentation.
EOF
echo "✓ Quick start guide created"

echo ""
echo "==================================="
echo "Deployment Complete!"
echo "==================================="
echo ""
echo "Deployed files in: $DEPLOY_DIR/"
ls -lh "$DEPLOY_DIR/"
echo ""
echo "Next steps:"
echo "  1. Test the deployment: python $DEPLOY_DIR/deploy_model_${TIMESTAMP}.py"
echo "  2. Integrate into your application"
echo "  3. Optional: Quantize model for smaller size"
echo ""
echo "✓ Option B model (99.15% accuracy) ready for production!"
