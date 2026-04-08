# TensorFlow Lite Quantization for Face Recognition Models

## 1. Quantization Techniques Overview

### Dynamic Range Quantization
- **What it does**: 8-bit quantizes weights, activations computed dynamically at inference
- **Benefits**: 4x smaller model size, 2x-3x CPU speedup, minimal accuracy loss
- **Implementation**: `converter.optimizations = [tf.lite.Optimize.DEFAULT]`

### Post-Training Quantization (Full Integer)
- **What it does**: Quantizes both weights and activations to int8 using representative dataset
- **Benefits**: 4x smaller model size, 3x+ speedup on CPU, Edge TPU, microcontrollers
- **Implementation**: Requires `representative_dataset` and `target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]`

### Quantization-Aware Training (QAT)
- **What it does**: Simulates quantization effects during training for better accuracy
- **Benefits**: Best accuracy retention, especially for sensitive models
- **Implementation**: Uses `tensorflow_model_optimization` to annotate layers during training

## 2. int8 vs float16 Trade-offs for Face Recognition

### int8 Quantization
- **Accuracy**: Typically 94-98% of original FP32 accuracy for face recognition
- **Performance**: 3x+ speedup, 4x size reduction
- **Best for**: CPU inference, edge devices, mobile deployment
- **Risk**: Higher accuracy loss for sensitive embedding models

### float16 Quantization
- **Accuracy**: 97-99% of original FP32 accuracy (minimal loss)
- **Performance**: 2x smaller model size, GPU acceleration support
- **Best for**: GPU inference, when accuracy is critical
- **Risk**: Less compression than int8

**Face Recognition Specific**: Face embedding models (like FaceNet) are more sensitive to quantization than classification models due to the need for precise distance calculations.

## 3. Quantizing Keras Models with Custom Layers

### Custom FaceEmbedding Layer Support
```python
import tensorflow as tf
import numpy as np

def convert_custom_model_to_tflite(model_path, output_path):
    # Load the Keras model
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    
    # Create representative dataset for INT8 quantization
    def representative_dataset():
        for _ in range(100):
            data = np.random.random_sample([1, 224, 224, 3]).astype(np.float32)
            yield [data]
    
    # Configure converter for full integer quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # Convert model
    quantized_model = converter.convert()
    
    # Save quantized model
    with open(output_path, 'wb') as f:
        f.write(quantized_model)
    
    return quantized_model
```

### Handling Lambda Layers (FaceNet Specific)
```python
# For FaceNet models with Lambda layers
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TFLite builtin ops
    tf.lite.OpsSet.SELECT_TF_OPS     # Fallback to TensorFlow ops
]
converter.allow_custom_ops = True
```

## 4. Performance Benchmarks

### General Model Performance (Based on Industry Data)
| Quantization Type | Model Size | Speedup | Accuracy Retention |
|-------------------|------------|---------|-------------------|
| FP32 (Baseline)   | 100%       | 1x      | 100%              |
| Float16           | 50%        | 1.5x    | 97-99%            |
| Dynamic Range     | 25%        | 2-3x    | 95-98%            |
| Full Integer      | 25%        | 3-4x    | 94-97%            |

### Face Recognition Specific Considerations
- **FaceNet Models**: INT8 typically retains 95-97% accuracy
- **Custom Embedding Models**: More sensitive, may need QAT for best results
- **Distance-Based Recognition**: Quantization can affect cosine similarity calculations

## 5. Best Practices for Maintaining Accuracy

### Before Quantization
1. **Validate Model Performance**: Establish FP32 baseline accuracy
2. **Check Input Range**: Ensure inputs are properly normalized
3. **Test Representative Data**: Use diverse face images for calibration

### During Quantization
```python
# Use proper representative dataset
def representative_dataset():
    # Use actual face data, not random noise
    for face_batch in face_dataset.take(100):
        yield [face_batch.numpy().astype(np.float32)]

# Consider float16 first for accuracy retention
converter.target_spec.supported_types = [tf.float16]

# For int8, use per-channel quantization for weights
converter.quantized_input_stats = {input_name: (mean, std)}
```

### After Quantization
1. **Validate Embeddings**: Compare distance metrics between original and quantized models
2. **Threshold Adjustment**: Recognition thresholds may need recalibration
3. **A/B Testing**: Test with actual recognition tasks, not just embedding similarity

### Accuracy Recovery Techniques
```python
# Fine-tune quantized model if accuracy drops significantly
import tensorflow_model_optimization as tfmot

# Apply QAT for better accuracy
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
quant_aware_model = quantize_annotate_model(model)

# Re-train for few epochs
quant_aware_model.compile(optimizer='adam', loss='triplet_loss')
quant_aware_model.fit(train_dataset, epochs=5)
```

## 6. Integration with RecognizerFactory Pattern

### TFLite Recognizer Implementation
```python
import tensorflow as tf
import numpy as np
from bp_face_recognition.models.interfaces import FaceRecognizer

class TFLiteFaceRecognizer(FaceRecognizer):
    def __init__(self, model_path: str, use_quantized: bool = True):
        self.model_path = model_path
        self.use_quantized = use_quantized
        self.interpreter = self._load_model()
        
    def _load_model(self):
        """Load TFLite model with proper configuration"""
        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()
        return interpreter
    
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract embedding using TFLite interpreter"""
        # Get input/output details
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        # Preprocess face image
        face_image = self._preprocess(face_image)
        
        # Set input tensor
        self.interpreter.set_tensor(input_details[0]['index'], face_image)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output embedding
        embedding = self.interpreter.get_tensor(output_details[0]['index'])
        return embedding[0]
    
    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for TFLite model"""
        face_image = cv2.resize(face_image, (224, 224))
        
        # Handle quantization input requirements
        if self.use_quantized:
            # Convert to uint8 for quantized models
            face_image = (face_image * 255).astype(np.uint8)
        else:
            face_image = face_image.astype(np.float32)
            
        return np.expand_dims(face_image, axis=0)
```

### Updated RecognizerFactory
```python
class RecognizerFactory:
    @staticmethod
    def get_recognizer(
        recognizer_type: str = "custom", 
        model_path: Optional[str] = None,
        use_quantized: bool = False
    ) -> FaceRecognizer:
        
        if recognizer_type == "facenet":
            return FaceNetRecognizer(model_path=model_path)
            
        if recognizer_type == "custom":
            return CustomFaceRecognizer(model_path=model_path)
            
        if recognizer_type == "tflite":
            return TFLiteFaceRecognizer(
                model_path=model_path, 
                use_quantized=use_quantized
            )
            
        raise ValueError(f"Unknown recognizer type: {recognizer_type}")
```

### Model Conversion Pipeline
```python
def convert_custom_model_pipeline(
    input_model_path: str,
    output_dir: str,
    quantization_type: str = "int8"
):
    """
    Complete pipeline to convert custom model to TFLite
    
    Args:
        input_model_path: Path to .keras model
        output_dir: Directory to save TFLite models
        quantization_type: 'int8', 'float16', or 'dynamic'
    """
    
    # Load original model for comparison
    original_model = tf.keras.models.load_model(input_model_path, compile=False)
    
    # Create different quantized versions
    converters = {
        'float16': create_float16_converter(original_model),
        'dynamic': create_dynamic_range_converter(original_model),
        'int8': create_int8_converter(original_model)
    }
    
    for q_type, converter in converters.items():
        if q_type == quantization_type or quantization_type == "all":
            tflite_model = converter.convert()
            output_path = f"{output_dir}/model_{q_type}.tflite"
            
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"Converted to {q_type}: {output_path}")
    
    # Generate accuracy comparison report
    generate_accuracy_report(input_model_path, output_dir)

def create_int8_converter(model):
    """Create converter with full integer quantization"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = create_representative_dataset()
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    return converter
```

## Implementation Recommendations

1. **Start with Float16**: Minimal accuracy loss, good starting point
2. **Test INT8**: For production deployment, validate with actual face recognition tasks
3. **Consider QAT**: If accuracy drops >3%, implement quantization-aware training
4. **Threshold Recalibration**: Re-tune recognition thresholds after quantization
5. **Comprehensive Testing**: Test with diverse demographics and lighting conditions

This framework provides a practical approach to quantizing the existing custom face recognition model while maintaining accuracy and integrating seamlessly with the current RecognizerFactory pattern.