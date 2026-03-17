import os
import numpy as np

# Workaround for tf2onnx/numpy compatibility
if not hasattr(np, "bool"):
    np.bool = bool
import tf2onnx
import tensorflow as tf
from mtcnn import MTCNN
from bp_face_recognition.config.settings import settings


def export_mtcnn():
    detector = MTCNN()
    models_dir = settings.MODELS_DIR / "mtcnn_onnx"
    models_dir.mkdir(parents=True, exist_ok=True)

    # MTCNN library has 3 stages: pnet, rnet, onet
    pnet = detector._stages[0].model
    rnet = detector._stages[1].model
    onet = detector._stages[2].model

    # Export P-Net
    spec = (tf.TensorSpec((None, None, None, 3), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(
        pnet, input_signature=spec, output_path=str(models_dir / "pnet.onnx")
    )

    # Export R-Net
    spec = (tf.TensorSpec((None, 24, 24, 3), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(
        rnet, input_signature=spec, output_path=str(models_dir / "rnet.onnx")
    )

    # Export O-Net
    spec = (tf.TensorSpec((None, 48, 48, 3), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(
        onet, input_signature=spec, output_path=str(models_dir / "onet.onnx")
    )

    print(f"MTCNN models exported to {models_dir}")


if __name__ == "__main__":
    export_mtcnn()
