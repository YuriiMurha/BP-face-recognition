import tensorflow as tf
from tensorflow.keras import layers, models


@tf.keras.utils.register_keras_serializable()
class L2NormalizeLayer(layers.Layer):
    """Custom L2 normalization layer for better serialization compatibility."""

    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


def create_embedding_model(
    backbone_type="EfficientNetB0", embedding_dim=128, input_shape=(224, 224, 3)
):
    """
    Creates a feature extractor model for Metric Learning.

    The model consists of a backbone CNN followed by an embedding head
    and an L2-Normalization layer to project embeddings onto a hypersphere.
    """
    if backbone_type == "EfficientNetB0":
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    elif backbone_type == "MobileNetV3Small":
        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_type}")

    # Freeze backbone initially
    base_model.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(embedding_dim, name="embedding_dense"),
            # L2 Normalization ensures Euclidean distance maps to Cosine Similarity
            L2NormalizeLayer(axis=1, name="l2_norm"),
        ]
    )

    return model


if __name__ == "__main__":
    # Quick verification
    model = create_embedding_model(embedding_dim=128)
    model.summary()
    print("\nOutput shape:", model.output_shape)
