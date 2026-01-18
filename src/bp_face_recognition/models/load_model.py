import tensorflow as tf
import os

def localization_loss(y_true, yhat, num_objects):
    """
    Compute localization loss for multiple objects, taking into account the actual number of objects.
    
    Args:
        y_true: Ground truth bounding boxes, shape [batch, max_objects, 4]
        yhat: Predicted bounding boxes, shape [batch, max_objects, 4]
        num_objects: Number of actual objects in each image, shape [batch, 1]
    """
    max_objects = tf.shape(y_true)[1]
    
    # Create a mask for valid objects
    object_mask = tf.sequence_mask(tf.squeeze(num_objects), maxlen=max_objects)
    object_mask = tf.cast(object_mask, tf.float32)
    
    # Expand mask for broadcasting with coordinates
    object_mask = tf.expand_dims(object_mask, -1)  # [batch, max_objects, 1]
    
    # Compute coordinate differences
    coord_diff = tf.reduce_sum(tf.square(y_true - yhat) * object_mask, axis=[1, 2])
    
    # Normalize by the number of objects in each image
    num_objects = tf.maximum(tf.cast(num_objects, tf.float32), 1.0)  # Avoid division by zero
    return tf.reduce_mean(coord_diff / num_objects)

def classification_loss(y_true, yhat, num_objects):
    """
    Compute classification loss for multiple objects, considering only valid objects.
    
    Args:
        y_true: Ground truth labels, shape [batch, max_objects]
        yhat: Predicted class probabilities, shape [batch, max_objects, num_classes]
        num_objects: Number of actual objects in each image, shape [batch, 1]
    """
    # Create object mask
    object_mask = tf.sequence_mask(tf.squeeze(num_objects), maxlen=tf.shape(y_true)[1])
    object_mask = tf.cast(object_mask, tf.float32)
    
    # Convert ground truth labels to one-hot
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.where(y_true < 0, 0, y_true)  # Replace -1 padding with 0
    y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(yhat)[-1])
    
    # Compute categorical crossentropy
    loss = tf.keras.losses.categorical_crossentropy(
        y_true_one_hot,
        yhat,
        from_logits=False
    )
    
    # Apply object mask
    loss = loss * object_mask
    
    # Normalize by number of objects
    num_objects = tf.maximum(tf.cast(num_objects, tf.float32), 1.0)
    return tf.reduce_mean(tf.reduce_sum(loss, axis=1) / num_objects)

# Path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(current_dir, 'models', 'facetracker.keras')

# When loading the model, pass a dictionary of custom objects
loaded_model = tf.keras.models.load_model(
    model_save_path, # Or model_save_path_h5
    custom_objects={
        'localization_loss': localization_loss,
        'classification_loss': classification_loss
    }
)

print("Model loaded successfully!")
loaded_model.summary()