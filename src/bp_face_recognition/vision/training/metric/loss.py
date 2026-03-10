import tensorflow as tf


def triplet_loss(margin=0.2):
    """
    Standard Triplet Loss implementation.
    L = max(d(a, p) - d(a, n) + margin, 0)
    """

    def loss(y_true, y_pred):
        # y_pred is expected to be a batch of embeddings [anchor, positive, negative] concatenated
        # However, usually we handle this via a custom training loop or a multi-input model.
        # For simpler integration, we'll implement a version that works with batches of
        # [anchor, positive, negative] in sequence.

        # Split the batch into anchor, positive and negative embeddings
        # This assumes the batch was prepared as [A1, P1, N1, A2, P2, N2, ...]
        anchor = y_pred[0::3]
        positive = y_pred[1::3]
        negative = y_pred[2::3]

        # Distance between anchor and positive
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        # Distance between anchor and negative
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

        # Loss
        basic_loss = pos_dist - neg_dist + margin
        return tf.reduce_mean(tf.maximum(basic_loss, 0.0))

    return loss


class TripletModel(tf.keras.Model):
    """
    Wrapper model to handle triplet inputs during training.
    """

    def __init__(self, base_model, margin=0.2):
        super().__init__()
        self.base_model = base_model
        self.margin = margin
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs, training=None, mask=None):
        # If inputs is a tuple/list (Anchor, Positive, Negative), just return Anchor embedding
        # This is mainly for building/tracing the model.
        if isinstance(inputs, (tuple, list)):
            return self.base_model(inputs[0], training=training)
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        # Unpack the data. Expected format: ((a, p, n), labels)
        (anchor, positive, negative), _ = data

        with tf.GradientTape() as tape:
            # Forward pass
            anchor_emb = self.base_model(anchor, training=True)
            positive_emb = self.base_model(positive, training=True)
            negative_emb = self.base_model(negative, training=True)

            # Calculate distance
            pos_dist = tf.reduce_sum(tf.square(anchor_emb - positive_emb), axis=-1)
            neg_dist = tf.reduce_sum(tf.square(anchor_emb - negative_emb), axis=-1)

            # Triplet Loss
            loss = tf.maximum(pos_dist - neg_dist + self.margin, 0.0)

        # Compute gradients
        trainable_vars = self.base_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]
