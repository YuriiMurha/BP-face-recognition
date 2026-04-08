# Custom Face Recognition Model Training Experiments

## Problem Analysis

### Root Cause: Model Collapse
The trained model outputs nearly identical embeddings (~0.999 similarity) for ANY input image, regardless of what's in the picture.

**Evidence:**
- 5 random noise images → similarity between all pairs: 0.999998 - 1.000000
- Database embeddings from 10 different photos of Yurii → similarity: 0.999998 - 0.999999
- Random image vs Yurii database → similarity: 0.999619

### Current Implementation Issues

1. **Training configuration:**
   - `steps_per_epoch = 10` (too few - only 80 images per epoch with batch_size=8)
   - Random triplet selection (no hard negative mining)
   - `margin = 0.2` (may be too small)
   - `learning_rate = 1e-4` (may be too high for fine-tuning)
   - No validation monitoring

2. **Dataset concerns:**
   - User registered only ~3-4 images effectively (low FPS during registration)
   - LFW has 36 identities in train set
   - Need to verify triplet generation is working correctly

---

## Experiment Plan

### Experiment A: Fix Triplet Loss Training

**Goal:** Debug why training collapsed and fix it

**Hypothesis:** Random triplet selection + too few steps + high learning rate = no learning

**Changes:**

| Parameter | Current | Proposed |
|-----------|---------|----------|
| steps_per_epoch | 10 | 50-100 |
| margin | 0.2 | 0.5-1.0 |
| learning_rate | 1e-4 | 1e-5 |
| batch_size | 8 | 16-32 |

**Additional fixes:**
1. Add validation loss monitoring
2. Implement semi-hard negative mining:
   - Select negatives that are harder than positive but within margin
   - This provides better gradient signal
3. Add learning rate scheduling (reduce on plateau)
4. Train for more epochs (50-100)

**Commands:**
```bash
# Train with fixed hyperparameters
make train-metric-wsl epochs=50 datasets="lfw,webcam,seccam" batch_size=16
```

**Success criteria:**
- Embeddings from different faces should have similarity < 0.9
- Validation loss should decrease
- Same person photos should have similarity > 0.95

---

### Experiment B: Contrastive Loss

**Goal:** Try pairwise loss instead of triplet loss

**Theory:** Contrastive loss can be more stable than triplet loss

**Implementation:**
```python
# In loss.py - new contrastive loss
def contrastive_loss(margin=1.0):
    def loss(y_true, y_pred):
        # y_pred = [embedding1, embedding2, label]
        # label = 1 if same, 0 if different
        emb1 = y_pred[0::2]
        emb2 = y_pred[1::2]
        
        distance = tf.sqrt(tf.reduce_sum(tf.square(emb1 - emb2), axis=1))
        same_loss = label * tf.square(distance)
        diff_loss = (1 - label) * tf.square(tf.maximum(margin - distance, 0))
        return tf.reduce_mean(same_loss + diff_loss)
    return loss
```

**Commands:**
```bash
# Train with contrastive loss (after implementing)
make train-contrastive-wsl epochs=50 datasets="lfw,webcam,seccam"
```

---

### Experiment C: Fine-tune Pre-trained Face Model

**Goal:** Use a pre-trained face recognition model and fine-tune

**Option 1: FaceNet (Keras)**
- Pre-trained on VGGFace2
- Already produces discriminative embeddings

**Option 2: InsightFace/ArcFace (via TensorFlow addons or custom)**
- State-of-the-art face recognition

**Option 3: Use as Feature Extractor + Simple Classifier**
- Don't train metric learning at all
- Use pre-trained model to extract features
- Train a simple classifier (SVM, softmax) on top

**Commands:**
```bash
# Option 1: Fine-tune FaceNet
make train-finetune-wsl epochs=20 model=facenet

# Option 3: Feature extraction + classifier
make train-classifier-wsl
```

---

### Experiment D: ArcFace Loss

**Goal:** Implement state-of-the-art angular margin loss

**Theory:** ArcFace adds angular margin in the angle space, more discriminative than Euclidean margin

**Implementation:**
```python
# ArcFace loss implementation
def arcface_loss(embedding_dim, num_classes, margin=0.5, scale=30):
    def loss(y_true, y_pred):
        # y_pred contains (embeddings, labels)
        embeddings, labels = y_pred
        
        # L2 normalize embeddings and weights
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        weights = tf.get_variable("arcface_weights", 
                                  shape=[num_classes, embedding_dim])
        weights = tf.nn.l2_normalize(weights, axis=1)
        
        # Calculate angles
        cos_theta = tf.matmul(embeddings, weights, transpose_b=True)
        cos_theta = tf.clip_by_value(cos_theta, -1.0, 1.0)
        
        # Add angular margin
        theta = tf.acos(cos_theta)
        target_logits = tf.cos(theta + margin)
        
        # One-hot encoding
        one_hot = tf.one_hot(labels, num_classes)
        
        # Final logits with scale
        logits = tf.where(one_hot > 0, target_logits, cos_theta) * scale
        
        # Cross entropy loss
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
    return loss
```

---

## Experiment Execution Order

1. **Experiment A** - Fix triplet loss (quickest to try)
2. **Experiment C** - Fine-tune pre-trained (most likely to work)
3. **Experiment B** - Contrastive loss (if A fails)
4. **Experiment D** - ArcFace (if we want SOTA)

---

## Verification Scripts

After each experiment, run:

```bash
# Test 1: Check embedding diversity
python scripts/test_embedding_diversity.py

# Test 2: Register and recognize
make register name="TestPerson"
make run

# Test 3: Similarity between different people
python scripts/test_recognition.py --person1=Yurii --person2=GeorgeBush
```

---

## Expected Outcomes

| Experiment | Success Criteria | Time |
|------------|------------------|------|
| A: Fixed Triplet | Different faces < 0.85 similarity | 30 min |
| B: Contrastive | Similar to A | 30 min |
| C: Fine-tune | Should work out of box | 20 min |
| D: ArcFace | Best discrimination | 45 min |

---

## Next Steps

1. Start with Experiment A (fix triplet loss)
2. If that fails, move to Experiment C (fine-tune pre-trained)
3. Keep dlib_v1 as fallback

**Immediate action:** Update trainer.py with Experiment A fixes and re-train
