# Datasets for Face Recognition

The dataset defines the boundary between what the model has internalized and what it must extrapolate, and in surveillance applications that boundary is unusually hostile: subjects walk past cameras rather than pose for them, illumination drifts with the time of day, and occlusions arrive without warning. A corpus assembled from a single capture regime, a laboratory webcam, a scraped web collection, a synthetic renderer, will produce a model that scores well on its own distribution and fails on every other.

This chapter documents the dataset used in this thesis, from the choice of sources through cropping, splitting, and augmentation to the final characteristics of the training corpus. Three sources are combined: a frontal webcam stream collected specifically for this work, footage from a ceiling-mounted IP camera in a shared office, and an automatically filtered subset of the Labeled Faces in the Wild (LFW) benchmark [CITE: huang2007lfw]. The three sources together cover 48 identities. The headline experiments in Chapter 6 use the 14-identity subset drawn from the webcam and security-camera sources only; the 34-identity LFW subset was retained for early model trials to verify the training pipeline against an established benchmark, but is excluded from the final closed-set comparison. The rest of the chapter explains how the corpus was assembled, what preprocessing was applied, and how the result is partitioned for training and evaluation.

## 5.2 Data Sources

The dataset is deliberately heterogeneous. Each source contributes a capture regime the other two lack, and the union covers the range of conditions a deployed system is expected to encounter. The breakdown is given in Table 4.1.

Counts in the table are post-augmentation totals, roughly two orders of magnitude larger than the raw cropped images.

**Table 4.1: Data sources used in the thesis corpus.**

| Source          | Identities | Augmented images | Capture regime                                   | Role in the experiments                                    |
|-----------------|-----------:|-----------------:|--------------------------------------------------|------------------------------------------------------------|
| Webcam          | 1          | 1,260            | Frontal, controlled indoor lighting              | Closed-set baseline; high-quality reference identity        |
| Security camera | 13         | 5,820            | Ceiling-mounted IP camera, natural poses         | Closed-set surveillance conditions; majority of the 14-class subset |
| LFW (filtered)  | 34         | 2,370            | In-the-wild web photos, mixed conditions         | Early trials to verify the training pipeline; excluded from the headline experiments |
| **Total**       | **48**     | **9,450**        |                                                  |                                                            |
| **14-class subset** | **14** | **7,080** | webcam + seccam | Headline experiments (Chapter 6) |

The webcam subset is small in identity count but dense in samples and is a clean reference point against which degraded captures can be compared. The security camera subset reproduces the conditions the final system is intended for: oblique angles, motion blur, variable illumination, subjects unaware of the camera. LFW [CITE: huang2007lfw] contributes the bulk of the identity count and was used at the beginning of the project to verify that the training and evaluation pipeline reproduced the expected behaviour on a benchmark with well-understood characteristics. Once the pipeline was validated, the headline closed-set comparison in Chapter 6 was confined to the 14-identity custom subset, because that subset matches the deployment domain the thesis targets.

The three sources differ not only in volume but in the type of variability they expose. Webcam images vary along intentional, small axes: slight pose changes, expression, while holding illumination and background constant. Security-camera images vary naturally in every direction at once. LFW images vary across photography sessions, decades, and imaging hardware. Combined, they produce a training signal that is neither uniformly easy nor uniformly hard.

## 5.3 Data Collection

**Webcam subset.** The webcam images were collected with a consumer-grade USB camera at a desk-height setup under consistent indoor lighting. The single subject enrolled here is the author, which makes this subset useful for interactive testing and for grounding qualitative comparisons: any failure on the webcam class is a failure under the easiest possible conditions. Small controlled variations, head tilts within roughly ±15°, minor expression changes, slight distance shifts, were introduced during capture so that the recognizer does not memorize a single pose.

**Security camera subset.** The security camera subset was collected from a ceiling-mounted IP surveillance unit covering a shared office area. Subjects were recorded during ordinary activity: walking through the room, sitting down, moving between desks, rather than posing for the camera. Thirteen identities are represented. Because the camera is mounted at ceiling height with a downward tilt, most faces appear at oblique angles; because subjects move while being recorded, a fraction of frames carry motion blur. These are precisely the distortions that a frontal-only corpus fails to model, and they are the main reason this subset was collected alongside the webcam data.

**LFW subset.** The LFW subset is not a fresh collection but a programmatic selection from a public benchmark. The full LFW distribution contains over thirteen thousand identities, most represented by a handful of images; stratified training splits require a minimum per-class sample count, so identities with fewer than fifty images were automatically excluded. The filter yields 34 identities and 2,370 augmented images, used to verify the training and evaluation pipeline on a well-understood public benchmark before the headline experiments were run on the custom subset. No manual curation of LFW identities was performed beyond the automatic threshold.

**Usage restrictions and retention.** The webcam and security-camera subsets were collected and used exclusively for thesis research; they are not redistributed, and no part of the corpus is uploaded to third-party services or cloud storage. The LFW subset is used under the terms of its original release [CITE: huang2007lfw] as a public academic benchmark. Raw images are retained locally for the duration of the research period so that reported experiments remain reproducible from the unmodified corpus.

## 5.4 Preprocessing

Raw images from three heterogeneous sources cannot be fed directly into a recognition model: they differ in resolution, framing, and orientation, and the training targets require a consistent per-face input. Preprocessing proceeds in three stages: face cropping, dataset splitting, and augmentation, applied in that order so that the split partitions raw images and the augmentation multiplies only the training side.

### 5.4.1 Face Cropping

Facial regions were isolated using LabelMe [CITE: russell2008labelme] to produce per-image bounding boxes. LabelMe is an open-source annotation tool that stores rectangular or polygonal regions as JSON alongside each image; for this work, rectangular regions aligned to the visible face extent were sufficient. LFW images were treated as pre-cropped and bypassed this stage, since the distribution is already face-centred.

After annotation, each bounding box was expanded by 10 % on all sides before the crop was written to disk. The margin is a deliberate choice: a bounding box tight to the face edge removes the hairline, jawline, and ear region, but those peripheral features are discriminative: they differ between identities that may otherwise look similar from a narrow crop. Ten per cent is empirically the point at which context is preserved without admitting enough background to confuse the recognizer; larger margins begin to include other subjects in crowded security-camera frames, and smaller margins degraded recognition accuracy in preliminary trials. Crops were resized to 160 × 160 pixels, the input resolution expected by the FaceNet backbone used downstream, and written as JPEG.

### 5.4.2 Dataset Splitting

Cropped images were partitioned into training, validation, and test sets with a 65 / 20 / 15 ratio (Table 4.2), stratified by identity so that every class appears in every split in its original proportion. The split is driven by a fixed random seed, which guarantees that repeated runs over the same raw corpus produce identical partitions and that reported metrics are reproducible.

**Table 4.2: Split percentages and role.**

| Split      | Fraction | Role                                         | Augmented? |
|------------|---------:|----------------------------------------------|:----------:|
| Training   | 65 %     | Parameter updates                            | Yes (60×)  |
| Validation | 20 %     | Early stopping, hyperparameter selection     | No         |
| Test       | 15 %     | Final held-out evaluation                    | No         |

A lower-than-typical training share (65 % rather than the customary 70–80 %) is chosen deliberately. Augmentation multiplies the training partition by a factor of sixty, so even after reserving a larger fraction for evaluation the number of training examples seen by the model is well into the hundreds of thousands. Reserving 20 % for validation gives a stable early-stopping signal with several tens of images per class, and 15 % for the held-out test set preserves a meaningful final estimate.

Stratification is applied at the raw level, before augmentation. This ordering is what prevents the augmented variants of a single raw image from being scattered across train, validation, and test: a leak that would inflate reported metrics without any gain in real generalization.

### 5.4.3 Data Augmentation

Augmentation was applied to the training split only, using the Albumentations library [CITE: buslaev2020albumentations]. Each raw training image was expanded into a fixed multiplier of augmented variants through random compositions drawn from the transform families in Table 4.3. The validation and test splits were left as raw crops so that reported metrics reflect behaviour on unaugmented, in-distribution images. All image counts elsewhere in this chapter and in Chapter 6 are post-augmentation training counts unless explicitly stated otherwise.

**Table 4.3: Augmentation transform families.**

| Family           | Transforms                                                       | Purpose                                               |
|------------------|------------------------------------------------------------------|-------------------------------------------------------|
| Geometric        | Horizontal flip, rotation (±15°), shift-scale-rotate, small crop | Pose and framing invariance                           |
| Photometric      | Brightness/contrast, gamma, hue-saturation-value shifts          | Robustness to illumination and white balance          |
| Noise and blur   | Gaussian noise, ISO noise, motion blur, Gaussian blur            | Low-light sensor noise, motion artefacts              |
| Occlusion        | Coarse dropout (random rectangular erasures)                     | Partial occlusion from clothing, hands, other objects |
| Compression      | JPEG compression at variable quality                             | Streaming and storage artefacts                       |

The multiplier was chosen empirically against validation accuracy: below a certain factor the augmented pool was too narrow and the model overfit to the raw frames; above that threshold, additional variants yielded diminishing gains and longer training without a corresponding accuracy improvement. The transform families were selected to cover the failure modes observed in the security-camera subset, which supplies most of the difficulty. The transform pipeline is described below; bounding-box coordinates are not tracked alongside the pixel transformations, which is why the ground-truth detection test set in Chapter 6 is reported on raw frames rather than augmented variants.

**Augmentation Pipeline:**

| Transform                  | Parameters                   | Probability | Purpose                        |
| -------------------------- | ---------------------------- | ----------- | ------------------------------ |
| Resize                     | 224×224 px                   | 1.00        | Standardize input for training |
| Horizontal Flip            | —                            | 0.50        | Exploit facial symmetry        |
| Random Brightness Contrast | brightness=0.2, contrast=0.2 | 0.30        | Simulate lighting changes      |
| Random Gamma               | gamma=(80, 120)              | 0.30        | Exposure variation             |
| RGB Shift                  | r=20, g=20, b=20             | 0.20        | White balance simulation       |
| Gauss Noise                | var_limit=(10, 50)           | 0.20        | Sensor noise                   |
| Blur                       | blur_limit=3                 | 0.20        | Motion/defocus blur            |

Each transform is applied independently with its specified probability. An image may receive multiple transforms (e.g., brightness change AND flip AND blur) or none at all. The 224×224 resize standardises the augmented variants written to disk; at training time each variant is downsampled to the 160×160 input expected by the FaceNet backbone.

## 5.5 Dataset Characteristics

The full corpus contains 48 identities. After cropping, splitting, and augmentation the training pool comprises 9,450 augmented images, of which 7,080 belong to the 14-identity webcam and security-camera subset used for the headline closed-set experiments and the remaining 2,370 belong to the LFW subset used in early pipeline verification.

Class balance is imperfect. The distribution is long-tailed: the security-camera identities each contribute on the order of several hundred augmented images, the LFW identities between fifty and a few hundred per class depending on their native LFW frequency, and the single webcam identity contributes 1,260 augmented images, more than any other individual class. Stratified splitting preserves these relative frequencies across train, validation, and test, so the imbalance is visible in every partition rather than concentrated in one. Because the imbalance is structural, different sources intrinsically supply different volumes, it cannot be removed by collecting more data, and evaluation therefore uses balanced accuracy and macro-averaged F1 (see Chapter 3) rather than raw accuracy, which would be dominated by the better-represented classes.

A clarification is worth making once explicitly so that the numbers later in this thesis are unambiguous. The image counts reported throughout (1,260, 5,820, 2,370, 7,080, 9,450) are augmented counts: each raw cropped image is expanded into multiple augmented variants and it is those variants that the model sees during training. The validation and test partitions remain unaugmented. When figures in Chapter 6 refer to "training samples", the count is the augmented count; when they refer to evaluation samples, the count is the raw cropped count of the test split. The 14-identity closed-set subset (7,080 augmented training images) is the substrate for every recognition experiment reported in Chapter 6, with cross-validation, embedding-geometry analysis, and verification evaluation all derived from it.

