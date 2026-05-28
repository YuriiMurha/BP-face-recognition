# Problem Formulation

The assignment for this thesis defines four tasks: to survey the artificial-intelligence methods used for face detection and recognition, to curate a dataset of real-life images from several cameras, to train and evaluate multiple models and judge their fitness for real deployment, and to document the resulting system. This chapter states each task and summarises how the thesis meets it. The technical detail is deferred to the chapters that follow.

**Survey of detection and recognition methods.** The work reviews the methods that matter for a surveillance setting and implements four detectors --- MediaPipe BlazeFace, MTCNN, the Haar cascade, and the dlib HOG detector --- together with a FaceNet recognition backbone and three strategies for adapting it. The literature review in Chapter 2 and the algorithmic description in Chapter 3 cover this task.

**A real-life multi-camera dataset.** A dataset of 14 identities was collected from a webcam and an indoor security camera under everyday conditions, cropped to faces, and expanded by augmentation to roughly 7,000 training images. A separate set of 19 surveillance frames was hand-annotated with 26 face boxes to give the detection benchmark a ground truth. Chapter 4 documents the dataset.

**Training, evaluation, and a deployment judgement.** Three fine-tuning strategies --- frozen transfer learning, progressive unfreezing, and triplet-loss metric learning --- were trained on the same data and compared on accuracy, embedding quality, and open-set verification. Progressive unfreezing reached 99.2% closed-set accuracy and was chosen as the default. The detector benchmark and the recognition pipeline run at video rate on a CPU, which establishes fitness for real deployment. Chapter 5 describes the implementation and Chapter 6 reports the results.

**Documentation.** The system ships with a user manual and a system manual (Appendices A and B), and a build that regenerates every reported figure and table from the repository.
