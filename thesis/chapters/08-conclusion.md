# Chapter 8: Conclusion

This thesis presented the design, implementation, and evaluation of a face detection and recognition system optimized for real-time surveillance applications. The work addressed two fundamental computer vision tasks — detecting faces in surveillance camera frames and recognizing identities from detected faces — within the specific constraints of resource-limited edge deployment. This concluding chapter summarizes the key findings, reflects on their implications for the research questions stated in Chapter 1, acknowledges the limitations of the current work, and identifies promising directions for future investigation.

## 8.1 Summary of Contributions and Answers to the Research Questions

The primary contribution of this thesis is a comprehensive empirical comparison of three fine-tuning strategies for adapting pre-trained FaceNet models to domain-specific face recognition tasks. While transfer learning and fine-tuning are well-established practices in deep learning, the systematic comparison of frozen transfer learning, progressive unfreezing, and triplet loss adaptation on a common dataset provides practical guidance for practitioners facing similar domain adaptation challenges.

The experimental results demonstrated that **progressive unfreezing achieved the best recognition accuracy at 99.15%**, significantly outperforming both frozen transfer learning (92.84%) and triplet loss fine-tuning (94.63%). For face detection, MediaPipe BlazeFace provided the best real-time performance at 238.3 frames per second, while MTCNN offered the highest recall at the cost of substantial latency. The combined detection and recognition pipeline provides a practical system for real-time surveillance face recognition with near-perfect accuracy on known identities.

The findings can be mapped directly onto the four research questions stated in Chapter 1.3.

**RQ1** asked which face detection method achieves the best speed–quality trade-off on indoor surveillance frames measured against manually annotated ground truth at IoU = 0.5. The detection benchmark in Chapter 7 answers this with a separation between speed and recall: MediaPipe BlazeFace dominates on speed (238.3 FPS) but pays a steep recall penalty (F1 = 0.250), while MTCNN dominates on quality (F1 = 0.706) at 3.5 FPS. Haar Cascade and dlib HOG fail to occupy useful intermediate points on this curve — they are dominated on both axes. For a real-time deployment the answer is MediaPipe BlazeFace with the recall caveat acknowledged; for offline forensic processing the answer is MTCNN.

**RQ2** asked whether transfer learning of a pre-trained FaceNet model can clear 95% closed-set classification accuracy on a 14-class surveillance dataset of approximately 7,000 augmented images, and which fine-tuning strategy delivers that accuracy most reliably. The answer is yes, with the caveat that the strategy matters: progressive unfreezing reaches 99.15% with F1 = 0.991, well above the 95% target, while frozen transfer learning falls short at 92.84% and triplet loss fine-tuning lands at 94.63%. Progressive unfreezing is therefore identified as the most reliable strategy on a dataset of this size and is selected as the production default.

**RQ3** asked whether triplet-loss fine-tuning produces a more strongly separated embedding space than cross-entropy fine-tuning, and whether the better geometry translates into better open-set verification. The first half of this question is answered affirmatively in Chapter 7.4: triplet loss produces a separation ratio of 3.72 against progressive unfreezing's 1.90, and the smallest intra-class L2 distance (0.337) of the three strategies. The second half is answered negatively for the dataset evaluated here: progressive unfreezing wins every verification metric (lowest EER at 0.090, highest TAR at fixed FAR, highest AUC at 0.971). The qualitative geometric advantage of triplet loss did not translate into operating-point superiority because the verification pairs were drawn from the same closed identity pool that progressive unfreezing was trained on, an effect discussed at length in Chapter 7.4.3.

**RQ4** asked about the robustness of the accuracy figures to random seed variation. The five-seed cross-validation in Chapter 7.4.4 produced a meaningful uncertainty bound around the headline numbers: progressive unfreezing's seed-averaged accuracy is 94.11% with a standard deviation of 0.59 percentage points, against the single-split number of 99.15%. The cross-validated picture is materially less optimistic than the single-split picture, particularly for triplet loss whose seed-to-seed variance is dramatic. The single-split numbers should therefore be read as upper-bound on optimistic seeds, not as expected performance.

## 8.2 Practical Recommendations

For production deployment of a surveillance face recognition system, the experimental results support the following recommendations:

**Maximum accuracy on known identities**: Use Progressive Unfreezing. The 99.15% accuracy with F1-score of 0.991 is near-perfect performance for a 14-class problem. Training time of 50 minutes is acceptable for a one-time model preparation step.

**Rapid prototyping and constrained resources**: Use Transfer Learning. At 92.84% accuracy in 4 minutes of training, it provides a strong baseline with minimal investment. This approach is suitable for evaluating whether the dataset and detection pipeline are sufficient before committing to longer training.

**Balanced accuracy and speed**: Consider running only Phases 1 and 2 of Progressive Unfreezing. Based on the phase results, this would yield approximately 96% accuracy in roughly 25 minutes of training, providing a useful middle ground.

**Open-set recognition with novel identities**: Triplet Loss is the only strategy that produces embeddings directly suitable for open-set recognition without modification. If the deployment scenario requires identifying individuals not seen during training or dynamically registering new identities without retraining, Triplet Loss is the appropriate choice even with lower closed-set classification accuracy, because its embedding geometry is the one most likely to generalize beyond the training population.

**Real-time detection**: Use MediaPipe BlazeFace for production deployment where 30+ FPS is required. Its 238.3 FPS performance provides substantial headroom for the recognition stage while maintaining acceptable recall for frontal faces at typical surveillance distances.

**High-recall detection**: Use MTCNN for offline batch processing or forensic analysis where missing a face is more costly than processing time. Its multi-scale pyramid approach detects faces that single-scale methods miss.

## 8.3 Limitations and Future Work

The current work has several limitations that suggest directions for future research.

**Cross-dataset generalization.** All recognition results are specific to the custom 14-class dataset. Future work should evaluate cross-dataset performance to assess generalization to different cameras, lighting conditions, and demographic distributions.

**In-distribution verification only.** Chapter 7.4.3 reports a quantitative open-set verification evaluation — EER, TAR at fixed FAR operating points, AUC, and ROC and DET curves — but the 5,000 positive and 5,000 negative pairs are sampled from the same 14 identities used during training. This measures *in-distribution* verification rather than true open-set generalization to unseen identities. A larger held-out benchmark with disjoint identities (for example an LFW-style verification protocol) would provide a more rigorous open-set measurement and would more fairly differentiate metric-learning approaches like Triplet Loss from closed-set-trained approaches like Progressive Unfreezing.

**Semi-hard negative mining.** The triplet loss results used random online mining. Implementing semi-hard negative mining as described in the original FaceNet paper [CITE: Schroff et al. 2015] would likely improve triplet loss performance substantially.

**Larger-scale evaluation.** The current dataset of 7,080 augmented images across 14 identities is modest by modern standards. Scaling to hundreds or thousands of identities would test the limits of the progressive unfreezing approach and may reveal different optimal strategies.

**Edge deployment optimization.** While this work focused on training strategies, deployment optimization techniques such as quantization, pruning, and knowledge distillation could further reduce model size and inference latency for resource-constrained edge devices.

**Temporal consistency.** Real surveillance systems process video streams rather than isolated frames. Incorporating temporal consistency constraints — requiring that identities remain stable across consecutive frames — could improve recognition accuracy in video contexts.

## 8.4 Closing Remarks

Face recognition in surveillance contexts presents a unique combination of technical challenges: real-time processing requirements, variable lighting and pose conditions, and the need for high accuracy on specific known identities. This thesis demonstrated that progressive unfreezing of pre-trained FaceNet models provides an effective solution to these challenges, achieving 99.15% accuracy on the 14-identity closed-set evaluation with training times measured in minutes rather than hours.

Several insights emerged from this work that extend beyond the specific implementation. Progressive adaptation beats abrupt change: the success of progressive unfreezing suggests that gradual adaptation strategies may be broadly applicable to domain adaptation problems. Training objective must match evaluation metric: the underperformance of triplet loss on classification accuracy, despite its theoretical appeal for face recognition, highlights the importance of aligning training objectives with evaluation metrics. Class imbalance has nonlinear effects: the substantial improvement on under-represented classes under progressive unfreezing suggests that backbone adaptation may be particularly beneficial for minority classes, with implications for fairness in deployed face recognition systems. And speed and accuracy are not always fungible: the detection results show that no single method dominates both axes, so system designers must make explicit choices about which criterion to prioritize based on deployment context.

The comparison of fine-tuning strategies revealed that the choice of adaptation approach has substantial impact on both accuracy and deployment characteristics. Practitioners should select strategies based on their specific requirements: frozen transfer learning for rapid prototyping, progressive unfreezing for maximum accuracy on known identities, and triplet loss for open-set scenarios requiring dynamic enrollment. As face recognition technology continues to advance, the tension between accuracy, efficiency, and flexibility will remain central to system design. The empirical findings of this thesis provide a foundation for navigating these trade-offs in surveillance applications, contributing to the broader goal of deploying reliable, efficient, and ethically responsible face recognition systems.
