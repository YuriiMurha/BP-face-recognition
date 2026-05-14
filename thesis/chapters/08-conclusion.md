# Chapter 8: Conclusion

This thesis presented the design, implementation, and evaluation of a face detection and recognition system optimized for real-time surveillance applications. The work addressed two fundamental computer vision tasks—detecting faces in surveillance camera frames and recognizing identities from detected faces—within the specific constraints of resource-limited edge deployment. This concluding chapter summarizes the key findings, reflects on their implications for both research and practice, acknowledges the limitations of the current work, and identifies promising directions for future investigation.

## 8.1 Summary of Contributions

The primary contribution of this thesis is a comprehensive empirical comparison of three fine-tuning strategies for adapting pre-trained FaceNet models to domain-specific face recognition tasks. While transfer learning and fine-tuning are well-established practices in deep learning, the systematic comparison of frozen transfer learning, progressive unfreezing, and triplet loss adaptation on a common dataset provides practical guidance for practitioners facing similar domain adaptation challenges.

The experimental results demonstrated that **progressive unfreezing achieved the best recognition accuracy at 99.15%**, significantly outperforming both frozen transfer learning (92.84%) and triplet loss fine-tuning (94.63%). For face detection, MediaPipe BlazeFace provided the best real-time performance at 325.6 FPS, while MTCNN offered the highest recall at the cost of substantial latency. The combined detection and recognition pipeline provides a practical system for real-time surveillance face recognition with near-perfect accuracy on known identities.

The results confirmed two of the four initial hypotheses: H1 (transfer learning achieves above 90% accuracy) and H2 (progressive unfreezing exceeds 95% accuracy). H3 (triplet loss achieves the highest accuracy) was disproven, yielding the insight that mining strategy is critical to triplet loss success. H4 (progressive unfreezing offers the best accuracy-to-time ratio) was partially confirmed: while Transfer Learning offers better efficiency in terms of accuracy per minute, Progressive Unfreezing achieves the best final accuracy with moderate training time and substantially outperforms the slower Triplet Loss.

## 8.2 Practical Recommendations

For production deployment of a surveillance face recognition system, the experimental results support the following recommendations:

**Maximum accuracy on known identities**: Use Progressive Unfreezing. The 99.15% accuracy with F1-score of 0.991 is near-perfect performance for a 14-class problem. Training time of 50 minutes is acceptable for a one-time model preparation step.

**Rapid prototyping and constrained resources**: Use Transfer Learning. At 92.84% accuracy in 4 minutes of training, it provides a strong baseline with minimal investment. This approach is suitable for evaluating whether the dataset and detection pipeline are sufficient before committing to longer training.

**Balanced accuracy and speed**: Consider running only Phases 1 and 2 of Progressive Unfreezing. Based on the phase results, this would yield approximately 96% accuracy in roughly 25 minutes of training, providing a useful middle ground.

**Open-set recognition**: Triplet Loss is the only strategy that produces embeddings directly suitable for open-set recognition without modification. If the deployment scenario requires identifying individuals not seen during training or dynamically registering new identities without retraining, Triplet Loss is the appropriate choice even with lower classification accuracy.

**Real-time detection**: Use MediaPipe BlazeFace for production deployment where 30+ FPS is required. Its 325.6 FPS performance provides substantial headroom for the recognition stage while maintaining acceptable recall for frontal faces at typical surveillance distances.

**High-recall detection**: Use MTCNN for offline batch processing or forensic analysis where missing a face is more costly than processing time. Its multi-scale pyramid approach detects faces that single-scale methods miss.

## 8.3 Key Insights and Lessons Learned

Several insights emerged from this work that extend beyond the specific implementation:

**Progressive adaptation beats abrupt change.** The success of progressive unfreezing suggests that gradual adaptation strategies may be broadly applicable to domain adaptation problems. Rather than viewing fine-tuning as a binary choice between frozen and fully trainable, the phased approach respects the hierarchical structure of learned representations and prevents catastrophic forgetting.

**Training objective must match evaluation metric.** The underperformance of triplet loss on classification accuracy, despite its theoretical appeal for face recognition, highlights the importance of aligning training objectives with evaluation metrics. A model trained to optimize embedding distances cannot be fairly evaluated on classification accuracy without acknowledging the methodological mismatch.

**Class imbalance has nonlinear effects.** The dramatic improvement for underrepresented classes (up to +44.4 percentage points for Stranger_9) under progressive unfreezing suggests that backbone adaptation may be particularly beneficial for minority classes. This finding has implications for fairness in face recognition systems, where underrepresented demographic groups often suffer from reduced accuracy.

**Speed and accuracy are not always fungible.** The detection results show that no single method dominates both speed and recall. This fundamental trade-off means that system designers must make explicit choices about which criterion to prioritize based on deployment context, rather than expecting optimization to resolve the tension.

## 8.4 Limitations and Future Work

The current work has several limitations that suggest directions for future research:

**Cross-dataset generalization.** All recognition results are specific to the custom 14-class dataset. Future work should evaluate cross-dataset performance to assess generalization to different cameras, lighting conditions, and demographic distributions.

**Open-set evaluation.** The system supports open-set recognition, but no quantitative evaluation was conducted. Future work should measure true acceptance rate, false acceptance rate, and equal error rate across various similarity thresholds.

**Semi-hard negative mining.** The triplet loss results used random online mining. Implementing semi-hard negative mining as described in the original FaceNet paper [CITE: Schroff et al. 2015] would likely improve triplet loss performance substantially.

**Larger-scale evaluation.** The current dataset of 7,080 images across 14 identities is modest by modern standards. Scaling to hundreds or thousands of identities would test the limits of the progressive unfreezing approach and may reveal different optimal strategies.

**Edge deployment optimization.** While this work focused on training strategies, deployment optimization techniques such as quantization, pruning, and knowledge distillation could further reduce model size and inference latency for resource-constrained edge devices.

**Temporal consistency.** Real surveillance systems process video streams rather than isolated frames. Incorporating temporal consistency constraints—requiring that identities remain stable across consecutive frames—could improve recognition accuracy in video contexts.

## 8.5 Closing Remarks

Face recognition in surveillance contexts presents a unique combination of technical challenges: real-time processing requirements, variable lighting and pose conditions, and the need for high accuracy on specific known identities. This thesis demonstrated that progressive unfreezing of pre-trained FaceNet models provides an effective solution to these challenges, achieving 99.15% accuracy with training times measured in minutes rather than hours or days.

The comparison of fine-tuning strategies revealed that the choice of adaptation approach has substantial impact on both accuracy and deployment characteristics. Practitioners should select strategies based on their specific requirements: frozen transfer learning for rapid prototyping, progressive unfreezing for maximum accuracy on known identities, and triplet loss for open-set scenarios requiring dynamic enrollment.

As face recognition technology continues to advance, the tension between accuracy, efficiency, and flexibility will remain central to system design. The empirical findings of this thesis provide a foundation for navigating these trade-offs in surveillance applications, contributing to the broader goal of deploying reliable, efficient, and ethically responsible face recognition systems.
