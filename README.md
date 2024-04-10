# Supervised Cross-modal Contrastive Learning Framework for Audio-Visual Coding (SCLAV)
This is an open source repository for our paper [SCLAV: Supervised Cross-modal Contrastive Learning for Audio-Visual Coding](https://doi.org/10.1145/3581783.3613805) based on the pytorch framework. In this paper, we propose a Supervised Cross-modal Contrastive Learning Framework for Audio-Visual Coding (SCLAV).
![F2](https://github.com/Supersunn/SCLAV/assets/45378715/9a0e3015-8cd9-40bb-b7ea-7611d5bf2b74)

# Supervised Cross-modal Contrastive Loss (SCL Loss)
Different flows for self-supervised contrastive learning and our proposed supervised cross-modal contrastive learning. The self-supervised contrastive learning (as shown in subfigure (a)) contrasts each anchor and its augmentation against the remaining negatives of the entire batch, which can be viewed as a clustering problem. Our supervised cross-modal contrastive learning (as shown in subfigure (b)) produces a classification hypersphere guided by weak labels. All samples of the same category are compared as positives with the negatives from the rest samples of the batch, and the SCL loss can narrow the distance between different modal representations of the same sample. 

![F1_1](https://github.com/Supersunn/SCLAV/assets/45378715/12896592-d1df-4bed-b87c-eb050e720fef)
(1) Self-supervised Contrastive Learning
![F1_2](https://github.com/Supersunn/SCLAV/assets/45378715/850dfe2a-bca4-4670-96d9-c762318871c3)
(2) Supervised Cross-modal Contrastive Learning

# Thank you for your interest in our work.
@inproceedings{sun2023sclav,
  title={SCLAV: Supervised Cross-modal Contrastive Learning for Audio-Visual Coding},
  author={Sun, Chao and Chen, Min and Cheng, Jialiang and Liang, Han and Zhu, Chuanbo and Chen, Jincai},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={261--270},
  year={2023}
}
