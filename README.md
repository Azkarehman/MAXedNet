# Intelligent Healthcare System for IoMT-Integrated Sonography: Leveraging Multi-Scale Self-Guided Attention Networks and Dynamic Feature Fusion

## Cloud-based thyroid nodule segmentation using IoMT-enabled sonography


## Abstract

Through the Internet of Medical Things (IoMT), early diagnosis of various critical diseases has been revolutionized, particularly via sonography in thyroid nodule identification. Despite its benefits, accurate thyroid nodule segmentation remains challenging due to the heterogeneity of nodules in terms of shape, size, and visual characteristics. This complexity underscores the necessity for improved Computer-Aided Diagnosis (CAD) methods that can provide robust assistance to radiologists. Subsequently, this study introduces a multiscale self-guided network leveraging a novel Dynamic Self-Distillation (DSD) training framework to significantly enhance thyroid nodule segmentation. The developed architecture captures rich contextual dependencies by capitalizing on  self-guided attention mechanisms, thus fusing the local features with corresponding global dependencies while adaptively highlighting interdependent channel maps. Irrelevant information from coarse multiscale features is filtered out using self-guided attention mechanisms, leading to the generation of refined feature maps. These maps, in turn, facilitate the creation of accurate thyroid tumor segmentation masks. The novel DSD mechanism, implemented to train the architecture, dynamically selects the teacher branch based on performance relative to the ground truth label, and computes distillation losses for each student branch. Evaluation on two publicly available datasets reveals the superior performance of our framework over its degraded versions and existing state-of-the-art techniques, demonstrating the promising potential of our proposed approach to be employed for thyroid nodule segmentation in IoMT.

IOMT

![](https://github.com/Azkarehman/Thyroid-Nodule-Segmentation/blob/main/model_.png)

![](https://github.com/Azkarehman/Thyroid-Nodule-Segmentation/blob/main/modules.png)
