# Intelligent Healthcare System for IoMT-Integrated Sonography: Leveraging Multi-Scale Self-Guided Attention Networks and Dynamic Feature Fusion

## Cloud-based thyroid nodule segmentation using IoMT-enabled sonography

![](https://github.com/Azkarehman/MAXedNet/blob/a524f41602835d8b590437adb879887764de8863/figures/fig.png)

## Abstract

Through the Internet of Medical Things (IoMT), early diagnosis of various critical diseases has been revolutionized, particularly via sonography in thyroid nodule identification. Despite its benefits, accurate thyroid nodule segmentation remains challenging due to the heterogeneity of nodules in terms of shape, size, and visual characteristics. This complexity underscores the necessity for improved Computer-Aided Diagnosis (CAD) methods that can provide robust assistance to radiologists. Subsequently, this study introduces a multiscale self-guided network leveraging a novel Dynamic Self-Distillation (DSD) training framework to significantly enhance thyroid nodule segmentation. The developed architecture captures rich contextual dependencies by capitalizing on  self-guided attention mechanisms, thus fusing the local features with corresponding global dependencies while adaptively highlighting interdependent channel maps. Irrelevant information from coarse multiscale features is filtered out using self-guided attention mechanisms, leading to the generation of refined feature maps. These maps, in turn, facilitate the creation of accurate thyroid tumor segmentation masks. The novel DSD mechanism, implemented to train the architecture, dynamically selects the teacher branch based on performance relative to the ground truth label, and computes distillation losses for each student branch. Evaluation on two publicly available datasets reveals the superior performance of our framework over its degraded versions and existing state-of-the-art techniques, demonstrating the promising potential of our proposed approach to be employed for thyroid nodule segmentation in IoMT.

## MAXedNet Architecture for Thyroid Nodule Segmentation:

MAXedNet is a specialized deep learning framework designed for segmenting thyroid nodules, inspired by the work of Sinha et al. (2020). It incorporates a DenseNet-121 backbone, known for efficient parameter use and improved feature propagation. The architecture processes the input through two convolutional layers, followed by the DenseNet structure. Within this structure, feature maps from four key intermediate layers are used. These maps are upscaled to construct initial segmentation masks, utilizing coarse features for a comprehensive representation at various abstraction levels.

The architecture also integrates attention mechanisms to combine low-level and abstract features effectively, enhancing segmentation quality. These mechanisms refine feature maps and help in accurately delineating nodules.

A novel aspect of MAXedNet is the introduction of a dynamic self-distillation (DSD) mechanism. This approach involves knowledge exchange among four branches processing the refined features. In the DSD framework, one branch serves as the 'teacher' and the others as 'students.' The students learn from both ground truth labels and the teacher branch, enhancing the model's robustness and generalizability.

The below diagrams depicts the model and its modules in detail.
![](https://github.com/Azkarehman/MAXedNet/blob/main/fig/Model_diagram.png)
![](https://github.com/Azkarehman/MAXedNet/blob/main/fig/modules.png)


## Results: Comparison with State-of-the-Art Segmentation Networks

The performance of MAXed-Net was rigorously tested against leading segmentation networks to assess its capabilities in thyroid nodule segmentation. The comparison involved several state-of-the-art models including ATT-U-Net, CE-NET, DeepLab v3+, MG-U-Net, SANet, HDA-Res U-Net, and UGNet. These models were evaluated on the DDTI and TN3K public datasets, with the results summarized in two tables.

To ensure a fair comparison, experimental settings such as epochs and batch sizes were aligned with those used for MAXed-Net. The learning rate and loss functions were adopted from the original papers of each model. If these details were unspecified in the original papers, the learning rate and loss function parameters from our MAXed-Net study were used.
![](https://github.com/Azkarehman/MAXedNet/blob/main/fig/results.png)

