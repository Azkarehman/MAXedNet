# MAX*ed*-Net: Multiscale Mutually Distilled Self-Guided Attention Network for Thyroid Nodule Segmentation

## Architecture

The nodules found in the thyroid are of various sizes, both small and large. A large number of traditional convolutional neural networks have a local receptive field, this is fine for specific small nodules but can cause intra-class inconsistency for larger nodules since the context information is not encoded properly. To deal with this problem we have first encoded features from multiple scales and then used the attention mechanisms to build a link between those features. The feature from multiple scales is concatenated together and then combined with the features from each scale. These combined features are then fed to guided attention modules which consist of the channel and spatial attention modules. Channel and spatial attention modules help to integrate local features with their global dependencies. The overall architecture of our model is depicted in Figures below whereas figure displays the details of attention module.

Since at each level the features come at different resolutions they all need to be upsampled using bilinear interpolation to a common resolution before being concatenated together. These concatenated features are convolved to create a single multi-scale feature map.

![](https://github.com/Azkarehman/Thyroid-Nodule-Segmentation/blob/main/model_.png)

[](https://github.com/Azkarehman/Thyroid-Nodule-Segmentation/blob/main/modules.png)
