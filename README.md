This repository contains the ability to conduct transfer-learning on the Models tf_efficientnetv2_s.in1k, vit_tiny_patch16_224.augreg_in21k_ft_in1k and convnextv2_pico.fcmae_ft_in1k. Also it is possible to train these models from scratch or to train a MiniVGG. The training process uses Crossvalidation. I want to remember the reader, that this is just a fast implementation and only serves as an entry point for transfer learning, there are many possibilities to reach better results. FOr example one could use Data Augmentation, Ensembling or just plainly transfer learn bigger Models. You can use main.py to visuallize the test results, which look as following for the Model **ConvNeXtV2**:

![alt text](https://github.com/MaxUhl98/ComputerVisionDemonstration/blob/main/demonstration_images/ConvNeXt_V2/ConvNeXtV2.png)


## Confusion Matrix:

![alt text](https://github.com/MaxUhl98/ComputerVisionDemonstration/blob/main/demonstration_images/ConvNeXt_V2/ConvNeXtV2_Konfusionsmatrix.png)

