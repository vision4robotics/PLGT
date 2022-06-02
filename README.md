# 基于在线逐像素伽马变换的自适应无人机图像增强


该项目包含PLGT的代码。

# 摘要
无人机的智能化应用往往离不开视觉目标跟踪，但是诸多研究指出：在黑夜条件下最前沿的视觉跟踪器性能亦会大幅下降。另外，主流基于Siamese神经网络的视觉跟踪器在初始化阶段通常需要人工标注跟踪目标，而黑夜条件下人工初始化目标标注的精度亦会大幅降低。因此，夜间视觉目标跟踪严重限制无人机智能化应用与发展。既然在初始化目标标注阶段和后续跟踪阶段，光照条件良好的图像均必不可少，为何不利用图像增强技术将低照度图像增强为符合人眼特性的理想照度图像呢？
近年来，图像增强方法与神经网络的结合已经成为研究的主流。但是大部分研究都仅立足于摄影领域痛点，一方面忽略无人机复杂动态场景下的特有诸多挑战，另一方面前沿图像增强方法复杂且参数庞大，导致无法在无人机机载平台上实现高精度、强鲁棒的实时应用。为解决这些问题，本文提出一种基于逐像素伽马变换的自适应图像增强方法（Adaptive Image Enhancer with Pixel-Level Gamma Transform, PLGT）。
在无人机复杂动态场景下，伽马变换作为一种传统增强方法，具有鲁棒性差的问题。因此， PLGT提出一种复合神经网络。具体地，其卷积神经网络分支利用局部信息生成增幅蒙版，实现像素级差异化增强，从而有效应对夜间无人机复杂动态场景下人造光源多、目标尺寸小等显著性挑战。此外，Transformer分支利用全局光照信息动态调节网络参数，以有效应对无人机复杂动态场景下光照快速变化等挑战。PLGT亦针对在夜间无人机上实际应用问题进行专门优化设计。具体地，其通过引入深度可分离卷积（Depthwise Seperable Convolution, DSC）和降采样过程，分别优化卷积神经网络分支和Transformer分支运算效率。PLGT亦提出一种软截断函数以解决暗部噪点过度增强问题。另外，PLGT包含一组精心设计的无参考损失函数，可利用不成对数据集进行训练，以节省搜集成对数据集所需巨大成本。
为验证PLGT在人工初始化目标标注的实用性，本文在经典低照度图像基准数据集上与其他前沿增强方法进行对比，证明PLGT在提升人眼感知特性方面的优势。此外，本文在公开黑夜无人机跟踪数据集UAVDark135上，将PLGT作为多种前沿视觉跟踪器的预处理算法，证明其即插即用的通用性。在同一数据集下，本文亦验证PLGT相较于其他增强方法在视觉目标跟踪方面的优势。最后，本文在一个典型无人机平台上进行实际测试，测试结果表明PLGT可以有效帮助无人机视觉跟踪器在夜间实现高精度、强鲁棒的实时目标跟踪。

![image](https://github.com/haolindong/images_store/blob/187ca383ba64b14fd3d5fd4ebd5e6a1af0128fdc/pipeline.png)

# 联系方式 
董昊林

邮箱: 1851146@tongji.edu.cn

符长虹

邮箱: changhongfu@tongji.edu.cn

# Demonstration running instructions

### Requirements

1.Python 3.7.10

2.Pytorch 1.10.1

4.torchvision 0.11.2

5.cuda 11.3.1

>Download the package, extract it and follow two steps:
>
>1. Put test images in data/test_data/, put training data in data/train_data/.
>
>2. For testing, run:
>
>     ```
>     python lowlight_test.py
>     ```
>     You can find the enhanced images in data/result/. Some examples have been put in this folder.
>   
>3. For training, run:
>
>     ```
>     python lowlight_train.py
>     ```



# Acknowledgements

We sincerely thank the contribution of `Chongyi Li` for his previous work Zero-DCE (https://github.com/Li-Chongyi/Zero-DCE).
