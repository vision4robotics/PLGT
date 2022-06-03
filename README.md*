# Adaptive UAV Image Enhancer Based on Online Pixel-Level Gamma Transform

This project contains the code of PLGT.

# Abstract

Intelligent applications of unmanned aerial vehicle (UAV) often requires accurate vision-based object tracking. However, many studies have shown that the tracking performance of state-of-the-art (SOTA) trackers is decreased significantly in dark conditions. Moreover, the leading-edge Siamese trackers require manual object selection during initialization. However, the precision of manual object initialization is also decreased greatly under low-light conditions. The nighttime object tracking has limited the intelligent applications and development of UAV. Since the images with ideal illumination conditions are essential in the initial selection stage and the subsequent tracking stage, why not use enhancers to enhance low-light images which are ideal for human eyes?

In recent years, the combination of traditional image enhancement methods and neural networks has become mainstream studies. However, most of these methods are designed for photography applications. On the one hand, the challenges of UAV applictions are ignored. On the other hand, these methods with complex algorithms can hardly be processed in real-time on UAV platforms. To address these problems, this thesis proposes an adaptive image enhancer with pixel-level Gamma transform, i.e., PLGT.

Gamma transform, as a traditional enhancement method,  is of poor robustness in complex and dynamic UAV conditions. Therefore, PLGT proposes a novel composite network. Convolutional neural network (CNN) branch processes the local information to generate a range mask, which can realize pixel-level enhancement and effectively cope with the nighttime UAV challenges such as artificial light sources and small target size. Transformer branch uses the global illumination information to dynamically adjust network parameters, thereby coping with illumination variation. PLGT is also specifically optimized for practical UAV applications in nighttime situations. To improve the computation efficiency of CNN branch and Transformer branch in PLGT, depthwise seperatable convolution and downsampling are employed, respectively. A soft truncation function is propesed to avoid over-enhancement of dark area noise. In addition, PLGT includes a set of well-designed non-reference loss functions to realize unsupervised training and save the cost of collecting paired datasets.

To verify the practicality of PLGT in the initial manuel selection stage, this thesis compares PLGT with other cutting-edge image enhancers on image enhancement benchmarks to prove the advantages of PLGT in improving human perception. This thesis applies PLGT as a preprocessing stage for multiple leading-edge trackers in the nighttime UAV tracking benchmark, i.e., UAVDark135, to prove its universality of “plug and play”. Moreover, this thesis compares PLGT with other state-of-the-art image enhancers on UAVDark135 to verify its advantage in the UAV tracking stage. Finally, in a real-world test on a typical UAV platform, PLGT helps the tracker achieve real-time stable tracking at night. This thesis evaluates the proposed approach on a typical UAV platform, the results show that PLGT helps vision-based UAV trackers at night to achieve accurate, robust, and real-time tracking.

![image](https://github.com/haolindong/images_store/blob/main/plgt.png)

# 联系方式 
董昊林

邮箱: 1851146@tongji.edu.cn

符长虹

邮箱: changhongfu@tongji.edu.cn

# 运行指令示例

### 运行环境需求

1.Python 3.7.10

2.Pytorch 1.10.1

4.torchvision 0.11.2

5.cuda 11.3.1

>下载并提取代码，接下来按以下步骤执行:
>
>1. 将测试图片放入 data/test_data/，部分示例测试图片已放入该文件夹中。将训练图片放入 data/train_data/。
>
>2. 为测试该方法，需运行:
>
>     ```
>     python lowlight_test.py
>     ```
>     你可以在 data/result/ 中找到测试结果，部分示例结果已放入该文件夹中。
>   
>3. 为训练该方法，需运行:
>
>     ```
>     python lowlight_train.py
>     ```

