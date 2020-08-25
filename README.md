1. 桥梁识别

    - 图像旋转等数据增强方法。

    - 划分训练集、验证集，根据图像中桥梁的数目（桥梁大小、图片大小等？）信息进行分层抽样。

    - 调参，early stop 防止过拟合。


2. SAR 图像语义分割

    - 四幅图叠成一幅图。

What is up

The purpose of this repo is to setting up **deep learning models** for **object detection and segmentation**.

# To do list

## Compare **the formats and shapes of segments** and understand how to transfer between different formats. 

We also need to compare different formats of images, such as CCD and SAR. In the task 3 SAR images segmentation, a SAR image include three SAR versions of the same location.

1. Download 4 images with ships from [the airbus chanllenge](https://www.kaggle.com/c/airbus-ship-detection). Extract the RLE of segments. Plot the images and the segments. (CCD images?)

    - I have downloaded all the images from the airbus challenges. You need to conda install kaggle and create the token from the kaggle account. The data is around 25 GB.
 
    - I prioritize **the third step** focusing on the gaofen chanllenge. We need to submit the docked image by the end of summer holiday.
 

2. Download 4 images from [`sarship` folder](https://www.jianguoyun.com/p/DbVLM8oQ3cTHBhiZx68D). Extract the segments. Plot the images and segments. (SAR images?)

    - We can skip this step. We may study the sarship example in future.

3. Download 4 images from [the task 2, 3 of the challenge](http://sw.chreos.org/datasetlist). Extract the segments. Plot the images and segments. (CCD images for task 2 and SAR images for task 3).

    - I have downloaded the data of the four tasks. Tasks 1,2,4 use the normal images (CCD format) while task 3 use the SAR images. Task 1,3,4 are similar, they are all about **pixels segmentation**, among which task 1 is the simplest (single object segmentation). Task 2 is **object segmentation** in which a rectangle box is used to segment the bridge. 
    
    - I suggest to take part in the tasks 2 and 3. 

4. Download 2 images from [`hou` folder](https://www.jianguoyun.com/p/DXdgvqgQ3cTHBhidx68D). Extract the segments. Plot the images and the segments. (CCD and SAR images?)

    - We skip this step at this moment.

## Study and compare different deep learning models in terms of 

1. [**their powers (classification, detection, semantic segmentation, instance segmentation)**](https://mp.weixin.qq.com/s/e2Zdrinw9RbVbpneYYIzxw), 

2. the calibration environment and requirements (such as tensorflow, keras, pytorch, detectron, etc), 

3. their structures. 

I list several relevant deep learning models as follows. Note that the links I provided simply follow the above bearclub link which may not be the optimal one.

- Detection 

    1. [Region Convolutional Neural Network (R-CNN)](https://github.com/yangxue0827/RCNN)

    2. [Spatial Pyramid Pooling Network (SPP-Net)](https://github.com/chengjunwen/spp_net)

    3. [Fast R-CNN](https://github.com/rbgirshick/fast-rcnn)

    4. [Faster R-CNN](https://github.com/shaoqingren/faster_rcnn)

- Semantic segmentation

    5. [Fully Convolutional Neural Network (FCNN)](https://github.com/MarvinTeichmann/tensorflow-fcn)

    6. [U-Net](https://github.com/jakeret/tf_unet)
    
        [U-Net used in the airbus challenge](https://www.kaggle.com/kmader/baseline-u-net-model-part-1)
    
        There are several U-Net models and their deviations in [the Notebooks](https://www.kaggle.com/c/airbus-ship-detection/notebooks). It seems that the segments are always rectangles. The champion team name is "Rectangle is all you need". 

    7. [V-Net](https://github.com/mattmacy/vnet.pytorch)

    8. U-Net + DNN

- Instance segmentation

    9. [Mask R-CNN](https://github.com/matterport/Mask_RCNN)
    
        See [the introduction of Mask R-CNN](https://zhuanlan.zhihu.com/p/48381892)
    
        See [the general introduction](https://mp.weixin.qq.com/s?__biz=MzA5MjEyMTYwMg==&mid=2650242752&idx=1&sn=2bf846319d56abe741a53854fa3a4eb2&chksm=887221adbf05a8bbe8455adc875c2f2270cee78a6c814fcf717eb42d314f9ecb6c84a2440b72&scene=21#wechat_redirect) of R-CNN and Mask R-CNN.

## Discuss which model above is suitable for the following five problems:


1. [Airbus challenge](https://www.kaggle.com/c/airbus-ship-detection)

    - Skip at this moment. Single object detection model should be enough, such as U-Net.

2. [AIR-SARShip](https://www.jianguoyun.com/p/DbVLM8oQ3cTHBhiZx68D)

    - Skip at this moment. This problem is similar to the airbus challenge. But it use the SAR images rather than the normal image. The paper compares the performance of different models.

4. [Task 2: single object detection of bridge](http://sw.chreos.org/competitionlist)

    - Task 2 of Gaofen challenge is similar to the above two problems. **We start with the baseline of U-Net model.** I suggest you to establish a U-Net model and calibrate it on the task 2 bridge data after you are familar with the format of images and segmentation files.

5. [Task 3: semantic/instance segmentation of 6 objects](http://sw.chreos.org/competitionlist)

3. ['hou' folder](https://www.jianguoyun.com/p/DXdgvqgQ3cTHBhidx68D)

    - Skip at this moment. This problem is similar to task 2 of bridge detection. I am not sure whether we need to classify different types of ship such as warships, fish boats etc.

## Understand the loss functions for detection and segmentation in the gaofen challenge

### Loss function for detection: [$F_2$ score](https://www.kaggle.com/c/airbus-ship-detection/overview/evaluation) 

The predictive performance is evaluated on the **$F_2$ score** at different **intersection over union (IoU)** thresholds. The IoU of *a proposed set of object pixels* $A$ and *a set of true object pixels* $B$ is calculated as:
$$IoU(A,B)=\frac{A\cap B}{A\cup B}.$$

The metric sweeps over a range of IoU thresholds, at each point calculating an $F_2$ score. The threshold values $t$ range from 0.5 to 0.95 with a step size of 0.05: $t\in\mathcal{T}=\{0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95\}$. In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value $t$, the $F_2$ score value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects. The following equation is equivalent to $F_2$ score when $\beta$ is set to 2:
$$ F_\beta(t) = \frac{(1 + \beta^2) \cdot TP(t)}{(1 + \beta^2) \cdot TP(t) + \beta^2 \cdot FN(t) + FP(t)}.$$

- A true positive (TP) is counted when a single predicted object matches aground truth object with an IoU above the threshold.

- A false positive (FP) indicates a predicted object had no associated ground truth object.

- A false negative (FN) indicates a ground truth object had no associated predicted object. 

The average $F_2$ score of a single image is then calculated as the mean of the above $F_2$ score values at each IoU threshold:
$$\frac{1}{|\mathcal{T}|} \sum_{t\in\mathcal{T}} F_2(t).$$
Lastly, the score returned by the competition metric is the mean taken over the individual average $F_2$ scores of each image in the test dataset.

### Loss function for segmentation


# Goals (we will investigate the following after finishing the above)

By studying two examples, we aim to achieve the following four goals:

1. Study [the deep learning models](https://mp.weixin.qq.com/s/StibMN4buP8-mvzwXr56jg) for ship detection/segmentation and similar tasks of objects segmentation. 

2. Be familar with the important steps in image preprocessing, setting up a deep learning mdoel and calibrating the model in [a cloud computing server](http://cc.ruc.edu.cn/help/manual/login/).

3. Improve the predictive performance of deep learning models by tuning the hyper-parameters and modifying the model structure.

4. Attend [2020 Gaofen Challenge on Automated High-Resolution Earth Observation Image Interpretation](http://sw.chreos.org).



