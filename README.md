# keras-ctpn

[TOC]

1. [Introduction](#Introduction) 
2. [Predicting](#Predicting)
3. [Training](#Training)
4. [Examples](#Examples)<br>
   4.1 [ICDAR2015](#ICDAR2015)<br>
   4.1.1 [With boundary refined](#With boundary being refined)<br>
   4.1.2 [Without boundary refined](#Without boundary being refined)<br>
   4.1.3 [Flipping horizontally](#Flipping horizontally)<br>
   4.2 [ICDAR2017](#ICDAR2017)<br>
   4.3 [Other datasets](#Other datasets)
5. [toDoList](#toDoList)
6. [Summary](#Summary)

## Introduction

​         This project is a kera application of [CPTN: Detecting Text in Natural Image with Connectionist Text Proposal Network](https://arxiv.org/abs/1609.03605) . The code is refer to [keras-faster-rcnn](https://github.com/yizt/keras-faster-rcnn) ; and the model is trained and tested on ICDAR2015 and ICDAR2017.

​         Links: [keras-ctpn](https://github.com/yizt/keras-ctpn)

​         Chinese version of the original paper:[CTPN.md](https://github.com/yizt/cv-papers/blob/master/CTPN.md)

**Result**：

​        1500 images from ICDAR2015, 1000 from training and 500 for testing ：Recall: 37.07 % Precision: 42.94 % Hmean: 39.79 %;

**Key points**:

a.Using resnet50

b.Input image size is 720*720; (extending the heigh to 720,keeping the heigh/width ratio unchanged, padding the width if it is less than 720;this is different to the original paper which sets the width as 600; using the size 1024*1024 when doing the prediction

c.batch_size is 4, there are 128 anchors for each image, the ratio of the number of positive samples and negative samples is 1:1;

d. The weights in the loss function for classification, boundary box regression and boundary refining are set as 1:1:1; while in the original paper they are set as 1:1:2

e.The boundary refining and boundary box regression use same positve anchors, which is different to the setting in original paper.
f.The boundary refining contributes to the performance, which is contract to some researchers views.

g.由于有双向GRU，水平翻转会影响效果(见样例[做数据增广-水平翻转](#做数据增广-水平翻转))

h.随机裁剪做数据增广，网络不收敛




## Predicting

a. Code can be downloat from the following link:

```bash
git clone https://github.com/yizt/keras-ctpn
```



b. Pretrained model can be download from the following link

​    Model for ICDAR2015 ：[ctpn.h5](https://pan.baidu.com/s/1XeQN0H1_FdTPBwH1GDlW_w) security code：k7yu ; [google drive](https://drive.google.com/file/d/1n1OeN99BP4NdFOXA1CaYom7O3S985Nd6/view?usp=sharing)

c.Change the following setting in config.py

```python
	WEIGHT_PATH = '/tmp/ctpn.h5'
```

d. Text detection

```shell
python predict.py --image_path image_3.jpg
```

## Evaluation

a. run the following statement, then compress the output (i.e., the ".txt" file) in to a zip file
```shell
python evaluate.py --weight_path /tmp/ctpn.140.h5 --image_dir /opt/dataset/OCR/ICDAR_2015/test_images/ --output_dir /tmp/output_2015/
```

b. Submit for online evaluation
   submit the zip file to :http://rrc.cvc.uab.es/?ch=4&com=mymethods&task=1

## Training

a. Dataset
```shell
#icdar2013
wget http://rrc.cvc.uab.es/downloads/Challenge2_Training_Task12_Images.zip
wget http://rrc.cvc.uab.es/downloads/Challenge2_Training_Task1_GT.zip
wget http://rrc.cvc.uab.es/downloads/Challenge2_Test_Task12_Images.zip
```

```shell
#icdar2015
wget http://rrc.cvc.uab.es/downloads/ch4_training_images.zip
wget http://rrc.cvc.uab.es/downloads/ch4_training_localization_transcription_gt.zip
wget http://rrc.cvc.uab.es/downloads/ch4_test_images.zip
```

```shell
#icdar2017
wget -c -t 0 http://datasets.cvc.uab.es/rrc/ch8_training_images_1~8.zip
wget -c -t 0 http://datasets.cvc.uab.es/rrc/ch8_training_localization_transcription_gt_v2.zip
wget -c -t 0 http://datasets.cvc.uab.es/rrc/ch8_test_images.zip
```



b. resnet50 and the trained model
```shell
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
```



c. Modify the settings in config.py，as the following

```python
	# pretrained model
    PRE_TRAINED_WEIGHT = '/opt/pretrained_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    # the path of the dataset
    IMAGE_DIR = '/opt/dataset/OCR/ICDAR_2015/train_images'
    IMAGE_GT_DIR = '/opt/dataset/OCR/ICDAR_2015/train_gt'
```

d.training

```shell
python train.py --epochs 50
```





## Examples

### ICDAR2015

#### With boundary being refined

![](image_examples/icdar2015/img_8.1.jpg)

![](image_examples/icdar2015/img_200.1.jpg)

#### Without boundary being refined
![](image_examples/icdar2015/img_8.0.jpg)

![](image_examples/icdar2015/img_200.0.jpg)

#### Flipping horizontally
![](image_examples/flip1.png)
![](image_examples/flip2.png)

### ICDAR2017


![](image_examples/icdar2017/ts_img_01000.1.jpg)

![](image_examples/icdar2017/ts_img_01001.1.jpg)

### Othere datasets
![](image_examples/bkgd_1_0_generated_0.1.jpg)
![](image_examples/a2.png)
![](image_examples/a1.png)
![](image_examples/a3.png)
![](image_examples/a0.png)

## toDoList

1. 侧边细化(已完成)
2. ICDAR2017数据集训练(已完成)
3. 检测文本行坐标映射到原图(已完成)
4. 精度评估(已完成)
5. 侧边回归,限制在边框内(已完成)
6. 增加水平翻转(已完成)
7. 增加随机裁剪(已完成)



### Summary

1. ctpn is good at detecting the words in horizontal 
2. The network is senstive to the dataset; Model training on ICDAR2017 has poor performance on ICDAR2015 
3. 推测由于双向GRU，网络有存储记忆的缘故？在使用随机裁剪作数据增广时网络不收敛，使用水平翻转时预测结果也水平对称出现
