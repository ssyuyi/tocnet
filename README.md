# tocnet
TOCNet: A Target Occlusion Contrast Network for License Plate Detection in Waterways

## Content

- [1. Introduction]()
- [2. environmental preparation]()
- [3. datasets]()
- [4. train]()
- [5. test]()
- [5. LICENSE]()

## 1. Introduction
With the continuous development of waterway trade, intelligent waterway supervision is gradually becoming an important way to maintain waterway safety. In this field, ship license plate (SLP) detection is importance for intelligent waterway supervision. However, in practical surveillance scenarios, detecting ship license plates (SLPs) is challenging because of small size, and text areas in complex backgrounds can also be mistakenly identified as SLPs. To solve these problems, we propose a Target Occlusion Contrast Network (TOCNet) for SLP detection. Firstly, to mitigate the effects of complex backgrounds, a novel Target Occlusion Contrast Learning (TOCL) approach is proposed, which is used to widen the decision boundary of the model and reduce false alarms. Secondly, to solve the problem that small targets are difficult to be detected, we design a Similarity Fusion Module (SFM), which effectively improves the detection performance of the model for small targets by efficient multi-scale features fusion. Finally, to further improve the detection accuracy, a Task-oriented Decoupled Head (TDH) is proposed, where TDH effectively utilizes the advantageous properties of different scale features through selective fusion. Experimental results on both the SLPD dataset and the UFPR-ALPR dataset show that our proposed modules effectively improve the detection performance of the model for license plates, and the proposed TOCNet model exhibits a better detection performance than the other existing state-of-the-art (SOTA) models. Our code and dataset is publicly available at  https://github.com/ssyuyi/tocnet

## 2. environmental preparation
##### Step 0. Download and install Miniconda from [the official website](https://docs.conda.io/en/latest/miniconda.html).
##### Step 1. Create a conda environment and activate it.
```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```
##### Step 2. Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.
```angular2html
conda install pytorch torchvision -c pytorch
```
##### Step 3. Install MMEngine and MMCV using MIM.
```angular2html
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```
##### Step 4. Install MMEngine and MMCV using MIM.
```angular2html
git clone https://github.com/ssyuyi/tocnet.git
cd tocnet
pip install -v -e .
```

## 3. datasets
| dataseet                 | link                                                              |
|:-------------------------|-------------------------------------------------------------------|
| SLPD (include json file) | [password: SLPD](https://pan.baidu.com/s/1MZn0vcpPw-2LlBb71LXq-w) |
| UFPR-ALPR                | [author web](https://github.com/raysonlaroca/ufpr-alpr-dataset)   |

Note that we have only released the test portion of dataset SLPD, and will release the full training data when the paper is accepted.
Dataset UFPR-ALPR is the experimental data from . Students can contact them for free access to [UFPR-ALPR](https://github.com/raysonlaroca/ufpr-alpr-dataset).

After downloading, please save the dataset as structured in the following way

```angular2html
-- .......
-- configs_wm
-- tools
-- data
    -- slpd
        -- images (all images in it)
        -- train.json
        -- test.json
    -- ufpr-alpr
        -- images
        -- train.json
        -- test.jasn
```
**Note that the train.json will be upload after our paper is accepted.**

## 4. train
```angular2html
python tools/train.py configs_wm/tocnet.py
```

## 5. test
| dataseet     | Recall | precise   | F1  | AP  | FPS | pth                |
|:-------------|--------|-----------|-----|-----|-----|--------------------|
| SLPD         | 83.0   | 92.2 | 87.4 | 86.9 | 60.1 | [password: SLPD](https://pan.baidu.com/s/1Fd_xmP1yRYRgvkox8V0WBQ) |
| UFPR-ALPR    | 98.0   | 98.3 | 98.1 | 98.7 | 60.1 | [password: SLPD](https://pan.baidu.com/s/1ccIHu6Tgl-4zg-lBuPjzHA ) |
```angular2html
python tools/test.py configs_wm/tocnet.py [your pth file]
```
**Note!**

**When using the slpd dataset, set the attribute `data_root` of `configs_wm/base/coco_detection.py` to `data/slpd`**

**When using the ufpr_alpr dataset, set the attribute `data_root` of `configs_wm/base/coco_detection.py` to `data/ufpr_alpr`****


## 5. LICENSE
This project is released under the [Apache 2.0 license](./LICENSE) license.

