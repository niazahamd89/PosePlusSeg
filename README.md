# PosePlusSeg (AAAI 2022)

Official code repo for the paper "Joint Human Pose Estimation and Instance Segmentation with PosePlusSeg"[arXiv] version will be available soon.

# Model Architecture 
![](demo_result/0001.png)

# Setup environment

```
- python==3.6
- conda install -c conda-forge matplotlib==2.0.2
- conda install -c conda-forge opencv OR pip install opencv-python
- conda install -c conda-forge pycocotools
- conda install -c anaconda scikit-image
- conda install tensorflow-gpu==1.13.1
```

**Recomendation:** tensorflow 1.13 & coda 10.

# Download data

### COCO 2017

- [COCO 2017 Train images 118K/18GB](http://images.cocodataset.org/zips/train2017.zip)
- [COCO 2017 Val images 5K/1GB](http://images.cocodataset.org/zips/val2017.zip)
- [COCOPersons Train Annotation (person_keypoints_train2017_pose2seg.json) [166MB]](https://github.com/liruilong940607/Pose2Seg/releases/download/data/person_keypoints_train2017_pose2seg.json)
- [COCOPersons Val Annotation (person_keypoints_val2017_pose2seg.json) [7MB]](https://github.com/liruilong940607/Pose2Seg/releases/download/data/person_keypoints_val2017_pose2seg.json)

### Hint 

Person keypoint dataset is a subset of COCO2017 dataset ([COCO 2017 Train images 118K/18GB](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)). We train our model only on human instances key points and segmentation by introducing a multi task system. 


# Setup data

The `coco2017` folder should be like this:
``` 
├── coco2017
│   ├── annotations  
│   │   ├── person_keypoints_train2017.json 
│   │   ├── person_keypoints_val2017.json 
│   ├── train2017  
│   │   ├── ####.jpg  
│   ├── val2017  
│   │   ├── ####.jpg  

```

# Training

 Run the `python train.py` for training the model. 

# Note

1. Please correctly give the path to the `dataset folder` and `check point files` in the `config.py` file. 
2. Currently we only support single-gpu training (Recommended: TITAN RTX).

# Testing

Please lookout the `PosePlusSeg_Test` folder for testing the model. 

# Results

#### Human Pose Estimation
<img src="demo_result/pose.png" width="500" height="600">

#### Human Instance Segmentation

<img src="demo_result/seg.png" width="500" height="600">

#### Joint Human Pose Estimation and Instance Segmentation

<img src="demo_result/PosePlusSeg.png" width="500" height="600">
