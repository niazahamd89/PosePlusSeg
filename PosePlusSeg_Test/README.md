# Environment
```
pip install tensorflow-gpu==1.13.2
pip install matplotlib==2.0.2
pip install imantics

```

# Data Setup

The `COCO2017` folder should be like this:
``` 
├── COCO2017
│   ├── annotations  
│   │   ├── person_keypoints_val2017.json 
│   │   ├── image_info_test2017.json 
│   ├── val2017  
│   │   ├── ####.jpg  
│   ├── test2017  
│   │   ├── ####.jpg  

```

# Generate the Keypoints and binary mask

Run the `demo.py` file to get the keypoints coordinates and confidence scores of each keypoint and store it in a `txt` file (`keypoints.txt`). `demo.py` also produces the binary mask information and stores it into `txt` file (`segmentation.txt`).   

# Convert the keypoints and mask into COCO formate

Run the `python making_coco_keypoint.py` file which accesses the information from the `keypoint.txt` file and convert the keypoints into COCO format (keypoint.json). The `making_coco_segmentation.py` gets the segmentation information from the `segmentation.txt` and converts it into RLE COCO format (segmentation.json). 

# Find mAP

Access the COCO [`cocoeval.py`](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI). Set the `keypoint.json` file path for keypoint test and `segmentation.json` path for segmentation test. Our system pretrained model can be downloaded [here](https://drive.google.com/drive/folders/1QlHmzX7Cdz6Yjcpr3_yuCbFDFKI-GCxY?usp=sharing)
