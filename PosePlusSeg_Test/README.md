# Setup environment
```
pip install tensorflow-gpu==1.13.2
pip install matplotlib==2.0.2
pip install imantics

```

# Setup data

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

Run the `demo.py` file to get the keypoints coordinates and confidence scores of each keypoint and store it in a txt file (keypoints.txt). `demo.py` also produce the binary mask information and stores it txt file (segmentation.txt).   
