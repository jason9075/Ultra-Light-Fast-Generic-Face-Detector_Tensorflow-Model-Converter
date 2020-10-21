# Ultra-Light-Fast-Generic-Face-Detector-Tensorflow Converter

 This project is base on [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB). You can use this script to converter origin model to tensorflow version.
 
## Run
Covert model
```Python
 python3 ./convert_tensorflow.py --net_type <RFB|slim>
```

Inference on image
```Python
 python3 ./det_image.py --net_type <RFB|slim> --img_path <path>
```

Inference on Webcam
```Python
 python3 ./det_webcam.py --net_type <RFB|slim>
```
 
##  Result Comparison
 Pytorch Slim
 
![img1](https://github.com/jason9075/Ultra-Light-Fast-Generic-Face-Detector_Tensorflow-Model-Converter/blob/master/imgs/test_output_origin_slim.jpg)

 Tensorflow Slim

![img1](https://github.com/jason9075/Ultra-Light-Fast-Generic-Face-Detector_Tensorflow-Model-Converter/blob/master/imgs/test_output_slim.jpg)

 Pytorch RFB

![img1](https://github.com/jason9075/Ultra-Light-Fast-Generic-Face-Detector_Tensorflow-Model-Converter/blob/master/imgs/test_output_origin_RFB.jpg)

 Tensorflow RFB

![img1](https://github.com/jason9075/Ultra-Light-Fast-Generic-Face-Detector_Tensorflow-Model-Converter/blob/master/imgs/test_output_RFB.jpg)
 
##  Reference
- [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)

