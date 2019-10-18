number detection and recognition based on [AdvancedEast](https://github.com/huoyijie/AdvancedEASTg) and [CRNN](https://arxiv.org/abs/1507.05717)
____
Detection and Crop:
<img src="https://github.com/yl305237731/number_detection_recognition/blob/master/demo/re1.jpg">

____
Recognition:

* 22046298859.jpg : 22046298859
* 97785067838.jpg : 97785067838
* 84999825604.jpg : 84999825604
* 99851544924.jpg : 99851544924
* 28510715459.jpg : 28510715459
* 12233418739.jpg : 12233418739
* 41679405336.jpg : 41679405336
* 37774346979.jpg : 37774346979

____
limitations:
*When the two models are test on their respective validation sets , they can reach an acc of about 0.9. However, the number of the training data for recognizer I generated is horizontal, and the number in the crop image after the detection result introduces the rotation and other factors, resulting in poor results when used in combination.*


----
# Detection

## training
* prepare training data, data format refer to [ICPR](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.3bcad780oQ9Ce4&raceId=231651)
* modify params in cfg.py
* run python preprocess.py to resize image and generator .npy training files
* run python label.py
* run python train.py, train the network
## testing
* modify your images' dir in predict.py, and run python predict.py, then we will get three outputs: bounding box on origin 
images, the cropped image, and coordinates(txt file).

*more details please refer to [AdvancedEast](https://github.com/huoyijie/AdvancedEASTg)*

-----
# Recognition

## training
* prepare training data, data format refer to [MJSynth data](http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz)
* modify params in cfg.py
* modify input_shape=(None, 50,7,512) in train.py line 55, the input_shape is refer to your bn_shape = bn4.get_shape() in network.py
* run python train.py
## testing
* modify your images' dir in predict.py, then run python predict.py
----