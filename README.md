number detection and recognition based on [AdvancedEast](https://github.com/huoyijie/AdvancedEASTg) and [CRNN](https://arxiv.org/abs/1507.05717)

Detection and Crop:
<img src="https://github.com/yl305237731/VOS_seg/blob/master/output/00001.jpg" height="200" width="300">

Recognition:

* 22046298859.jpg : 22046298859
* 97785067838.jpg : 97785067838
* 84999825604.jpg : 84999825604
* 99851544924.jpg : 99851544924
* 28510715459.jpg : 28510715459
* 12233418739.jpg : 12233418739
* 41679405336.jpg : 41679405336
* 37774346979.jpg : 37774346979


limitations:
*When the two models are test on their respective validation sets , they can reach an acc of about 0.9. However, the number of the training data for recognizer I generated is horizontal, and the number in the crop image after the detection result introduces the rotation and other factors, resulting in poor results when used in combination.*
