Number recognition based on CRNN(cnn + bi_lstm)

train:
* prepare your dataset, format like the [MJSynth data](http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz)
* modify config params in cfg.py, see default values.
* python train.py

predict:
* specify test images dir
* python predict.py