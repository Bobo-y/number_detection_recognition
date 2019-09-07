import os
import cv2
import numpy as np
import warnings
from network import CRNN
import cfg
import sys
import keras.backend as ktf


warnings.filterwarnings('ignore')
_, model = CRNN(cfg.width, cfg.height, cfg.label_len, cfg.characters).network()
model.load_weights(cfg.save_model_path)
# model.summary()


def predict(infer_model, img_path):
    img = cv2.imread(img_path)
    img_size = img.shape
    if (img_size[1] / img_size[0] * 1.0) < 6:
        img_reshape = cv2.resize(img, (int(31.0 / img_size[0] * img_size[1]), cfg.height))

        mat_ori = np.zeros((cfg.height, cfg.width - int(31.0 / img_size[0] * img_size[1]), 3), dtype=np.uint8)
        out_img = np.concatenate([img_reshape, mat_ori], axis=1).transpose([1, 0, 2])
    else:
        out_img = cv2.resize(img, (cfg.width, cfg.height), interpolation=cv2.INTER_CUBIC)
        out_img = np.asarray(out_img)
        out_img = out_img.transpose([1, 0, 2])

    y_pred = infer_model.predict(np.expand_dims(out_img, axis=0))
    shape = y_pred[:, 2:, :].shape
    ctc_decode = ktf.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
    out = ktf.get_value(ctc_decode)[:, :cfg.label_len]
    result = ''.join([cfg.characters[k] for k in out[0]])
    return result


def main():
    imgs_list = os.listdir('test')
    result_txt = open('result.txt', 'a+')
    for img_name in imgs_list:
        result = predict(model, os.path.join('test', img_name))
        result = img_name + " : " + result + "\n"
        print(result)
        result_txt.write(result)
    result_txt.close()


if __name__ == '__main__':
    main(sys.argv)

