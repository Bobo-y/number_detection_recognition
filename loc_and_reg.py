from keras.models import load_model
from PIL import Image, ImageDraw
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from util import *
import numpy as np
import os
import cv2
import keras.backend as ktf
import cfg


def location(model, img_path, pixel_threshold):
    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.image_size)
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = model.predict(x)
    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)
    results = []
    with Image.open(img_path) as im:
        d_wight, d_height = resize_image(im, cfg.image_size)
        scale_ratio_w = d_wight / im.width
        scale_ratio_h = d_height / im.height
        im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        for score, geo, s in zip(quad_scores, quad_after_nms,
                                 range(len(quad_scores))):
            if np.amin(score) > 0:
                rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
                im = crop_rectangle(im, rescaled_geo)
                results.append(im)
    return results


def recognition(model, img):
    img_size = img.shape
    if (img_size[1] / img_size[0] * 1.0) < 6:
        img_reshape = cv2.resize(img, (int(31.0 / img_size[0] * img_size[1]), cfg.height))

        mat_ori = np.zeros((cfg.height, cfg.width - int(31.0 / img_size[0] * img_size[1]), 3), dtype=np.uint8)
        out_img = np.concatenate([img_reshape, mat_ori], axis=1).transpose([1, 0, 2])
    else:
        out_img = cv2.resize(img, (cfg.width, cfg.height), interpolation=cv2.INTER_CUBIC)
        out_img = np.asarray(out_img)
        out_img = out_img.transpose([1, 0, 2])

    y_pred = model.predict(np.expand_dims(out_img, axis=0))
    shape = y_pred[:, 2:, :].shape
    ctc_decode = ktf.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
    out = ktf.get_value(ctc_decode)[:, :cfg.label_len]
    result = ''.join([cfg.characters[k] for k in out[0]])
    return result


def main():
    location_model = load_model(cfg.location_model)
    recognition_model = load_model(cfg.recognition_model)
    imgs = os.listdir('test')
    result_txt = open('result.txt', 'a+')
    for im in imgs:
        img_path = os.path.join('test', im)
        ims_re = location(location_model, img_path, cfg.pixel_threshold)
        if len(ims_re) > 0:
            for i in range(len(ims_re)):
                re_text = recognition(recognition_model, ims_re[i])
                result = im + " : " + re_text + "\n"
                result_txt.write(result)
    result_txt.close()


main()